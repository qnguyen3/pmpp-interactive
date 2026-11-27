import { useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { useNarrow } from '../../lib/narrow';
import { LabsPage, Reveal, benchRow, defineLabs, measure, type BenchSet } from '../../components/LabKit';
import { Bench, Beyond, Source, Today } from '../../components/notes';
import { fmt, num, type BenchRow, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { FEATURES, TARGETS, summarize, type Cell, type FeatureId, type Status } from '../../sim/gemmhw';

// ---------------------------------------------------------------- questions
export const questionsS159: Record<string, Question> = {
  'ch15/15.9/q-wgmma': {
    kind: 'choice',
    id: 'ch15/15.9/q-wgmma',
    prompt: (
      <Translated vi={<>Một kernel được build cho sm_90a với WGMMA có chạy được trên GPU compute capability 10.0 không?</>}>
        <>Does a kernel built for sm_90a with WGMMA run on a compute capability 10.0 GPU?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Không: mã build cho một target \"a\" không thể chạy trên các thế hệ sau."}>{'No: code built for an "a" target cannot run on later generations.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Có: GPU mới hơn chạy được mã build cho GPU cũ hơn, và 10.0 mới hơn 9.0."}>{'Yes: newer GPUs run code built for older ones, and 10.0 is newer than 9.0.'}</Translated>,
        why: <Translated vi={"Đó là quy tắc thông thường, và các target \"a\" là ngoại lệ: PTX cho chúng \"cannot be run on later generation devices\" (không thể chạy trên thiết bị thế hệ sau)."}>{'That is the usual rule, and "a" targets are the exception: PTX for them "cannot be run on later generation devices".'}</Translated>,
      },
      {
        label: <Translated vi={"Có, nhưng chậm hơn, qua đường mma mức warp mà 10.0 giữ lại cho mã cũ."}>{'Yes, but more slowly, through the warp-level mma path that 10.0 keeps for older code.'}</Translated>,
        why: <Translated vi={"Quy tắc cho các target \"a\" là mã như vậy hoàn toàn không thể chạy trên các thế hệ sau."}>{'The rule for "a" targets is that such code cannot be run on later generations at all.'}</Translated>,
      },
      {
        label: <Translated vi={"Chỉ khi nó cũng được build cho sm_90, thứ mà mọi bản build sm_90a đều bao gồm."}>{'Only if it was also built for sm_90, which every sm_90a build includes.'}</Translated>,
        why: <Translated vi={"Một bản build sm_90 hoàn toàn không thể chứa WGMMA: nó \"Requires sm_90a\" (cần sm_90a)."}>{'An sm_90 build cannot contain WGMMA at all: it "Requires sm_90a".'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Các target riêng cho kiến trúc như sm_90a "do not follow the onion layer model" (không theo mô hình lớp hành). Thứ tương
            ứng trên Blackwell là tcgen05 với tensor memory. Một kernel muốn dùng tensor core trên cả Hopper lẫn Blackwell cho trung
            tâm dữ liệu cần hai đường mã, hoặc một thư viện có sẵn chúng.
          </p>
        }
      >
        <p>
          Architecture-specific targets such as sm_90a "do not follow the onion layer model". Blackwell&rsquo;s counterpart
          is tcgen05 with tensor memory. A kernel that wants tensor cores on both Hopper and data-centre Blackwell needs two
          code paths, or a library that has them.
        </p>
      </Translated>
    ),
    source: 'ch15-facts A6 (PTX ISA 9.5)', // i18n-ignore: citation only
  },
  'ch15/15.9/q-tmem-rtx': {
    kind: 'choice',
    id: 'ch15/15.9/q-tmem-rtx',
    prompt: (
      <Translated
        vi={
          <>
            Một binary CUDA phải chạy GEMM của nó trên phép nhân tensor core lớn nhất mà mỗi GPU cung cấp, trên compute capability
            9.0, 10.0 và 12.x (RTX 50). Nó phải mang theo những đường mã nào?
          </>
        }
      >
        <>
          One CUDA binary must run its GEMM on the largest tensor-core multiply each GPU offers, on compute capability
          9.0, 10.0 and 12.x (RTX 50). Which code paths does it have to carry?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Ba: WGMMA trong bản build sm_90a, tcgen05 với TMEM trong bản build sm_100a, mma mức warp cho 12.x, chọn lúc chạy."}>{'Three: WGMMA in an sm_90a build, tcgen05 with TMEM in an sm_100a build, warp-level mma for 12.x, chosen at run time.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Một bản build sm_90a với WGMMA: GPU mới hơn chạy được mã build cho GPU cũ hơn."}>{'One sm_90a build with WGMMA: newer GPUs run code built for older ones.'}</Translated>,
        why: <Translated vi={"Đó là quy tắc thông thường, và các target \"a\" là ngoại lệ của nó: mã của chúng \"cannot be run on later generation devices\" (không thể chạy trên thiết bị thế hệ sau)."}>{'That is the usual rule, and "a" targets are its exception: their code "cannot be run on later generation devices".'}</Translated>,
      },
      {
        label: <Translated vi={"Hai: WGMMA cho 9.0, và tcgen05 với TMEM cho cả hai họ Blackwell, 10.0 và 12.x."}>{'Two: WGMMA for 9.0, and tcgen05 with TMEM for both Blackwell families, 10.0 and 12.x.'}</Translated>,
        why: <Translated vi={"12.x cũng là Blackwell, nhưng sm_120 không nằm trong các target của tcgen05. Nó không có TMEM."}>{'12.x is Blackwell too, but sm_120 is not among tcgen05\'s targets. It has no TMEM.'}</Translated>,
      },
      {
        label: <Translated vi={"Một đường mma mức warp: nó chạy trên cả ba và dùng tensor core của chúng, nên phủ được mọi GPU."}>{'One warp-level mma path: it runs on all three and uses their tensor cores, so it covers every GPU.'}</Translated>,
        why: <Translated vi={"Nó đúng là chạy ở mọi nơi, nên nó là đường dự phòng. Nhưng nó giữ mọi toán hạng và biến tích lũy trong thanh ghi, nên trên 9.0 và 10.0 nó bỏ phí phép nhân lớn hơn."}>{'It does run everywhere, which makes it the fallback. But it keeps every operand and the accumulator in registers, so on 9.0 and 10.0 it leaves the larger multiply unused.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            WGMMA chỉ có trong bản build sm_90a, và mã build cho một target "a" không chạy trên các thế hệ sau. tcgen05 và TMEM
            thuộc về các họ sm_100 và sm_110, không phải 12.x. Vì vậy 12.x lùi về mma mức warp. Các thư viện mang theo tất cả những
            đường này và chọn lúc chạy, đó là một phần lý do gọi thư viện là lựa chọn mặc định.
          </p>
        }
      >
        <p>
          WGMMA exists only in the sm_90a build, and code built for an "a" target does not run on later generations.
          tcgen05 and TMEM belong to the sm_100 and sm_110 families, not to 12.x. So 12.x falls back to warp-level mma.
          Libraries carry all of these and choose at run time, which is part of why calling them is the default.
        </p>
      </Translated>
    ),
    source: 'ch15-facts A6 (PTX ISA 9.5 target notes); compute capabilities from ch04-facts A12',
    vi: { source: "ch15-facts A6 (ghi chú target của PTX ISA 9.5); compute capability từ ch04-facts A12" },
  },
  'ch15/15.9/q-ldgsts': {
    kind: 'choice',
    id: 'ch15/15.9/q-ldgsts',
    prompt: (
      <Translated
        vi={
          <>
            Bạn đã viết lại <code>loadTile</code> với <code>cuda::memcpy_async</code>, và biểu đồ bộ nhớ của Nsight Compute vẫn cho
            thấy việc lấp đầy bộ nhớ chia sẻ đi qua tệp thanh ghi. Bạn làm gì tiếp theo?
          </>
        }
      >
        <>
          You rewrote <code>loadTile</code> with <code>cuda::memcpy_async</code>, and Nsight Compute&rsquo;s memory chart
          still shows the shared-memory fills going through the register file. What do you do next?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Build cho compute capability 8.0 trở lên và tìm LDGSTS trong SASS; nếu không có, dùng các primitive __pipeline_memcpy_async hoặc CUTLASS."}>{'Build for compute capability 8.0 or later and look for LDGSTS in the SASS; if it is missing, use the __pipeline_memcpy_async primitives or CUTLASS.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Không làm gì: khi bật tối ưu hóa, nvcc tự gộp mỗi lệnh nạp vector và lệnh ghi chia sẻ thành một LDGSTS."}>{'Nothing: with optimisation on, nvcc fuses each vector load and shared store into one LDGSTS by itself.'}</Translated>,
        why: <Translated vi={"Sách nói vậy, nhưng không trang nào của NVIDIA mà chúng tôi đọc nói vậy, và hướng dẫn của NVIDIA nói trình biên dịch có thể phát ra các cặp LDG và STS."}>{'The book says so, but no NVIDIA page we read does, and NVIDIA\'s guide says the compiler may emit LDG and STS pairs.'}</Translated>,
      },
      {
        label: <Translated vi={"Thêm một __syncthreads() sau mỗi lần sao chép, để các lần sao chép hoàn tất trước khi ô chia sẻ được đọc."}>{'Add a __syncthreads() after each copy, so the copies complete before the shared tile is read.'}</Translated>,
        why: <Translated vi={"Một rào chắn sau lần chờ là cần cho tính đúng đắn (ch06-facts A16), nhưng nó không thay đổi lệnh nào thực hiện việc sao chép."}>{'A barrier after the wait is needed for correctness (ch06-facts A16), but it does not change which instructions do the copy.'}</Translated>,
      },
      {
        label: <Translated vi={"Không làm gì: biểu đồ vẽ mọi lệnh ghi bộ nhớ chia sẻ đi qua tệp thanh ghi, bất kể lệnh nào tạo ra nó."}>{'Nothing: the chart draws every shared-memory store through the register file, whatever instruction made it.'}</Translated>,
        why: <Translated vi={"Hướng dẫn chẩn đoán của NVIDIA đọc đúng bức tranh này là \"the async-copy path is not actually being exercised\" (đường sao chép bất đồng bộ thực ra không được dùng)."}>{'NVIDIA\'s triage guide reads exactly this picture as "the async-copy path is not actually being exercised".'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            LDGSTS cần compute capability 8.0. <code>cuda::memcpy_async</code> có thể tạo ra nó nhưng không được bảo đảm; hướng dẫn
            nói các primitive cấp thấp hơn mới bảo đảm điều đó. Biểu đồ và SASS là cách bạn biết mình nhận được gì.
          </p>
        }
      >
        <p>
          LDGSTS needs compute capability 8.0. <code>cuda::memcpy_async</code> may produce it but is not guaranteed to;
          the guide says the lower-level primitives ensure it. The chart and the SASS are how you know which you got.
        </p>
      </Translated>
    ),
    source: 'ch15-facts A7; ch06-facts A16', // i18n-ignore: citation only
  },
  'ch15/15.9/q-tma-2d': {
    kind: 'choice',
    id: 'ch15/15.9/q-tma-2d',
    prompt: (
      <Translated vi={<><code>cuda::memcpy_async</code> có nạp được ô 128 × 8 của A qua TMA không?</>}>
        <>Can <code>cuda::memcpy_async</code> fetch the 128 × 8 tile of A through TMA?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Không: nó chỉ dùng tới TMA cho sao chép khối 1D; một ô 2D cần một tensor map và cp_async_bulk_tensor."}>{'No: it reaches TMA only for 1D bulk copies; a 2D tile needs a tensor map and cp_async_bulk_tensor.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Có, trên compute capability 9.0 trở lên: ở đó cuda::memcpy_async giao mọi lần sao chép ô bộ nhớ chia sẻ cho TMA."}>{'Yes, on compute capability 9.0 and later: there cuda::memcpy_async hands any shared-memory tile copy to TMA.'}</Translated>,
        why: <Translated vi={"Phần cứng thì có; API thì không phải cái sách nêu tên. Một ô 128 × 8 của một ma trận rộng hơn không phải một dải liên tục."}>{'The hardware is there; the API is not the one the book names. A 128 × 8 tile of a wider matrix is not one contiguous run.'}</Translated>,
      },
      {
        label: <Translated vi={"Có, nếu ô được đệm thành lũy thừa của hai để nó thành một dải liên tục."}>{'Yes, if the tile is padded to a power of two so that it is one contiguous run.'}</Translated>,
        why: <Translated vi={"Phần đệm không biến một ô 2D có bước nhảy thành một lần sao chép 1D liên tục."}>{'Padding does not make a strided 2D tile into a 1D contiguous copy.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Tám float mỗi hàng của ô nằm cách nhau K float trong bộ nhớ toàn cục. Một lần sao chép tensor khối cần một mô tả về cách
            bố trí đó, tức tensor map, được tạo trên host bằng <code>cuTensorMapEncodeTiled</code> và truyền cho kernel, kernel sao
            chép bằng <code>cuda::ptx::cp_async_bulk_tensor</code>.
          </p>
        }
      >
        <p>
          The tile's eight floats per row sit K floats apart in global memory. A bulk tensor copy needs a description of
          that layout, the tensor map, built on the host with <code>cuTensorMapEncodeTiled</code> and passed to the
          kernel, which copies with <code>cuda::ptx::cp_async_bulk_tensor</code>.
        </p>
      </Translated>
    ),
    source: 'ch15-facts A8', // i18n-ignore: citation only
  },
  'ch15/15.9/q-progression': {
    kind: 'choice',
    id: 'ch15/15.9/q-progression',
    prompt: (
      <Translated vi={<>Chuỗi tiến hóa từ WMMA đến WGMMA đến TMEM nhắm vào điều gì?</>}>
        <>What does the progression from WMMA to WGMMA to TMEM attack?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Áp lực thanh ghi, giới hạn của mục 15.7."}>{'Register pressure, the limit of section 15.7.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Băng thông bộ nhớ toàn cục, bằng cách nạp các ô toán hạng thành những phần lớn hơn."}>{'Global-memory bandwidth, by fetching operand tiles in larger pieces.'}</Translated>,
        why: <Translated vi={"Đó là việc của ô lớn hơn, và TMA. Những thứ này thay đổi nơi các toán hạng của phép nhân nằm."}>{'That is what larger tiles, and TMA, are for. These change where the operands of the multiply live.'}</Translated>,
      },
      {
        label: <Translated vi={"Xung đột bank trong bộ nhớ chia sẻ, bằng cách đọc toán hạng theo đơn vị rộng hơn."}>{'Bank conflicts in shared memory, by reading operands in wider units.'}</Translated>,
        why: <Translated vi={"Phần đệm (mục 15.6) xử lý chúng; các giao diện này là về thanh ghi."}>{'Padding (section 15.6) handles those; these interfaces are about registers.'}</Translated>,
      },
      {
        label: <Translated vi={"Phân kỳ warp, vì giờ cả một nhóm warp phát một lệnh."}>{'Warp divergence, since a whole warp group now issues one instruction.'}</Translated>,
        why: <Translated vi={"Không cái nào trong ba thay đổi cách xử lý rẽ nhánh. Chúng thay đổi nơi toán hạng và kết quả nằm."}>{'None of the three changes how branches are handled. They change where the operands and results live.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            MMA mức warp giữ mọi đầu vào và đầu ra trong thanh ghi. WGMMA nạp các ô toán hạng của nó từ bộ nhớ chia sẻ. Với tcgen05,
            kết quả được ghi vào TMEM. Mỗi bước giải phóng thanh ghi, thứ mà mục 15.7 thấy là tài nguyên khan hiếm.
          </p>
        }
      >
        <p>
          Warp-level MMA keeps every input and output in registers. WGMMA loads its operand tiles from shared memory. With
          tcgen05 the result is written to TMEM. Each step frees registers that section 15.7 found to be the scarce resource.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.9, p. 369; ch15-facts A6 (NVIDIA\'s triage guide)',
    vi: { source: "ch. 15 §15.9, p. 369; ch15-facts A6 (hướng dẫn chẩn đoán của NVIDIA)" },
  },
  'ch15/15.9/q-fp32-default': {
    kind: 'choice',
    id: 'ch15/15.9/q-fp32-default',
    prompt: (
      <Translated
        vi={
          <>
            Trên một GPU có tensor core, kernel M3 FP32 của bạn chạy bằng một phần ba tốc độ của <code>cublasSgemm</code> ở batch
            256, với thư viện ở chế độ toán mặc định. Bạn có thể loại bỏ lời giải thích nào mà không cần đo?
          </>
        }
      >
        <>
          On a GPU with tensor cores your FP32 M3 kernel runs at a third of <code>cublasSgemm</code>&rsquo;s speed at
          batch 256, the library in its default math mode. Which explanation can you rule out without measuring?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Rằng cuBLAS âm thầm tính bằng TF32 trên tensor core của nó."}>{'That cuBLAS quietly computes in TF32 on its tensor cores.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Rằng cuBLAS chọn một kernel được tinh chỉnh cho hình dạng này."}>{'That cuBLAS picked a kernel tuned for this shape.'}</Translated>,
        why: <Translated vi={"Không điều gì bạn biết loại bỏ được nó, nên nó vẫn là một ứng viên; một bản profile cho biết kernel nào đã chạy sẽ trả lời bạn."}>{'Nothing you know rules it out, so it stays a live candidate; a profile of which kernel ran would tell you.'}</Translated>,
      },
      {
        label: <Translated vi={"Rằng cuBLAS chồng lấp việc nạp ô với tính toán qua các lần sao chép bất đồng bộ."}>{'That cuBLAS overlaps its tile loads with arithmetic through asynchronous copies.'}</Translated>,
        why: <Translated vi={"Ở nơi phần cứng có LDGSTS hoặc TMA, một thư viện có thể dùng chúng. Một ứng viên."}>{'Where the hardware has LDGSTS or TMA, a library can use them. A candidate.'}</Translated>,
      },
      {
        label: <Translated vi={"Rằng cuBLAS tinh chỉnh hình dạng ô và độ sâu pipeline vượt xa kernel của sách."}>{'That cuBLAS tunes tile shapes and pipeline depth beyond the book\'s kernel.'}</Translated>,
        why: <Translated vi={"Cũng là một ứng viên: kernel của sách là một thiết kế cố định."}>{'Also a candidate: the book\'s kernel is one fixed design.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Chế độ mặc định của cuBLAS giữ ít nhất độ chính xác bạn yêu cầu; TF32 và chế độ giả lập BF16x9 phải chủ động bật, cái sau
            cũng có thể bật qua biến môi trường trên compute capability 10.x, nên hãy kiểm tra chế độ toán mà lab 15.6 in ra. XF32 của
            hipBLAS và độ chính xác giảm của MPS cũng phải chủ động bật. Thứ còn lại là kỹ thuật.
          </p>
        }
      >
        <p>
          cuBLAS&rsquo;s default mode keeps at least the precision you asked for; TF32 and the BF16x9 emulation are
          opt-in, the latter also through environment variables on compute capability 10.x, so check the math mode lab
          15.6 prints. hipBLAS&rsquo;s XF32 and MPS&rsquo;s reduced precision are opt-in too. What is left is technique.
        </p>
      </Translated>
    ),
    source: 'ch15-facts A10, B6, C6', // i18n-ignore: citation only
  },
  'ch15/15.9/q-rdna': {
    kind: 'choice',
    id: 'ch15/15.9/q-rdna',
    prompt: (
      <Translated
        vi={
          <>
            GEMM HIP của bạn phải dùng lệnh ma trận trên một GPU CDNA và trên một GPU RDNA3. Bảng tính năng HIP không cho thấy Matrix
            Core nào trên RDNA. Bạn xây gì?
          </>
        }
      >
        <>
          Your HIP GEMM must use matrix instructions on a CDNA GPU and on an RDNA3 one. The HIP features table shows no
          Matrix Cores on RDNA. What do you build?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Hai đường chọn theo target lúc biên dịch: builtin MFMA cho CDNA (gfx9), builtin WMMA cho RDNA3 (gfx11)."}>{'Two paths chosen per target at compile time: MFMA builtins for CDNA (gfx9), WMMA builtins for RDNA3 (gfx11).'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Builtin MFMA ở mọi nơi: chúng là lệnh ma trận của AMD trên mọi GPU hãng làm ra."}>{'MFMA builtins everywhere: they are AMD\'s matrix instructions on every GPU it makes.'}</Translated>,
        why: <Translated vi={"MFMA thuộc về các bộ tính năng CDNA (gfx9). RDNA3 có một họ khác, các builtin WMMA."}>{'MFMA belongs to the CDNA (gfx9) feature sets. RDNA3 has a different family, the WMMA builtins.'}</Translated>,
      },
      {
        label: <Translated vi={"Một đường ma trận chỉ cho CDNA và FMA thường trên RDNA3, vì bảng tính năng không liệt kê Matrix Core nào ở đó."}>{'A matrix path for CDNA only and plain FMAs on RDNA3, since the features table lists no Matrix Cores there.'}</Translated>,
        why: <Translated vi={"Bảng và trình biên dịch không khớp nhau. Bộ tính năng gfx11 và gfx12 của trình biên dịch có WMMA, và llama.cpp gọi các builtin đó."}>{'The table and the compiler disagree. The compiler\'s gfx11 and gfx12 feature sets include WMMA, and llama.cpp calls those builtins.'}</Translated>,
      },
      {
        label: <Translated vi={"Một đường viết bằng nvcuda::wmma, biên dịch qua header HIP của khóa học."}>{'One path written with nvcuda::wmma, compiled through the course\'s HIP header.'}</Translated>,
        why: <Translated vi={"HIP không có nvcuda::wmma. Đường C++ trên AMD là rocWMMA, viết khác, nên mã ma trận nằm ngoài tệp .cu dùng chung."}>{'HIP has no nvcuda::wmma. The C++ route on AMD is rocWMMA, spelled differently, so matrix code stays out of the shared .cu.'}</Translated>,
      },
    ],
    explain: (
      <Translated vi={<p>Khi bảng của một nhà cung cấp và trình biên dịch của họ không khớp nhau, trình biên dịch quyết định mã của bạn dùng được gì. <code>mma.cuh</code> của llama.cpp tách đúng theo cách này.</p>}>
        <p>Where a vendor&rsquo;s table and its compiler disagree, the compiler decides what your code can use. llama.cpp&rsquo;s <code>mma.cuh</code> makes exactly this split.</p>
      </Translated>
    ),
    source: 'ch15-facts B4, D1; infra-facts B12', // i18n-ignore: citation only
  },
  'ch15/15.9/q-handwrite': {
    kind: 'choice',
    id: 'ch15/15.9/q-handwrite',
    prompt: (
      <Translated vi={<>Khi nào bạn nên tự viết GEMM thay vì gọi thư viện?</>}>
        <>When should you write your own GEMM rather than call the library?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Khi thư viện thiếu phép toán đó: một định dạng lượng tử hóa, một epilogue được gộp, một hình dạng mà nó phục vụ kém."}>{'When the library lacks the operation: a quantised format, a fused epilogue, a shape it serves badly.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Cho mọi mô hình bạn triển khai: một kernel chuyên biệt cho hình dạng của nó thắng một thư viện tổng quát."}>{'For every model you deploy: a kernel specialised to its shapes beats a general library.'}</Translated>,
        why: <Translated vi={"Các thư viện đã mang mọi tối ưu hóa trong chương này và hơn thế (tensor core, TMEM, sao chép bất đồng bộ)."}>{'The libraries already carry every optimisation in this chapter and more (tensor cores, TMEM, asynchronous copies).'}</Translated>,
      },
      {
        label: <Translated vi={"Để học, không phải để phát hành: thư viện của nhà cung cấp phủ mọi phép toán một mô hình cần, nên các kỹ thuật chỉ có ích ở các kernel khác."}>{'To learn, not to ship: the vendor libraries cover every operation a model needs, so the techniques pay off in other kernels.'}</Translated>,
        why: <Translated vi={"Sách nói lập trình viên hiếm khi viết GEMM từ đầu, không phải không bao giờ: một thư viện không thể chạy một phép toán mà nó không có."}>{'The book says programmers rarely write GEMM from scratch, not never: a library cannot run an operation it does not have.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Ý của sách là các thư viện hiện thân cho chương này (p. 368). Ý của khóa học này, cho mục tiêu của bạn: lý do để tự viết
            là một định dạng hay phép gộp mà thư viện không cung cấp, đó là hướng module X2 đi tới.
          </p>
        }
      >
        <p>
          The book's point is that the libraries embody this chapter (p. 368). Ours, for your goals: the reason to hand-write
          is a format or fusion the library does not offer, which is where module X2 goes.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.9, p. 368; when to hand-write is our reading',
    vi: { source: "ch. 15 §15.9, p. 368; khi nào nên tự viết là cách hiểu của khóa học này" },
  },
};

// ---------------------------------------------------------------- explorer: what can this target use?
const STATUS_TEXT: Record<Status, string> = {
  get yes() {
    return tr('yes', 'có');
  },
  get no() {
    return tr('no', 'không');
  },
  get undocumented() {
    return tr('not documented', 'không được công bố');
  },
};
const STATUS_CHIP: Record<Status, string> = { yes: 'ok', no: 'fail', undocumented: 'global' };

function CellView({ c }: { c: Cell }) {
  return (
    <>
      <span className={'chip ' + STATUS_CHIP[c.status]} data-status={c.status}>{STATUS_TEXT[c.status]}</span>{' '}
      <span>{c.what}</span> <span className="muted" style={{ fontSize: '0.8rem' }}>({c.items.join(', ')})</span>
    </>
  );
}

function HardwareBench() {
  const [a, setA] = useState('cuda-9.0a');
  const [b, setB] = useState('');
  const A = summarize(a);
  const B = b ? summarize(b) : null;
  const narrow = useNarrow();
  const differs = (f: FeatureId) => !!B && B.cells[f].status !== A.cells[f].status;
  const pick = (v: string, set: (x: string) => void, label: string, allowNone: boolean) => (
    <label className="field" style={{ maxWidth: '100%' }}>
      {label}
      <select value={v} aria-label={label} onChange={(e) => set(e.target.value)} style={{ maxWidth: '100%' }}>
        {allowNone ? <option value="">{tr('nothing', 'không có')}</option> : null}
        {(['CUDA', 'HIP', 'Metal'] as const).map((bk) => (
          <optgroup key={bk} label={bk}>
            {TARGETS.filter((t) => t.backend === bk).map((t) => (
              <option key={t.id} value={t.id}>{t.label}</option>
            ))}
          </optgroup>
        ))}
      </select>
    </label>
  );
  return (
    <Bench
      title={tr('What can this target use?', 'Target này dùng được gì?')}
      task={tr(
        'Find a target where WGMMA code cannot run, one without TMEM, one with undocumented copies.',
        'Tìm một target mà mã WGMMA không chạy được, một target không có TMEM, một target có các lần sao chép không được công bố.',
      )}
      foot={
        <Translated
          vi={
            <>
              Mỗi ô đến từ một bảng đã được kiểm thử (src/sim/gemmhw.ts) xây từ notes/today/ch15-facts.md. Một mục trần như A6 nằm
              trong ch15-facts; "infra", "ch03" và "ch06" là tên các tệp facts khác; "ch15 table" là bảng backend của tệp đó. "Không
              được công bố" mô tả tài liệu, không phải con chip.
            </>
          }
        >
          <>
            Every cell comes from a tested table (src/sim/gemmhw.ts) built from notes/today/ch15-facts.md. A bare item such
            as A6 is in ch15-facts; "infra", "ch03" and "ch06" name the other facts files; "ch15 table" is that file's
            backend table. "Not documented" describes the documentation, not the silicon.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        {pick(a, setA, tr('Target', 'Target'), false)}
        {pick(b, setB, tr('Compare with', 'So sánh với'), true)}
      </div>
      {A.toolkit || B?.toolkit ? (
        <p style={{ margin: '0 0 0.7rem' }} data-toolkit>
          {A.toolkit ? <><b>{A.target.label}:</b> <CellView c={A.toolkit} /></> : null}
          {B?.toolkit ? <><br /><b>{B.target.label}:</b> <CellView c={B.toolkit} /></> : null}
        </p>
      ) : null}
      {narrow ? (
        // On a phone: one card per feature, the targets stacked inside it.
        <div data-hw-cards>
          {FEATURES.map((f) => (
            <div key={f.id} className="panel" data-feature={f.id} style={{ marginBottom: '0.6rem' }}
              data-differs={B && differs(f.id) ? 1 : undefined}>
              <p style={{ margin: '0 0 0.4rem' }}>
                <b>{tr(f.label, { warpMma: 'Phép nhân ma trận mức warp', bigMma: 'Phép nhân ma trận lớn hơn (nhóm warp, tensor memory, hoặc một phép toán thư viện trong kernel)', asyncCopy: 'Sao chép bất đồng bộ từ toàn cục sang chia sẻ', bulkCopy: 'Sao chép khối tensor nhiều chiều', templateLib: 'Thư viện GEMM dạng template', vendorGemm: 'GEMM của nhà cung cấp để so sánh (FP32)' }[f.id])}</b> <span className="muted">({f.book})</span>
                {B && differs(f.id) ? <b className="ui" style={{ fontSize: '0.8rem' }}> <Translated vi="khác nhau">differs</Translated></b> : null}
              </p>
              <p style={{ margin: 0 }} data-col="a">
                {B ? <span className="muted">{A.target.label}: </span> : null}
                <CellView c={A.cells[f.id]} />
              </p>
              {B ? (
                <p style={{ margin: '0.4rem 0 0' }} data-col="b">
                  <span className="muted">{B.target.label}: </span>
                  <CellView c={B.cells[f.id]} />
                </p>
              ) : null}
            </div>
          ))}
        </div>
      ) : (
        <table className="data" style={{ margin: 0 }} data-hw-table>
          <thead>
            <tr>
              <th>
                <Translated vi="Tính năng (tên trong sách)">Feature (the book's name)</Translated>
              </th>
              <th>{A.target.label}</th>
              {B ? <th>{B.target.label}</th> : null}
            </tr>
          </thead>
          <tbody>
            {FEATURES.map((f) => (
              <tr key={f.id} data-feature={f.id} data-differs={B && differs(f.id) ? 1 : undefined}>
                <td style={{ minWidth: '9rem' }}>
                  {tr(f.label, { warpMma: 'Phép nhân ma trận mức warp', bigMma: 'Phép nhân ma trận lớn hơn (nhóm warp, tensor memory, hoặc một phép toán thư viện trong kernel)', asyncCopy: 'Sao chép bất đồng bộ từ toàn cục sang chia sẻ', bulkCopy: 'Sao chép khối tensor nhiều chiều', templateLib: 'Thư viện GEMM dạng template', vendorGemm: 'GEMM của nhà cung cấp để so sánh (FP32)' }[f.id])} <span className="muted">({f.book})</span>
                  {B && differs(f.id) ? <><br /><b className="ui" style={{ fontSize: '0.8rem' }}><Translated vi="khác nhau">differs</Translated></b></> : null}
                </td>
                <td data-col="a"><CellView c={A.cells[f.id]} /></td>
                {B ? <td data-col="b"><CellView c={B.cells[f.id]} /></td> : null}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- your M3 against the library
// Lab 15.6 (milestone M3, labs/ch15/lab6_linear) records rows "m2", "m3" and "vendor" with
// batch, in and out, on the 1024 -> 4096 layer and on a 4096 -> 4096 one; this page reads the
// first. The m, k, n form is how labs 15.1 to 15.5 report the same layer.
const LABS = defineLabs('ch15', [{ num: '15.6', dir: 'lab6_linear', name: 'M3 fast linear layer', vi: { name: 'lớp tuyến tính nhanh M3' } }] as const);
const [M3] = LABS;
const BATCH = 256;
const atBatch = (r: BenchRow) => num(r.m) === BATCH && num(r.k) === 1024 && num(r.n) === 4096;
const rowAt = (s: BenchSet, name: string) =>
  benchRow(s, name, { batch: BATCH, in: 1024, out: 4096 }) ?? s.rows.find((r) => r.name === name && atBatch(r));

/** Speed of a row: GFLOP/s if reported, else 1 / seconds (same problem, so the ratio is the same). */
const speed = (r: BenchRow | undefined) => num(r?.gflops) ?? (num(r?.seconds) ? 1 / num(r?.seconds)! : undefined);

function percentOfVendor(reports: Reports) {
  return measure(reports, M3, (s: BenchSet) => {
    const mine = rowAt(s, 'm3');
    const lib = rowAt(s, 'vendor');
    const a = speed(mine);
    const v = speed(lib);
    if (!a || !v) return null;
    const unit = num(mine?.gflops) != null && num(lib?.gflops) != null ? 'GFLOP/s' : tr('runs per second', 'lần chạy mỗi giây');
    return { value: (100 * a) / v, detail: tr(`${fmt(a)} against ${fmt(v)} ${unit} at batch ${BATCH}`, `${fmt(a)} so với ${fmt(v)} ${unit} ở batch ${BATCH}`) };
  });
}

export function S159() {
  return (
    <>
      <Translated
        vi={
          <p>
            Sách kết thúc chương bằng việc chỉ ra rằng lập trình viên hiếm khi viết kernel này từ đầu: các nhà cung cấp phát hành thư
            viện với tất cả các tối ưu hóa này và hơn thế, trên những tính năng phần cứng mà sách chỉ nêu tên. Trang này phân loại
            những gì thư viện và phần cứng bổ sung, phần nào trong đó mỗi backend cho phép kernel của chính bạn dùng hôm nay, và
            kernel của bạn còn cách thư viện bao xa.
          </p>
        }
      >
        <p>
          The book ends the chapter by pointing out that programmers rarely write this kernel from scratch: vendors ship
          libraries with all of these optimisations and more, on hardware features the book only names. This page sorts
          out what the libraries and the hardware add, which of it each backend lets your own kernel use today, and how far
          your kernel is from the library.
        </p>
      </Translated>

      <h2>
        <Translated vi="Thư viện">Libraries</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Các thư viện của NVIDIA là cuBLAS, bản hiện thực giao diện BLAS tiêu chuẩn của hãng, có cả phép nhân ma trận; cuDNN, các
            hàm cho mạng nơ-ron sâu mà bên dưới là nhân ma trận; CUTLASS, một thư viện mã nguồn mở để đưa phép nhân ma trận vào các
            kernel lớn hơn ở nhiều cấp; và cuTile, nơi bạn viết mã mảng còn trình biên dịch lo việc tối ưu hóa.
          </p>
        }
      >
        <p>
          NVIDIA's are cuBLAS, its implementation of the standard BLAS interface, matrix multiply included; cuDNN, routines
          for deep neural networks that multiply matrices underneath; CUTLASS, an open-source library for building matrix
          multiplication into larger kernels at several levels; and cuTile, where you write array code and the compiler does
          the optimising.
        </p>
      </Translated>
      <Source section="15.9" pages="368" />
      <Translated
        vi={
          <p>
            Với tất cả những thứ đó sẵn có, lập trình viên hiếm khi viết một GEMM từ đầu. Dù vậy các kỹ thuật vẫn đáng có, vì chúng áp
            dụng được xa hơn GEMM nhiều.
          </p>
        }
      >
        <p>
          With all that available, programmers rarely write a GEMM from scratch. The techniques are worth having anyway,
          because they apply far beyond GEMM.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch15-facts A9, B6, B7, C5, C6; infra-facts A12)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              cuTile giờ là CUDA Tile (Python; C++ từ toolkit 13.3). AMD: hipBLAS, rocBLAS, Composable Kernel, rocWMMA. Apple: MPS và
              Metal Performance Primitives, không có thư viện GEMM dạng template.
            </p>
          }
        >
          <p>
            cuTile is now CUDA Tile (Python; C++ from toolkit 13.3). AMD: hipBLAS, rocBLAS, Composable Kernel, rocWMMA.
            Apple: MPS and Metal Performance Primitives, no template GEMM library.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Có một lý do để tự viết mà sách để ngầm, và đó là lý do quan trọng cho công việc của bạn: thư viện không có phép toán đó.
            Một định dạng trọng số lượng tử hóa phải được bung ra thành các dải thanh ghi của mục 15.4, một epilogue gộp vào lần ghi,
            một hình dạng mà thư viện phục vụ kém. Ngoài ra, hãy gọi thư viện. <a href={href('x2')}>Module X2</a> viết loại đầu tiên.
          </p>
        }
      >
        <p>
          There is a reason to write your own that the book leaves implicit, and it is the one that matters for your work:
          the library does not have the operation. A quantised weight format that has to be expanded into the register
          strips of section 15.4, an epilogue fused into the store, a shape the library serves badly. Otherwise, call the
          library. <a href={href('x2')}>Module X2</a> writes the first kind.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Khi nào nên tự viết là cách hiểu của khóa học này, dựa trên nhận xét của sách rằng lập trình viên hiếm khi viết GEMM từ đầu.</p>}>
          <p>When to hand-write is our reading, built on the book's remark that programmers rarely write GEMM from scratch.</p>
        </Translated>
      </Beyond>
      <Ask q={questionsS159['ch15/15.9/q-handwrite']} />

      <h2>
        <Translated vi="Tensor core">Tensor cores</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Từ Volta trở đi, một số GPU NVIDIA có tensor core: những đơn vị mà mỗi cái thực hiện một phép nhân ma trận nhỏ trong một
            lệnh. Bạn tiếp cận chúng qua trình biên dịch, qua một giao diện đặc biệt hoặc, thường gặp nhất, qua cuBLAS, cuDNN và
            CUTLASS.
          </p>
        }
      >
        <p>
          From Volta on, some NVIDIA GPUs have tensor cores: units that each perform a small matrix multiplication in one
          instruction. You reach them through the compiler, through a special interface or, most often, through cuBLAS,
          cuDNN and CUTLASS.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 against the cuBLAS 13.5 docs and nvcc 13.5 (ch15-facts A5)',
          '2026-11-20, đối chiếu với tài liệu cuBLAS 13.5 và nvcc 13.5 (ch15-facts A5)',
        )}
      >
        <Translated
          vi={
            <p>
              Volta là compute capability 7.0. nvcc 13.5 liệt kê sm_75 trở lên, không có sm_70, dù không có câu nào của NVIDIA nói
              trình biên dịch đã bỏ nó.
            </p>
          }
        >
          <p>
            Volta is compute capability 7.0. nvcc 13.5 lists sm_75 and later, no sm_70, though no single NVIDIA sentence
            says the compiler dropped it.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Sách nêu hai lợi ích. Một lệnh giờ làm việc của nhiều lệnh, giải tỏa giới hạn về giải mã và phát lệnh. Và các đơn vị này
            có thông lượng rất cao, nhờ phần cứng mức warp để trao đổi nội dung thanh ghi và nhờ phép tính độ chính xác thấp hơn.
          </p>
        }
      >
        <p>
          The book names two gains. One instruction now does the work of many, which relieves the limit on decoding and
          issuing instructions. And the units have very high throughput, helped by warp-level hardware for exchanging
          register contents and by lower-precision arithmetic.
        </p>
      </Translated>
      <Source section="15.9" pages="368" />

      <h2>
        <Translated vi="WMMA, WGMMA và tensor memory">WMMA, WGMMA and tensor memory</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách lần theo ba thế hệ của giao diện. WMMA (warp matrix multiply and accumulate) làm việc từng warp một. WGMMA (warp group
            matrix multiply accumulate), từ Hopper, cho một nhóm warp cùng hợp tác trên một phép nhân lớn hơn. Nó có thể lấy đầu vào
            thẳng từ bộ nhớ chia sẻ, giúp tiết kiệm thanh ghi, và nó chạy bất đồng bộ với các lệnh khác, nên việc chồng lấp những lần
            chờ dài ít phụ thuộc vào trình biên dịch hơn. Blackwell thêm tensor memory, TMEM: bộ lưu trữ trên chip cho đầu vào và đầu
            ra của các phép toán tensor core, giúp giảm thêm áp lực lên thanh ghi.
          </p>
        }
      >
        <p>
          The book traces three generations of the interface. WMMA (warp matrix multiply and accumulate) works one warp at
          a time. WGMMA (warp group matrix multiply accumulate), from Hopper, has a group of warps cooperate on a larger
          multiplication. It can take its inputs straight from shared memory, which saves registers, and it runs
          asynchronously with other instructions, so overlapping long waits depends less on the compiler. Blackwell adds
          tensor memory, TMEM: on-chip storage for the inputs and outputs of tensor-core operations, which takes still more
          pressure off the registers.
        </p>
      </Translated>
      <Source section="15.9" pages="369" />
      <Translated
        vi={
          <p>
            Hãy đọc ba thế hệ như một ý tưởng. Mục 15.7 thấy rằng thanh ghi giới hạn kernel của chương này: riêng ô biến tích lũy đã
            chiếm 64 thanh ghi mỗi luồng. Mỗi thế hệ chuyển một thứ ra khỏi tệp thanh ghi. Đầu tiên đầu vào chuyển sang bộ nhớ chia
            sẻ, rồi biến tích lũy chuyển sang TMEM.
          </p>
        }
      >
        <p>
          Read the three as one idea. Section 15.7 finds that registers limit this chapter's kernel: the accumulator tile
          alone takes 64 per thread. Each generation moves something out of the register file. First the inputs go to
          shared memory, then the accumulator goes to TMEM.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 against PTX ISA 9.5 and the Nsight Compute triage guide (ch15-facts A6; infra-facts A12)',
          '2026-11-20, đối chiếu với PTX ISA 9.5 và hướng dẫn chẩn đoán của Nsight Compute (ch15-facts A6; infra-facts A12)',
        )}
      >
        <Translated
          vi={
            <p>
              WMMA vẫn được ghi trong tài liệu cho compute capability 7.0 trở lên, không phải "chỉ trước Hopper". WGMMA "Requires
              sm_90a" (cần sm_90a), và mã build cho một target "a" "cannot be run on later generation devices" (không thể chạy trên
              thiết bị thế hệ sau). tcgen05, với TMEM, chỉ nhắm vào các họ sm_100 và sm_110: không phải compute capability 12.x (RTX
              50). Tóm tắt của chính NVIDIA: MMA mức warp giữ đầu vào và đầu ra trong thanh ghi, WGMMA nạp các ô toán hạng từ bộ nhớ
              chia sẻ, và thế hệ thứ năm ghi kết quả vào TMEM.
            </p>
          }
        >
          <p>
            WMMA is still documented for compute capability 7.0 and up, not "pre-Hopper only". WGMMA "Requires sm_90a", and
            code built for an "a" target "cannot be run on later generation devices". tcgen05, with TMEM, targets the sm_100
            and sm_110 families only: not compute capability 12.x (RTX 50). NVIDIA's own summary: warp MMA keeps inputs and
            outputs in registers, WGMMA loads operand tiles from shared memory, and the fifth generation writes its result
            to TMEM.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS159['ch15/15.9/q-progression']} />
      <Ask q={questionsS159['ch15/15.9/q-wgmma']} />
      <Ask q={questionsS159['ch15/15.9/q-tmem-rtx']} />

      <h2>
        <Translated vi="Di chuyển dữ liệu không qua thanh ghi">Moving data without the registers</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Khi tensor core làm phép tính nhanh, giới hạn có thể chuyển sang việc đưa dữ liệu giữa bộ nhớ toàn cục và bộ nhớ chia sẻ:
            băng thông, các lệnh tốn cho việc di chuyển nó, và các thanh ghi nó chiếm trên đường đi. Lệnh ghi vector (mục 15.5) giúp
            ích: mỗi luồng ghi nhiều hơn mỗi lệnh, ít chi phí phụ hơn và hợp nhất truy cập tốt hơn. Nhưng sao chép từ bộ nhớ toàn cục
            sang bộ nhớ chia sẻ bằng một lệnh nạp vector và một lệnh ghi vector vẫn tốn hai bước qua thanh ghi.
          </p>
        }
      >
        <p>
          Once tensor cores make the arithmetic fast, the limit can move to getting data between global and shared memory:
          the bandwidth, the instructions spent moving it, and the registers it occupies on the way. Vector stores (section
          15.5) help: each thread stores more per instruction, with less overhead and better coalescing. But copying from
          global to shared memory with a vector load and a vector store still takes two steps through registers.
        </p>
      </Translated>
      <Source section="15.9" pages="369" />
      <Translated
        vi={
          <p>
            LDGSTS là một lệnh nạp từ bộ nhớ toàn cục thẳng vào bộ nhớ chia sẻ, không qua thanh ghi, và phần cứng làm việc đó ở chế độ
            nền, nên việc chồng lấp nó với tính toán không còn phụ thuộc vào cách trình biên dịch lập lịch. Sách nói nvcc nhận ra một
            lệnh nạp vector theo sau là một lệnh ghi bộ nhớ chia sẻ và tự phát ra LDGSTS, và rằng <code>cuda::memcpy_async</code> từ
            libcu++ dùng nó. TMA, từ Hopper, làm điều tương tự cho cả tensor nhiều chiều, cũng không qua thanh ghi, thông qua{' '}
            <code>cuda::memcpy_async</code> hoặc các thư viện.
          </p>
        }
      >
        <p>
          LDGSTS is one instruction that loads from global memory straight into shared memory, without registers, and the
          hardware does it in the background, so overlapping it with arithmetic no longer depends on the compiler's
          scheduling. The book says nvcc recognises a vector load followed by a shared-memory store and emits LDGSTS on its
          own, and that <code>cuda::memcpy_async</code> from libcu++ uses it. TMA, from Hopper, does the same for whole
          multi-dimensional tensors, again without registers, through <code>cuda::memcpy_async</code> or the libraries.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Hai trong các khẳng định đó không đúng như được viết. NVIDIA không công bố phép gộp nào như vậy: hướng dẫn của hãng nói
            trình biên dịch có thể phát ra các cặp LDG và STS cho mã thông thường, và hướng dẫn chẩn đoán của hãng bảo bạn tự chuyển
            mã hoặc dùng CUTLASS. <code>cuda::memcpy_async</code> có thể tạo ra LDGSTS nhưng không được bảo đảm; các primitive{' '}
            <code>__pipeline_memcpy_async</code> thì được bảo đảm. Và <code>cuda::memcpy_async</code> chỉ dùng tới TMA cho sao chép
            khối một chiều vào bộ nhớ chia sẻ của chính khối đó. Một ô GEMM hai chiều cần một tensor map tạo trên host bằng{' '}
            <code>cuTensorMapEncodeTiled</code> và <code>cuda::ptx::cp_async_bulk_tensor</code>. Để biết bạn nhận được gì, hãy xem:
            LDGSTS trong SASS đã biên dịch, hoặc biểu đồ bộ nhớ của Nsight Compute.
          </p>
        }
      >
        <p>
          Two of those claims do not hold as written. NVIDIA documents no such fusion: its guide says the compiler may
          emit LDG and STS pairs for ordinary code, and its triage guide says to convert the code yourself or use CUTLASS.{' '}
          <code>cuda::memcpy_async</code> may produce LDGSTS but is not guaranteed to; the{' '}
          <code>__pipeline_memcpy_async</code> primitives are. And <code>cuda::memcpy_async</code> reaches TMA only for
          one-dimensional bulk copies into the block's own shared memory. A two-dimensional GEMM tile needs a tensor map
          built on the host with <code>cuTensorMapEncodeTiled</code> and <code>cuda::ptx::cp_async_bulk_tensor</code>. To
          know what you got, look: LDGSTS in the compiled SASS, or Nsight Compute's memory chart.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 against the CUDA 13.5 programming guide, PTX ISA 9.5 and the CCCL docs (ch15-facts A7, A8, A11)',
          '2026-11-20, đối chiếu với CUDA 13.5 programming guide, PTX ISA 9.5 và tài liệu CCCL (ch15-facts A7, A8, A11)',
        )}
      >
        <Translated
          vi={
            <p>
              LDGSTS: compute capability 8.0 trở lên. TMA: 9.0 trở lên. Trình biên dịch là nvcc và thư viện là libcu++ (libcudacxx, giờ
              là một phần của CCCL); "multidimensional tenors" của sách nghĩa là tensor.
            </p>
          }
        >
          <p>
            LDGSTS: compute capability 8.0 and later. TMA: 9.0 and later. The compiler is nvcc and the library is libcu++
            (libcudacxx, now part of CCCL); the book's "multidimensional tenors" means tensors.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS159['ch15/15.9/q-ldgsts']} />
      <Ask q={questionsS159['ch15/15.9/q-tma-2d']} />

      <h2>
        <Translated vi="Trên HIP và Metal">On HIP and Metal</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách chỉ nói về NVIDIA. Trên AMD, các lệnh ma trận là builtin của trình biên dịch. GPU CDNA có lệnh MFMA, trong đó có một
            lệnh với đầu vào và đầu ra FP32; CDNA 4 (gfx950) thiếu dạng xf32 độ chính xác giảm. RDNA3 và RDNA4 có builtin WMMA, dù
            bảng tính năng HIP không cho thấy Matrix Core nào trên RDNA. rocWMMA là con đường thư viện C++. Với sao chép bất đồng bộ,
            intrinsic mà hướng dẫn của AMD nêu tên không tồn tại trong ROCm 10.2, còn builtin toàn cục-sang-LDS thực sự tồn tại thì
            không có trong hướng dẫn HIP, nên khóa học này không dạy nó. Trên HIP, pipeline phần mềm của mục 15.8 giữ nguyên dạng phần
            mềm.
          </p>
        }
      >
        <p>
          The book covers NVIDIA only. On AMD the matrix instructions are compiler builtins. CDNA
          GPUs have MFMA instructions, including one with FP32 inputs and outputs; CDNA 4 (gfx950) lacks the
          reduced-precision xf32 form. RDNA3 and RDNA4 have WMMA builtins, even though the HIP features table shows no
          Matrix Cores on RDNA. rocWMMA is the C++ library route. For asynchronous copies, the intrinsic AMD's guide names
          does not exist in ROCm 10.2, and the global-to-LDS builtin that does exist is not in the HIP guide, so this course
          does not teach it. On HIP, section 15.8's software pipeline stays in its software form.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 against the ROCm 10.2 compiler sources and HIP docs (ch15-facts B4, B5; infra-facts B12)',
          '2026-11-20, đối chiếu với mã nguồn trình biên dịch ROCm 10.2 và tài liệu HIP (ch15-facts B4, B5; infra-facts B12)',
        )}
      >
        <Translated
          vi={
            <p>
              <code>__builtin_amdgcn_mfma_f32_16x16x4f32</code> nhận và trả về FP32; các dạng RDNA là{' '}
              <code>__builtin_amdgcn_wmma_*_w32</code> và <code>_w32_gfx12</code>.{' '}
              <code>__builtin_amdgcn_async_work_group_copy</code> trong hướng dẫn không xuất hiện ở đâu trong trình biên dịch.
            </p>
          }
        >
          <p>
            <code>__builtin_amdgcn_mfma_f32_16x16x4f32</code> takes and returns FP32; the RDNA forms are{' '}
            <code>__builtin_amdgcn_wmma_*_w32</code> and <code>_w32_gfx12</code>. The guide's{' '}
            <code>__builtin_amdgcn_async_work_group_copy</code> occurs nowhere in the compiler.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Metal có <code>simdgroup_float8x8</code> với <code>simdgroup_multiply_accumulate</code> trên Apple7 trở lên và, với Metal
            4, <code>matmul2d</code> của Metal Performance Primitives, hỗ trợ float × float ra float và, từ OS 26.4, các định dạng số
            nguyên 4 bit. Apple không công bố đơn vị ma trận nào đằng sau cả hai, nên khóa học này không bao giờ gọi chúng là tensor
            core. Metal hoàn toàn không công bố sao chép bất đồng bộ nào.
          </p>
        }
      >
        <p>
          Metal has <code>simdgroup_float8x8</code> with <code>simdgroup_multiply_accumulate</code> on Apple7 and later and,
          with Metal 4, Metal Performance Primitives' <code>matmul2d</code>, which supports float × float into float and,
          from OS 26.4, 4-bit integer formats. Apple documents no matrix unit behind either, so this course never calls
          them tensor cores. Metal documents no asynchronous copy at all.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 against the MSL 4.1 specification (ch15-facts C5; infra-facts C10; ch06-facts C7)',
          '2026-11-20, đối chiếu với đặc tả MSL 4.1 (ch15-facts C5; infra-facts C10; ch06-facts C7)',
        )}
      >
        <Translated vi={<p>Apple khuyên dùng <code>matmul2d</code> thay cho <code>simdgroup_matrix</code>. Đặc tả MSL 4.1 không có "async".</p>}>
          <p>Apple recommends <code>matmul2d</code> over <code>simdgroup_matrix</code>. The MSL 4.1 specification has no "async".</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Với mã của chính bạn, quy tắc là quy tắc trong danh sách bẫy tính khả chuyển của khóa học này: intrinsic tensor core và
            matrix core đến từ các thư viện khác nhau ở mỗi nhà cung cấp, không có cách viết chung, nên chúng nằm ngoài tệp{' '}
            <code>.cu</code> dùng chung phục vụ CUDA, HIP và các trình giả lập.
          </p>
        }
      >
        <p>
          For your own code the rule is the one in this course's list of portability traps: tensor and matrix-core
          intrinsics come from different libraries on each vendor, with no shared spelling, so they stay out of the shared{' '}
          <code>.cu</code> file that serves CUDA, HIP and the emulators.
        </p>
      </Translated>
      <HardwareBench />
      <Translated
        vi={
          <p>
            Những điều cần tìm: mã WGMMA build cho H100 không chạy trên GPU compute capability 10.0; RTX 50 không có TMEM; card RDNA3
            có lệnh ma trận; và không có gì trên Metal hay HIP công bố một lần sao chép bất đồng bộ mà bạn có thể gọi.
          </p>
        }
      >
        <p>
          Things to find: WGMMA code built for an H100 does not run on a compute capability 10.0 GPU; an RTX 50 has no
          TMEM; an RDNA3 card does have matrix instructions; and nothing on Metal or HIP documents an asynchronous copy you
          can call.
        </p>
      </Translated>
      <Ask q={questionsS159['ch15/15.9/q-rdna']} />

      <h2>
        <Translated vi="Các backend thật tiếp cận phần cứng thế nào">How real backends reach the hardware</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Backend CUDA của llama.cpp không dùng WMMA. <code>mma.cuh</code> của nó phát trực tiếp các lệnh PTX{' '}
            <code>mma.sync</code> và <code>ldmatrix</code> trên NVIDIA, và các builtin MFMA và WMMA trên AMD, từ cùng một tệp. PyTorch
            gửi một phép matmul float trên CUDA đến cuBLASLt hoặc cuBLAS, và trên ROCm có thể gửi nó đến Composable Kernel.
          </p>
        }
      >
        <p>
          llama.cpp's CUDA backend does not use WMMA. Its <code>mma.cuh</code> issues PTX <code>mma.sync</code> and{' '}
          <code>ldmatrix</code> instructions directly on NVIDIA, and the MFMA and WMMA builtins on AMD, from the same file.
          PyTorch sends a float matmul on CUDA to cuBLASLt or cuBLAS, and on ROCm can send it to Composable Kernel.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Đọc từ mã nguồn của hai dự án ngày 2026-11-20 (ch15-facts D1, D4).</p>}>
          <p>Read from the two projects' sources on 2026-11-20 (ch15-facts D1, D4).</p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Kernel của bạn so với thư viện">Your kernel against the library</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lab 15.6, mốc dự án M3, đo lớp tuyến tính nhanh của bạn và thư viện của nhà cung cấp cạnh nhau trên lớp 1024 → 4096 (
            <a href={href('ch15', 'labs')}>lab</a>). Cả ba mốc so sánh của nhà cung cấp đều tính FP32 theo mặc định, nên khoảng cách
            bạn thấy là ở kỹ thuật, không phải độ chính xác. Hãy chốt một con số trước.
          </p>
        }
      >
        <p>
          Lab 15.6, milestone M3, times your fast linear layer and the vendor library side by side on the 1024 → 4096 layer
          (<a href={href('ch15', 'labs')}>labs</a>). All three vendor baselines compute FP32 by default, so whatever gap
          you see is technique, not precision. Commit a number first.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 (ch15-facts A10, B6, C6; not in the book)',
          '2026-11-20 (ch15-facts A10, B6, C6; không có trong sách)',
        )}
      >
        <Translated
          vi={
            <p>
              cuBLAS: TF32 và BF16x9 phải chủ động bật. hipBLAS: XF32 phải chủ động bật. MPS: độ chính xác giảm là một tùy chọn bạn đặt.
            </p>
          }
        >
          <p>
            cuBLAS: TF32 and BF16x9 are opt-in. hipBLAS: XF32 is opt-in. MPS: reduced precision is an option you set.
          </p>
        </Translated>
      </Today>
      <LabsPage labs={LABS}>
        <Predict
          id="ch15/15.9/p-m3-vs-vendor"
          prompt={
            <Translated vi={<>Ở batch {BATCH}, kernel M3 của bạn sẽ đạt bao nhiêu phần trăm tốc độ của thư viện nhà cung cấp trên thiết bị của bạn?</>}>
              <>At batch {BATCH}, what percentage of the vendor library's speed will your M3 kernel reach on your device?</>
            </Translated>
          }
          unit={tr('% of the library', '% của thư viện')}
        >
          {(v) => (
            <Reveal
              committed={v}
              measure={percentOfVendor}
              unit={tr('% of the library', '% của thư viện')}
              reading={
                <Translated
                  vi={
                    <>
                      100% là tốc độ của thư viện. Đây là số đo trên máy của bạn với kernel nào đã qua các phép kiểm thử, không phải giá trị kỳ
                      vọng; thư viện dùng những kỹ thuật mà chương này chỉ nêu tên.
                    </>
                  }
                >
                  <>
                    100% would be the library's speed. These are measurements on your machine with whichever kernel passed
                    its tests, not expected values; the library uses techniques this chapter only names.
                  </>
                </Translated>
              }
              missing={
                <Translated
                  vi={
                    <>
                      Chạy <code>python3 labs/pmpp.py run 15.6 --solution --bench</code> trên một máy có GPU cho lời giải tham chiếu (hoặc{' '}
                      <code>run 15.6 --bench</code> khi kernel của chính bạn đã qua), rồi tải lại trang.
                    </>
                  }
                >
                  <>
                    Run <code>python3 labs/pmpp.py run 15.6 --solution --bench</code> on a machine with a GPU for the
                    reference solution (or <code>run 15.6 --bench</code> once your own kernel passes), then reload.
                  </>
                </Translated>
              }
            />
          )}
        </Predict>
      </LabsPage>
      <Ask q={questionsS159['ch15/15.9/q-fp32-default']} />
    </>
  );
}
