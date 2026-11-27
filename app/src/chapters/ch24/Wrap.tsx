import type { ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Beyond, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useProgress } from '../../lib/progress';
import { fmt, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { PLAN, useAllChapters } from '../registry';
import { GOALS, gpuBackends, rollup } from '../../sim/booksteps';
import { BOOK, DEVICES, VENDOR, linearLayerRatio, marks, pcieRate, threshold, tileFor, type RowId } from '../../sim/hwtable';

// Chapter 24 has no code, so this page shows no kernel and needs no snippet file.
// Every number its questions state comes from the model the start page runs on
// (sim/hwtable.ts) or from a vendor figure quoted in notes/today/ch24-facts.md;
// Wrap.test.ts pins both.

const row = (id: RowId) => BOOK.rows.find((r) => r.id === id)!;
const device = (key: string) => DEVICES.find((d) => d.key === key)!;
const thr = (key: string) => threshold(device(key).peak, device(key).bw);
const THR_2007 = thr('book2007');
const THR_2025 = thr('book2025');
const THR_TC = thr('book2025tc');

// ---- q-gap: device memory against the host link, inside one column of Fig. 24.1.
export const GAP = {
  y2007: row('membw').from! / row('pcie').from!,
  y2025: row('membw').to / row('pcie').to,
  // The same quotient on NVIDIA's documents for one board each (facts A3, A5, A6, A10, A12).
  vendor2007: VENDOR.g80.membw.value / pcieRate(VENDOR.g80.pcieGen, 16),
  vendor2025: VENDOR.b200.membw.value / pcieRate(VENDOR.b200.pcieGen, 16),
};

// ---- q-perboard: NVIDIA's HGX B200 page gives FP32 for a board of eight (facts A8);
// the reference architecture gives bandwidth per GPU, "up to 8 TB/s" (facts A10).
export const HGX = { boardFp32Tflops: 600, gpus: 8, gpuBandwidthTBs: 8 };
const PER_GPU_THR = threshold((HGX.boardFp32Tflops / HGX.gpus) * 1000, HGX.gpuBandwidthTBs * 1000);
const MIXED_THR = threshold(HGX.boardFp32Tflops * 1000, HGX.gpuBandwidthTBs * 1000);
const OUT128 = marks(1).find((m) => m.id === 'out128')!.ratio;

// ---- q-mi355x: AMD's published peaks for the Instinct MI355X (facts B4, B1).
export const MI355X = { fp32Tflops: 157.3, fp32MatrixTflops: 157.3, fp16MatrixPflops: 2.5, bandwidthTBs: 8 };
const AMD_THR = threshold(MI355X.fp32Tflops * 1000, MI355X.bandwidthTBs * 1000);
const AMD_FP16_THR = threshold(MI355X.fp16MatrixPflops * 1e6, MI355X.bandwidthTBs * 1000);
const AMD_T = tileFor(AMD_THR);

// ---- q-next: the book's 16-bit linear layer at batch 1 (ch. 20 §20.6, p. 505), then 8 and 4 bits.
export const DECODE = {
  b16: linearLayerRatio(1, 1024, 4096, 2),
  b8: linearLayerRatio(1, 1024, 4096, 1),
  b4: linearLayerRatio(1, 1024, 4096, 0.5),
};

export const questionsWrap: Record<string, Question> = {
  'ch24/wrap/q-gap': {
    kind: 'numeric',
    id: 'ch24/wrap/q-gap',
    prompt: (
      <Translated
        vi={
          <>
            Ở cột 2025 của bảng trong sách, bộ nhớ device cung cấp {fmt(row('membw').to)} GB/s còn đường truyền tới host{' '}
            {row('pcie').to} GB/s mỗi chiều. Một mô hình quá lớn so với bộ nhớ device vẫn chạy được bằng cách truyền
            trọng số từ bộ nhớ host cho mỗi token. Tính trên mỗi token, chuyển trọng số theo cách đó lâu hơn bao nhiêu lần
            so với đọc chúng từ bộ nhớ device?
          </>
        }
      >
        In the 2025 column of the book's table, device memory delivers {fmt(row('membw').to)} GB/s and the host link{' '}
        {row('pcie').to} GB/s each way. A model too large for device memory can still be run by streaming its weights
        from host memory for every token. Per token, how many times longer does moving the weights take that way than
        reading them from device memory?
      </Translated>
    ),
    answer: GAP.y2025,
    tol: 0.02,
    unit: 'times',
    explain: (
      <Translated
        vi={
          <p>
            {fmt(row('membw').to)} / {row('pcie').to} = {fmt(GAP.y2025)}. Ở cột 2007 cùng thương số đó là{' '}
            {row('membw').from} / {row('pcie').from} = {fmt(GAP.y2007)}. Vậy cái giá của việc đi qua đường truyền tới host
            thay vì đọc bộ nhớ device đã tăng từ khoảng {Math.round(GAP.y2007)} lần lên hơn một trăm lần. Trên số liệu của
            chính NVIDIA cho mỗi năm một bo mạch, nó tăng từ {fmt(GAP.vendor2007)} lên {fmt(GAP.vendor2025)}: số khác,
            cùng chiều hướng. Cả hai đều là tốc độ đỉnh, nên phép tính này chỉ đếm byte và không gì khác. Đó là phép tính
            đứng sau việc giữ trọng số, KV cache và việc lấy mẫu trên device.
          </p>
        }
      >
      <p>
        {fmt(row('membw').to)} / {row('pcie').to} = {fmt(GAP.y2025)}. In the 2007 column the same quotient is{' '}
        {row('membw').from} / {row('pcie').from} = {fmt(GAP.y2007)}. So the cost of crossing the host link instead of
        reading device memory went from about {Math.round(GAP.y2007)} times to over a hundred. On NVIDIA's own figures
        for one board each it went from {fmt(GAP.vendor2007)} to {fmt(GAP.vendor2025)}: different numbers, same
        direction. Both are peak rates, so this counts the bytes and nothing else. It is the arithmetic behind keeping
        weights, the KV cache and sampling on the device.
      </p>
      </Translated>
    ),
    source: 'our analysis on the figures of Fig. 24.1, p. 578; vendor figures from notes/today/ch24-facts.md, A3 to A6, A10, A12',
    vi: {
      unit: 'lần',
      source: 'phân tích của chúng tôi trên số liệu của Fig. 24.1, p. 578; số liệu nhà sản xuất từ notes/today/ch24-facts.md, A3 đến A6, A10, A12',
    },
  },
  'ch24/wrap/q-perboard': {
    kind: 'choice',
    id: 'ch24/wrap/q-perboard',
    prompt: (
      <Translated
        vi={
          <>
            Một đồng nghiệp viết: "Trang HGX B200 của NVIDIA ghi FP32 là {HGX.boardFp32Tflops} TFLOPS, và băng thông bộ
            nhớ lên tới {HGX.gpuBandwidthTBs} TB/s mỗi GPU. Vậy ngưỡng là {HGX.boardFp32Tflops} / {HGX.gpuBandwidthTBs} ={' '}
            {fmt(MIXED_THR)} FLOP/B, và GEMM float32 của chúng ta với ô đầu ra 128 × 128, ở mức {fmt(OUT128)} FLOP/B, bị
            giới hạn bởi bộ nhớ." Sai ở đâu?
          </>
        }
      >
        A colleague writes: "NVIDIA's HGX B200 page gives FP32 as {HGX.boardFp32Tflops} TFLOPS, and memory bandwidth is
        up to {HGX.gpuBandwidthTBs} TB/s per GPU. So the threshold is {HGX.boardFp32Tflops} / {HGX.gpuBandwidthTBs} ={' '}
        {fmt(MIXED_THR)} FLOP/B, and our float32 GEMM with 128 × 128 output tiles, at {fmt(OUT128)} FLOP/B, is
        memory-bound." What is wrong?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không có gì sai. TFLOPS chia cho TB/s là FLOP trên mỗi byte, nên đơn vị khớp và ngưỡng đúng như đã viết.">
            Nothing. TFLOPS divided by TB/s is FLOP per byte, so the units check out and the threshold stands as written.
          </Translated>
        ),
        why: (
          <Translated vi="Đơn vị đúng là triệt tiêu nhau, nên lỗi này dễ bị bỏ qua. Đơn vị không nói gì về việc hai con số có mô tả cùng một phần cứng hay không.">
            The units do cancel, which is why this slip is easy to miss. Units say nothing about whether both numbers describe the same hardware.
          </Translated>
        ),
      },
      {
        label: (
          <Translated
            vi={`${HGX.boardFp32Tflops} TFLOPS là cho cả bo mạch gồm ${HGX.gpus} GPU. Tính cho mỗi GPU, ngưỡng là khoảng ${fmt(PER_GPU_THR, 2)} FLOP/B, và kernel nằm trên ngưỡng đó.`}
          >
            {`The ${HGX.boardFp32Tflops} TFLOPS is for the whole board of ${HGX.gpus} GPUs. Per GPU the threshold is about ${fmt(PER_GPU_THR, 2)} FLOP/B, and the kernel is above it.`}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Một GEMM nên được so với số liệu tensor core, không phải số FP32, và như vậy ngưỡng còn cao hơn nữa.">
            A GEMM should be measured against the tensor-core figure, not the FP32 one, which puts the threshold higher still.
          </Translated>
        ),
        why: (
          <Translated vi="Chỉ khi kernel chạy trên tensor core, qua một thư viện, CUDA Tile, hoặc khi bật TF32. Một GEMM float32 viết bằng phép tính số học thông thường bị giới hạn bởi số liệu FP32.">
            Only if the kernel runs on the tensor cores, through a library, CUDA Tile, or TF32 switched on. A float32 GEMM written with ordinary arithmetic is limited by the FP32 figure.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Băng thông phải là của đường truyền tới host, vì các ma trận bắt đầu ở bộ nhớ host và phải đi qua nó trước.">
            {'The bandwidth should be the host link\'s, since the matrices start in host memory and cross it first.'}
          </Translated>
        ),
        why: (
          <Translated vi="Ngưỡng so sánh phép tính số học với lưu lượng mà kernel tạo ra tới bộ nhớ device, nơi các toán hạng của nó nằm. Đường truyền tới host chỉ bị đi qua một lần, trước khi kernel chạy.">
            {"The threshold compares arithmetic with the traffic the kernel makes to device memory, where its operands live. The host link is crossed once, before the kernel runs."}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trang HGX của NVIDIA ghi {HGX.boardFp32Tflops} TFLOPS cho một bo mạch gồm {HGX.gpus} GPU, tức là{' '}
            {HGX.boardFp32Tflops / HGX.gpus} mỗi GPU, và kiến trúc tham chiếu của họ ghi lên tới{' '}
            {HGX.gpuBandwidthTBs} TB/s mỗi GPU. {fmt((HGX.boardFp32Tflops / HGX.gpus) * 1000)} GFLOP/s /{' '}
            {fmt(HGX.gpuBandwidthTBs * 1000)} GB/s = {fmt(PER_GPU_THR)} FLOP/B, nên một kernel {fmt(OUT128)} FLOP/B bị giới hạn bởi tính toán theo mô hình của mục 5.1. Chia
            phép tính số học của cả bo mạch cho băng thông của một GPU đã thổi phồng ngưỡng lên {HGX.gpus} lần. Chính bảng
            của sách mắc cùng loại lỗi này hai lần: cột 2007 lấy băng thông từ một bo mạch và dung lượng từ một bo mạch
            khác, và hàng NVLink đếm cả hai chiều trong khi hàng PCIe chỉ đếm một chiều.
          </p>
        }
      >
      <p>
        NVIDIA's HGX page prints {HGX.boardFp32Tflops} TFLOPS for a board of {HGX.gpus}, which is{' '}
        {HGX.boardFp32Tflops / HGX.gpus} per GPU, and its reference architecture prints up to{' '}
        {HGX.gpuBandwidthTBs} TB/s per GPU. {fmt((HGX.boardFp32Tflops / HGX.gpus) * 1000)} GFLOP/s /{' '}
        {fmt(HGX.gpuBandwidthTBs * 1000)} GB/s = {fmt(PER_GPU_THR)} FLOP/B, so a {fmt(OUT128)} FLOP/B kernel is compute-bound by the section 5.1 model. Dividing
        a board's arithmetic by one GPU's bandwidth overstated the threshold {HGX.gpus} times. The book's own table
        makes the same kind of slip twice: its 2007 column takes bandwidth from one board and capacity from another,
        and its NVLink row counts both directions where the PCIe row counts one.
      </p>
      </Translated>
    ),
    source: 'method of ch. 5 §5.1, pp. 94–97; output tile from ch. 15 §15.2, p. 351; NVIDIA figures from notes/today/ch24-facts.md, A8, A10',
    vi: {
      source: 'phương pháp của ch. 5 §5.1, pp. 94–97; ô đầu ra từ ch. 15 §15.2, p. 351; số liệu NVIDIA từ notes/today/ch24-facts.md, A8, A10',
    },
  },
  'ch24/wrap/q-mi355x': {
    kind: 'numeric',
    id: 'ch24/wrap/q-mi355x',
    prompt: (
      <Translated
        vi={
          <>
            Trang của AMD cho Instinct MI355X ghi, trong số các hàng khác: peak FP32 {MI355X.fp32Tflops} TFLOPs; peak FP32
            matrix {MI355X.fp32MatrixTflops} TFLOPs; peak FP16 matrix {MI355X.fp16MatrixPflops} PFLOPs; peak memory
            bandwidth {MI355X.bandwidthTBs} TB/s. Ngưỡng theo mục 5.1 cho một phép nhân ma trận float32 trên chip này là bao nhiêu?
          </>
        }
      >
        AMD's page for its Instinct MI355X prints, among other rows: peak FP32 {MI355X.fp32Tflops} TFLOPs; peak FP32
        matrix {MI355X.fp32MatrixTflops} TFLOPs; peak FP16 matrix {MI355X.fp16MatrixPflops} PFLOPs; peak memory
        bandwidth {MI355X.bandwidthTBs} TB/s. What is the section 5.1 threshold for a float32 matmul on this part?
      </Translated>
    ),
    answer: Number(AMD_THR.toFixed(2)),
    tol: 0.02,
    unit: 'FLOP/B',
    explain: (
      <Translated
        vi={
          <>
            <p>
              {fmt(MI355X.fp32Tflops * 1000)} GFLOP/s / {fmt(MI355X.bandwidthTBs * 1000)} GB/s = {fmt(AMD_THR)} FLOP/B.
              Số liệu FP16 matrix sẽ cho {fmt(AMD_FP16_THR)}, nhưng nó dành cho đầu vào 16-bit, không phải dữ liệu float32 của bạn.
              Trên chip này số liệu FP32 matrix bằng số liệu thường, và AMD chú thích TF32 là "support through software
              emulation", nên không có bước nhảy kiểu TF32 nào để chọn.
            </p>
            <p>
              Một ô bộ nhớ chia sẻ (shared memory) hình vuông sẽ cần T = {AMD_T} để đạt tới ngưỡng đó. Một ô đầu ra 128 × 128 ({fmt(OUT128)} FLOP/B)
              vượt qua nó; sinh token với batch 1, khoảng {fmt(DECODE.b16, 2)} FLOP/B ở 16 bit, thì không. Đây là các số đỉnh
              AMD công bố cho một sản phẩm, không phải thiết bị của bạn.
            </p>
          </>
        }
      >
      <>
        <p>
          {fmt(MI355X.fp32Tflops * 1000)} GFLOP/s / {fmt(MI355X.bandwidthTBs * 1000)} GB/s = {fmt(AMD_THR)} FLOP/B.
          The FP16 matrix figure would give {fmt(AMD_FP16_THR)}, but it is for 16-bit inputs, not your float32 data.
          On this part the FP32 matrix figure equals the plain one, and AMD footnotes TF32 as "support through software
          emulation", so there is no TF32-style jump to choose.
        </p>
        <p>
          A square shared-memory tile would need T = {AMD_T} to reach it. A 128 × 128 output tile ({fmt(OUT128)} FLOP/B)
          clears it; batch-1 generation, about {fmt(DECODE.b16, 2)} FLOP/B in 16 bits, does not. These are AMD's
          published peaks for one product, not your device.
        </p>
      </>
      </Translated>
    ),
    source: "method of ch. 5 §5.1, pp. 94–97; AMD's product page as quoted in notes/today/ch24-facts.md, B4 and B1",
    vi: {
      source: 'phương pháp của ch. 5 §5.1, pp. 94–97; trang sản phẩm của AMD như được trích trong notes/today/ch24-facts.md, B4 và B1',
    },
  },
  'ch24/wrap/q-next': {
    kind: 'choice',
    id: 'ch24/wrap/q-next',
    prompt: (
      <Translated
        vi={
          <>
            Việc sinh từng token của mô hình của bạn chạy trên Metal với batch 1 và trọng số 16-bit, và các phép đo của bạn
            cho thấy các phép nhân ma trận-vector của nó gần chạm trần băng thông. Bước tiếp theo nào trong khóa học này
            nâng tốc độ mà trần đó cho phép?
          </>
        }
      >
        Token-by-token generation of your model runs on Metal at batch 1 with 16-bit weights, and your measurements put
        its matrix-vector products close to the bandwidth ceiling. Which next step in this course raises the speed that
        ceiling allows it?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Phân ô thanh ghi của chương 15: các ô đầu ra lớn hơn được giữ trên mỗi luồng (thread).">
            {"Chapter 15's register tiling: larger output tiles held per thread."}
          </Translated>
        ),
        why: (
          <Translated vi="Ô đầu ra lớn hơn chia sẻ các toán hạng đã nạp cho nhiều đầu ra hơn. Với batch 1 mỗi trọng số chỉ dùng cho một đầu ra, nên không có gì để chia sẻ.">
            Larger output tiles share fetched operands across more outputs. At batch 1 each weight is used for one output only, so there is nothing to share.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Bật TF32 cho matmul, để các phép nhân chạy trên phần cứng ma trận.">
            Turning on TF32 for matmul, so the products run on the matrix hardware.
          </Translated>
        ),
        why: (
          <Translated vi="Nó nâng một trần tính toán mà kernel còn ở rất xa bên dưới, và TF32 không nằm trong các kiểu của matmul2d trên Metal.">
            {'It raises a compute ceiling the kernel is far below, and TF32 is not among the types of Metal\'s matmul2d.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Module X2: lưu trọng số ở 8 hoặc 4 bit và giải lượng tử hóa trong kernel.">
            Module X2: store the weights in 8 or 4 bits and dequantise in the kernel.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Module X1: chuyển việc lấy mẫu lên GPU, để CPU không bao giờ phải đọc một hàng logits.">
            Module X1: move sampling onto the GPU, so the CPU never reads a row of logits.
          </Translated>
        ),
        why: (
          <Translated vi="Việc đó gỡ phần lấy mẫu khỏi CPU cho mỗi token, và trên Metal không có bản sao logits nào để tiết kiệm: một chi phí khác. Byte nằm ở trọng số.">
            That takes the sampling work off the CPU for every token, and on Metal there is no logits copy to save: a different cost. The weights are where the bytes are.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hai phép toán cho mỗi trọng số và một lần nạp nó: tỉ lệ được quyết định bởi số byte trên mỗi trọng số. Trong lớp
            ví dụ của sách nó khoảng {fmt(DECODE.b16, 2)} FLOP/B ở 16 bit; ở 8 bit là {fmt(DECODE.b8, 2)}, ở 4 bit{' '}
            {fmt(DECODE.b4, 2)}. Một kernel bị giới hạn bởi bộ nhớ tăng tốc theo đúng tỉ lệ đó, trước khi phép tính số học
            rẻ đi chút nào. Đòn bẩy còn lại là batching, chương 20 và 22, nơi một trọng số đã nạp phục vụ nhiều chuỗi. Mốc
            dự án của X2 là M9.
          </p>
        }
      >
      <p>
        Two operations per weight and one fetch of it: the ratio is set by the bytes per weight. In the book's example
        layer it is about {fmt(DECODE.b16, 2)} FLOP/B at 16 bits; at 8 bits it is {fmt(DECODE.b8, 2)}, at 4 bits{' '}
        {fmt(DECODE.b4, 2)}. A memory-bound kernel speeds up in proportion, before any arithmetic gets cheaper. The
        other lever is batching, chapters 20 and 22, where one fetched weight serves several sequences. X2's milestone is
        M9.
      </p>
      </Translated>
    ),
    source: 'ch. 20 §20.6, p. 505, for the 16-bit layer; the bit widths and the module are ours; notes/today/ch24-facts.md, C1, for matmul2d',
    vi: {
      source: 'ch. 20 §20.6, p. 505, cho lớp 16-bit; các độ rộng bit và module là của khóa học này; notes/today/ch24-facts.md, C1, cho matmul2d',
    },
  },
};

// The modules and appendices after chapter 24, with the place in PyTorch's source that
// holds the same kind of code (notes/today/ch24-facts.md, D2 and D3).
const NEXT: { id: string; name: string; pages: string; adds: string; pytorch: ReactNode }[] = [
  {
    id: 'appA',
    get name() {
      return tr('Appendix A, numerical considerations', 'Phụ lục A, các vấn đề về số học');
    },
    pages: 'pp. 583–600',
    get adds() {
      return tr('Why summation order changes a result, and the lower-precision formats that quantisation starts from. Feeds X2.', 'Vì sao thứ tự cộng làm thay đổi kết quả, và các định dạng độ chính xác thấp hơn mà lượng tử hóa bắt đầu từ đó. Dẫn vào X2.');
    },
    pytorch: <span className="muted"><Translated vi="không nêu ở đây">none named here</Translated></span>,
  },
  {
    id: 'appC',
    get name() {
      return tr('Appendix C, CUDA memories, address spaces and pointers', 'Phụ lục C, các bộ nhớ CUDA, không gian địa chỉ và con trỏ');
    },
    pages: 'pp. 623–629',
    get adds() {
      return tr("The book's memory models, set against unified memory on Apple silicon.", 'Các mô hình bộ nhớ của sách, đặt cạnh unified memory trên Apple silicon.');
    },
    pytorch: (
      <Translated vi={<><code>aten/src/ATen/mps/</code>: phía runtime của Metal, trong đó có <code>MPSAllocator.mm</code></>}>
        <><code>aten/src/ATen/mps/</code>: the Metal runtime side, among it <code>MPSAllocator.mm</code></>
      </Translated>
    ),
  },
  {
    id: 'x1',
    get name() {
      return tr('X1, sampling kernels', 'X1, các kernel lấy mẫu');
    },
    get pages() {
      return tr('not in the book', 'không có trong sách');
    },
    get adds() {
      return tr('Greedy, top-k and top-p sampling on the device, assembled from chapters 10, 11, 12 and 14. Milestone M5.', 'Lấy mẫu greedy, top-k và top-p trên device, ghép từ các chương 10, 11, 12 và 14. Mốc dự án M5.');
    },
    pytorch: (
      <Translated vi={<><code>Sort.metal</code>, <code>SortTopK.h</code>, <code>ScanKernel.metal</code> trong <code>native/mps/kernels/</code></>}>
        <><code>Sort.metal</code>, <code>SortTopK.h</code>, <code>ScanKernel.metal</code> in <code>native/mps/kernels/</code></>
      </Translated>
    ),
  },
  {
    id: 'x2',
    get name() {
      return tr('X2, quantised kernels', 'X2, các kernel lượng tử hóa');
    },
    get pages() {
      return tr('not in the book', 'không có trong sách');
    },
    get adds() {
      return tr("int8 and 4-bit-weight GEMVs and the quantised decode step, anchored on appendix A and chapter 15's intensity analysis. Milestone M9.", 'Các GEMV int8 và trọng số 4-bit, cùng bước decode lượng tử hóa, dựa trên phụ lục A và phân tích cường độ tính toán của chương 15. Mốc dự án M9.');
    },
    pytorch: (
      <Translated vi={<><code>Quantized.metal</code> trong <code>native/mps/kernels/</code></>}>
        <><code>Quantized.metal</code> in <code>native/mps/kernels/</code></>
      </Translated>
    ),
  },
  {
    id: 'x3',
    get name() {
      return tr('X3, reading real backends', 'X3, đọc các backend thật');
    },
    get pages() {
      return tr('not in the book', 'không có trong sách');
    },
    get adds() {
      return tr('Find the kernel behind a given op in llama.cpp, MLX and PyTorch, and explain it; then run a small generated Llama-style model on the course\'s kernels. Milestone M10.', 'Tìm kernel đứng sau một op cho trước trong llama.cpp, MLX và PyTorch, và giải thích nó; rồi chạy một mô hình nhỏ kiểu Llama được sinh ra trên các kernel của khóa học. Mốc dự án M10.');
    },
    pytorch: (
      <Translated vi={<><code>native/cuda/</code>, <code>native/mps/kernels/</code>, và <code>tools/amd_build/</code> cho AMD</>}>
        <><code>native/cuda/</code>, <code>native/mps/kernels/</code>, and <code>tools/amd_build/</code> for AMD</>
      </Translated>
    ),
  },
];
const NAME: Record<string, string> = { metal: 'Metal', cuda: 'CUDA', hip: 'HIP' };
const servedBy = (id: string) => GOALS.filter((g) => g.chapters.includes(id)).map((g) => g.label);

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const all = useAllChapters();
  const built = new Map((all ?? []).filter((c) => c.sections.length).map((c) => [c.id, c]));
  // A link to the chapter's first page (plain text only if a chapter failed to load).
  const to = (id: string, text: ReactNode) => {
    const c = built.get(id);
    return c ? <a href={href(c.id, c.sections[0].id)}>{text}</a> : <>{text}</>;
  };

  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch24/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;

  const record = Object.values(rollup(PLAN, progress.sections, reports));
  const sectionsDone = record.reduce((n, r) => n + r.sectionsDone, 0);
  const sectionsWritten = (all ?? []).reduce((n, c) => n + c.sections.length, 0);
  const labsRun = record.reduce((n, r) => n + r.labs, 0);
  const labsPassing = record.reduce((n, r) => n + r.labsPassing, 0);
  const gpus = gpuBackends(reports);

  return (
    <>
      <h2 style={{ marginTop: 0 }}><Translated vi="Những gì mang theo">What to carry forward</Translated></h2>
      <Translated
        vi={
        <p>
          Chương 24 không dạy kỹ thuật mới nào. Những gì nó để lại là một bảng và ba thói quen để đọc bất kỳ bảng nào
          giống nó, điều bạn sẽ cần mỗi khi một bảng thông số hay một tuyên bố benchmark đến tay bạn.
        </p>
        }
      >
      <p>
        Chapter 24 teaches no new technique. What it leaves is one table and three habits for reading any table like
        it, which you will need every time a data sheet or a benchmark claim reaches you.
      </p>
      </Translated>
      <Source section="24.2" pages="578" figs={tr('Fig. 24.1', 'Hình 24.1')} />
      <ul data-carry>
        <Translated
          vi={
          <li>
            <b>Tính lại trước khi trích dẫn.</b> Chia lại mọi tỉ lệ, và kiểm tra rằng cả hai số của một thương cùng đếm
            một thứ: một bo mạch, một chiều, một loại phép tính. Chính bảng của sách sai một lần về phép tính và một lần
            về đơn vị, ngoài ra còn trộn lẫn bo mạch và chiều truyền.
          </li>
          }
        >
        <li>
          <b>Recompute before you quote.</b> Divide every ratio again, and check that both numbers of a quotient count
          the same thing: one board, one direction, one kind of arithmetic. The book's own table fails once on
          arithmetic and once on units, and mixes boards and directions besides.
        </li>
        </Translated>
        <Translated
          vi={
          <li>
            <b>Hỏi xem một con số đếm cái gì.</b> Một con số được công bố là giá trị đỉnh, không phải phép đo, và nó đi kèm
            các điều kiện: "up to", mỗi GPU hay mỗi bo mạch, dense hay có sparsity, TF32 hay FP32. Số liệu cho thiết bị
            của bạn đến từ chính các lần chạy của bạn (lab 5.1, <code>python3 labs/pmpp.py doctor</code>), không bao giờ
            từ một bảng thông số.
          </li>
          }
        >
        <li>
          <b>Ask what a figure counts.</b> A published figure is a peak, not a measurement, and it comes with
          qualifiers: "up to", per GPU or per board, dense or with sparsity, TF32 or FP32. Numbers for your own device
          come from your own runs (lab 5.1, <code>python3 labs/pmpp.py doctor</code>), never from a data sheet.
        </li>
        </Translated>
        <Translated
          vi={
          <li>
            <b>Đọc các hàng như giới hạn cho kernel của bạn.</b> Hai hàng đầu cho ngưỡng của mục 5.1. Trên số liệu của
            sách nó tăng từ khoảng {fmt(THR_2007, 2)} FLOP/B lên {fmt(THR_2025)} ở FP32 và lên {fmt(THR_TC)} khi so với
            số liệu tensor core, mức mà không ô bộ nhớ chia sẻ (shared memory) nào đạt tới: việc tái sử dụng phải đến từ
            ô thanh ghi, batching hoặc ít byte hơn cho mỗi giá trị. Đường truyền tới host tăng ít nhất, nên chỉ đi qua nó
            một lần: giữ trọng số, KV cache và việc lấy mẫu trên device. Bước cuối đó là cách đọc của chúng tôi; sách chỉ
            in các con số.
          </li>
          }
        >
        <li>
          <b>Read the rows as limits on your kernel.</b> The top two give the threshold of section 5.1. On the book's
          figures it rose from about {fmt(THR_2007, 2)} FLOP/B to {fmt(THR_2025)} in FP32 and to {fmt(THR_TC)} against
          the tensor-core figure, which no shared-memory tile reaches: the reuse has to come from register tiles,
          batching or fewer bytes per value. The host link grew least, so cross it once: keep weights, the KV cache and
          sampling on the device. That last step is our reading; the book only prints the numbers.
        </li>
        </Translated>
      </ul>

      <h2><Translated vi="Bài kiểm tra tổng hợp">A mixed check</Translated></h2>
      <p>
        <Translated vi="Các câu này đưa ba thói quen vào việc trên những con số mới và những câu hỏi mới về bảng.">
          These put the three habits to work on new numbers and new questions about the table.
        </Translated>
        {asked ? tr(` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.`, ` Đến giờ trong chương này bạn đã trả lời đúng ${right} trên ${asked} câu, tính theo câu trả lời mới nhất của mỗi câu.`) : ''}
      </p>
      <Ask q={questionsWrap['ch24/wrap/q-gap']} />
      <Ask q={questionsWrap['ch24/wrap/q-perboard']} />
      <Ask q={questionsWrap['ch24/wrap/q-mi355x']} />
      <Ask q={questionsWrap['ch24/wrap/q-next']} />

      <h2><Translated vi="Bài tập của sách">The book's exercises</Translated></h2>
      <Translated
        vi={
        <p>
          Không có: chương 24 không có mã và không có phần bài tập. Thứ nó có là một bản đồ của sách và một bảng, và mỗi
          con số trong đó được tính trên <a href={href('ch24', 'start')}>trang bắt đầu</a>.
        </p>
        }
      >
      <p>
        There are none: chapter 24 has no code and no exercise section. What it has is a map of the book and one table,
        and each of their numbers is worked on the <a href={href('ch24', 'start')}>start page</a>.
      </p>
      </Translated>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="worked">
        <thead>
          <tr><th><Translated vi="Con số của sách">The book's number</Translated></th><th><Translated vi="Trang">Page</Translated></th><th><Translated vi="Được tính ở đâu tại đây">Where it is worked here</Translated></th></tr>
        </thead>
        <tbody>
          <tr><td><Translated vi="Ba bước: chương 2 đến 6, 7 đến 15 và 16 đến 23">Three steps: chapters 2 to 6, 7 to 15 and 16 to 23</Translated></td><td>577–578</td><td><Translated vi={'"Ba bước của sách" và câu hỏi đầu tiên'}>"The book's three steps" and the first question</Translated></td></tr>
          <tr><td><Translated vi="232x và 638x cho thông lượng tính toán">232x and 638x for compute throughput</Translated></td><td>578</td><td><Translated vi={'"Kiểm tra bảng", và câu hỏi về điều gì sẽ làm 638x đúng'}>"Check the table", and the question on what would make 638x right</Translated></td></tr>
          <tr><td><Translated vi="93x cho băng thông bộ nhớ, 128x cho dung lượng">93x for memory bandwidth, 128x for capacity</Translated></td><td>578</td><td><Translated vi={'"Kiểm tra bảng", và các tỉ lệ của nó được tính lại trên số liệu của NVIDIA'}>"Check the table", and its ratios recomputed on NVIDIA's figures</Translated></td></tr>
          <tr><td><Translated vi="8x cho đường truyền tới host">8x for the host link</Translated></td><td>578</td><td><Translated vi="dự đoán đầu tiên, và nút chọn thế hệ PCIe của các tỉ lệ tính lại">the first prediction, and the PCIe generation control of the recomputed ratios</Translated></td></tr>
          <tr><td><Translated vi="1,800 TB/s cho NVLink">1,800 TB/s for NVLink</Translated></td><td>578</td><td><Translated vi={'"Kiểm tra bảng", và nút "NVLink được đếm" của nó'}>"Check the table", and its "NVLink counted" control</Translated></td></tr>
          <tr><td><Translated vi="638× và 93× trong phần văn bản">638× and 93× in the running text</Translated></td><td>578</td><td><Translated vi="cùng phép kiểm tra đó: văn bản lặp lại bảng">the same check: the text repeats the table</Translated></td></tr>
        </tbody>
      </table>

      <h2><Translated vi="Chỗ PDF có vẻ sai">Where the PDF looks wrong</Translated></h2>
      <Translated
        vi={
        <p>
          Bốn trang, và phần lớn các lỗi nằm trong một bảng duy nhất. Hai ô mâu thuẫn với chính phép tính hoặc đơn vị
          của bảng (hai mục đầu dưới đây). Một nhãn và thế hệ đường truyền tới host năm 2007 sai so với tài liệu của
          NVIDIA (hai mục tiếp theo). Bốn mục nữa mô tả phần cứng lỏng lẻo hơn vẻ ngoài của bảng, và hai mục cuối là cách
          diễn đạt và một cách nhóm lỏng lẻo. Mỗi sửa chữa đến từ tài liệu của NVIDIA hoặc từ phép tính trên chúng (tốc
          độ PCIe lấy từ bảng của nhân Linux, facts A5); không có gì được đo. Đây là cách đọc của chúng tôi trên một bản
          PDF, nên nếu bản của bạn khác, hãy tin bản của bạn.
        </p>
        }
      >
      <p>
        Four pages, and most of the slips sit in the one table. Two cells contradict the table's own arithmetic or
        units (the first two items below). One label and the 2007 host-link generation are wrong against NVIDIA's
        documents (the next two). Four more items describe the hardware more loosely than the table looks, and the
        last two are wording and a loose grouping. Each correction comes from NVIDIA's documents or from arithmetic
        on them (PCIe rates from the Linux kernel's table, facts A5); nothing was measured. These are our readings of
        one PDF, so if your copy differs, trust your copy.
      </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch24-facts A1, A3 to A9, A11, A13, A15)', '2026-11-20 (ch24-facts A1, A3 đến A9, A11, A13, A15)')}>
        <Translated
          vi={
          <p>
            Hai điểm vẫn còn bỏ ngỏ: 86.4 GB/s dựa trên một bài đăng diễn đàn, và không trang NVIDIA nào chúng tôi đọc
            cho biết thế hệ PCI Express của 8800 GTX.
          </p>
          }
        >
        <p>
          Two points stay open: 86.4 GB/s rests on a forum post, and no NVIDIA page we read gives the 8800 GTX's PCI
          Express generation.
        </p>
        </Translated>
      </Today>
      <ul data-errata>
        <Translated
          vi={
          <li>
            Fig. 24.1 và đoạn văn bên cạnh (p. 578) ghi tỉ lệ tensor core là 638x. Trên các đầu vào được in, 2200 /
            0.345 = 6,377; so với 0.3456, giá trị đỉnh mà chính các hệ số của NVIDIA cho ra, nó là 6,366. NVIDIA in 2.2
            petaFLOPS cho hàng đó, nên ô sai là ô tỉ lệ.
          </li>
          }
        >
        <li>
          Fig. 24.1 and the text beside it (p. 578) give the tensor-core ratio as 638x. On the printed inputs 2200 /
          0.345 = 6,377; against 0.3456, the peak NVIDIA's own factors give, it is 6,366. NVIDIA prints 2.2 petaFLOPS
          for that row, so the ratio is the wrong cell.
        </li>
        </Translated>
        <Translated vi={<li>NVLink được in là "1,800 TB/s". Số liệu của NVIDIA là 1,800 GB/s, và nó đếm cả hai chiều, trong khi hàng PCIe đếm mỗi chiều: 900 GB/s so với 64.</li>}>
          <li>NVLink is printed as "1,800 TB/s". NVIDIA's figure is 1,800 GB/s, and it counts both directions, where the PCIe row counts each way: 900 GB/s against 64.</li>
        </Translated>
        {/* Non-breaking space: Wrap.test.ts requires the printed label to occur once in this file. */}
        <Translated vi={<li>Nhãn hàng "Tensor Core&nbsp;FP32" không phải là một độ chính xác mà NVIDIA liệt kê. Con số đó là TF32 trên Tensor Core, tính có sparsity; dense là một nửa.</li>}>
          <li>The row label "Tensor Core FP32" is not a precision NVIDIA lists. The figure is TF32 on Tensor Cores, counted with sparsity; dense is half.</li>
        </Translated>
        <Translated vi={<li>"Gen2 x16" cho bo mạch 2007: thông số của NVIDIA cho Tesla C870 1.5 GB ghi thế hệ 1, 4 GB/s mỗi chiều, khiến mức tăng của đường truyền tới host khoảng 16x, không phải 8x. Không trang NVIDIA nào chúng tôi đọc cho biết thế hệ của 8800 GTX.</li>}>
          <li>"Gen2 x16" for the 2007 board: NVIDIA's specification for the 1.5 GB Tesla C870 says generation 1, 4 GB/s each way, which makes the host-link growth about 16x, not 8x. No NVIDIA page we read gives the 8800 GTX's generation.</li>
        </Translated>
        <Translated vi={<li>Cột 2007 trộn hai bo mạch. 1.5 GB là của Tesla C870. Các trang NVIDIA chúng tôi đọc không in 86.4 GB/s; nó khớp với bus 384-bit ở 900 MHz của GeForce 8800 GTX 768 MB và một bài đăng diễn đàn năm 2007 trích bảng của NVIDIA. Bộ nhớ 800 MHz của C870 cho 76.8 GB/s.</li>}>
          <li>The 2007 column mixes two boards. 1.5 GB is the Tesla C870's. NVIDIA's pages we read do not print 86.4 GB/s; it matches the 768 MB GeForce 8800 GTX's 384-bit bus at 900 MHz and a 2007 forum post quoting NVIDIA's table. The C870's 800 MHz memory gives 76.8 GB/s.</li>
        </Translated>
        <Translated vi={<li>80 TFLOPS của cột 2025 khớp với số liệu tủ rack GB200 NVL72 của NVIDIA chia cho 72, và 192 GB của nó khớp với mức "up to" của một tài liệu kỹ thuật. Với các bo mạch HGX B200 đã xuất xưởng, tài liệu của NVIDIA cho 75 TFLOPS và 180 GB mỗi GPU.</li>}>
          <li>The 2025 column's 80 TFLOPS matches NVIDIA's GB200 NVL72 rack figure divided by 72, and its 192 GB matches the "up to" of one technical brief. For shipped HGX B200 boards NVIDIA's documents give 75 TFLOPS and 180 GB per GPU.</li>
        </Translated>
        <Translated vi={<li>Năm ở đầu các cột. G80 lên kệ vào tháng 11 năm 2006 và CUDA Toolkit đầu tiên ra sau đó vào tháng 6 năm 2007, nên "2007" là mốc của phần mềm. B200 được công bố vào tháng 3 năm 2024, và quý đầu tiên NVIDIA bán Blackwell số lượng lớn kết thúc vào tháng 1 năm 2025. Các trang của NVIDIA đã liệt kê những chip mới hơn.</li>}>
          <li>The column heads' years. The G80 reached shops in November 2006 and the first CUDA Toolkit followed in June 2007, so "2007" is the software date. The B200 was announced in March 2024, and NVIDIA's first quarter of Blackwell volume sales ended in January 2025. NVIDIA's pages already list newer parts.</li>
        </Translated>
        <Translated vi={<li>Bảng không nói các con số của nó là gì. Tất cả đều là giá trị đỉnh được công bố, không con số nào là phép đo.</li>}>
          <li>The table does not say what its figures are. All of them are published peaks, none a measurement.</li>
        </Translated>
        <Translated vi={<li>Cách diễn đạt, như được in: "a 18-year comparison" (chú thích hình, p. 578), "massively computing devices" (p. 578, có lẽ là massively parallel), "the answer is a definitely no" (p. 579).</li>}>
          <li>Wording, as printed: "a 18-year comparison" (caption, p. 578), "massively computing devices" (p. 578, presumably massively parallel), "the answer is a definitely no" (p. 579).</li>
        </Translated>
        <Translated vi={<li>Không phải lỗi, mà là một cách nhóm lỏng lẻo: chương 22 và 23 được tính là "advanced patterns and applications" (pp. 577–578).</li>}>
          <li>Not an error, a loose grouping: chapters 22 and 23 count as "advanced patterns and applications" (pp. 577–578).</li>
        </Translated>
      </ul>

      <h2><Translated vi="Những gì chương này bỏ qua">What this chapter left out</Translated></h2>
      <Translated
        vi={
        <p>
          Trang bắt đầu giữ lại ba bước và bảng. Mọi thứ khác trong chương là triển vọng của các tác giả, và nó ở lại
          trong sách.
        </p>
        }
      >
      <p>
        The start page keeps the three steps and the table. Everything else in the chapter is the authors' outlook,
        and it stays in the book.
      </p>
      </Translated>
      <Source section="24.2" pages="578–580" />
      <ul data-left>
        <Translated vi={<li>Tác động được nêu cho HPC, AI, phân tích dữ liệu, tài chính, sản xuất và y học, và chương 19 và 20 là các ví dụ học sâu (p. 578).</li>}>
          <li>The impact claimed for HPC, AI, data analytics, finance, manufacturing and medicine, and chapters 19 and 20 as the deep-learning examples (p. 578).</li>
        </Translated>
        <Translated
          vi={
          <li>
            Sự tăng số GPU cho mỗi tác vụ, các hyperscaler và hạ tầng mạng đi theo (p. 579). Các tác giả viết về hàng chục
            hay thậm chí hàng trăm nghìn GPU trong mỗi tác vụ huấn luyện. NVIDIA xác nhận một cụm huấn luyện 100,000
            GPU; việc một tác vụ dùng hết chúng, và mốc bắt đầu năm 2017, chúng tôi không xác minh được (ch24-facts A17).
            Bản thân NCCL nằm ở chương 23, §23.4 (pp. 563–568), mà khóa học này rút gọn trong trang{' '}
            <a href={href('ch23', '23.4')}>“Events, priorities, and who waits”</a>.
          </li>
          }
        >
        <li>
          The growth in GPUs per job, hyperscalers and the networking that followed (p. 579). The authors write of tens
          or even hundreds of thousands of GPUs in each training job. NVIDIA confirms a training cluster of 100,000
          GPUs; that one job used all of them, and the start date of 2017, we could not verify (ch24-facts A17). NCCL
          itself is in chapter 23, §23.4 (pp. 563–568), which this course condenses on the page{' '}
          <a href={href('ch23', '23.4')}>“Events, priorities, and who waits”</a>.
        </li>
        </Translated>
        <Translated vi={<li>Phạm vi bài toán đã mở rộng thế nào từ ấn bản đầu năm 2010, từ ma trận dày và Monte Carlo tới ma trận thưa và các phương pháp đồ thị (p. 579).</li>}>
          <li>How the range of problems broadened since the first edition of 2010, from dense matrices and Monte Carlo to sparse and graph methods (p. 579).</li>
        </Translated>
        <Translated
          vi={
          <li>
            Luận điểm rằng tính toán có thiết bị đo hỗ trợ nay dẫn đầu, với ba ví dụ: GPS và xe tự lái, MRI, PET và y học
            cá thể hóa, nguồn sáng và mặt nạ quang khắc tính toán (pp. 579–580), cùng phần mở rộng sang sáng tạo nội dung,
            khám phá thuốc và thời tiết (p. 580).
          </li>
          }
        >
        <li>
          The thesis that computing assisted by instruments now leads, with its three examples: GPS and self-driving,
          MRI, PET and individualised medicine, light sources and computed lithography masks (pp. 579–580), and its
          extension to content authoring, drug discovery and weather (p. 580).
        </li>
        </Translated>
        <Translated vi={<li>Truy cập song song vào bộ lưu trữ như cơ hội tiếp theo, và lời kết (p. 580).</li>}>
          <li>Parallel access to storage as the next opening, and the closing remarks (p. 580).</li>
        </Translated>
        <Translated vi={<li>Trang 581 để trống và p. 582 là trang phân cách của Part 4, các phụ lục.</li>}>
          <li>Page 581 is blank and p. 582 is the divider for Part 4, the appendices.</li>
        </Translated>
      </ul>
      <Translated
        vi={
        <p>
          Cố ý không xây dựng ở đây: một lab, vì không gì trong chương là một khẳng định mà kernel có thể kiểm tra (các
          lần sao chép từ host sang device được đo trong lab của {to('ch02', tr("chapter 2's", 'chương 2'))} và quay lại ở {to('ch23', tr('chapter 23', 'chương 23'))}); một
          hoạt cảnh có thuyết minh, vì không gì trong đó diễn ra theo thời gian; và bất kỳ cột 2007-so-với-hôm-nay nào cho
          AMD hay Apple. Apple không công bố số FLOPS nào, nên với Apple thì không thể xây dựng được.
        </p>
        }
      >
      <p>
        Not built here, on purpose: a lab, because nothing in the chapter is a claim a kernel can test (host-to-device
        copies are measured in {to('ch02', tr("chapter 2's", 'chương 2'))} labs and come back in {to('ch23', tr('chapter 23', 'chương 23'))}); a narrated
        animation, because nothing in it unfolds in time; and any 2007-against-today column for AMD or Apple. Apple
        publishes no FLOPS figure, so for Apple it cannot be built.
      </p>
      </Translated>

      <h2><Translated vi="Dự án đến lúc này">The project so far</Translated></h2>
      <Translated
        vi={
        <p data-project>
          Chương 24 không có lab và không thêm mốc dự án nào. Trên toàn khóa học bạn đã đánh dấu hoàn thành {sectionsDone} trên{' '}
          {sectionsWritten} mục đã viết.{' '}
          {gpus.length === 0
            ? 'Chưa có kết quả lab nào từ GPU được ghi trên máy này.'
            : `Trong ${labsRun} lab đã chạy trên máy này, ${labsPassing} lab qua trong bản work/ của chính bạn trên ${gpus.map((b) => NAME[b] ?? b).join(' và ')}; một lab bạn chưa bắt đầu được tính là chưa qua.`}{' '}
          <a href={href('ch24', 'start')}>Bản đồ các bước trên trang bắt đầu</a> cho thấy cùng bản ghi đó theo từng chương,
          kèm các mốc dự án của mỗi mục tiêu.
        </p>
        }
      >
      <p data-project>
        Chapter 24 has no lab and adds no milestone. Across the course you have marked {sectionsDone} of{' '}
        {sectionsWritten} written sections done.{' '}
        {gpus.length === 0
          ? 'No lab results from a GPU are recorded on this machine yet.'
          : `Of the ${labsRun} labs run on this machine, ${labsPassing} pass in your own work/ copy on ${gpus.map((b) => NAME[b] ?? b).join(' and ')}; a lab you have not started counts as not passing.`}{' '}
        The <a href={href('ch24', 'start')}>steps map on the start page</a> shows the same record chapter by chapter,
        with each goal's milestones.
      </p>
      </Translated>

      <h2><Translated vi="Đi tiếp đâu">Where next</Translated></h2>
      <Translated
        vi={
        <p>
          Sách tiếp tục với các phụ lục. Phụ lục B, kiến thức cơ bản về học sâu (pp. 601–622), được bỏ qua ở đây vì bạn
          đã biết nó. Phụ lục A và C có trang riêng trong khóa học này.
        </p>
        }
      >
      <p>
        The book continues with its appendices. Appendix B, deep learning basics (pp. 601–622), is skipped here because
        you already know it. Appendices A and C get their own pages in this course.
      </p>
      </Translated>
      <Source pages="582" />
      <Translated
        vi={
        <p>
          Sau chúng là ba module mà sách không có, mỗi module được xây từ các chương bạn đã đọc. Hãy chọn theo mục tiêu bạn
          đang theo đuổi.
        </p>
        }
      >
      <p>
        After them come three modules the book does not have, each built from chapters you will have read. Pick by the
        goal you are working on.
      </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Các module và mốc dự án của chúng là kế hoạch của khóa học này, không phải của sách.</p>}>
          <p>The modules and their milestones are this course's plan, not the book's.</p>
        </Translated>
      </Beyond>
      {/* On a phone the table scrolls inside its own box; the page does not. */}
      <div className="wide" style={{ overflowX: 'auto', maxWidth: '62rem', marginBottom: '1.4rem' }}>
        <table className="data" style={{ margin: 0, minWidth: '34rem' }} data-table="next">
          <thead>
            <tr><th><Translated vi="Tiếp theo">Next</Translated></th><th><Translated vi="Nó thêm gì">What it adds</Translated></th><th><Translated vi="Mã cùng loại trong PyTorch">Same kind of code in PyTorch</Translated></th></tr>
          </thead>
          <tbody>
            {NEXT.map((n) => (
              <tr key={n.id} data-next={n.id}>
                <td style={{ minWidth: '10rem' }}>
                  <b>{to(n.id, n.name)}</b>
                  <div className="muted">{n.pages}</div>
                  <div data-goals><Translated vi="cho:">for:</Translated> {servedBy(n.id).join(', ')}</div>
                </td>
                <td>{n.adds}</td>
                <td>{n.pytorch}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Translated
        vi={
        <p>
          Cột cuối là nơi PyTorch giữ loại kernel tương tự. Kernel CUDA được viết tay trong{' '}
          <code>aten/src/ATen/native/cuda/</code> và kernel Metal trong <code>aten/src/ATen/native/mps/kernels/</code>. Với
          AMD, bước build dịch mã nguồn CUDA bằng hipify (<code>tools/amd_build/build_amd.py</code>): cùng ý tưởng với
          một tệp <code>.cu</code> duy nhất cho CUDA và HIP của khóa học này, nhưng làm bằng một bước dịch thay vì một
          header. Header của khóa học này không làm mọi thứ khả chuyển: các intrinsic warp <code>_sync</code>, cooperative
          groups, header half-precision và intrinsic matrix-core vẫn cần guard hoặc mask có kiểu.
        </p>
        }
      >
      <p>
        The last column is where PyTorch keeps the same kind of kernel. CUDA kernels are hand-written in{' '}
        <code>aten/src/ATen/native/cuda/</code> and Metal kernels in <code>aten/src/ATen/native/mps/kernels/</code>. For
        AMD the build translates the CUDA sources with hipify (<code>tools/amd_build/build_amd.py</code>): the same idea
        as this course's one <code>.cu</code> file for CUDA and HIP, done by a translation step instead of a header.
        This course's header does not make everything portable: warp <code>_sync</code> intrinsics, cooperative
        groups, half-precision headers and matrix-core intrinsics still need guards or typed masks.
      </p>
      </Translated>
      <Today checked={tr('2026-11-20 against the PyTorch repository (ch24-facts D2, D3)', '2026-11-20 so với kho mã PyTorch (ch24-facts D2, D3)')}>
        <Translated
          vi={
          <p>
            Chỉ tên tệp. Toán tử nào chạy qua các tệp <code>.metal</code> này, và toán tử nào qua MPSGraph trong{' '}
            <code>operations/</code>, chưa được kiểm tra; đó là việc của module X3.
          </p>
          }
        >
        <p>
          File names only. Which operators run through these <code>.metal</code> files, and which through MPSGraph in{' '}
          <code>operations/</code>, was not checked; that is module X3's work.
        </p>
        </Translated>
      </Today>
    </>
  );
}
