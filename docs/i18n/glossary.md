# Vietnamese glossary

One term, one translation, across every chapter, narration script and lab README. Change a row
here first, then the content; do not invent a new rendering in one chapter.

Conventions:

- Address the reader as **bạn**. Plain, direct prose, as in the English. No honorific padding.
- **First use in a page:** where the Vietnamese column shows English in parentheses, give it
  on the term's first use in a page, e.g. "bộ nhớ chia sẻ (shared memory)"; after that, the
  Vietnamese alone. Terms without parentheses in the column never take them.
- **Kept in English:** identifiers, API names, commands, file paths, code, equations, units,
  backend names (CUDA, HIP, Metal, emu, race), product names (PyTorch, MLX, llama.cpp), and the
  project's verification vocabulary: `verified on Metal`, `emulator-checked`,
  `front-end-checked`, `unverified`. Those four are statements of fact with fixed meanings;
  never translate or paraphrase them.
- **Numbers exactly as in the English**, separators included (65,536 stays 65,536; 0.083 stays
  0.083). A Vietnamese reader of technical material reads them this way, and changing the
  separator would make 65,536 read as 65.536 elsewhere in the course.
- **Citations** ("ch. 5 §5.4, p. 110", "Fig. 3.11") may stay as written; translate only the
  prose around them ("Hình 3.11" for a figure named in a sentence is fine).
- "the book" is **sách**; "the book's example" is **ví dụ của sách**; "ours" (material not in
  the book) is **của khóa học này** or **của chúng tôi**.

| English | Tiếng Việt | Note |
|---|---|---|
| parallel computing | tính toán song song | |
| thread | luồng (thread) | |
| thread block, block | khối luồng, khối (block) | |
| grid | lưới (grid) | |
| kernel | kernel | |
| launch (a kernel) | khởi chạy | |
| host / device | host / device (thiết bị) | "device" as hardware in general: thiết bị |
| warp | warp | |
| wavefront (AMD) | wavefront | |
| SIMD group (Metal) | SIMD group | |
| lane | lane | |
| streaming multiprocessor (SM) | streaming multiprocessor (SM) | |
| core | lõi (core) | |
| register | thanh ghi (register) | |
| shared memory | bộ nhớ chia sẻ (shared memory) | |
| global memory | bộ nhớ toàn cục (global memory) | |
| constant memory | bộ nhớ hằng (constant memory) | |
| local memory | bộ nhớ cục bộ (local memory) | |
| threadgroup memory (Metal) | bộ nhớ threadgroup | |
| cache, L1, L2 | bộ nhớ đệm (cache), L1, L2 | |
| memory tier | tầng bộ nhớ | |
| occupancy | occupancy (mức chiếm dụng) | keep "occupancy" after first use |
| memory coalescing | hợp nhất truy cập bộ nhớ (coalescing) | "coalesced": được hợp nhất |
| bank conflict | xung đột bank (bank conflict) | |
| thread divergence | phân kỳ luồng (divergence) | |
| synchronization | đồng bộ hóa | |
| barrier | rào chắn (barrier) | |
| race condition, data race | race condition (tranh chấp dữ liệu) | |
| atomic operation | phép toán nguyên tử (atomic) | |
| privatization | riêng tư hóa (privatization) | |
| contention | tranh chấp | |
| tiling, tile | phân ô (tiling), ô (tile) | |
| halo | vùng biên (halo) | |
| thread coarsening | gộp việc cho luồng (thread coarsening) | |
| register tiling | phân ô thanh ghi | |
| reduction | phép rút gọn (reduction) | |
| scan, prefix sum | scan, tổng tiền tố (prefix sum) | inclusive/exclusive: bao gồm / loại trừ |
| filter, stream compaction | lọc, nén gọn (compaction) | |
| merge, co-rank | trộn (merge), co-rank | |
| sort, radix sort | sắp xếp, radix sort | |
| histogram | histogram | |
| convolution | tích chập (convolution) | |
| stencil | stencil | |
| sparse matrix | ma trận thưa | CSR, ELL, COO stay as is |
| graph traversal, frontier | duyệt đồ thị, frontier | |
| stride | bước nhảy (stride) | |
| row-major / column-major | thứ tự hàng (row-major) / thứ tự cột (column-major) | |
| throughput | thông lượng | |
| latency | độ trễ | |
| bandwidth | băng thông | |
| arithmetic intensity | cường độ tính toán | FLOP/B stays |
| memory-bound / compute-bound | bị giới hạn bởi bộ nhớ / bị giới hạn bởi tính toán | |
| roofline | roofline | |
| threshold (roofline knee) | ngưỡng | |
| ceiling, roof | trần, mái | |
| speed of light (bound) | tốc độ ánh sáng (speed of light) | |
| peak (throughput) | đỉnh, thông lượng đỉnh | |
| floating-point operation | phép toán dấu phẩy động | FLOP, FLOPS stay |
| matrix multiplication | phép nhân ma trận | GEMM, GEMV stay |
| naive (kernel) | ngây thơ | "kernel ngây thơ" |
| quantization / dequantize | lượng tử hóa / giải lượng tử hóa | |
| sampling | lấy mẫu | greedy, top-k, top-p stay |
| logits, token, attention, KV cache, softmax | giữ nguyên | |
| inference, decode, prefill | suy luận, decode, prefill | |
| batch | batch | "batching" stays |
| stream (CUDA) | stream | |
| pinned memory | bộ nhớ ghim (pinned memory) | |
| unified memory | unified memory | |
| emulator | trình giả lập (emulator) | |
| tolerance | dung sai | |
| driver (a lab's main program) | chương trình điều khiển (driver) | |
| dispatch | dispatch | |
| linear layer | lớp tuyến tính | |
| activations | activation | |
| stack frame | khung stack (stack frame) | |
| generalise | tổng quát hóa | |
| through-line project | dự án xuyên suốt | |
| benchmark | benchmark, đo hiệu năng | |
| lab | lab, bài thực hành | |
| starter / work / solution | mã khởi đầu (starter) / bản làm của bạn (work) / lời giải tham chiếu (solution) | directory names stay |
| planted defect | lỗi cài sẵn | |
| bounds check | kiểm tra biên | |
| prediction, predict first | dự đoán, dự đoán trước | |
| review (spaced) | ôn tập | |
| simplification (note) | điều được đơn giản hóa | |
| narration, caption | thuyết minh, phụ đề | |
| beat (of a narrated animation) | phần | |
| explorer, stepper | trình khám phá, trình chạy từng bước | |
| milestone (project) | mốc dự án | M0 to M10 stay |
| chunk (of a pipelined transfer) | chunk | |
| buffer | buffer | "bộ nhớ đệm" is cache; never use it for a buffer |
| staging buffer | buffer trung chuyển (staging buffer) | |
| copy (a transfer, memcpy) | sao chép, lần sao chép | "bản sao" only for a copy of data that is kept |
| copy engine | copy engine | |
| pageable memory | bộ nhớ phân trang (pageable) | |
| staged (pipelined in chunks) | chia giai đoạn (staged) | |
| fill / drain (a pipeline) | nạp / rút | |
| fuse, fused, kernel fusion | gộp, kernel gộp (fused), gộp kernel (kernel fusion) | not "hợp nhất", which is coalescing |
| ridge point (roofline) | điểm gãy (ridge point) | |
| subnormal | số subnormal | |
| event (CUDA) | event | |
