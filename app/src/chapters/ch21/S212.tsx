import { Ask, Predict, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useProgress } from '../../lib/progress';
import { href } from '../../lib/router';
import { PairsExplorer } from './PairsExplorer';
import { ScatterGatherScene } from './ScatterGatherScene';

// Section 21.2 of the app covers the book's §§21.2 to 21.4 (pp. 515–522): scatter versus
// gather, thread coarsening, and coalescing-aware ownership. Code shown here is the part
// below the marker line of each file in labs/ch21/snippets (snippets.test.ts ties them).

const SCATTER = `__constant__ float atoms[CHUNK_SIZE * 4];  // x, y, z, charge of each atom in the chunk

__global__ void cenergy(float* energygrid, dim3 grid, float gridspacing, float z, int numatoms) {
    int n = (blockIdx.x * blockDim.x + threadIdx.x) * 4;  // this thread's atom
    if (n >= numatoms * 4) return;                        // added: Fig. 21.5 has no bound
    int k = z / gridspacing;                              // which slice
    float dz = z - atoms[n + 2];  // the same for every point of the slice
    float dz2 = dz * dz;
    float charge = atoms[n + 3];
    for (int j = 0; j < (int)grid.y; j++) {
        float y = gridspacing * (float)j;
        float dy = y - atoms[n + 1];  // the same for every point of a row
        float dy2 = dy * dy;
        int row = grid.x * grid.y * k + grid.x * j;
        for (int i = 0; i < (int)grid.x; i++) {
            float x = gridspacing * (float)i;
            float dx = x - atoms[n];
            // every thread adds to every point: the update must be atomic
            atomicAdd(&energygrid[row + i], charge / sqrtf(dx * dx + dy2 + dz2));
        }
    }
}`;

const GATHER = `__constant__ float atoms[CHUNK_SIZE * 4];  // x, y, z, charge of each atom in the chunk

__global__ void cenergy(float* energygrid, dim3 grid, float gridspacing, float z, int numatoms) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= (int)grid.x || j >= (int)grid.y) return;  // added: Fig. 21.6 relies on padding
    int atomarrdim = numatoms * 4;
    int k = z / gridspacing;
    float y = gridspacing * (float)j;  // recomputed by every thread
    float x = gridspacing * (float)i;
    float energy = 0.0f;
    for (int n = 0; n < atomarrdim; n += 4) {  // every lane reads the same atom
        float dx = x - atoms[n];
        float dy = y - atoms[n + 1];
        float dz = z - atoms[n + 2];
        energy += atoms[n + 3] / sqrtf(dx * dx + dy * dy + dz * dz);
    }
    // one plain write, to a point no other thread touches
    energygrid[grid.x * grid.y * k + grid.x * j + i] += energy;
}`;

const PRINTED = `__constant__ float atoms[CHUNK_SIZE*4];
#define COARSEN_FACTOR 4
void __global__ cenergy(float *energygrid, dim3 grid, float gridspacing,
                        float z, int numatoms) {
    int i = blockIdx.x * blockDim.x*COARSEN_FACTOR + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int atomarrdim = numatoms * 4;
    int k = z / gridspacing;
    float y = gridspacing * (float) j;
    float x = gridspacing * (float) i;
    float energy0 = 0.0f;
    float energy1 = 0.0f;
    float energy2 = 0.0f;
    float energy3 = 0.0f;
    // add up the contributions of every atom
    for (int n=0; n<atomarrdim; n+=4) {
        float dx0 = x - atoms[n  ];
        float dx1 = dx0 +   gridspacing;
        float dx2 = dx0 + 2*gridspacing;
        float dx3 = dx0 + 3*gridspacing;
        float dy = y - atoms[n+1];
        float dz = z - atoms[n+2];
        float dysqdzsq = dy*dy + dz*dz;
        float charge = atoms[n+3];
        energy0 += charge / sqrtf(dx0*dx0 + dysqdzsq);
        energy1 += charge / sqrtf(dx1*dx1 + dysqdzsq);
        energy2 += charge / sqrtf(dx2*dx2 + dysqdzsq);
        energy3 += charge / sqrtf(dx3*dx3 + dysqdzsq);
    }
    energygrid[grid.x*grid.y*k + grid.x*j + i  ] += energy0;
    energygrid[grid.x*grid.y*k + grid.x*j + i+1] += energy1;
    energygrid[grid.x*grid.y*k + grid.x*j + i+2] += energy2;
    energygrid[grid.x*grid.y*k + grid.x*j + i+3] += energy3;
}`;

const ADJACENT = `__constant__ float atoms[CHUNK_SIZE * 4];
#define COARSEN_FACTOR 4

__global__ void cenergy(float* energygrid, dim3 grid, float gridspacing, float z, int numatoms) {
    // corrected: the thread index is scaled, so thread t owns points 4t .. 4t + 3
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * COARSEN_FACTOR;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= (int)grid.y) return;
    int atomarrdim = numatoms * 4;
    int k = z / gridspacing;
    float y = gridspacing * (float)j;
    float x = gridspacing * (float)i;
    float energy0 = 0.0f, energy1 = 0.0f, energy2 = 0.0f, energy3 = 0.0f;
    for (int n = 0; n < atomarrdim; n += 4) {
        float dx0 = x - atoms[n];  // one load of the atom serves four points
        float dx1 = dx0 + gridspacing;
        float dx2 = dx0 + 2 * gridspacing;
        float dx3 = dx0 + 3 * gridspacing;
        float dy = y - atoms[n + 1];
        float dz = z - atoms[n + 2];
        float dysqdzsq = dy * dy + dz * dz;  // shared by the four points of the row
        float charge = atoms[n + 3];
        energy0 += charge / sqrtf(dx0 * dx0 + dysqdzsq);
        energy1 += charge / sqrtf(dx1 * dx1 + dysqdzsq);
        energy2 += charge / sqrtf(dx2 * dx2 + dysqdzsq);
        energy3 += charge / sqrtf(dx3 * dx3 + dysqdzsq);
    }
    float* row = energygrid + grid.x * grid.y * k + grid.x * j;
    // four write statements; in each, neighbouring lanes are 4 floats apart
    if (i < (int)grid.x) row[i] += energy0;
    if (i + 1 < (int)grid.x) row[i + 1] += energy1;
    if (i + 2 < (int)grid.x) row[i + 2] += energy2;
    if (i + 3 < (int)grid.x) row[i + 3] += energy3;
}`;

const STRIDED = `__constant__ float atoms[CHUNK_SIZE * 4];
#define COARSEN_FACTOR 4

__global__ void cenergy(float* energygrid, dim3 grid, float gridspacing, float z, int numatoms) {
    int i = blockIdx.x * blockDim.x * COARSEN_FACTOR + threadIdx.x;  // line 05, as printed
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= (int)grid.y) return;
    int atomarrdim = numatoms * 4;
    int k = z / gridspacing;
    float y = gridspacing * (float)j;
    float x = gridspacing * (float)i;
    float energy0 = 0.0f, energy1 = 0.0f, energy2 = 0.0f, energy3 = 0.0f;
    for (int n = 0; n < atomarrdim; n += 4) {
        float dx0 = x - atoms[n];
        // the thread's points are blockDim.x apart, and so are their distances
        float dx1 = dx0 + blockDim.x * gridspacing;
        float dx2 = dx0 + 2 * blockDim.x * gridspacing;
        float dx3 = dx0 + 3 * blockDim.x * gridspacing;
        float dy = y - atoms[n + 1];
        float dz = z - atoms[n + 2];
        float dysqdzsq = dy * dy + dz * dz;
        float charge = atoms[n + 3];
        energy0 += charge / sqrtf(dx0 * dx0 + dysqdzsq);
        energy1 += charge / sqrtf(dx1 * dx1 + dysqdzsq);
        energy2 += charge / sqrtf(dx2 * dx2 + dysqdzsq);
        energy3 += charge / sqrtf(dx3 * dx3 + dysqdzsq);
    }
    float* row = energygrid + grid.x * grid.y * k + grid.x * j;
    int B = blockDim.x;
    // four write statements; in each, neighbouring lanes write neighbouring floats
    if (i < (int)grid.x) row[i] += energy0;
    if (i + B < (int)grid.x) row[i + B] += energy1;
    if (i + 2 * B < (int)grid.x) row[i + 2 * B] += energy2;
    if (i + 3 * B < (int)grid.x) row[i + 3 * B] += energy3;
}`;

/** Lines of PRINTED the erratum hunt refers to (1-based). */
export const PRINTED_LINES = { index: 5, writes: [30, 31, 32, 33] };

const STATUS = 'front-end-checked; emulator-checked (python3 labs/ch21/snippets/check.py); unverified on CUDA or HIP hardware'; // i18n-ignore: the verification vocabulary and a command
const TIERS: Record<string, Tier> = {
  atoms: 'const', energygrid: 'global', row: 'global',
  energy: 'reg', energy0: 'reg', energy1: 'reg', energy2: 'reg', energy3: 'reg', charge: 'reg', dysqdzsq: 'reg',
  dx: 'reg', dy: 'reg', dz: 'reg', dx0: 'reg', dx1: 'reg', dx2: 'reg', dx3: 'reg', dy2: 'reg', dz2: 'reg',
  i: 'reg', j: 'reg', k: 'reg', n: 'reg', x: 'reg', y: 'reg', B: 'reg', atomarrdim: 'reg',
};

export const questionsS212: Record<string, Question> = {
  'ch21/21.2/q-why-no-atomic': {
    kind: 'choice',
    id: 'ch21/21.2/q-why-no-atomic',
    prompt: (
      <Translated vi={<>Mọi luồng của kernel gather đều đọc mọi nguyên tử của chunk. Vì sao nó không cần phép toán nguyên tử nào?</>}>
        <>Every thread of the gather kernel reads every atom of the chunk. Why does it need no atomic operation?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Các lần đọc chung không thể gây race condition; chỉ các lần ghi vào một vị trí mới có thể, và mỗi luồng gather là người ghi duy nhất của điểm của nó.">Shared reads cannot race; only writes to one location can, and each gather thread is the only writer of its point.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Vì các nguyên tử nằm trong bộ nhớ hằng, nơi không kernel nào ghi được, nên cả các lần đọc lẫn các cập nhật lưới đều không thể gây race condition.">Because the atoms sit in constant memory, which no kernel can write, so neither the reads nor the grid updates can race.</Translated>, why: <Translated vi="Điều đó bảo vệ các nguyên tử. Câu hỏi là về lưới, thứ mà mỗi lần khởi chạy đều cập nhật.">That protects the atoms. The question is about the grid, which every launch updates.</Translated> },
      { label: <Translated vi="Vì host khởi chạy kernel một lần cho mỗi chunk, nên cập nhật của các luồng khác nhau không bao giờ chồng lấn về thời gian.">Because the host launches the kernel once per chunk, so the updates of different threads never overlap in time.</Translated>, why: <Translated vi="Một lần khởi chạy cho mỗi chunk sắp thứ tự các chunk. Bên trong một lần khởi chạy, các luồng vẫn chạy cùng nhau.">One launch per chunk orders the chunks. Inside one launch the threads still run together.</Translated> },
      { label: <Translated vi="Vì một phép += trên một float trong bộ nhớ toàn cục là một lệnh đọc-sửa-ghi duy nhất không thể chia cắt trên các GPU hiện nay.">Because a += on a float in global memory is a single indivisible read-modify-write instruction on current GPUs.</Translated>, why: <Translated vi="Nó là một lần nạp, một phép cộng và một lần lưu. Hai luồng có thể xen vào giữa chúng.">It is a load, an add and a store. Two threads can interleave between them.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một race condition cần hai luồng chạm vào một vị trí, trong đó ít nhất một luồng ghi. Mọi sự chia sẻ trong
            kernel gather đều là đọc: mọi lane đọc cùng một nguyên tử. Việc ghi là riêng tư: luồng (i, j) là luồng duy
            nhất từng ghi điểm (i, j). Mỗi lần khởi chạy cộng đóng góp của một chunk bằng một phép += thường vì không luồng
            nào khác chạm vào điểm đó trong lần khởi chạy.
          </p>
        }
      >
        <p>
          A race needs two threads touching one location with at least one of them writing. All the sharing in the gather
          kernel is reading: every lane reads the same atom. The writing is private: thread (i, j) is the only one that ever
          writes point (i, j). Each launch adds one chunk's contribution with a plain += because no other thread touches that
          point during the launch.
        </p>
      </Translated>
    ),
    source: 'ch. 21 §21.2, pp. 518–519',
  },
  'ch21/21.2/q-dilemma': {
    kind: 'choice',
    id: 'ch21/21.2/q-dilemma',
    prompt: (
      <Translated
        vi={
          <>
            Kernel gather tính lại <code>y</code> trong mọi luồng, việc mà Fig. 21.4 tuần tự đã đưa ra khỏi vòng lặp
            trong của nó. Vì sao chấp nhận điều đó?
          </>
        }
      >
        <>
          The gather kernel recomputes <code>y</code> in every thread, work that the sequential Fig. 21.4 had hoisted out of
          its inner loop. Why accept that?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Đưa ra ngoài vòng lặp đặt vòng lặp nguyên tử ở ngoài cùng, nên các luồng sẽ dùng chung đầu ra và cần phép toán nguyên tử. Tính lại y thì rẻ, và gộp việc cho luồng lấy lại phần lớn chi phí đó bên trong một luồng.">Hoisting puts the atom loop outermost, so threads would share outputs and need atomics. Recomputing y is cheap, and coarsening wins most of it back inside one thread.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Phép tính trên GPU là miễn phí so với lưu lượng bộ nhớ, nên việc thừa bao nhiêu cũng không tốn gì và không cần lấy lại.">Arithmetic on a GPU is free compared with memory traffic, so redundant work of any amount costs nothing and need not be recovered.</Translated>, why: <Translated vi="Không miễn phí: sách gọi đây là sự đánh đổi giữa lượng tính toán và mức độ song song.">Not free: the book calls this a trade-off between the amount of calculation and the degree of parallelism.</Translated> },
      { label: <Translated vi="Tính lại y là cần thiết cho tính đúng: một kernel tính y một lần sẽ cộng các nguyên tử của mỗi điểm theo thứ tự khác, nên tổng của nó sẽ khác Fig. 21.4.">Recomputing y is needed for correctness: a kernel that computed y once would add each point's atoms in another order, and so its sums would differ from Fig. 21.4.</Translated>, why: <Translated vi="Thứ tự khác chỉ làm thay đổi phần làm tròn float, điều mà chương này coi là đúng, nên tính đúng không phụ thuộc vào nó. Phép hoán đổi vòng lặp hợp lệ vì các lần lặp độc lập với nhau.">Another order changes only the float rounding, which this chapter counts as correct, so correctness does not depend on it. The loop interchange is legal because the iterations are independent.</Translated> },
      { label: <Translated vi="Thật ra nó không bị tính lại: trình biên dịch nhận ra sự lặp lại giữa các luồng và thay vào đó tính y một lần cho mỗi khối.">It is not really recomputed: the compiler spots the repetition across the threads and computes y once per block instead.</Translated>, why: <Translated vi="Trình biên dịch chỉ đưa phép tính ra ngoài vòng lặp trong phạm vi một luồng. Các luồng khác là những lần thực thi khác của mã.">A compiler hoists within one thread. Other threads are other executions of the code.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mã tuần tự đã tối ưu có được phần tiết kiệm nhờ đặt các nguyên tử ở ngoài cùng, và chính thứ tự đó biến một
            phiên bản song song thành scatter. Lối thoát của sách: song song hóa thứ tự kém tối ưu hơn trước, chấp nhận một
            chút phép tính thừa ở mỗi luồng, rồi lấy lại sự tái sử dụng bằng gộp việc cho luồng (thread coarsening), như
            phần còn lại của trang này làm.
          </p>
        }
      >
        <p>
          The optimised sequential code bought its savings by putting the atoms outermost, and that order is exactly what
          turns a parallel version into a scatter. The book's way out: parallelise the less optimised order first, accept a
          little redundant arithmetic per thread, then recover the reuse with thread coarsening, as the rest of this page does.
        </p>
      </Translated>
    ),
    source: 'ch. 21 §21.2, p. 518 (the dilemma) and pp. 518–519',
    vi: { source: 'ch. 21 §21.2, p. 518 (thế khó) và pp. 518–519' },
  },
  'ch21/21.2/q-deterministic': {
    kind: 'choice',
    id: 'ch21/21.2/q-deterministic',
    prompt: (
      <Translated
        vi={
          <>
            Với <code>torch.use_deterministic_algorithms(True)</code>, lượt lan truyền ngược của <code>nn.Embedding</code>{' '}
            chạy được trên một tensor CUDA nhưng báo lỗi trên một tensor MPS. Vì sao?
          </>
        }
      >
        <>
          With <code>torch.use_deterministic_algorithms(True)</code>, the backward pass of <code>nn.Embedding</code> runs
          on a CUDA tensor but raises an error on an MPS tensor. Why?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Trên CUDA, các hàng của mỗi token có một người ghi (sau một lần sắp xếp, khi có nhiều chỉ số); MPS cộng chúng vào gradient bằng phép toán nguyên tử float.">On CUDA each token's rows get one writer (after a sort, for many indices); MPS adds them to the gradient with float atomics.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Metal không có phép toán nguyên tử float, nên backend MPS lùi về một kernel CPU, điều mà chế độ tất định không cho phép.">Metal has no float atomics, so the MPS backend falls back to a CPU kernel, which deterministic mode does not allow.</Translated>, why: <Translated vi="Nó có (atomic_float, Apple7 trở lên), và kernel MPS dùng chúng; đó là lý do nó không tất định.">It has them (atomic_float, Apple7 and later), and the MPS kernel uses them; that is why it is not deterministic.</Translated> },
      { label: <Translated vi="CUDA bảo đảm thứ tự các phép cộng nguyên tử vào một địa chỉ, nên đường nguyên tử của nó tất định; Metal không hứa như vậy.">CUDA guarantees the order of atomic additions to one address, so its atomic path is deterministic; Metal makes no such promise.</Translated>, why: <Translated vi="NVIDIA ghi rằng thứ tự đó là không xác định. Vì thế PyTorch bỏ qua đường nguyên tử gộp của nó ở chế độ tất định.">NVIDIA documents that order as undefined. PyTorch skips its fused atomic path in deterministic mode for that reason.</Translated> },
      { label: <Translated vi="Backend MPS tích lũy gradient ở độ chính xác một nửa, và tổng độ chính xác một nửa không thể làm cho lặp lại được trên bất kỳ GPU nào.">The MPS backend accumulates the gradient in half precision, and half-precision sums cannot be made repeatable on any GPU.</Translated>, why: <Translated vi="Vấn đề không phải độ chính xác; mà là thứ tự các phép cộng.">Precision is not the issue; the order of the additions is.</Translated> },
    ],
    explain: (
      <>
        <Translated
          vi={
            <p>
              Một framework, hai lựa chọn. Trên CUDA và ROCm, với hơn 3,072 chỉ số, PyTorch sắp xếp chúng, việc này chia các
              hàng vào ngăn theo id token, rồi cộng mỗi ngăn với một người ghi; với ít chỉ số hơn, mỗi khối sở hữu một dải cột
              đặc trưng và một warp dẫn đầu gộp các token trùng nhau, cũng theo cách tất định. Nó chỉ đi đường nguyên tử hợp
              nhất khi điều đó có lợi và chế độ tất định đang tắt. Trên MPS nó dùng scatter với phép toán nguyên tử float.
              Thứ tự các phép toán nguyên tử đến là không xác định và phép cộng float không có tính kết hợp, nên các bit có
              thể đổi từ lần chạy này sang lần chạy khác. Chế độ tất định đòi hỏi gather.
            </p>
          }
        >
          <p>
            One framework, two choices. On CUDA and ROCm, with more than 3,072 indices PyTorch sorts them, which bins the
            rows by token id, and sums each bin with one writer; with fewer, each block owns a range of feature columns and
            a leader warp merges duplicate tokens, again deterministically. It takes a fused atomic path only when that
            helps and deterministic mode is off.
            On MPS it scatters with float atomics. The order in which atomics land is undefined and float addition is not
            associative, so the bits can change from run to run. Deterministic mode asks for the gather.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              <code>get_rows_back</code> của llama.cpp (CUDA và HIP, chỉ cho huấn luyện) là gather tổng trực tiếp của
              Fig. 21.6 với một phép so sánh bằng thay cho 1/r; chúng tôi không tìm thấy phiên bản Metal nào.
            </p>
          }
        >
          <p>
            llama.cpp's <code>get_rows_back</code> (CUDA and HIP, training only) is Fig. 21.6's direct-summation gather with
            an equality test in place of 1/r; we found no Metal version.
          </p>
        </Translated>
      </>
    ),
    source: 'ours, not the book: notes/today/ch21-facts.md D1 to D3; infra A1',
    vi: { source: 'của khóa học này, không phải của sách: notes/today/ch21-facts.md D1 đến D3; infra A1' },
  },
  'ch21/21.2/q-ex2': {
    kind: 'choice',
    id: 'ch21/21.2/q-ex2',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 2. Gộp việc theo hệ số tám cắt số lần nạp bộ nhớ hằng cho mỗi nguyên tử từ 32 (tám luồng Fig. 21.6)
            xuống 4. Việc nào trên mỗi điểm vẫn giữ nguyên y như cũ?
          </>
        }
      >
        <>
          Exercise 2. Coarsening by eight cuts the constant-memory loads per atom from 32 (eight Fig. 21.6 threads) to 4.
          Which per-point work stays exactly as it was?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Mỗi điểm vẫn tự làm phép chia, căn bậc hai, dx², một phép cộng bên trong căn và phép tích lũy của riêng nó.">Each point still does its own division, square root, dx², one addition inside the root and accumulate.</Translated>, correct: true },
      { label: <Translated vi="Phép trừ tạo ra dy, thứ mà mỗi điểm trong tám điểm của hàng vẫn cần tính.">The subtraction that forms dy, which every one of the eight points of the row still needs to compute.</Translated>, why: <Translated vi="dy giống nhau cho cả hàng; luồng đã gộp việc tính nó một lần cho mỗi nguyên tử, không phải tám lần.">dy is the same for the whole row; the coarsened thread computes it once per atom, not eight times.</Translated> },
      { label: <Translated vi="Phép nhân dy·dy: mỗi điểm trong tám điểm vẫn bình phương bản dy của riêng nó trước khi lấy căn.">The multiplication dy·dy: each of the eight points still squares its own copy of dy before taking the root.</Translated>, why: <Translated vi="Nó được gộp vào dysqdzsq một lần cho mỗi nguyên tử.">It is folded into dysqdzsq once per atom.</Translated> },
      { label: <Translated vi="Nhánh của vòng lặp, vì mỗi điểm trong tám điểm vẫn chạy vòng lặp riêng qua các nguyên tử của chunk.">The loop branch, because each of the eight points still runs its own loop over the atoms of the chunk.</Translated>, why: <Translated vi="Một luồng chạy một vòng lặp: một nhánh cho mỗi nguyên tử thay vì tám.">One thread runs one loop: one branch per atom instead of eight.</Translated> },
    ],
    explain: (
      <>
        <Translated
          vi={
            <p>
              Đếm từ đoạn mã, cho mỗi nguyên tử: tám luồng Fig. 21.6 thực hiện 32 lần nạp, 24 phép trừ, 24 phép nhân, 24
              phép cộng, 8 phép chia, 8 phép căn bậc hai và 8 nhánh vòng lặp. Một luồng gộp việc theo hệ số tám thực hiện 4
              lần nạp, 3 phép trừ, 10 phép nhân (16 nếu các tích m·gridspacing vẫn nằm trong vòng lặp), 24 phép cộng, 8 phép
              chia, 8 phép căn bậc hai và 1 nhánh. Đặt F = 8 trong bảng phép toán ở trên để thấy điều đó.
            </p>
          }
        >
          <p>
            Counted from the listing, per atom: eight Fig. 21.6 threads make 32 loads, 24 subtractions, 24 multiplications, 24
            additions, 8 divisions, 8 square roots and 8 loop branches. One thread coarsened by eight makes 4 loads, 3
            subtractions, 10 multiplications (16 if the m·gridspacing products stay in the loop), 24 additions, 8 divisions, 8
            square roots and 1 branch. Set F = 8 in the operation table above to see it.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Phép cộng ở cả hai bên đều là 24, nhưng chúng không phải cùng các phép cộng. Một điểm Fig. 21.6 làm hai phép
              cộng bên trong căn; một điểm đã gộp việc làm một. Phép thứ hai, dy² + dz², là phép mà gộp việc chia sẻ: nó được
              tạo một lần cho mỗi nguyên tử, và 7 phép cộng tạo ra dx1 đến dx7 bù vào phần chênh lệch.
            </p>
          }
        >
          <p>
            The additions total 24 on both sides, but they are not the same additions. A Fig. 21.6 point makes two inside
            its root; a coarsened point makes one. The second, dy² + dz², is the one coarsening shares: it is formed once per
            atom, and the 7 additions that form dx1 to dx7 take up the difference.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Thứ mà gộp việc loại bỏ là thứ các điểm dùng chung. Mỗi điểm vẫn cần khoảng cách riêng và 1/r riêng, và phép
              chia cùng căn bậc hai là những phép toán đắt còn lại trong vòng lặp.
            </p>
          }
        >
          <p>
            What coarsening removes is what the points share. Every point still needs its own distance and its own 1/r, and
            the division and square root are the expensive operations left in the loop.
          </p>
        </Translated>
      </>
    ),
    source: 'ch. 21 Exercise 2, p. 528; notes/today/ch21-facts.md E1',
    vi: { source: 'ch. 21 Bài tập 2, p. 528; notes/today/ch21-facts.md E1' },
  },
  'ch21/21.2/q-printed': {
    kind: 'choice',
    id: 'ch21/21.2/q-printed',
    prompt: (
      <Translated
        vi={
          <>
            Fig. 21.8 như bản in tính <code>i = blockIdx.x*blockDim.x*COARSEN_FACTOR + threadIdx.x</code> và ghi{' '}
            <code>i</code>, <code>i+1</code>, <code>i+2</code>, <code>i+3</code>. Với <code>blockDim.x = 32</code>, khối 0
            ghi những điểm nào trong 128 điểm của nó?
          </>
        }
      >
        <>
          Fig. 21.8 as printed computes <code>i = blockIdx.x*blockDim.x*COARSEN_FACTOR + threadIdx.x</code> and writes{' '}
          <code>i</code>, <code>i+1</code>, <code>i+2</code>, <code>i+3</code>. With <code>blockDim.x = 32</code>, which of
          its 128 points does block 0 write?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Các điểm 0 đến 34: phần lớn trong số đó (3 đến 31) bị ghi bốn lần bằng một phép += thường, còn các điểm 35 đến 127 không bao giờ được ghi.">Points 0 to 34: most of them (3 to 31) four times with a plain +=, and points 35 to 127 never.</Translated>, correct: true },
      { label: <Translated vi="Các điểm 0 đến 127, mỗi điểm một lần: luồng t sở hữu 4t đến 4t + 3, bốn điểm liền kề cho mỗi luồng.">Points 0 to 127, each once: thread t owns 4t to 4t + 3, four adjacent points per thread.</Translated>, why: <Translated vi="Đó là quyền sở hữu dự định, và nó cần threadIdx.x nhân với 4.">That is the intended ownership, and it needs threadIdx.x multiplied by 4.</Translated> },
      { label: <Translated vi="Các điểm 0 đến 31, mỗi điểm một lần: mỗi luồng một điểm, như trong Fig. 21.6, vì i có cùng dạng.">Points 0 to 31, each once: one point per thread, as in Fig. 21.6, since i has the same form.</Translated>, why: <Translated vi="Mỗi luồng ghi bốn điểm, từ i đến i+3.">Each thread writes four points, i to i+3.</Translated> },
      { label: <Translated vi="Các điểm 0 đến 127, mỗi điểm bốn lần bằng một phép += thường, nên dùng phép toán nguyên tử sẽ làm nó đúng.">Points 0 to 127, each four times with a plain +=, so atomics would make it right.</Translated>, why: <Translated vi="Luồng 31 bắt đầu ở 31 và dừng ở 34; không gì vươn xa hơn.">Thread 31 starts at 31 and stops at 34; nothing reaches beyond.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Luồng t bắt đầu ở t, sau luồng hàng xóm một điểm, và ghi bốn điểm, nên khối phủ từ 0 đến 31 + 3 = 34. Các điểm
            bên trong được ghi bởi bốn luồng, mỗi luồng cộng toàn bộ thế, bằng một phép += không nguyên tử. 93 điểm còn lại
            trong đoạn của khối không bao giờ được chạm tới. Khoảng cách của mỗi luồng đúng cho các điểm nó ghi; quyền sở
            hữu thì sai. Văn bản ở p. 521 mô tả các luồng cách nhau bốn điểm, đúng là điều mà việc nhân threadIdx.x với hệ
            số cho ra.
          </p>
        }
      >
        <p>
          Thread t starts at t, one point after its neighbour, and writes four points, so the block covers 0 to 31 + 3 = 34.
          Interior points are written by four threads, each adding the full potential, with a non-atomic +=. The other 93
          points of the block's stretch are never touched. Each thread's distances are right for the points it writes; the
          ownership is wrong. The text on p. 521 describes threads four points apart, which is what scaling threadIdx.x by
          the factor gives.
        </p>
      </Translated>
    ),
    source: 'ch. 21 Fig. 21.8, p. 520, against p. 521; notes/today/ch21-facts.md E2',
    vi: { source: 'ch. 21 Fig. 21.8, p. 520, đối chiếu với p. 521; notes/today/ch21-facts.md E2' },
  },
  'ch21/21.2/q-ex3': {
    kind: 'choice',
    id: 'ch21/21.2/q-ex3',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 3, cụ thể hóa. Fig. 21.10 với <code>blockDim.x = 128</code> trên một lát rộng 1,024 điểm. Bạn nâng hệ
            số gộp việc từ 4 lên 16. Bạn mất gì?
          </>
        }
      >
        <>
          Exercise 3, made concrete. Fig. 21.10 with <code>blockDim.x = 128</code> on a slice 1,024 points wide. You raise
          the coarsening factor from 4 to 16. What do you lose?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Một khối trải 2,048 vị trí x, nên một nửa trong 16 điểm của mỗi luồng rơi ra ngoài hàng: 16 bộ tích lũy cho 8 điểm việc.">A block spans 2,048 x positions, so half of each thread's 16 points fall past the row: 16 accumulators for 8 points of work.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Không mất gì: gộp việc loại bỏ các lần nạp thừa, nên trên một lát rộng thế này, hệ số lớn hơn ít nhất cũng nhanh bằng hệ số nhỏ hơn.">Nothing: coarsening removes redundant loads, so a larger factor is at least as fast as a smaller one on a slice this wide.</Translated>, why: <Translated vi="Nó loại bỏ các lần nạp thừa, không loại bỏ các ô rảnh; những điểm thừa ngoài hàng là chi phí thuần.">It removes redundant loads, not idle slots; the extra points past the row are pure cost.</Translated> },
      { label: <Translated vi="Các lần ghi không còn được hợp nhất, vì 16 điểm của mỗi luồng giờ nằm xa nhau hơn và một câu lệnh bị trải rộng ra.">The writes stop being coalesced, because each thread's 16 points now lie further apart and a statement spreads out.</Translated>, why: <Translated vi="Trong Fig. 21.10, bước nhảy vẫn là blockDim.x, nên mỗi câu lệnh ghi của một warp vẫn là một dải liên tục. Mất hợp nhất là một chi phí thật của việc giao nhiều việc hơn cho mỗi luồng, nhưng nó đến từ quyền sở hữu liền kề, điều mà Fig. 21.10 tránh được.">In Fig. 21.10 the stride is still blockDim.x, so each write statement of a warp stays one contiguous run. Lost coalescing is a real cost of more work per thread, but it comes from adjacent ownership, which Fig. 21.10 avoids.</Translated> },
      { label: <Translated vi="Kernel giờ cần phép toán nguyên tử, vì 16 điểm của một luồng vươn vào đoạn mà khối kế tiếp sở hữu.">The kernel now needs atomics, because a thread's 16 points reach into the stretch the next block owns.</Translated>, why: <Translated vi="Mỗi điểm vẫn có đúng một chủ sở hữu; khối kế tiếp bắt đầu sau đoạn của khối này.">Each point still has exactly one owner; the next block starts after this block's stretch.</Translated> },
    ],
    explain: (
      <>
        <Translated
          vi={
            <p>
              Sách nêu hai chi phí của việc giao nhiều việc hơn cho mỗi luồng (p. 521): nhiều thanh ghi hơn cho mỗi luồng, và
              mất hợp nhất nếu các đầu ra của một luồng liền kề nhau. Fig. 21.10 tránh được cái thứ hai, nên ở đây bạn trả
              cái thứ nhất: mười sáu bộ tích lũy và mười sáu khoảng cách cho mỗi luồng, điều có thể làm giảm số warp vừa trên
              SM.
            </p>
          }
        >
          <p>
            The book names two costs of more work per thread (p. 521): more registers per thread, and the loss of coalescing
            if a thread's outputs are adjacent. Fig. 21.10 avoids the second, so here you pay the first: sixteen accumulators
            and sixteen distances per thread, which can lower how many warps fit.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Chi phí thứ ba, được vẽ trong Fig. 21.9 là "padding waste" (lãng phí do đệm) nhưng không được bàn trong văn
              bản, là độ thô của phân chia. Đoạn của một khối gồm hệ số × blockDim.x điểm, nên một hàng không phải bội số của
              nó để lại các luồng rảnh hoặc làm việc trên những điểm không tồn tại. Với một khối cho mỗi hàng, cũng có ít song
              song hơn để chia ra.
            </p>
          }
        >
          <p>
            A third cost, drawn in Fig. 21.9 as "padding waste" but not discussed in the text, is granularity. A block's
            stretch is factor × blockDim.x points, so a row that is not a multiple of it leaves threads idle or working on
            points that do not exist. With one block per row there is also less parallelism to go round.
          </p>
        </Translated>
      </>
    ),
    source: 'ch. 21 Exercise 3, p. 528; p. 521; Fig. 21.9, p. 522',
    vi: { source: 'ch. 21 Bài tập 3, p. 528; p. 521; Fig. 21.9, p. 522' },
  },
  'ch21/21.2/q-wavefront': {
    kind: 'choice',
    id: 'ch21/21.2/q-wavefront',
    prompt: (
      <Translated vi={<>Trên một GPU AMD Instinct, một wavefront có 64 lane. Bước nhảy của Fig. 21.10 ở đó có còn đúng không?</>}>
        <>On an AMD Instinct GPU a wavefront has 64 lanes. Is Fig. 21.10's stride still right there?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Đúng. Bước nhảy là blockDim.x, không phải độ rộng warp, và với blockDim.x từ 64 trở lên, một câu lệnh là một dải 256 byte.">Yes. The stride is blockDim.x, not the warp width, and at blockDim.x of 64 or more a statement is one 256-byte run.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Không: trên AMD, bước nhảy phải là 64, độ rộng wavefront, thay vì blockDim.x, nếu không các lane của một wavefront sẽ chồng lấn.">No: on AMD the stride has to be 64, the wavefront width, instead of blockDim.x, or the lanes of a wavefront overlap.</Translated>, why: <Translated vi="blockDim.x đã là bước nhảy, và nó được chọn là bội số của độ rộng wavefront.">blockDim.x already is the stride, and it is chosen as a multiple of the wavefront width.</Translated> },
      { label: <Translated vi="Không: AMD hợp nhất theo từng nửa wavefront 32 lane, nên bước nhảy phải là 32 float, bất kể blockDim.x là bao nhiêu.">No: AMD coalesces per half wavefront of 32 lanes, so the stride has to be 32 floats, whatever blockDim.x is.</Translated>, why: <Translated vi="Half-warp là một đơn vị của NVIDIA từ năm 2007. AMD ghi nhận việc gom địa chỉ của một wavefront thành các dòng bộ nhớ đệm.">Half-warps are a 2007 NVIDIA unit. AMD documents grouping a wavefront's addresses into cache lines.</Translated> },
      { label: <Translated vi="Không quan trọng: GPU AMD không hợp nhất các truy cập toàn cục, nên mọi mẫu ghi đều tốn cùng một số dòng.">It does not matter: AMD GPUs do not coalesce global accesses, so every write pattern costs the same number of lines.</Translated>, why: <Translated vi="AMD ghi nhận chính việc gom đó, thành các dòng 64 hoặc 128 byte.">AMD documents exactly that grouping, into 64- or 128-byte lines.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lane t của một câu lệnh ghi base + t + m·blockDim.x, nên các lane cạnh nhau luôn ghi các float cạnh nhau, bất kể
            độ rộng nhóm. Thứ mà độ rộng thay đổi là một câu lệnh vươn xa đến đâu: 128 byte cho 32 lane, 256 byte cho 64.
            Với <code>blockDim.x = 32</code>, một wavefront 64 lane chứa hai hàng của khối, nên mỗi câu lệnh trở thành hai
            dải, cách nhau một hàng. Đặt chế độ xem câu lệnh ghi thành 64 lane và blockDim.x = 32 để thấy điều đó.
          </p>
        }
      >
        <p>
          Lane t of a statement writes base + t + m·blockDim.x, so neighbouring lanes always write neighbouring floats, whatever
          the group width. What the width changes is how far one statement reaches: 128 bytes for 32 lanes, 256 bytes for
          64. With <code>blockDim.x = 32</code> a 64-lane wavefront holds two rows of the block, so each statement becomes two
          runs, a row apart. Set the write-statement view to 64 lanes and blockDim.x = 32 to see it.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch21-facts.md B1, B3, E2; ch05-facts B5', // i18n-ignore: a citation
  },
  'ch21/21.2/q-ex4': {
    kind: 'choice',
    id: 'ch21/21.2/q-ex4',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 4. Bạn giữ quyền sở hữu liền kề và thay bốn phép <code>+=</code> vô hướng bằng một lần nạp{' '}
            <code>float4</code>, bốn phép cộng và một lần lưu <code>float4</code> cho mỗi luồng. Điều gì phải đúng với lát?
          </>
        }
      >
        <>
          Exercise 4. You keep adjacent ownership and replace the four scalar <code>+=</code> with one <code>float4</code>{' '}
          load, four adds and one <code>float4</code> store per thread. What must be true of the slice?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Mọi hàng phải bắt đầu tại một ranh giới 16 byte: grid.x là bội số của 4, hoặc các hàng được cấp phát với một pitch.">Every row must start on a 16-byte boundary: grid.x a multiple of 4, or rows allocated with a pitch.</Translated>, correct: true },
      { label: <Translated vi="Không gì cả: mọi buffer float đều có thể truy cập như float4 tại bất kỳ địa chỉ nào; một buffer lệch căn chỉ chậm hơn.">Nothing: any float buffer can be accessed as float4 at any address; a misaligned one is only slower.</Translated>, why: <Translated vi="Một truy cập 16 byte cần một địa chỉ căn theo 16 byte; hướng dẫn của NVIDIA nói như vậy.">A 16-byte access needs a 16-byte-aligned address; NVIDIA's guidance says so.</Translated> },
      { label: <Translated vi="blockDim.x phải là bội số của 4, để mọi luồng của khối nhận được trọn một float4.">blockDim.x must be a multiple of 4, so that every thread of the block gets a whole float4.</Translated>, why: <Translated vi="Căn chỉnh là chuyện của địa chỉ, không phải chuyện một khối có bao nhiêu luồng.">Alignment is about addresses, not about how many threads a block has.</Translated> },
      { label: <Translated vi="Khối phải đưa các đầu ra của nó vào bộ nhớ chia sẻ trước rồi lưu chúng từ đó như một ô (tile).">The block must first stage its outputs in shared memory and store them from there as one tile.</Translated>, why: <Translated vi="Không cần: mỗi luồng đã sở hữu bốn float liền kề trong thanh ghi.">Not needed: each thread already owns four adjacent floats in registers.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Bốn điểm của luồng t bắt đầu ở phần tử 4t trong đoạn của nó, và đoạn đó bắt đầu xa thêm trọn một hàng cho mỗi
            hàng. Nếu <code>grid.x</code> không phải bội số của 4, các float4 của hàng thứ hai bị lệch căn. Với các hàng được
            căn chỉnh, một warp khi đó ghi 512 byte liên tục trong một yêu cầu: 16 sector, mức tối ưu của NVIDIA cho truy
            cập 128 bit. Việc ép kiểu <code>float*</code> thành <code>float4*</code> của sách là thực hành từ blog của
            NVIDIA, không có trong các hướng dẫn của hãng; trên Metal, cách được ghi nhận trong tài liệu là khai báo buffer
            là <code>device float4*</code>.
          </p>
        }
      >
        <p>
          Thread t's four points start at element 4t of its stretch, and the stretch starts a whole row further on for every
          row. If <code>grid.x</code> is not a multiple of 4, the second row's float4s are misaligned. With aligned rows a warp
          then writes 512 contiguous bytes in one request: 16 sectors, NVIDIA's optimum for 128-bit accesses. The book's
          cast of a <code>float*</code> to <code>float4*</code> is NVIDIA blog practice, not in its guides; on Metal the
          documented way is to declare the buffer <code>device float4*</code>.
        </p>
      </Translated>
    ),
    source: 'ch. 21 Exercise 4, p. 528; p. 521; notes/today/ch21-facts.md A9, C4; infra-facts A6, C7',
    vi: { source: 'ch. 21 Bài tập 4, p. 528; p. 521; notes/today/ch21-facts.md A9, C4; infra-facts A6, C7' },
  },
};

/**
 * The coarsened explorer, the fix and the corrected listing open only after the erratum
 * question has an answer, so none of them gives it away. (The first explorer offers only
 * scatter and gather for the same reason.)
 */
function AfterPrinted() {
  const { progress } = useProgress();
  if (!progress.answers['ch21/21.2/q-printed'])
    return (
      <Translated
        vi={
          <p className="narrate" data-gate="q-printed">
            Hãy trả lời câu hỏi tìm lỗi in ở trên trước. Trình khám phá và cách sửa phía sau chạy dòng 05 như bản in và sẽ để
            lộ đáp án.
          </p>
        }
      >
        <p className="narrate" data-gate="q-printed">
          Answer the erratum question above first. The explorer and the fix that follow run the printed line 05 and would
          give it away.
        </p>
      </Translated>
    );
  return (
    <>
      <PairsExplorer
        title={tr('Who writes where: coarsened kernels', 'Ai ghi vào đâu: các kernel đã gộp việc')}
        task={
          <Translated vi="Bắt đầu với dòng 05 như bản in và tìm số người ghi mỗi điểm lớn hơn 1 cùng các điểm gạch chéo. Rồi chuyển sang hai cách ánh xạ đã sửa.">
            Start with the printed line 05 and look for writers per point above 1 and hatched points. Then switch to the two corrected mappings.
          </Translated>
        }
        views={['mapping']}
        initialKernel="printed" /* i18n-ignore: a kernel id */
        kernels={['printed', 'adjacent', 'strided']}
        foot={
          <Translated vi="Mô hình: src/sim/dcs.ts. Các kernel đã gộp việc tô các điểm của mỗi luồng cùng một sắc độ; bảng bên dưới hình đếm khối lượng việc cho mỗi nguyên tử từ các đoạn mã.">
            Model: src/sim/dcs.ts. Coarsened kernels draw each thread's points in the same shade; the table below the drawing counts work per atom from the listings.
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Cách sửa là một phép nhân: nhân chỉ số luồng với hệ số, để luồng t bắt đầu ở điểm 4t trong đoạn của khối nó. Đó là
            quyền sở hữu mà văn bản mô tả (p. 521), và là phiên bản mà lab 21.1 cung cấp làm mốc cơ sở. Chuyển trình khám phá
            sang "coarsened, adjacent": mọi điểm có một người ghi và các tổng khớp nhau.
          </p>
        }
      >
        <p>
          The fix is one multiplication: scale the thread index by the factor, so thread t starts at point 4t of its block's
          stretch. That is the ownership the text describes (p. 521), and it is the version lab 21.1 provides as its baseline.
          Switch the explorer to "coarsened, adjacent": every point has one writer and the sums match.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Cùng mô hình với trình khám phá scatter và gather ở trên: một lát 2D, warp 32 lane, các lần ghi được đếm, không
              được đo thời gian. Bảng phép toán đếm phép tính trong các đoạn mã, không phải các lệnh đã biên dịch.
            </p>
          }
        >
          <p>
            The same model as the scatter and gather explorer above: a 2D slice, warps of 32 lanes, writes counted, not
            timed. The operation table counts the listings' arithmetic, not compiled instructions.
          </p>
        </Translated>
      </Simplification>
      <Code
        tabs={[{ label: tr('Fig. 21.8, corrected', 'Fig. 21.8, đã sửa'), code: ADJACENT, tiers: TIERS, status: STATUS }]}
        caption={
          <Translated
            vi={
              <>
                Từ <code>labs/ch21/snippets/coarsen_adjacent.cu</code>: dòng 05 nhân chỉ số luồng với hệ số; một phép kiểm tra
                biên cho mỗi điểm thay cho phần đệm.
              </>
            }
          >
            <>
              From <code>labs/ch21/snippets/coarsen_adjacent.cu</code>: line 05 scales the thread index; one bound check per
              point replaces the padding.
            </>
          </Translated>
        }
      />
    </>
  );
}

export function S212() {
  return (
    <>
      <Translated
        vi={
          <p>
            Phép tính của chương là một tổng trên mọi cặp. Tại mọi điểm của một lưới (grid) 3D, thế tĩnh điện là tổng trên mọi
            nguyên tử của điện tích nguyên tử chia cho khoảng cách từ nó đến điểm đó. Sách xử lý từng lát 2D của lưới một, nên
            khối lượng việc cho mỗi lát là số nguyên tử nhân số điểm trong lát. Bỏ phần hóa học đi, trang này nói về một câu
            hỏi mà mọi kernel trên mọi cặp đều gặp: luồng (thread) nào ghi đầu ra nào.
          </p>
        }
      >
        <p>
          The chapter's computation is a sum over all pairs. At every point of a 3D grid, the electrostatic potential is the
          sum over all atoms of the atom's charge divided by its distance to the point. The book handles one 2D slice of the
          grid at a time, so the work per slice is the number of atoms times the number of points in the slice. Strip away the
          chemistry and this page is about one question that every all-pairs kernel faces: which thread writes which output.
        </p>
      </Translated>
      <Source section="21.1" pages="514–515" figs="Fig. 21.2" />

      <h2>
        <Translated vi="Hai thứ tự vòng lặp">Two loop orders</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Phiên bản C đơn giản của sách (Fig. 21.3) lặp qua các hàng của lát, rồi qua các điểm của một hàng, rồi qua các
            nguyên tử, và ghi mỗi điểm một lần. Phiên bản tối ưu (Fig. 21.4) hoán đổi các vòng lặp để các nguyên tử nằm ngoài
            cùng. Điều đó mang lại khoản tiết kiệm thật: khoảng cách theo z giống nhau cho mọi điểm của lát, nên nó được bình
            phương một lần cho mỗi nguyên tử, và khoảng cách theo y giống nhau dọc một hàng, nên nó được bình phương một lần
            cho mỗi hàng. Phép hoán đổi hợp lệ vì mọi lần lặp độc lập với nhau: cùng các số hạng được cộng, theo một thứ tự
            khác. Cái giá là giờ mỗi điểm được cập nhật bằng <code>+=</code> một lần cho mỗi nguyên tử thay vì được ghi một
            lần.
          </p>
        }
      >
        <p>
          The book's plain C version (Fig. 21.3) loops over the rows of the slice, then over the points of a row, then over the
          atoms, and writes each point once. Its optimised version (Fig. 21.4) interchanges the loops so that the atoms are
          outermost. That buys real savings: the z distance is the same for every point of the slice, so it is squared once
          per atom, and the y distance is the same along a row, so it is squared once per row. The interchange is legal because
          every iteration is independent: the same terms get added, in a different order. The price is that each point is now
          updated with <code>+=</code> once per atom instead of being written once.
        </p>
      </Translated>
      <Source section="21.2" pages="515–517" figs="Figs. 21.3, 21.4" />

      <h2>
        <Translated vi="Scatter: mỗi nguyên tử một luồng">Scatter: one thread per atom</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Song song hóa phiên bản tối ưu theo cách hiển nhiên thì mỗi luồng nhận một lần lặp của vòng ngoài: một nguyên tử.
            Rồi nó tự chạy cả hai vòng lặp lưới và <em>rải (scatter)</em> đóng góp của nguyên tử mình tới mọi điểm của lát.
            Mọi điểm nhận một cập nhật từ mọi luồng, nên mọi cập nhật phải là phép toán nguyên tử (atomic).
          </p>
        }
      >
        <p>
          Parallelise the optimised version the obvious way and each thread takes one iteration of the outer loop: one atom.
          It then runs both grid loops itself and <em>scatters</em> its atom's contribution to every point of the slice. Every
          point receives an update from every thread, so every update has to be atomic.
        </p>
      </Translated>
      <Source section="21.2" pages="517–518" figs="Fig. 21.5" />
      <Code
        tabs={[{ label: tr('scatter (Fig. 21.5)', 'scatter (Fig. 21.5)'), code: SCATTER, tiers: TIERS, status: STATUS }]}
        caption={
          <Translated
            vi={
              <>
                Fig. 21.5 như bài học trình bày, từ <code>labs/ch21/snippets/scatter.cu</code>. Đoạn mã in không có số nguyên
                tử và không có kiểm tra biên; bản này thêm cả hai, và tính chỉ số lát theo cách Fig. 21.6 làm.{' '}
                <code>atomicAdd</code> kiểu cũ (legacy) được giữ như bản in. <TierLegend tiers={['const', 'global', 'reg']} />
              </>
            }
          >
            <>
              Fig. 21.5 as the lesson shows it, from <code>labs/ch21/snippets/scatter.cu</code>. The listing has no atom count
              and no bound; this one adds both, and computes the slice index the way Fig. 21.6 does. The legacy{' '}
              <code>atomicAdd</code> is kept as printed. <TierLegend tiers={['const', 'global', 'reg']} />
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Các nguyên tử đến từ đâu? Bộ nhớ hằng (constant memory) nhỏ, nên host đưa chúng vào theo từng chunk{' '}
            <code>CHUNK_SIZE</code>. Với mỗi lát, nó chép lát sang device, rồi với mỗi chunk chép chunk vào mảng{' '}
            <code>__constant__</code> và khởi chạy kernel, kernel này cộng đóng góp của chunk đó bằng <code>+=</code>; sau chunk
            cuối, lát được chép về host.
          </p>
        }
      >
        <p>
          Where do the atoms come from? Constant memory is small, so the host feeds them in chunks of{' '}
          <code>CHUNK_SIZE</code>. For each slice it copies the slice to the device, then for each chunk copies the chunk into
          the <code>__constant__</code> array and launches the kernel, which adds that chunk's contribution with{' '}
          <code>+=</code>; after the last chunk the slice goes back to the host.
        </p>
      </Translated>
      <Source section="21.2" pages="517" />
      <Predict
        id="ch21/21.2/p-lockstep"
        prompt={
          <Translated
            vi={
              <>
                Một warp của kernel scatter chạy hai vòng lặp lưới đồng bộ từng bước. Trong một lệnh <code>atomicAdd</code>, 32
                lane của nó nhắm tới bao nhiêu địa chỉ lưới khác nhau?
              </>
            }
          >
            <>
              A warp of the scatter kernel runs the two grid loops in lockstep. In one <code>atomicAdd</code> instruction, how
              many distinct grid addresses do its 32 lanes target?
            </>
          </Translated>
        }
        unit={tr('addresses', 'địa chỉ')}
      >
        {(v) => (
          <p>
            <Translated
              vi={
                <>
                  <b>Một.</b> Mọi luồng đi qua các điểm theo cùng thứ tự, nên ở mỗi lệnh, mọi lane cộng vào cùng một điểm: một
                  va chạm của cả warp trên một địa chỉ ở mỗi bước. Các lane chỉ khác nhau ở giá trị chúng cộng, và phần cứng
                  áp dụng các phép toán nguyên tử cùng địa chỉ lần lượt từng cái.
                </>
              }
            >
              <b>One.</b> Every thread walks the points in the same order, so at each instruction all the lanes add to the same
              point: a full-warp collision on one address at every step. The lanes differ only in the value they add, and the
              hardware applies same-address atomics one after another.
            </Translated>{' '}
            {v === 1 ? tr('That is what you had.', 'Đúng như bạn đã trả lời.') : v === 32 ? tr('Thirty-two would be one point per lane. That is the gather kernel, below: there the threads own the points.', 'Ba mươi hai sẽ là mỗi lane một điểm. Đó là kernel gather, ở bên dưới: ở đó các luồng sở hữu các điểm.') : ''}
          </p>
        )}
      </Predict>
      <ScatterGatherScene />
      <Translated
        vi={
          <p>
            Hoạt ảnh cho thấy hai chi phí mà sách không nói rõ. Thứ nhất, thứ tự mà một hàng đợi các phép toán nguyên tử cùng
            địa chỉ được áp dụng là không xác định, và phép cộng float không có tính kết hợp, nên tổng của kernel scatter có thể
            khác nhau ở các bit cuối giữa các lần chạy. Mỗi luồng gather cộng các nguyên tử của nó theo thứ tự của vòng lặp,
            nên các bit của nó không bao giờ đổi. Thứ hai, mỗi lane của kernel scatter đọc một nguyên tử khác nhau từ bộ nhớ
            hằng, đúng trường hợp NVIDIA nói là bị tuần tự hóa: một lần truy cập cho mỗi địa chỉ khác nhau. Không nhà cung cấp
            nào đưa ra tỷ lệ tốc độ giữa scatter và gather.
          </p>
        }
      >
        <p>
          The animation shows two costs that the book does not spell out. First, the order in which a queue of same-address
          atomics lands is undefined, and float addition is not associative, so the scatter kernel's sums can differ from one
          run to the next in the last bits. Each gather thread adds its atoms in the loop's order, so its bits never change.
          Second, the scatter kernel's lanes each read a different atom from constant memory, which is the case NVIDIA says
          is serialised: one access per distinct address. No vendor gives a speed ratio for scatter against gather.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/infra-facts.md A1, B1, C1; ch10-facts A9; ch07-facts A3; ch21-facts A6, C2)">{/* i18n-ignore: a date and a citation */}
        <Translated
          vi={
            <p>
              Cả hai chi phí đều lấy từ tài liệu của nhà cung cấp, không phải từ sách. NVIDIA gọi <code>atomicAdd</code> cho
              float là legacy (hãy dùng <code>cuda::atomic_ref</code>). HIP có thể biên dịch nó thành một vòng lặp
              compare-and-swap nếu không có <code>-munsafe-fp-atomics</code>. Metal cần <code>device atomic_float*</code>{' '}
              (Apple7 trở lên) và không ghi nhận chi phí nào.
            </p>
          }
        >
          <p>
            Both costs are from vendor documents, not the book. NVIDIA calls float <code>atomicAdd</code> legacy (use{' '}
            <code>cuda::atomic_ref</code>).
            HIP may compile it to a compare-and-swap loop without <code>-munsafe-fp-atomics</code>. Metal needs{' '}
            <code>device atomic_float*</code> (Apple7 and later) and documents no cost.
          </p>
        </Translated>
      </Today>
      <Simplification>
        <Translated
          vi={
            <p>
              Hoạt ảnh: một hàng sáu điểm và bốn nguyên tử, với các nhóm bốn và sáu luồng đóng vai một warp. Hai thứ tự hàng đợi
              được chọn để cho thấy hiệu ứng; trên GPU, thứ tự là không xác định. Không có thời gian.
            </p>
          }
        >
          <p>
            The animation: one row of six points and four atoms, with groups of four and six threads standing in for a
            warp. The two queue orders are picked to show the effect; on a GPU the order is undefined. No time.
          </p>
        </Translated>
      </Simplification>

      <h2>
        <Translated vi="Gather: mỗi điểm một luồng">Gather: one thread per point</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Cách thay thế là giao cho mỗi luồng một điểm và để nó <em>gom (gather)</em> đóng góp của mọi nguyên tử. Việc đó cần
            thứ tự vòng lặp của Fig. 21.3 đơn giản, và một thay đổi: để giao cho mỗi luồng một lần lặp của hai vòng ngoài,
            chúng phải lồng nhau hoàn hảo, nên tọa độ y chuyển vào trong và mọi luồng tính lại nó. Sách gọi đây là đổi phép tính
            lấy độ song song. Đổi lại, mỗi luồng thực hiện một phép <code>+=</code> thường cho mỗi lần khởi chạy, vào một điểm
            không luồng nào khác chạm tới, và ở mỗi bước mọi lane của một warp đọc cùng một nguyên tử: trường hợp tốt nhất của
            bộ nhớ đệm hằng, một địa chỉ phục vụ cả warp.
          </p>
        }
      >
        <p>
          The alternative is to give each thread one point and let it <em>gather</em> the contributions of all atoms. That
          needs the loop order of the plain Fig. 21.3, and one change: to give each thread one iteration of the two outer
          loops, they must be perfectly nested, so the y coordinate moves inside and every thread computes it again. The book
          calls this a trade of calculation for parallelism. In return each thread makes one plain <code>+=</code> per launch,
          to a point no other thread touches, and at every step all the lanes of a warp read the same atom: the constant
          cache's best case, one address served to the whole warp.
        </p>
      </Translated>
      <Source section="21.2" pages="518–519" figs="Fig. 21.6" />
      <Code
        tabs={[{ label: tr('gather (Fig. 21.6)', 'gather (Fig. 21.6)'), code: GATHER, tiers: TIERS, status: STATUS }]}
        caption={
          <Translated
            vi={
              <>
                Fig. 21.6 từ <code>labs/ch21/snippets/gather.cu</code>, với một phép kiểm tra biên trên <code>i</code> và{' '}
                <code>j</code>. Kernel in trong sách không có và dựa vào một lát được đệm thành các khối trọn vẹn ("padding
                waste" của Fig. 21.9).
              </>
            }
          >
            <>
              Fig. 21.6 from <code>labs/ch21/snippets/gather.cu</code>, with a bound check on <code>i</code> and{' '}
              <code>j</code>. The printed kernel has none and relies on a slice padded to whole blocks (the "padding waste" of
              Fig. 21.9).
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Trong trình khám phá, hãy bắt đầu với kernel scatter. Mọi điểm có số người ghi bằng số nguyên tử; đánh dấu ô{' '}
            <code>+=</code> thường để xem việc bỏ phép toán nguyên tử gây ra điều gì. Rồi chuyển sang gather, nhấp vào một điểm
            và đọc phép tính chỉ số của luồng ghi nó. Kéo các nguyên tử để di chuyển chúng.
          </p>
        }
      >
        <p>
          In the explorer, start with the scatter kernel. Every point has as many writers as there are atoms; tick the plain{' '}
          <code>+=</code> box to see what losing the atomic does. Then switch to gather, click a point and read its writer's
          index arithmetic. Drag the atoms to move them.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một lát 2D, khoảng cách đơn vị, các nguyên tử ở một độ cao cố định phía trên. Warp 32 lane, độ rộng của NVIDIA và
              giả định của sách (ch05-facts B5). Các phép toán nguyên tử cùng địa chỉ đến theo một thứ tự cố định; không có phép
              toán nguyên tử, lần lưu của lane cao nhất được giữ lại. Trên GPU, cả hai đều không xác định. Không có thời gian,
              không có bộ nhớ đệm.
            </p>
          }
        >
          <p>
            A 2D slice, unit spacing, atoms a fixed height above it. Warps of 32 lanes, NVIDIA's width and the book's
            assumption (ch05-facts B5). Same-address atomics land in a fixed order; without the atomic, the highest lane's
            store survives. On a GPU both are undefined. No time, no caches.
          </p>
        </Translated>
      </Simplification>
      <PairsExplorer
        title={tr('Who writes where: scatter and gather', 'Ai ghi vào đâu: scatter và gather')}
        task={
          <Translated vi="Đếm số người ghi mỗi điểm trong scatter, rồi trong gather. Để ý cả số địa chỉ hằng.">
            Count the writers per point in scatter, then in gather. Watch the constant-address count too.
          </Translated>
        }
        views={['mapping']}
        initialKernel="scatter" /* i18n-ignore: a kernel id */
        kernels={['scatter', 'gather']}
        foot={
          <Translated vi="Mô hình: src/sim/dcs.ts. Các điểm được tô theo luồng ghi chúng; ×n đánh dấu một điểm có n người ghi; các điểm gạch chéo không bao giờ được ghi. Kết quả là float32 theo thứ tự riêng của mỗi kernel, so với một tổng trực tiếp float64.">
            Model: src/sim/dcs.ts. Points are shaded by the thread that writes them; ×n marks a point with n writers; hatched points are never written. Results are float32 in each kernel's own order, against a float64 direct sum.
          </Translated>
        }
      />
      <Ask q={questionsS212['ch21/21.2/q-why-no-atomic']} />
      <Translated
        vi={
          <p>
            Đây là một thế khó mà sách quay lại nhiều lần: mã tuần tự đã tối ưu thường khó song song hóa hơn mã đơn giản. Khoản
            tiết kiệm của nó đến từ thứ tự vòng lặp biến thành scatter. Kernel gather từ bỏ chúng, và mỗi luồng làm nhiều phép
            tính hơn lẽ ra phải làm. Rủi ro là các luồng chậm hơn ăn bớt phần mà song song đã giành được.
          </p>
        }
      >
        <p>
          This is a dilemma the book returns to: the optimised sequential code is often harder to parallelise than the plain
          one. Its savings came from the loop order that turns into a scatter. The gather kernel gives them up, and each thread
          does more arithmetic than it would have. The risk is that the slower threads eat into what parallelism gained.
        </p>
      </Translated>
      <Source section="21.2" pages="518" />
      <Ask q={questionsS212['ch21/21.2/q-dilemma']} />
      <Translated
        vi={
          <p>
            Bạn đã gặp lựa chọn này nếu từng huấn luyện một mô hình ngôn ngữ. Lượt lan truyền ngược của một phép tra embedding
            cộng hàng gradient của mỗi token vào hàng trọng số của id token đó, và nhiều token dùng chung một id: đó là scatter
            của Fig. 21.5, với id token thay cho các điểm lưới.
          </p>
        }
      >
        <p>
          You have met this choice before if you have trained a language model. The backward pass of an embedding lookup adds
          each token's gradient row into the weight row of its token id, and many tokens share an id: the scatter of Fig. 21.5,
          with token ids in place of grid points.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>Cầu nối này là của khóa học này. Những gì PyTorch và llama.cpp làm được ghi trong ch21-facts D1 đến D3.</p>
          }
        >
          <p>This bridge is ours. What PyTorch and llama.cpp do is recorded in ch21-facts D1 to D3.</p>
        </Translated>
      </Beyond>
      <Ask q={questionsS212['ch21/21.2/q-deterministic']} />

      <h2>
        <Translated vi="Gộp việc: một lần nạp, nhiều điểm">Coarsening: one load, several points</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Kernel gather vẫn thực hiện bốn lần nạp bộ nhớ hằng cho mỗi nguyên tử ở mỗi điểm. Sách chỉ ra rằng các lệnh nạp đó
            dùng phần cứng lẽ ra có thể nâng thông lượng dấu phẩy động, và chúng tốn năng lượng. Bốn điểm cạnh nhau trong một
            hàng dùng chung khoảng cách theo y và z cùng điện tích của nguyên tử, và chỉ khác nhau ở x (Fig. 21.7). Vậy hãy gộp
            bốn luồng thành một: luồng nạp mỗi nguyên tử một lần, giữ nó trong thanh ghi (register), và tính bốn điểm.
          </p>
        }
      >
        <p>
          The gather kernel still makes four constant-memory loads per atom for each point. The book points out that those
          load instructions use hardware that could otherwise raise floating-point throughput, and they cost energy. Four neighbouring points of a row share their y and z distances and the atom's
          charge, and differ only in x (Fig. 21.7). So fuse four threads into one: the thread loads each atom once, keeps it in
          registers, and computes four points.
        </p>
      </Translated>
      <Source section="21.3" pages="519–520" figs="Fig. 21.7" />
      <Predict
        id="ch21/21.2/p-loads"
        prompt={
          <Translated
            vi={
              <>
                Sau khi gộp việc theo hệ số bốn, một luồng thực hiện bao nhiêu lần nạp bộ nhớ hằng cho mỗi nguyên tử cho bốn
                điểm của nó? (Bốn luồng Fig. 21.6 thực hiện 16.)
              </>
            }
          >
            <>
              After coarsening by four, how many constant-memory loads does one thread make per atom for its four points?
              (Four Fig. 21.6 threads make 16.)
            </>
          </Translated>
        }
        unit={tr('loads', 'lần nạp')}
      >
        {(v) => (
          <p>
            <Translated
              vi={
                <>
                  <b>Bốn:</b> x, y, z và điện tích, mỗi thứ một lần. dy, dz và dy² + dz² cũng được tính một lần và giữ
                  trong thanh ghi.
                </>
              }
            >
              <b>Four:</b> x, y, z and the charge, once each. dy, dz and dy² + dz² are also computed once and kept in
              registers.
            </Translated>{' '}
            {v === 4 ? tr('That is what you had.', 'Đúng như bạn đã trả lời.') : v === 1 ? tr('One would be the whole atom in a single load; the listings read four floats, one at a time.', 'Một sẽ là cả nguyên tử trong một lần nạp; các đoạn mã đọc bốn float, lần lượt từng cái.') : ''}
          </p>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Trước khi nhìn vào trình khám phá, hãy đọc kernel đã gộp việc như sách in. Mỗi luồng giữ bốn bộ tích lũy, tính{' '}
            <code>dx1</code> đến <code>dx3</code> từ <code>dx0</code>, và ghi bốn điểm cạnh nhau ở các dòng 30 đến 33. Rồi hãy
            nhìn kỹ dòng 05.
          </p>
        }
      >
        <p>
          Before you look at the explorer, read the coarsened kernel as the book prints it. Each thread keeps four accumulators,
          computes <code>dx1</code> to <code>dx3</code> from <code>dx0</code>, and writes four neighbouring points on lines 30
          to 33. Then look hard at line 05.
        </p>
      </Translated>
      <Source section="21.3" pages="520" figs="Fig. 21.8" />
      <Code
        tabs={[{ label: tr('Fig. 21.8 as printed: do not use', 'Fig. 21.8 như bản in: đừng dùng'), code: PRINTED, tiers: TIERS, status: 'front-end-checked; emulator-checked to fail as described' }]}
        highlight={[PRINTED_LINES.index, ...PRINTED_LINES.writes]}
        caption={
          <Translated
            vi={
              <>
                34 dòng mã của Fig. 21.8 với cách đánh số của sách, từ <code>labs/ch21/snippets/coarsen_printed.cu</code>; chú
                thích ở dòng 15 được viết lại. Chỉ giữ cho việc tìm lỗi in.
              </>
            }
          >
            <>
              The 34 code lines of Fig. 21.8 with the book's numbering, from <code>labs/ch21/snippets/coarsen_printed.cu</code>;
              the comment on line 15 is reworded. Kept for the erratum hunt only.
            </>
          </Translated>
        }
      />
      <Ask q={questionsS212['ch21/21.2/q-printed']} label={tr('Erratum hunt', 'Tìm lỗi in')} />
      <AfterPrinted />
      <Translated
        vi={
          <p>
            Nó tiết kiệm được bao nhiêu? Sách đếm từ 16 lần nạp và 48 phép toán xuống còn 4 lần nạp và 24 phép toán. Các đoạn mã
            không hẳn nói vậy. Đếm từ mã, bốn luồng Fig. 21.6 thực hiện 40 phép tính cho mỗi nguyên tử và luồng đã gộp việc 27,
            hoặc 25 nếu các tích 2·gridspacing và 3·gridspacing được đưa ra ngoài vòng lặp. Sách đếm 12 phép chia trong khi mã có
            4, và không đếm phép căn bậc hai nào; sách còn đưa 9 phép toán cho mỗi điểm ở p. 519 và 12 ở p. 521 (ch21-facts E1).
            Bảng phép toán trong trình khám phá kernel đã gộp việc đếm chúng cho bất kỳ hệ số nào.
          </p>
        }
      >
        <p>
          How much does it save? The book counts 16 loads and 48 operations down to 4 loads and 24 operations. The listings
          do not quite say that. Counted from the code, four Fig. 21.6 threads make 40 arithmetic operations per atom and the
          coarsened thread 27, or 25 if the 2·gridspacing and 3·gridspacing products are hoisted. The book counts 12
          divisions where the code has 4, and counts no square roots at all; it even gives 9 operations per point on p. 519
          and 12 on p. 521 (ch21-facts E1). The operation table in the coarsened explorer counts them for any factor.
        </p>
      </Translated>
      <Source section="21.3" pages="520–521" />
      <Translated
        vi={
          <p>
            Cái giá là thanh ghi. Giờ một luồng giữ bốn bộ tích lũy, bốn khoảng cách, số hạng dùng chung và điện tích, và nhiều
            thanh ghi hơn cho mỗi luồng có thể nghĩa là ít warp thường trú hơn. Sách nói kernel đã gộp việc vẫn nằm trong giới
            hạn, nên occupancy (mức chiếm dụng) không bị ảnh hưởng, nhưng không đưa con số, trình biên dịch hay thiết bị nào để
            đối chiếu; trang tóm tắt nói cách xem con số của bạn.
          </p>
        }
      >
        <p>
          The cost is registers. A thread now holds four accumulators, four distances, the shared term and the charge, and
          more registers per thread can mean fewer warps resident. The book says the coarsened kernel stays within the limit,
          so occupancy does not suffer, but gives no count, compiler or device to check that against; the summary page says
          how to see yours.
        </p>
      </Translated>
      <Source section="21.3" pages="521" />
      <Ask q={questionsS212['ch21/21.2/q-ex2']} label={tr('Book exercise', 'Bài tập của sách')} />
      <Ask q={questionsS212['ch21/21.2/q-ex3']} label={tr('Book exercise', 'Bài tập của sách')} />

      <h2>
        <Translated vi="Hợp nhất truy cập: ai sở hữu bốn điểm nào">Coalescing: who owns which four</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một lần profile kernel đã gộp việc cho thấy các lần ghi của nó kém hiệu quả. Lấy một trong bốn câu lệnh ghi của nó.
            Lane t ghi điểm 4t + m trong đoạn của khối, nên các lane cạnh nhau cách nhau bốn float và 32 lần ghi của một warp
            trải trên 128 float, bỏ qua ba float giữa mỗi cặp. Bạn đã đếm chi phí bằng sector 32 byte ở trang bắt đầu; giờ hãy
            đếm nó trên phần cứng của một nhà cung cấp khác.
          </p>
        }
      >
        <p>
          A profile of the coarsened kernel shows its writes are inefficient. Take one of its four write statements. Lane t
          writes point 4t + m of the block's stretch, so neighbouring lanes are four floats apart and a warp's 32 writes are
          spread over 128 floats with three skipped between each pair. You counted the cost in 32-byte sectors on the start
          page; now count it on another vendor's hardware.
        </p>
      </Translated>
      <Source section="21.4" pages="521" />
      <Predict
        id="ch21/21.2/p-adjacent-lines"
        prompt={
          <Translated
            vi={
              <>
                Cùng câu lệnh ghi đó trên một GPU AMD Instinct: một wavefront 64 lane, lane t ghi phần tử base + 4t + m, với
                base nằm trên một ranh giới 128 byte. AMD gom địa chỉ của một wavefront thành các dòng 64 hoặc 128 byte. Một câu
                lệnh đó chạm tới bao nhiêu dòng 128 byte?
              </>
            }
          >
            <>
              The same write statement on an AMD Instinct GPU: a wavefront of 64 lanes, lane t writing element base + 4t + m,
              with base on a 128-byte boundary. AMD groups a wavefront's addresses into 64- or 128-byte lines. How many
              128-byte lines does that one statement touch?
            </>
          </Translated>
        }
        unit={tr('lines', 'dòng')}
      >
        {(v) => (
          <p>
            <Translated
              vi={
                <>
                  <b>8.</b> 64 lane trải trên 256 float, 1,024 byte, và mỗi dòng 128 byte trong đoạn đó chứa 8 float đang
                  được ghi, trên tổng số 32. Bốn câu lệnh chạm tới 32 dòng. Với bước nhảy của Fig. 21.10, ở bên dưới, một câu
                  lệnh là 256 byte liên tục: 2 dòng, tổng cộng 8.
                </>
              }
            >
              <b>8.</b> The 64 lanes span 256 floats, 1,024 bytes, and each 128-byte line in that span holds 8 of the floats
              being written, out of 32. The four statements touch 32 lines. With the stride of Fig. 21.10, below, a statement
              is 256 contiguous bytes: 2 lines, 8 in all.
            </Translated>{' '}
            {v === 8
              ? tr('That is what you had.', 'Đúng như bạn đã trả lời.')
              : v === 2
                ? tr('Two is the coalesced case: 64 floats side by side. Here they are four apart.', 'Hai là trường hợp được hợp nhất: 64 float nằm cạnh nhau. Ở đây chúng cách nhau bốn.')
                : v === 16
                  ? tr('Sixteen is the count in 32-byte sectors for a 32-lane warp. Here the lanes are twice as many and the lines four times as large.', 'Mười sáu là số sector 32 byte cho một warp 32 lane. Ở đây số lane gấp đôi và mỗi dòng lớn gấp bốn.')
                  : ''}
          </p>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Cách sửa là chia điểm theo cách khác. Giao blockDim.x điểm đầu tiên của đoạn cho các luồng của khối theo thứ tự, rồi
            blockDim.x điểm kế tiếp cho cùng các luồng đó, và cứ thế cho đến khi mỗi luồng có bốn điểm (Fig. 21.9). Khi đó luồng t
            sở hữu t, t + blockDim.x, t + 2·blockDim.x và t + 3·blockDim.x, và trong mọi câu lệnh ghi, các lane cạnh nhau ghi các
            float cạnh nhau. Dòng 05 không đổi; thứ thay đổi là các độ lệch ghi và, cùng với chúng, các khoảng cách theo x:{' '}
            <code>dx1 = dx0 + blockDim.x * gridspacing</code>. Hai thứ phải thay đổi cùng nhau, nếu không luồng sẽ tính đúng
            tổng cho sai điểm.
          </p>
        }
      >
        <p>
          The fix is to hand the points out differently. Give the block's first blockDim.x points of the stretch to its threads
          in order, then the next blockDim.x points to the same threads, and so on until each thread has four (Fig. 21.9).
          Thread t then owns t, t + blockDim.x, t + 2·blockDim.x and t + 3·blockDim.x, and in every write statement neighbouring
          lanes write neighbouring floats. Line 05 is unchanged; what changes is the write offsets and, with them, the x
          distances: <code>dx1 = dx0 + blockDim.x * gridspacing</code>. The two must change together, or the thread computes
          the right sums for the wrong points.
        </p>
      </Translated>
      <Source section="21.4" pages="521–522" figs="Figs. 21.9, 21.10" />
      <Code
        tabs={[{ label: tr('blockDim.x stride (Fig. 21.10)', 'bước nhảy blockDim.x (Fig. 21.10)'), code: STRIDED, tiers: TIERS, status: STATUS }]}
        caption={
          <Translated
            vi={
              <>
                Fig. 21.10 từ <code>labs/ch21/snippets/coarsen_strided.cu</code>, với một phép kiểm tra biên cho mỗi điểm. So
                sánh các độ lệch ghi và các dòng dx với Fig. 21.8 đã sửa.
              </>
            }
          >
            <>
              Fig. 21.10 from <code>labs/ch21/snippets/coarsen_strided.cu</code>, with a bound check per point. Compare the
              write offsets and the dx lines with the corrected Fig. 21.8.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Mỗi nhà cung cấp đếm chi phí bằng đơn vị riêng. NVIDIA đếm sector 32 byte cho mỗi warp, và bộ đếm là số sector mỗi
            yêu cầu của Nsight Compute, với 4 là mức tối ưu cho truy cập 32 bit. AMD gom địa chỉ của một wavefront thành các dòng
            64 hoặc 128 byte. Apple không ghi nhận quy tắc nào, nên trên Mac, lab 21.1 đo nó (ch21-facts A8, B3, C4). Hãy đi qua
            từng câu lệnh ghi của một warp với mỗi kiểu sở hữu, rồi đổi độ rộng nhóm, hình dạng khối và đơn vị.
          </p>
        }
      >
        <p>
          Each vendor counts the cost in its own unit. NVIDIA counts 32-byte sectors per warp, and Nsight Compute's sectors
          per request is the counter, 4 being the optimum for 32-bit accesses. AMD groups a wavefront's addresses into 64- or
          128-byte lines. Apple documents no rule, so on a Mac lab 21.1 measures it (ch21-facts A8, B3, C4). Step through
          the write statements of one warp under each ownership, then change the group width, the block shape and the unit.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một warp của khối (0, 0), từng câu lệnh một; không bộ nhớ đệm nào giữ một đoạn giữa các câu lệnh. Các hàng 1,024
              float trên một ranh giới 256 byte. Phép <code>+=</code> cũng đọc theo cùng mẫu.
            </p>
          }
        >
          <p>
            One warp of block (0, 0), one statement at a time; no cache keeps a segment between statements. Rows of 1,024
            floats on a 256-byte boundary. The <code>+=</code> also reads with the same pattern.
          </p>
        </Translated>
      </Simplification>
      <PairsExplorer
        title={tr('Write statements, one warp at a time', 'Các câu lệnh ghi, từng warp một')}
        task={
          <Translated vi="Đi qua từng câu lệnh với quyền sở hữu liền kề, rồi với bước nhảy, rồi với một lần lưu float4.">
            Step through the statements for adjacent ownership, then for the stride, then for one float4 store.
          </Translated>
        }
        views={['writes']}
        foot={
          <Translated vi={<>Mô hình: writeStatements của src/sim/dcs.ts, đếm bằng warpAccess từ src/sim/coalesce.ts (chương 6). Đậm: các byte mà các lane ghi trong câu lệnh này; nhạt: các câu lệnh trước; viền: các đoạn mà câu lệnh này chạm tới.</>}>
            <>Model: src/sim/dcs.ts writeStatements, counted with warpAccess from src/sim/coalesce.ts (chapter 6). Dark: bytes the lanes write in this statement; pale: earlier statements; outlined: the segments this statement touches.</>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Fig. 21.9 ghi nhãn các luồng của nó là "up to 8 potentials, skipping by half-warps" (tối đa 8 thế, nhảy theo
            half-warp), điều không khớp với mã (hệ số 4, bước nhảy blockDim.x) lẫn cách hợp nhất truy cập được ghi nhận ngày nay,
            theo từng warp (ch21-facts A3). Nhãn còn lại của nó, 64 đến 256 luồng mỗi khối, thì khớp với khuyến nghị hiện tại
            của NVIDIA và AMD (A4, B3).
          </p>
        }
      >
        <p>
          Fig. 21.9 labels its threads "up to 8 potentials, skipping by half-warps", which matches neither the code (a factor
          of 4, a stride of blockDim.x) nor how coalescing is documented today, per warp (ch21-facts A3). Its other label, 64
          to 256 threads per block, does match NVIDIA's and AMD's current advice (A4, B3).
        </p>
      </Translated>
      <Source section="21.4" pages="522" figs="Fig. 21.9" />
      <Ask q={questionsS212['ch21/21.2/q-wavefront']} />
      <Translated
        vi={
          <p>
            Sách nhắc tới lối thoát thứ hai cho hệ số bốn: giữ quyền sở hữu liền kề và ghi bốn kết quả bằng một lần lưu vector.
            Mỗi luồng nạp bốn float của nó như một float4, cộng bốn tổng và lưu float4 trở lại. Khi đó lần lưu của một warp là
            512 byte liên tục trong một yêu cầu. Hãy thử "one float4 store" trong trình chạy từng bước. Chương 15 đã gặp cùng vấn
            đề với các ô đầu ra của phép nhân ma trận (<a href={href('ch15', '15.5')}>mục 15.5</a>).
          </p>
        }
      >
        <p>
          The book mentions a second way out for a factor of four: keep the adjacent ownership and write the four results with
          one vector store. Each thread loads its four floats as one float4, adds its four sums and stores the float4 back. A
          warp's store is then 512 contiguous bytes in one request. Try "one float4 store" in the stepper. Chapter 15 met the
          same problem for the output tiles of a matrix multiplication (<a href={href('ch15', '15.5')}>section 15.5</a>).
        </p>
      </Translated>
      <Source section="21.4" pages="521" />
      <Ask q={questionsS212['ch21/21.2/q-ex4']} label={tr('Book exercise', 'Bài tập của sách')} />

      <h2>
        <Translated vi="Đo nó">Measure it</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Không điều nào ở trên cho bạn biết kernel gather nhanh hơn kernel scatter bao nhiêu trên GPU của bạn, hay bước nhảy
            đáng giá bao nhiêu: không nhà cung cấp nào ghi nhận một tỷ lệ, và Apple hoàn toàn không ghi nhận quy tắc hợp nhất
            nào. Trong <a href={href('ch21', 'labs')}>lab 21.1</a>, bạn viết các kernel scatter, gather và strided cho CUDA, HIP
            và Metal, kiểm thử chúng với một tổng trên CPU, và đo thời gian so với mốc cơ sở adjacent.
          </p>
        }
      >
        <p>
          None of this tells you how much faster the gather kernel is than the scatter kernel on your GPU, or what the stride
          is worth: no vendor documents a ratio, and Apple documents no coalescing rule at all. In{' '}
          <a href={href('ch21', 'labs')}>lab 21.1</a> you write the scatter, gather and strided kernels for CUDA, HIP and
          Metal, test them against a CPU sum, and time them against the adjacent baseline.
        </p>
      </Translated>
    </>
  );
}
