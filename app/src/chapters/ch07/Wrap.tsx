import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Beyond, Source, type Tier } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// A kernel of our own, for the first question. Checked copy: labs/ch07/snippets/wrap_causal_conv1d.cu
// (Wrap.test.ts ties the two, runs a model of the kernel and the file itself on the emulator for
// every n up to 400, and pins the numbers the other questions state).
const CAUSAL_CONV1D = `// A causal 1D convolution: y[i] = w[0]*x[i-3] + w[1]*x[i-2] + w[2]*x[i-1] + w[3]*x[i],
// with x[j] = 0 for j < 0. Each output looks back K - 1 inputs, so the halo is one-sided.
// One thread per output; grid = ceil(n / BLOCK) blocks of BLOCK threads.
#define BLOCK 128
#define K 4

__constant__ float w[K];

__global__ void causalConv1d(const float* x, float* y, int n) {
    __shared__ float tile[BLOCK + K - 1];
    int t = threadIdx.x;
    int i = blockIdx.x * BLOCK + t;
    if (i >= n) return;                     // no output past the end

    tile[t + K - 1] = x[i];                 // this thread's own input
    if (t < K - 1) {                        // the halo: the K - 1 inputs before the block
        int j = i - (K - 1);
        tile[t] = (j >= 0) ? x[j] : 0.0f;   // before the start of x: a ghost zero
    }
    __syncthreads();

    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += w[k] * tile[t + k];
    y[i] = acc;
}`;
const CAUSAL_TIERS: Record<string, Tier> = { x: 'global', y: 'global', tile: 'shared', w: 'const', t: 'reg', i: 'reg', j: 'reg', k: 'reg', acc: 'reg' };

// ---------------------------------------------------------------- the numbers the questions state
/** The causal kernel's sizes, and the remainders n mod BLOCK for which it is wrong (pinned by Wrap.test.ts). */
export const CAUSAL = { block: 128, taps: 4, wrongRemainders: [1, 2] } as const;

/** Exercise 7 on one case: a 1000 x 1000 image, a 5 x 5 filter, 28 x 28 output tiles. */
export const EX7 = { n: 1000, m: 5, t: 28 } as const;
export const EX7_BLOCKS = Math.ceil(EX7.n / EX7.t) ** 2;
export const EX7_THREADS_712 = (EX7.t + EX7.m - 1) ** 2; // Fig. 7.12: the block is the input tile
export const EX7_THREADS_715 = EX7.t ** 2; // Fig. 7.15: the block is the tile
export const EX7_EXTRA = EX7_BLOCKS * (EX7_THREADS_712 - EX7_THREADS_715);

/** Section 7.5's count for one interior block, in 3D: t^3 outputs, an (t + m - 1)^3 input tile, 4-byte floats. */
export const ai3d = (t: number, m: number) => (2 * t ** 3 * m ** 3) / (4 * ((t + m - 1) ** 3 + t ** 3));
/** Exercise 10 with a 5 x 5 x 5 filter: the largest cube of threads within 1024 is 10 on a side. */
export const EX10 = { maxThreads: 1024, m: 5, inTile: Math.floor(Math.cbrt(1024 + 0.5)) } as const;
export const EX10_OUT = EX10.inTile - (EX10.m - 1);
export const EX10_AI = ai3d(EX10_OUT, EX10.m);
export const EX10_IDEAL = EX10.m ** 3 / 4;

const n = (v: number) => v.toLocaleString('en-US');

export const questionsWrap: Record<string, Question> = {
  'ch07/wrap/q-early-return': {
    kind: 'choice',
    id: 'ch07/wrap/q-early-return',
    prompt: (
      <Translated vi={<>Kernel ở trên tính một phép tích chập nhân quả ngắn, đúng dạng của ví dụ phục vụ LLM ở mục 7.5. Với những độ dài <code>n</code> nào nó cho đầu ra sai?</>}>
<>
        The kernel above computes a short causal convolution, the shape of section 7.5&rsquo;s LLM-serving example. For
        which lengths <code>n</code> does it produce a wrong output?
      </>
</Translated>
    ),
    options: [
      {
        label: <Translated vi="Không độ dài nào: các luồng return không sở hữu đầu ra nào, và rào chắn không chờ chúng.">{'For none: the threads that return own no output, and the barrier does not wait for them.'}</Translated>,
        why: <Translated vi="Rào chắn quả thật không chờ chúng. Nhưng những luồng nào ghi tile[0], tile[1] và tile[2], và chúng có còn tồn tại hết trong khối cuối không?">{'The barrier indeed does not wait for them. But which threads write tile[0], tile[1] and tile[2], and do they all still exist in the last block?'}</Translated>,
      },
      {
        label: <Translated vi="Bất cứ khi nào n không phải bội số của 128.">{'Whenever n is not a multiple of 128.'}</Translated>,
        why: <Translated vi="Với 3 đầu ra trở lên ở khối cuối, các luồng 0, 1 và 2 đều tồn tại và nạp toàn bộ vùng biên. Một luồng vắng mặt ở sau điểm cuối không sở hữu ô nhớ nào mà một đầu ra hợp lệ đọc.">{'With 3 or more outputs in the last block, threads 0, 1 and 2 all exist and load the whole halo. A missing thread past the end owns no slot that a valid output reads.'}</Translated>,
      },
      { label: <Translated vi="Khi khối cuối chỉ có 1 hoặc 2 đầu ra: n = 129 và n = 130 sai, n = 131 đúng.">{'When the last block has only 1 or 2 outputs: n = 129 and n = 130 are wrong, n = 131 is right.'}</Translated>, correct: true },
      {
        label: <Translated vi="Với mọi n: một rào chắn sau một lệnh return sớm làm khối bị treo.">{'For every n: a barrier after an early return hangs the block.'}</Translated>,
        why: <Translated vi="Hướng dẫn của NVIDIA nói rào chắn chờ các luồng chưa thoát, và Apple nói điều tương tự cho Apple silicon (ch04-facts A6, C3). Kernel chạy xong; câu hỏi là nó đã đọc gì.">{"NVIDIA's guide says a barrier waits for the threads that have not exited, and Apple says the same for Apple silicon (ch04-facts A6, C3). The kernel finishes; the question is what it read."}</Translated>,
      },
    ],
    explain: (
      <Translated vi={<p>Các luồng ở sau điểm cuối return trước khi nạp gì cả, và không có gì bị treo. Thiệt hại nằm ở những gì chúng không nạp. Các ô nhớ vùng biên <code>tile[0]</code>, <code>tile[1]</code> và <code>tile[2]</code> được ghi bởi các luồng 0, 1 và 2. Khi khối cuối có 3 đầu ra trở lên thì các luồng đó tồn tại, và mọi ô nhớ mà một đầu ra hợp lệ đọc đều đã được ghi. Với 1 hoặc 2 đầu ra, luồng 2 (và có thể cả luồng 1) đã return, và đầu ra 0 đọc một ô nhớ không ai ghi: trên GPU là bất cứ gì đang nằm ở đó, trên trình giả lập là các byte <code>0xCD</code>. Vậy kernel sai đúng khi{' '}<code>n mod 128</code> bằng 1 hoặc 2. Cách sửa là cấu trúc của mục 7.5: không return sớm; mọi luồng đều nạp (một số 0 của ô ma nếu ở sau điểm cuối), tới rào chắn, và chỉ sau đó mới kiểm tra <code>i &lt; n</code> trước khi tính và ghi. AMD không ghi trong tài liệu quy tắc nào cho các luồng đã thoát tại rào chắn, thêm một lý do để đưa mọi luồng tới đó.</p>}>
<p>
        Threads past the end return before loading anything, and nothing hangs. The damage is in what they did not
        load. The halo slots <code>tile[0]</code>, <code>tile[1]</code> and <code>tile[2]</code> are written by threads 0,
        1 and 2. When the last block has 3 or more outputs those threads exist, and every slot a valid output reads has
        been written. With 1 or 2 outputs, thread 2 (and perhaps thread 1) has returned, and output 0 reads a slot nobody
        wrote: on a GPU whatever was there, on the emulator <code>0xCD</code> bytes. So the kernel is wrong exactly when{' '}
        <code>n mod 128</code> is 1 or 2. The fix is section 7.5&rsquo;s structure: no early return; every thread loads
        (a ghost zero past the end), reaches the barrier, and only then tests <code>i &lt; n</code> before computing and
        storing. AMD documents no rule for exited threads at a barrier, one more reason to bring every thread to it.
      </p>
</Translated>
    ),
    source:
      'a kernel of ours (labs/ch07/snippets/wrap_causal_conv1d.cu, run on the emulator for n = 1 to 400); barrier rule: ch04-facts A6, C3; the causal shape: ch07-facts D3',
    vi: { source: 'một kernel của chúng tôi (labs/ch07/snippets/wrap_causal_conv1d.cu, chạy trên trình giả lập với n = 1 đến 400); quy tắc rào chắn: ch04-facts A6, C3; dạng nhân quả: ch07-facts D3' },
  },
  'ch07/wrap/q-replicate': {
    kind: 'choice',
    id: 'ch07/wrap/q-replicate',
    prompt: (
      <Translated vi={<>Bạn cần kernel cơ bản của lab 7.1 xử lý các cạnh giống{' '}<code>padding_mode=&apos;replicate&apos;</code> của PyTorch: một phần tử lân cận bị thiếu lấy giá trị của pixel gần nhất bên trong ảnh. Thay đổi nhỏ nhất cho kernel là gì?</>}>
<>
        You need lab 7.1&rsquo;s basic kernel to treat the edges like PyTorch&rsquo;s{' '}
        <code>padding_mode=&apos;replicate&apos;</code>: a missing neighbour takes the value of the nearest pixel inside
        the image. What is the smallest change to the kernel?
      </>
</Translated>
    ),
    options: [
      {
        label: <Translated vi="Kẹp inRow và inCol vào trong ảnh thay vì kiểm tra chúng, và cộng dồn mọi tap.">{'Clamp inRow and inCol into the image instead of testing them, and accumulate every tap.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Giữ phép kiểm tra ô ma, và với mỗi tap bị bỏ qua thì cộng pixel biên gần nhất vào Pvalue.">{'Keep the ghost test, and for every skipped tap add the nearest edge pixel to Pvalue.'}</Translated>,
        why: <Translated vi="Mỗi tap bị bỏ qua có trọng số riêng. Giá trị được nhân bản phải được nhân với trọng số của tap đó, và đó chính là điều việc đọc phần tử đã kẹp làm.">{'Each skipped tap has its own weight. The replicated value has to be multiplied by that tap’s weight, which is what reading the clamped element does.'}</Translated>,
      },
      {
        label: <Translated vi="Lật bộ lọc cho các luồng gần cạnh, để các tap của chúng gập ngược vào trong ảnh.">{'Flip the filter for threads near an edge, so that their taps fold back into the image.'}</Translated>,
        why: <Translated vi="Việc lật thay đổi trọng số nào gặp đầu vào nào; nó không nhân bản gì cả.">{'Flipping changes which weight meets which input; it replicates nothing.'}</Translated>,
      },
      {
        label: <Translated vi="Không đổi gì trong kernel: khởi chạy thêm r luồng ở ngoài mỗi cạnh để cung cấp các bản sao.">{'Change nothing in the kernel: launch r extra threads past each edge to supply the copies.'}</Translated>,
        why: <Translated vi="Các luồng thêm tính những đầu ra không tồn tại (phép chặn đầu ra loại bỏ chúng), và các đầu ra thật ở cạnh vẫn bỏ qua các tap rơi vào ô ma của chúng.">{'Extra threads compute outputs that do not exist (the output guard drops them), and the real edge outputs still skip their ghost taps.'}</Translated>,
      },
    ],
    explain: (
      <Translated vi={<p>Phép kẹp là <code>inRow = min(max(inRow, 0), height - 1)</code>, và tương tự cho <code>inCol</code> với{' '}<code>width</code>. Với nó, mọi tap đều đọc một phần tử thật, và <code>if</code> của chính sách số 0 biến mất. Không còn tap bị bỏ qua, nên ở mức mã nguồn mọi lane của một warp ở cạnh chạy cùng các phép nhân-cộng; phép chặn đầu ra vẫn giữ. Sách nêu chính sách nhân bản nhưng không đưa kernel nào cho nó; phép kẹp là của khóa học này. Texture có thể làm điều tương tự bằng phần cứng (address mode Clamp của CUDA, <code>clamp_to_edge</code> của Metal), điều mà mục 7.1 có nhắc tới và khóa học, giống sách, không dùng.</p>}>
<p>
        The clamp is <code>inRow = min(max(inRow, 0), height - 1)</code>, and the same for <code>inCol</code> with{' '}
        <code>width</code>. With it, every tap reads a real element, and the zero policy&rsquo;s <code>if</code> disappears. There
        are no skipped taps, so at the source level every lane of an edge warp runs the same multiply-adds; the output
        guard stays. The book names the replicate policy but gives no kernel for it; the clamp is ours. Textures can do
        the same in hardware (CUDA&rsquo;s Clamp address mode, Metal&rsquo;s <code>clamp_to_edge</code>), which section
        7.1 mentions and the course, like the book, does not use.
      </p>
</Translated>
    ),
    source: 'ch. 7 §7.1, p. 162 (the policies) and §7.2, p. 166 (the ghost test); the clamp is ours; ch07-facts D1, A6, C5',
    vi: { source: 'ch. 7 §7.1, p. 162 (các chính sách) và §7.2, p. 166 (phép kiểm tra ô ma); phép kẹp là của chúng tôi; ch07-facts D1, A6, C5' },
  },
  'ch07/wrap/q-warp-index': {
    kind: 'choice',
    id: 'ch07/wrap/q-warp-index',
    prompt: (
      <Translated vi={<>Một kernel với khối 256 luồng đọc một bảng <code>__constant__</code> tại chỉ số{' '}<code>threadIdx.x / 32</code>. Một warp đọc bao nhiêu địa chỉ khác nhau, với warp rộng 32 và với wavefront rộng 64?</>}>
<>
        A kernel with 256-thread blocks reads a <code>__constant__</code> table at index{' '}
        <code>threadIdx.x / 32</code>. How many distinct addresses does one warp read, with 32-wide warps and with
        64-wide wavefronts?
      </>
</Translated>
    ),
    options: [
      { label: <Translated vi="1 và 1.">{'1 and 1.'}</Translated>, why: <Translated vi="Một wavefront rộng 64 phủ threadIdx.x từ 0 đến 63 (hoặc 64 đến 127, ...): hai giá trị của threadIdx.x / 32.">{'A 64-wide wavefront covers threadIdx.x from 0 to 63 (or 64 to 127, ...): two values of threadIdx.x / 32.'}</Translated> },
      { label: <Translated vi="1 và 2.">{'1 and 2.'}</Translated>, correct: true },
      { label: <Translated vi="32 và 64.">{'32 and 64.'}</Translated>, why: <Translated vi="Cả 32 lane của một warp bắt đầu tại một bội số của 32 có chung threadIdx.x / 32.">{'All 32 lanes of a warp that starts at a multiple of 32 share threadIdx.x / 32.'}</Translated> },
      { label: <Translated vi="8 và 8.">{'8 and 8.'}</Translated>, why: <Translated vi="threadIdx.x / 32 đếm các warp 32 lane trong khối; nó không phụ thuộc vào việc khối có 8 warp như vậy.">{'threadIdx.x / 32 counts warps of 32 within the block; it does not depend on the block having 8 of them.'}</Translated> },
    ],
    explain: (
      <Translated vi={<p>Trên một warp rộng 32, mọi lane tính cùng một chỉ số: một địa chỉ, trường hợp mà NVIDIA nói nhanh ngang một thanh ghi. Một wavefront rộng 64 trải qua hai giá trị, và bộ nhớ đệm vô hướng của AMD chỉ phục vụ các lần đọc <code>__constant__</code> khi chúng đồng nhất trên toàn wavefront; nếu không chúng bị tuần tự hóa. Hằng số 32 viết cứng chính là lỗi: hãy đánh chỉ số theo{' '}<code>threadIdx.x / warpSize</code>, hoặc theo thứ gì đó đồng nhất trong mỗi khối như <code>blockIdx.x</code>, và lần đọc là một địa chỉ trên cả hai.</p>}>
<p>
        On a 32-wide warp every lane computes the same index: one address, the case NVIDIA says is as fast as a register.
        A 64-wide wavefront spans two values, and AMD&rsquo;s scalar cache serves <code>__constant__</code> reads only
        when they are uniform across the wavefront; otherwise they are serialised. The literal 32 is the bug: index by{' '}
        <code>threadIdx.x / warpSize</code>, or by something uniform per block such as <code>blockIdx.x</code>, and the
        read is one address on both.
      </p>
</Translated>
    ),
    source: 'ch. 7 §7.4, p. 171; ch07-facts A3, B1; wavefront widths ch05-facts B5',
    vi: { source: 'ch. 7 §7.4, p. 171; ch07-facts A3, B1; độ rộng wavefront ch05-facts B5' },
  },
  'ch07/wrap/q-ex7': {
    kind: 'numeric',
    id: 'ch07/wrap/q-ex7',
    prompt: (
      <Translated vi={<>Bài tập 7 cho một trường hợp: ảnh {n(EX7.n)} × {n(EX7.n)}, bộ lọc {EX7.m} × {EX7.m} và ô đầu ra {EX7.t} × {EX7.t}. Tổng cộng, Fig. 7.12 khởi chạy nhiều hơn Fig. 7.15 bao nhiêu luồng với cùng ô đầu ra?</>}>
<>
        Exercise 7 on one case: a {n(EX7.n)} × {n(EX7.n)} image, a {EX7.m} × {EX7.m} filter and {EX7.t} × {EX7.t}{' '}
        output tiles. In all, how many more threads does Fig. 7.12 launch than Fig. 7.15 with the same output tile?
      </>
</Translated>
    ),
    answer: EX7_EXTRA,
    unit: 'threads',
    vi: { unit: 'luồng', source: 'ch. 7 Bài tập 7, p. 181; Figs. 7.12 và 7.15, pp. 174, 178' },
    explain: (
      <Translated vi={<p>Cả hai kernel có một khối cho mỗi ô đầu ra: ⌈{n(EX7.n)} / {EX7.t}⌉ = {Math.ceil(EX7.n / EX7.t)} mỗi cạnh,{' '}{n(EX7_BLOCKS)} khối. Khối của Fig. 7.12 là ô đầu vào, ({EX7.t} + {EX7.m} − 1)² = {n(EX7_THREADS_712)}{' '}luồng, với cùng số float bộ nhớ chia sẻ ({n(4 * EX7_THREADS_712)} byte). Khối của Fig. 7.15 là ô,{' '}{n(EX7_THREADS_715)} luồng và {n(4 * EX7_THREADS_715)} byte. Chênh lệch là {n(EX7_BLOCKS)} ×{' '}{EX7_THREADS_712 - EX7_THREADS_715} = {n(EX7_EXTRA)} luồng, mỗi luồng trong số đó nạp một phần tử vùng biên rồi ngồi không.</p>}>
<p>
        Both kernels have one block per output tile: ⌈{n(EX7.n)} / {EX7.t}⌉ = {Math.ceil(EX7.n / EX7.t)} per side,{' '}
        {n(EX7_BLOCKS)} blocks. Fig. 7.12&rsquo;s block is the input tile, ({EX7.t} + {EX7.m} − 1)² = {n(EX7_THREADS_712)}{' '}
        threads, with as many floats of shared memory ({n(4 * EX7_THREADS_712)} bytes). Fig. 7.15&rsquo;s block is the tile,{' '}
        {n(EX7_THREADS_715)} threads and {n(4 * EX7_THREADS_715)} bytes. The difference is {n(EX7_BLOCKS)} ×{' '}
        {EX7_THREADS_712 - EX7_THREADS_715} = {n(EX7_EXTRA)} threads, every one of which loads a halo element and then
        idles.
      </p>
</Translated>
    ),
    source: 'ch. 7 Exercise 7, p. 181; Figs. 7.12 and 7.15, pp. 174, 178',
  },
  'ch07/wrap/q-3d': {
    kind: 'numeric',
    id: 'ch07/wrap/q-3d',
    prompt: (
      <Translated vi={<>Bài tập 10 yêu cầu Fig. 7.12 trong 3D. Lấy bộ lọc {EX10.m} × {EX10.m} × {EX10.m}, tối đa {n(EX10.maxThreads)}{' '}luồng mỗi khối, và khối lập phương luồng lớn nhất vừa làm ô đầu vào. Cách đếm của mục 7.5 cho một khối nằm bên trong bao nhiêu FLOP/B?</>}>
<>
        Exercise 10 asks for Fig. 7.12 in 3D. Take a {EX10.m} × {EX10.m} × {EX10.m} filter, at most {n(EX10.maxThreads)}{' '}
        threads per block, and the largest cube of threads that fits as the input tile. What does section 7.5&rsquo;s
        count give for an interior block, in FLOP/B?
      </>
</Translated>
    ),
    answer: Math.round(EX10_AI * 100) / 100,
    tol: 0.015,
    unit: 'FLOP/B',
    vi: { source: 'ch. 7 Bài tập 10, p. 181; cách đếm của §7.5, p. 176, mở rộng sang 3D (của chúng tôi); Fig. 7.14, p. 177' },
    explain: (
      <Translated vi={<p>{EX10.inTile}³ = {n(EX10.inTile ** 3)} luồng thì vừa còn {EX10.inTile + 1}³ = {n((EX10.inTile + 1) ** 3)} thì không, nên ô đầu vào có cạnh {EX10.inTile} và ô đầu ra {EX10_OUT}. Mỗi khối: 2 · {EX10_OUT}³ · {EX10.m}³ = {n(2 * EX10_OUT ** 3 * EX10.m ** 3)} FLOP, so với {n(EX10.inTile ** 3)} float được nạp và {n(EX10_OUT ** 3)}{' '}được ghi, {n(4 * (EX10.inTile ** 3 + EX10_OUT ** 3))} byte: {EX10_AI.toFixed(1)} FLOP/B, khoảng một phần ba mức lý tưởng 3D m³/4 = {EX10_IDEAL}. Trong 2D, cùng bộ lọc và cùng ngân sách khối đạt 5.42 trên 6.25. Vùng biên là một lớp vỏ, và quanh một khối lập phương nhỏ như vậy phần lớn ô là vỏ: {n(EX10.inTile ** 3 - EX10_OUT ** 3)} trên{' '}{n(EX10.inTile ** 3)} phần tử. Chương 8 bàn về cách vượt qua giới hạn này.</p>}>
<p>
        {EX10.inTile}³ = {n(EX10.inTile ** 3)} threads fit and {EX10.inTile + 1}³ = {n((EX10.inTile + 1) ** 3)} do not,
        so the input tile is {EX10.inTile} on a side and the output tile {EX10_OUT}. Per block: 2 · {EX10_OUT}³ · {EX10.m}³
        = {n(2 * EX10_OUT ** 3 * EX10.m ** 3)} FLOP, against {n(EX10.inTile ** 3)} floats loaded and {n(EX10_OUT ** 3)}{' '}
        stored, {n(4 * (EX10.inTile ** 3 + EX10_OUT ** 3))} bytes: {EX10_AI.toFixed(1)} FLOP/B, about a third of the 3D
        ideal m³/4 = {EX10_IDEAL}. In 2D the same filter and block budget reach 5.42 of 6.25. A halo is a shell, and
        around a cube this small most of the tile is shell: {n(EX10.inTile ** 3 - EX10_OUT ** 3)} of{' '}
        {n(EX10.inTile ** 3)} elements. Chapter 8 is about getting past this limit.
      </p>
</Translated>
    ),
    source: 'ch. 7 Exercise 10, p. 181; the count of §7.5, p. 176, extended to 3D (ours); Fig. 7.14, p. 177',
  },
  'ch07/wrap/q-same-speed': {
    kind: 'choice',
    id: 'ch07/wrap/q-same-speed',
    prompt: (
      <Translated vi={<>Trên một GPU nào đó, kernel dùng bộ nhớ hằng của lab 7.1 chạy chỉ chênh vài phần trăm so với kernel cơ bản ở mọi bán kính. Cách hiểu nào phù hợp nhất?</>}>
<>
        On some GPU, lab 7.1&rsquo;s constant-memory kernel runs within a few percent of the basic kernel at every
        radius. Which reading fits best?
      </>
</Translated>
    ),
    options: [
      {
        label: <Translated vi="Trình biên dịch đã bỏ qua qualifier __constant__, nên bộ lọc vẫn đến từ DRAM ở mọi tap.">{'The compiler ignored the __constant__ qualifier, so the filter still came from DRAM on every tap.'}</Translated>,
        why: <Translated vi="Qualifier không phải một gợi ý: kernel không thể ghi F_c, và NVIDIA phục vụ nó qua bộ nhớ đệm hằng. Câu hỏi là có bao nhiêu lưu lượng DRAM để loại bỏ.">{'The qualifier is not a hint: kernels cannot write F_c, and NVIDIA serves it through the constant cache. The question is how much DRAM traffic there was to remove.'}</Translated>,
      },
      {
        label: <Translated vi="Các bộ nhớ đệm thông thường đã giữ bộ lọc cho kernel cơ bản, nên còn rất ít lưu lượng DRAM để loại bỏ.">{'Ordinary caches already held the filter for the basic kernel, leaving little DRAM traffic to remove.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Cả hai kernel đều bị giới hạn bởi tính toán, vì 0.25 và 0.5 FLOP/B nằm trên ngưỡng của thiết bị này.">{'Both kernels are compute-bound, since 0.25 and 0.5 FLOP/B are above this device\'s threshold.'}</Translated>,
        why: <Translated vi="Điều đó cần một ngưỡng dưới 0.25 FLOP/B: một thiết bị di chuyển hơn bốn byte cho mỗi phép toán nó làm được. Lab 5.1 đo ngưỡng của bạn.">{'That needs a threshold below 0.25 FLOP/B: a device that moves more than four bytes for every operation it can do. Lab 5.1 measures yours.'}</Translated>,
      },
      {
        label: <Translated vi="Bộ nhớ đệm hằng chỉ có lợi với bộ lọc lớn hơn working set 8 KB của nó, còn bộ lọc này nhỏ.">{'The constant cache only pays off for filters larger than its 8 KB working set, and this one is small.'}</Translated>,
        why: <Translated vi="Ngược lại: một bộ lọc vừa trong working set là trường hợp tốt nhất.">{'The opposite: a filter that fits in the working set is the best case.'}</Translated>,
      },
    ],
    explain: (
      <Translated vi={<p>Hệ số 2 của mô hình giả định mọi lần đọc <code>F</code> đều tới DRAM. Nhưng mọi luồng đọc cùng vài trăm byte, thứ mà bất kỳ bộ nhớ đệm nào cũng giữ sau lần chạm đầu tiên, như bản thân sách cũng nói (p. 168). Khi đó bộ nhớ hằng thay đổi bộ nhớ đệm nào trả lời, chứ không thay đổi bao nhiêu tới DRAM. Phần còn lại là đầu vào, một lần nạp{' '}<code>N</code> mỗi tap, và đó là thứ phân ô nhắm tới.</p>}>
<p>
        The model&rsquo;s factor of 2 assumes every read of <code>F</code> reaches DRAM. But every thread reads the same
        few hundred bytes, which any cache keeps after the first touch, as the book itself says (p. 168). Constant memory
        then changes which cache answers, not how much reaches DRAM. What remains is the input, one load of{' '}
        <code>N</code> per tap, and that is what tiling goes after.
      </p>
</Translated>
    ),
    source: 'ch. 7 §7.3, p. 168 and §7.4, p. 171; ch07-facts A3; lab 7.1',
    vi: { source: 'ch. 7 §7.3, p. 168 và §7.4, p. 171; ch07-facts A3; lab 7.1' },
  },
};

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = realBackends(reports).filter((b) => b !== 'race');
  const passing = (lab: string) => backends.filter((b) => labStatus(reports[b], lab, 'work') === 'pass');
  const l1 = passing('ch07/lab1_constconv');
  const l2 = passing('ch07/lab2_tiledconv');
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch07/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const labLine = (on: string[]) => (on.length ? tr(`passes on ${on.join(' and ')}`, `đạt trên ${on.join(' và ')}`) : tr('has not passed on a device here yet', 'chưa đạt trên thiết bị nào ở đây'));
  return (
    <>
      <h2 style={{ marginTop: 0 }}><Translated vi="Những gì cần mang theo">What to carry forward</Translated></h2>
      <ul>
        <Translated vi={<li>Một đầu ra tích chập là tổng có trọng số trên một vùng. Công thức của sách không lật bộ lọc, và PyTorch cũng tính như vậy. Một phần tử lân cận bị thiếu có giá trị gì (0, giá trị biên gần nhất, quấn vòng) là một chính sách bạn chọn.</li>}>
<li>
          A convolution output is a weighted sum over a patch. The book&rsquo;s formula does not flip the filter, which
          is what PyTorch computes too. What a missing neighbour is worth (zero, the nearest edge value, a wrap) is a
          policy you choose.
        </li>
</Translated>
        <Translated vi={<li>Mỗi đầu ra một luồng là đúng và đơn giản: một phép chặn cho đầu ra, và một phép kiểm tra (hoặc một phép kẹp) cho mỗi tap. Những warp nào phân kỳ ở các cạnh tùy vào hình dạng khối và độ rộng warp; hãy đọc độ rộng, đừng bao giờ viết cứng 32.</li>}>
<li>
          One thread per output is correct and simple: a guard for the output, and a test (or a clamp) for every tap.
          Which warps diverge at the edges depends on the block shape and the warp width; read the width, never write
          32.
        </li>
</Translated>
        <Translated vi={<li>Mức lý tưởng là m²/4 FLOP/B trong 2D và m/4 trong 1D với giá trị 4 byte, và nó tăng lên khi phần tử nhỏ đi. Kernel cơ bản đòi 0.25 bất kể bộ lọc: khoảng cách là phần tái sử dụng nó vứt bỏ. Một bộ lọc nhỏ không thể bị giới hạn bởi tính toán, dù kernel tốt đến đâu.</li>}>
<li>
          The ideal is m²/4 FLOP/B in 2D and m/4 in 1D for 4-byte values, and it scales up as the element shrinks. The
          basic kernel asks for 0.25 whatever the filter: the gap is the reuse it throws away. A small filter cannot be
          compute-bound, however good the kernel.
        </li>
</Translated>
        <Translated vi={<li>Bộ nhớ hằng, hoặc một đối số truyền theo giá trị, hợp với dữ liệu chỉ-đọc nhỏ mà mọi lane của một warp đọc ở cùng chỉ số. Chỉ số riêng cho từng lane là trường hợp tệ nhất của nó. Hãy giữ một symbol <code>__constant__</code>, kernel của nó và lệnh sao chép của nó trong một tệp.</li>}>
<li>
          Constant memory, or an argument passed by value, suits small read-only data that every lane of a warp reads at
          the same index. Per-lane indices are its worst case. Keep a <code>__constant__</code> symbol, its kernel and
          its copy in one file.
        </li>
</Translated>
        <Translated vi={<li>Khi ô đầu vào lớn hơn ô đầu ra, vùng biên có cái giá: cường độ tính toán là mức lý tưởng nhân với một hệ số chỉ tiến tới 1 khi ô rộng gấp nhiều lần bộ lọc, và kích thước khối giới hạn ô. Hãy ghi các số 0 của ô ma, dùng một rào chắn mà mọi luồng đều tới, và chuyển chỉ số không dấu sang <code>int</code>{' '}trước khi trừ.</li>}>
<li>
          When the input tile is bigger than the output tile, the halo costs: the intensity is the ideal times a factor
          that approaches 1 only for tiles several times wider than the filter, and the block size caps the tile. Store
          the ghost zeros, use one barrier that every thread reaches, and convert unsigned indices to <code>int</code>{' '}
          before subtracting.
        </li>
</Translated>
        <Translated vi={<li>Để vùng biên cho cache lo thì được các ô có kích thước lũy thừa của hai và giữ mọi lane bận rộn, nhưng lưu lượng của nó phụ thuộc vào một bộ nhớ đệm bạn không kiểm soát, và nó chia rẽ warp bên trong vòng lặp.</li>}>
<li>
          Leaving the halo to the cache gives tiles that are a power of two and keeps every lane busy, but its traffic
          depends on a cache you do not control, and it splits warps inside the loop.
        </li>
</Translated>
        <Translated vi={<li>Các mô hình đếm những gì một kernel đòi hỏi ở bộ nhớ. Bộ nhớ đệm quyết định thứ gì tới DRAM, và chỉ một phép đo cho biết bao nhiêu: hãy dự đoán, rồi chạy các lab.</li>}>
<li>
          The models count what a kernel asks of memory. Caches decide what reaches DRAM, and only a measurement says
          how much: predict, then run the labs.
        </li>
</Translated>
      </ul>
      <Source section="7.7" pages="179–180" />
      <Translated vi={<p>Sách khép lại bằng việc chỉ về phía trước. Stencil, chủ đề của chương 8, là các phép tích chập có hình dạng đặc biệt, và chương 8 cũng giải quyết giới hạn mà kích thước khối đặt lên ô. Các lớp tích chập, và cách các thư viện biến một phép tích chập thành một phép nhân ma trận, nằm ở <a href={href('ch19', '19.3')}>chương 19</a>.</p>}>
<p>
        The book closes by pointing ahead. Stencils, the subject of chapter 8, are convolutions with a special shape,
        and chapter 8 also takes on the limit that the block size puts on the tile. Convolutional layers, and the way
        libraries turn a convolution into a matrix multiplication, are <a href={href('ch19', '19.3')}>chapter 19</a>.
      </p>
</Translated>
      <Aside>
        <Translated vi={<p>Cố ý lược bỏ: phần giới thiệu chung về bộ nhớ đệm CPU ở pp. 170 đến 171 được rút gọn còn những gì NVIDIA, AMD và Apple ghi trong tài liệu hiện nay (bảng của mục 7.4), không có hoạt ảnh về độ trễ, vì không có số liệu độ trễ L2 nào kiểm chứng được. Phần ôn lại mô hình bộ nhớ của Fig. 7.8 (p. 168) chính là mục 5.2 của chương 5.</p>}>
<p>
          Left out on purpose: the general tour of CPU caches on pp. 170 to 171 is reduced to what NVIDIA, AMD and Apple
          document today (section 7.4&rsquo;s table), with no latency animation, because no L2 latency figure could be
          checked. Fig. 7.8&rsquo;s review of the memory model (p. 168) is chapter 5&rsquo;s section 5.2.
        </p>
</Translated>
      </Aside>

      <h2><Translated vi="Bài kiểm tra tổng hợp">A mixed check</Translated></h2>
      <p>
        <Translated vi="Các câu này dựa trên cả chương và trên những tình huống các mục chưa trình bày.">These draw on the whole chapter and on situations the sections did not show.</Translated>
        {asked ? tr(` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.`, ` Đến giờ trong chương này bạn đã trả lời đúng ${right} trên ${asked} câu hỏi, tính câu trả lời gần nhất của bạn cho mỗi câu.`) : ''}
      </p>
      <Code
        tabs={[
          {
            label: tr('CUDA (not a lab file: for this question only)', 'CUDA (không phải tệp lab: chỉ dùng cho câu hỏi này)'),
            code: CAUSAL_CONV1D,
            tiers: CAUSAL_TIERS,
            status: 'front-end-checked; emulator-checked',
          },
        ]}
      />
      <Ask q={questionsWrap['ch07/wrap/q-early-return']} label={tr('Find the defect', 'Tìm lỗi')} />
      <Ask q={questionsWrap['ch07/wrap/q-replicate']} />
      <Ask q={questionsWrap['ch07/wrap/q-warp-index']} />
      <Ask q={questionsWrap['ch07/wrap/q-ex7']} label={tr('Exercise 7', 'Bài tập 7')} />
      <Ask q={questionsWrap['ch07/wrap/q-3d']} label={tr('Exercise 10', 'Bài tập 10')} />
      <Ask q={questionsWrap['ch07/wrap/q-same-speed']} />

      <h2><Translated vi="Bài tập của sách">The book&rsquo;s exercises</Translated></h2>
      <Translated vi={<p>Chương kết thúc với mười một bài tập (pp. 180–181). Mỗi bài đều được giải hoặc được dựng sẵn ở đâu đó trong khóa học này; làm theo đúng cách diễn đạt của sách cũng đáng công.</p>}>
<p>
        The chapter ends with eleven exercises (pp. 180–181). Each is worked or set up somewhere in this course; the
        book&rsquo;s own wording is worth doing as well.
      </p>
</Translated>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }} data-table="exercises">
          <thead>
            <tr>
              <th><Translated vi="Bài tập">Exercise</Translated></th>
              <th><Translated vi="Nội dung">About</Translated></th>
              <th><Translated vi="Được giải ở đâu trong khóa học">Where it is worked here</Translated></th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1</td>
              <td><Translated vi="y[0] của Fig. 7.3">y[0] of Fig. 7.3</Translated></td>
              <Translated vi={<td>phần dự đoán trước cửa sổ trượt ở <a href={href('ch07', '7.1')}>7.1</a>, phần này ẩn y[0] cho tới khi bạn chốt</td>}>
<td>
                the prediction before the sliding window in <a href={href('ch07', '7.1')}>7.1</a>, which hides y[0] until
                you commit
              </td>
</Translated>
            </tr>
            <tr>
              <td>2</td>
              <td><Translated vi="Tự tay tính một phép tích chập 1D">A 1D convolution by hand</Translated></td>
              <Translated vi={<td>một câu hỏi và một bộ dữ liệu mẫu của trình khám phá ở <a href={href('ch07', '7.1')}>7.1</a></td>}>
<td>
                a question and an explorer preset in <a href={href('ch07', '7.1')}>7.1</a>
              </td>
</Translated>
            </tr>
            <tr>
              <td>3</td>
              <td><Translated vi="Năm bộ lọc 1D làm gì">What five 1D filters do</Translated></td>
              <Translated vi={<td>các bộ dữ liệu mẫu của trình khám phá và một câu hỏi ở <a href={href('ch07', '7.1')}>7.1</a></td>}>
<td>
                the explorer&rsquo;s presets and a question in <a href={href('ch07', '7.1')}>7.1</a>
              </td>
</Translated>
            </tr>
            <tr>
              <td>4, 5, 6</td>
              <td><Translated vi="Ô ma và phép nhân trong 1D, N×N và N1×N2">Ghost cells and multiplications in 1D, N×N and N1×N2</Translated></td>
              <Translated vi={<td>bộ đếm ô ma và các công thức của nó ở <a href={href('ch07', '7.1')}>7.1</a>, với một câu hỏi về bài tập 5</td>}>
<td>
                the ghost counter and its formulas in <a href={href('ch07', '7.1')}>7.1</a>, with a question on exercise 5
              </td>
</Translated>
            </tr>
            <tr>
              <td>7</td>
              <td><Translated vi="Khối, luồng và bộ nhớ chia sẻ cho Figs. 7.12 và 7.15">Blocks, threads and shared memory for Figs. 7.12 and 7.15</Translated></td>
              <Translated vi={<td>câu hỏi Bài tập 7 ở trên; kích thước ô trong trình khám phá của <a href={href('ch07', '7.5')}>7.5</a></td>}>
<td>
                the Exercise 7 question above; tile sizes in the explorer of <a href={href('ch07', '7.5')}>7.5</a>
              </td>
</Translated>
            </tr>
            <tr>
              <td>8, 9</td>
              <td><Translated vi="Figs. 7.7 và 7.9 trong 3D">Figs. 7.7 and 7.9 in 3D</Translated></td>
              <td><Translated vi="chưa viết mã; các thay đổi được liệt kê bên dưới, và lab 7.1 là nơi để chuyển chúng sang">not coded; the changes are listed below, and lab 7.1 is the place to port them</Translated></td>
            </tr>
            <tr>
              <td>10</td>
              <td><Translated vi="Fig. 7.12 trong 3D">Fig. 7.12 in 3D</Translated></td>
              <td><Translated vi="chỉ định kích thước và cường độ tính toán: câu hỏi Bài tập 10 ở trên">sizing and intensity only: the Exercise 10 question above</Translated></td>
            </tr>
            <tr>
              <td>11</td>
              <td><Translated vi="Khối là ô đầu ra">The block as the output tile</Translated></td>
              <Translated vi={<td>cách tổ chức thứ hai của trình khám phá và hai câu hỏi ở <a href={href('ch07', '7.5')}>7.5</a>; phép tích chập depthwise của MLX được xây theo cách này</td>}>
<td>
                the explorer&rsquo;s second organisation and two questions in <a href={href('ch07', '7.5')}>7.5</a>; MLX&rsquo;s
                depthwise convolution is built this way
              </td>
</Translated>
            </tr>
          </tbody>
        </table>
      </div>
      <Translated vi={<p>Với bài tập 8 và 9: thêm một đối số <code>depth</code> và chỉ số thứ ba từ <code>blockIdx.z</code>,{' '}<code>blockDim.z</code> và <code>threadIdx.z</code>; thêm vòng lặp bộ lọc thứ ba và một phép kiểm tra biên theo z; đọc{' '}<code>N[(inZ*height + inRow)*width + inCol]</code>; và trải phẳng bộ lọc trên (2r + 1)³ trọng số. Trong lab 7.1, bán kính lớn nhất, 7, khi đó sẽ cần 15³ = 3,375 float, 13,500 byte: nằm trong 64 KB bộ nhớ hằng của CUDA nhưng vượt working set 8 KB của bộ nhớ đệm của nó, và vượt 4 KB của Metal cho <code>setBytes</code>. Trình giả lập chạy được khối 3D, nên bản chuyển có thể được test mà không cần GPU.</p>}>
<p>
        For exercises 8 and 9: add a <code>depth</code> argument and a third index from <code>blockIdx.z</code>,{' '}
        <code>blockDim.z</code> and <code>threadIdx.z</code>; add a third filter loop and a bounds test in z; read{' '}
        <code>N[(inZ*height + inRow)*width + inCol]</code>; and flatten the filter over (2r + 1)³ weights. In lab 7.1 the
        largest radius, 7, would then need 15³ = 3,375 floats, 13,500 bytes: inside CUDA&rsquo;s 64 KB of constant memory
        but past the 8 KB working set of its cache, and past Metal&rsquo;s 4 KB for <code>setBytes</code>. The emulator
        runs 3D blocks, so the port can be tested without a GPU.
      </p>
</Translated>
      <Beyond>
        <Translated vi={<p>Các kích thước 3D là phép tính của chúng tôi, đối chiếu với ch07-facts A3 và C1.</p>}>
<p>The 3D sizes are our arithmetic, set against ch07-facts A3 and C1.</p>
</Translated>
      </Beyond>

      <h2><Translated vi="Những chỗ PDF có vẻ sai">Where the PDF looks wrong</Translated></h2>
      <Translated vi={<p>Bản sách dùng để xây khóa học này có những lỗi có vẻ là sơ suất sau trong chương 7, được liệt kê để bạn không bị rối khi đọc sách song song. Chỗ nào ảnh hưởng tới mã, các lab dùng mã đã sửa và đã test, và nói rõ chỗ khác với trang sách.</p>}>
<p>
        The copy this course was built from has these apparent slips in chapter 7, listed so that you are not thrown by
        them when you read the book alongside. Where one affects code, the labs use corrected, tested code and say where
        it differs from the page.
      </p>
</Translated>
      <ul data-list="errata">
        <Translated vi={<li>p. 159 nói bộ lọc Gaussian có thể làm sắc nét cạnh. Trong các công cụ bạn dùng, bộ lọc Gaussian là một phép làm mờ (torchvision mô tả <code>GaussianBlur</code> là như vậy), và khóa học này gọi nó là làm mượt.</li>}>
<li>
          p. 159 says Gaussian filters can sharpen edges. In the tools you use a Gaussian filter is a blur (torchvision
          documents <code>GaussianBlur</code> as one), and this course calls it smoothing.
        </li>
</Translated>
        <Translated vi={<li>Fig. 7.1 vẽ x[1] với một viền màu mảnh khác các ô còn lại; trông như một lỗi vẽ không mang ý nghĩa gì.</li>}>
<li>
          Fig. 7.1 draws x[1] with a thin coloured border unlike the other cells; it looks like a drawing artifact with
          no meaning.
        </li>
</Translated>
        <Translated vi={<li>p. 161 gọi tích vô hướng cho y[3] là phép tính &ldquo;for x[3]&rdquo;, và thiếu một dấu chấm trước &ldquo;Fig. 7.2&rdquo;.</li>}>
<li>
          p. 161 calls the inner product for y[3] the calculation &ldquo;for x[3]&rdquo;, and a full stop is missing
          before &ldquo;Fig. 7.2&rdquo;.
        </li>
</Translated>
        <Translated vi={<li>p. 163 gọi bộ lọc là M trong phần khai triển từng số hạng của P[2][2], trong khi hình gọi nó là f, và in số hạng thứ hai của hàng thứ ba là N<sub>2,1</sub>·M<sub>1,1</sub> trong khi quy luật cần M<sub>2,1</sub>. Các con số và tổng, 321, là đúng.</li>}>
<li>
          p. 163 names the filter M in the term-by-term expansion of P[2][2], where the figure calls it f, and prints the
          third row&rsquo;s second term as N<sub>2,1</sub>·M<sub>1,1</sub> where the pattern needs M<sub>2,1</sub>. The
          numbers and the sum, 321, are right.
        </li>
</Translated>
        <Translated vi={<li>p. 165 nói kernel nhận năm đối số, rồi liệt kê sáu.</li>}>
<li>p. 165 says the kernel receives five arguments, then lists six.</li>
</Translated>
        <Translated vi={<li>Fig. 7.7 không biên dịch được: nó đánh chỉ số các con trỏ thường F và P bằng hai chỉ số, và inRow cùng inCol không có kiểu. Nó cũng không có phép chặn đầu ra. <a href={href('ch07', '7.2')}>Mục 7.2</a> truy tìm các lỗi này; lab 7.1 có kernel đã sửa.</li>}>
<li>
          Fig. 7.7 does not compile: it indexes the plain pointers F and P with two subscripts, and inRow and inCol have
          no type. It also has no output guard. <a href={href('ch07', '7.2')}>Section 7.2</a> hunts these; lab 7.1 has
          the corrected kernel.
        </li>
</Translated>
        <Translated vi={<li>Fig. 7.9 cố định kích thước của F bằng một bán kính lúc biên dịch nhưng vẫn nhận bán kính r lúc chạy; hai giá trị phải khớp nhau. Lab 7.1 định kích thước F_c cho bán kính 7 và đánh chỉ số phẳng.</li>}>
<li>
          Fig. 7.9 fixes F&rsquo;s size with a compile-time radius but still takes the radius r at run time; the two
          must agree. Lab 7.1 sizes F_c for radius 7 and indexes it flat.
        </li>
</Translated>
        <Translated vi={<li>p. 168 nói hầu hết bộ lọc có bán kính 7 trở xuống, rồi giới hạn bộ lọc 3D ở 7³ = 343 phần tử, tức một bộ lọc rộng 7, không phải bán kính 7 (15³).</li>}>
<li>
          p. 168 says most filters have a radius of 7 or less, then bounds a 3D filter by 7³ = 343 elements, which is a
          filter 7 wide, not radius 7 (15³).
        </li>
</Translated>
        <Translated vi={<li>p. 169 gợi ý một khai báo <code>extern</code> khi kernel và mã host nằm ở các tệp khác nhau. Ở chế độ whole-program mặc định của nvcc đó là lỗi; nó cần <code>-rdc=true</code>. Cùng trang đó nói <code>cudaMemcpyToSymbol</code> báo cho runtime rằng dữ liệu sẽ không thay đổi; NVIDIA mô tả nó là một lệnh sao chép tới bất kỳ device symbol nào, còn tính chỉ-đọc đến từ <code>__constant__</code>.</li>}>
<li>
          p. 169 suggests an <code>extern</code> declaration when the kernel and the host code are in different files.
          In nvcc&rsquo;s default whole-program mode that is an error; it needs <code>-rdc=true</code>. The same page
          says <code>cudaMemcpyToSymbol</code> tells the runtime the data will not change; NVIDIA documents it as a copy
          to any device symbol, and read-only comes from <code>__constant__</code>.
        </li>
</Translated>
        <Translated vi={<li>p. 171 đặt L1 ở mức &ldquo;typically up to 128 KB&rdquo;: L1 và bộ nhớ chia sẻ hợp nhất của NVIDIA hiện là 96 đến 336 KB mỗi SM. Trang đó cũng có &ldquo;high-thought writes&rdquo; thay cho high-throughput.</li>}>
<li>
          p. 171 puts L1 at &ldquo;typically up to 128 KB&rdquo;: NVIDIA&rsquo;s unified L1 and shared memory is now 96 to
          336 KB per SM. It also has &ldquo;high-thought writes&rdquo; for high-throughput.
        </li>
</Translated>
        <Translated vi={<li>p. 172 có một chữ &ldquo;or&rdquo; thừa trong &ldquo;the input tiles are or 4× larger&rdquo;.</li>}>
<li>p. 172 has a stray &ldquo;or&rdquo; in &ldquo;the input tiles are or 4× larger&rdquo;.</li>
</Translated>
        <Translated vi={<li>Figs. 7.12 và 7.15 khai báo bộ lọc là F_c nhưng dùng nó là F, khai báo N_s không có kiểu phần tử, và viết 0.0 thay cho 0.0f. Lab 7.2 sửa tất cả những lỗi này.</li>}>
<li>
          Figs. 7.12 and 7.15 declare the filter as F_c and use it as F, declare N_s without an element type, and write
          0.0 for 0.0f. Lab 7.2 fixes all of these.
        </li>
</Translated>
        <Translated vi={<li>p. 175 nói luồng (5,5) dùng vùng có góc là N_s[5][5]; với bán kính 1 đó là luồng (6,6), luồng mà Fig. 7.13 vẽ (<a href={href('ch07', '7.5')}>mục 7.5</a> có hỏi). Cùng đoạn đó có &ldquo;threadId.y&rdquo;.</li>}>
<li>
          p. 175 says thread (5,5) uses the patch whose corner is N_s[5][5]; with radius 1 that is thread (6,6), the
          one Fig. 7.13 draws (<a href={href('ch07', '7.5')}>section 7.5</a> asks). The same paragraph has
          &ldquo;threadId.y&rdquo;.
        </li>
</Translated>
        <Translated vi={<li>p. 176 giới thiệu cường độ tính toán là số phép toán trên mỗi byte khối nạp, nhưng mẫu số đếm cả các lần ghi của nó, và như vậy là đúng.</li>}>
<li>
          p. 176 introduces the intensity as operations per byte the block loads, but the denominator rightly counts
          its stores too.
        </li>
</Translated>
        <Translated vi={<li>Fig. 7.15, dòng 21, đọc N_s mà không có <code>-FILTER_RADIUS</code> mà chính phép chặn của nó dùng, nên mọi tap đọc từ bộ nhớ chia sẻ đều đọc sai phần tử và một số rơi ra ngoài ô. Các phép kiểm tra <code>&gt;= 0</code> của nó áp lên giá trị không dấu và luôn đúng; phép chặn vẫn hoạt động vì nửa kia loại bỏ các giá trị bị quấn vòng. Số dòng của nó bị hỏng (24 đến 27 xuất hiện hai lần). <a href={href('ch07', '7.6')}>Mục 7.6</a> và phần B của lab 7.2 đề cập cả ba.</li>}>
<li>
          Fig. 7.15, line 21, reads N_s without the <code>-FILTER_RADIUS</code> its own guard uses, so every shared tap
          reads the wrong element and some fall past the tile. Its <code>&gt;= 0</code> tests are on unsigned values
          and always true; the guard still works because the other half rejects the wrapped values. Its line numbers
          are corrupted (24 to 27 appear twice). <a href={href('ch07', '7.6')}>Section 7.6</a> and lab 7.2&rsquo;s part B
          cover all three.
        </li>
</Translated>
        <Translated vi={<li>p. 178 thiếu một từ trong &ldquo;by the time a block needs to its halo cells&rdquo;.</li>}>
<li>p. 178 is missing a word in &ldquo;by the time a block needs to its halo cells&rdquo;.</li>
</Translated>
        <Translated vi={<li>p. 179 chỉ bạn sang chương 20 cho mạng nơ-ron tích chập; chú thích 1 và mục lục đặt chúng ở chương 19.</li>}>
<li>
          p. 179 sends you to chapter 20 for convolutional neural networks; footnote 1 and the contents put them in
          chapter 19.
        </li>
</Translated>
        <Translated vi={<li>Bài tập 1 hỏi P[0] trong Fig. 7.3, trong khi mảng đầu ra của hình được gọi là y.</li>}>
<li>Exercise 1 asks for P[0] in Fig. 7.3, whose output array is called y.</li>
</Translated>
      </ul>
      <Aside>
        <Translated vi={<p>Đây là cách chúng tôi đọc một bản PDF. Nếu bản của bạn khác, hãy tin bản của bạn. Các mục về extern, lệnh sao chép và kích thước L1 đến từ việc đối chiếu sách với tài liệu hiện hành (ch07-facts A1, A2, A5, D5).</p>}>
<p>
          These are our readings of one PDF. If your copy differs, trust your copy. The items about extern, the copy
          call and the L1 size come from checking the book against current documentation (ch07-facts A1, A2, A5, D5).
        </p>
</Translated>
      </Aside>

      <h2><Translated vi="Dự án đến lúc này">The project so far</Translated></h2>
      <p data-project={[l1.length ? 'lab1' : '', l2.length ? 'lab2' : ''].filter(Boolean).join(',') || 'none'}>
        <Translated vi={<>Chương 7 không thêm mốc nào cho dự án xuyên suốt, dự án đang ở M2: GEMM phân ô có kiểm tra biên của chương 5 và 6. Thứ dự án lấy từ chương này là kỹ thuật: truyền đối số theo giá trị cho mọi khối tham số nhỏ, phép đếm cường độ tính toán để định kích thước ô, và phân ô có vùng biên cho mọi cửa sổ trượt.{' '}</>}>
          Chapter 7 adds no milestone to the through-line project, which stands at M2: the tiled, boundary-checked GEMM of
          chapters 5 and 6. What the project takes from this chapter is technique: arguments by value for every small
          parameter block, the intensity count for sizing tiles, and halo tiling for any sliding window.{' '}
        </Translated>
        {backends.length === 0
          ? tr('No lab results are recorded on this machine yet.', 'Máy này chưa ghi nhận kết quả lab nào.')
          : tr(`On this machine your lab 7.1 ${labLine(l1)}, and your lab 7.2 ${labLine(l2)}.`, `Trên máy này, lab 7.1 của bạn ${labLine(l1)}, và lab 7.2 của bạn ${labLine(l2)}.`)}{' '}
        <Translated vi={<>Xem các <a href={href('ch07', 'labs')}>lab</a>.</>}>
          See the <a href={href('ch07', 'labs')}>labs</a>.
        </Translated>
      </p>
      <Translated vi={<p><a href={href('ch08', 'start')}>Chương 8</a> tiếp tục từ chỗ chương này dừng lại. Một stencil là một phép tích chập có trọng số đến từ một phương trình vi phân và có các điểm nằm dọc theo các trục, thường là trong 3D, nơi giới hạn kích thước khối mà bạn gặp ở bài tập 10 gây khó nhất. Lời giải của nó là gộp việc cho luồng (thread coarsening), cho phép một khối đi xuyên qua một ô sâu hơn số luồng của nó, và phân ô thanh ghi.</p>}>
<p>
        <a href={href('ch08', 'start')}>Chapter 8</a> picks up where this one stops. A stencil is a convolution whose
        weights come from a differential equation and whose points lie along the axes, usually in 3D, where the
        block-size limit you met in exercise 10 bites hardest. Its answers are thread coarsening, which lets one block
        march through a tile deeper than its threads, and register tiling.
      </p>
</Translated>
    </>
  );
}
