import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Source } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { INDEX } from '../../sim/addrwidth';
import { breakEven, BOOK_RULE } from '../../sim/zerocopy';
import { DerefMap } from './DerefMap';

// A program of our own, for one question: a managed table of layers that holds device
// pointers. Checked copy: labs/appC/snippets/wrap_layer_table.cu (Wrap.snippets.test.ts
// ties the two; ./verify.sh front-end-checks the file).
export const LAYER_TABLE = String.raw`#include <cstdio>

// Where one layer's keys and values live: two device pointers.
struct LayerKV {
    float* k;
    float* v;
};

// Block b copies the first key of layer b. The table of layers is read from
// memory, and then the pointer found in it is followed.
__global__ void firstKeys(const LayerKV* layers, float* out) {
    if (threadIdx.x == 0) out[blockIdx.x] = layers[blockIdx.x].k[0];
}

void checkLayers(int nLayers, size_t perLayer) {
    LayerKV* layers;
    float* out;
    cudaMallocManaged((void**)&layers, nLayers * sizeof(LayerKV));   // the table: managed
    cudaMallocManaged((void**)&out, nLayers * sizeof(float));
    for (int l = 0; l < nLayers; ++l) {
        cudaMalloc((void**)&layers[l].k, perLayer * sizeof(float));   // the keys: device memory
        cudaMalloc((void**)&layers[l].v, perLayer * sizeof(float));
        cudaMemset(layers[l].k, 0, perLayer * sizeof(float));
    }
    firstKeys<<<nLayers, 32>>>(layers, out);
    cudaDeviceSynchronize();
    printf("%f\n", out[0]);            // line A
    printf("%f\n", layers[0].k[0]);    // line B
}`;

// The numbers the questions ask for.
/** A KV cache laid out [layer][token][head][dim]: elements per layer. */
export const KV = { layers: 32, tokens: 32768, heads: 32, dim: 128 };
export const KV_PER_LAYER = KV.tokens * KV.heads * KV.dim;
/** An int offset reaches 2^31 − 1, so the smallest value of `layer` whose offset does not fit. */
export const KV_FIRST_BAD = Math.ceil(2 ** INDEX.int32.log2Count / KV_PER_LAYER);
/** The embedding question: one row of the table read per generated token. */
export const VOCAB = 128000;
export const EMBED_BREAK_EVEN = breakEven(1 / VOCAB, BOOK_RULE);

const n = (v: number) => v.toLocaleString('en-US');

export const questionsWrap: Record<string, Question> = {
  'appC/wrap/q-layer-table': {
    kind: 'choice',
    id: 'appC/wrap/q-layer-table',
    prompt: (
      <Translated
        vi={
          <>
            <code>checkLayers</code> ở trên chạy trên Linux với Unified Memory đầy đủ (<code>ConcurrentManagedAccess</code> = 1).
            Điều gì xảy ra ở dòng A và dòng B?
          </>
        }
      >
        <>
          <code>checkLayers</code> above runs on Linux with full Unified Memory (<code>ConcurrentManagedAccess</code> = 1).
          What happens at lines A and B?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Cả hai in ra 0: bảng là managed, nên mọi thứ đi tới qua nó đều được host đọc.">
            Both print 0: the table is managed, so everything reached through it is host-readable.
          </Translated>
        ),
        why: (
          <Translated vi="Dòng A ổn: out là managed và kernel đã xong. Dòng B đọc layers[0].k, điều đó ổn, rồi dereference nó trên host. Con trỏ đó đến từ cudaMalloc.">
            {'Line A is fine: out is managed and the kernel has finished. Line B reads layers[0].k, which is fine, and then dereferences it on the host. That pointer came from cudaMalloc.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="A in ra 0. B không được phép: nó dereference bộ nhớ cudaMalloc trên host, dù bảng là managed.">
            A prints 0. B is not allowed: it dereferences cudaMalloc memory on the host, though the table is managed.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Kernel thất bại: kernel không thể đi theo một con trỏ mà nó đọc từ bộ nhớ, chỉ đi theo con trỏ được truyền làm đối số.">
            {'The kernel fails: a kernel cannot follow a pointer that it read from memory, only one passed as an argument.'}
          </Translated>
        ),
        why: (
          <Translated vi="Nó đi theo được, miễn là vùng nhớ ở đầu kia là vùng mà kernel được dereference. Bảng là managed và các key là bộ nhớ device; cả hai đều hợp lệ.">
            {'It can, as long as the memory at the other end is memory the kernel may dereference. The table is managed and the keys are device memory; both qualify.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Cả hai thất bại: bộ nhớ managed không thể chứa con trỏ device, nên bảng không dùng được ở bên nào.">
            Both fail: managed memory cannot hold device pointers, so the table is unusable on either side.
          </Translated>
        ),
        why: (
          <Translated vi="Một con trỏ được lưu trong bộ nhớ chỉ là một con số, và bộ nhớ nào cũng chứa được nó. Điều quan trọng là ai dereference nó, và nó trỏ tới đâu.">
            {'A pointer stored in memory is just a number, and any memory can hold it. What matters is who dereferences it, and where it points.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Hãy đi theo mỗi con trỏ tới nơi nó trỏ. <code>layers</code> là managed, nên host được ghi bảng và kernel được
              đọc nó. Mỗi <code>k</code> là bộ nhớ <code>cudaMalloc</code>: kernel được đi theo nó, còn host chỉ được đưa nó
              cho các lời gọi API. Dòng B đọc con trỏ từ bộ nhớ managed không gặp vấn đề gì rồi dereference bộ nhớ device trên
              host, điều mà không thời kỳ nào của phụ lục này cho phép, kể cả nhất quán bằng phần cứng. Hãy sao chép giá trị ra
              bằng <code>cudaMemcpy</code>, hoặc đặt cả các key vào bộ nhớ managed.
            </p>
            <p>
              Có hai nơi lỗi này không lộ ra. Dưới HIP, AMD ghi nhận việc host đọc bộ nhớ <code>hipMalloc</code> là được phép,
              qua kênh kết nối, nên dòng B in ra. Trên trình giả lập CPU của khóa học, bộ nhớ host và device là một, nên lỗi
              cũng không bị phát hiện ở đó.
            </p>
          </>
        }
      >
        <>
          <p>
            Follow each pointer to where it points. <code>layers</code> is managed, so the host may write the table and
            the kernel may read it. Each <code>k</code> is <code>cudaMalloc</code> memory: the kernel may follow it, the
            host may only hand it to API calls. Line B reads the pointer from managed memory without trouble and then
            dereferences device memory on the host, which no era of this appendix allows, not even hardware coherence.
            Copy the value out with <code>cudaMemcpy</code>, or put the keys in managed memory too.
          </p>
          <p>
            Two places where this does not show. Under HIP, AMD documents host reads of <code>hipMalloc</code> memory as
            allowed, over the interconnect, so line B prints. On the course's CPU emulator host and device memory are one
            memory, so the mistake goes unnoticed there as well.
          </p>
        </>
      </Translated>
    ),
    source: 'book §C.3, p. 625 (pointers read from memory) and §C.6, pp. 626–627; appC-facts A12, B3; ch02-facts A17',
    vi: { source: 'sách §C.3, p. 625 (con trỏ đọc từ bộ nhớ) và §C.6, pp. 626–627; appC-facts A12, B3; ch02-facts A17' },
  },
  'appC/wrap/q-kv-offset': {
    kind: 'numeric',
    id: 'appC/wrap/q-kv-offset',
    prompt: (
      <Translated
        vi={
          <>
            Một KV cache chứa các key của {KV.layers} lớp trong một mảng giá trị nửa độ chính xác, bố trí theo [layer][token]
            [head][dim] với {n(KV.tokens)} token, {KV.heads} head và kích thước head là {KV.dim}. Một kernel tìm lớp của nó
            bằng <code>const __half* k = cache + layer * tokens * heads * dim</code>, trong đó mọi biến ở vế phải đều là{' '}
            <code>int</code>. Con trỏ rộng 64 bit. Giá trị nhỏ nhất của <code>layer</code> khiến con trỏ bị sai là bao nhiêu?
          </>
        }
      >
        <>
          A KV cache holds the keys of {KV.layers} layers in one array of half-precision values, laid out [layer][token]
          [head][dim] with {n(KV.tokens)} tokens, {KV.heads} heads and a head size of {KV.dim}. A kernel finds its layer
          with <code>const __half* k = cache + layer * tokens * heads * dim</code>, where every variable on the right is
          an <code>int</code>. Pointers are 64 bits wide. What is the smallest value of <code>layer</code> for which the
          pointer comes out wrong?
        </>
      </Translated>
    ),
    answer: KV_FIRST_BAD,
    explain: (
      <Translated
        vi={
          <>
            <p>
              Tích được tính bằng <code>int</code> trước khi cộng vào con trỏ. Một lớp là {n(KV.tokens)} × {KV.heads} ×{' '}
              {KV.dim} = 2<sup>27</sup> phần tử, và một <code>int</code> chứa tối đa 2<sup>31</sup> − 1. Lớp 15 bắt đầu tại 15
              × 2<sup>27</sup>, vẫn vừa; lớp {KV_FIRST_BAD} bắt đầu đúng tại 2<sup>31</sup>, không vừa. Tràn số có dấu là hành
              vi không xác định trong C++, và trên thực tế offset quay vòng thành số âm và con trỏ rơi vào trước cache.
            </p>
            <p>
              Con trỏ 64-bit không giúp được, vì phần hẹp là phép tính số học. Cả mảng là 8 GiB, còn xa mọi giới hạn địa chỉ
              của <a href={href('appC', 'C.4')}>mục C.4</a>. Hãy viết <code>(size_t)layer * tokens * heads * dim</code> để mọi
              phép nhân diễn ra ở 64 bit.
            </p>
          </>
        }
      >
        <>
          <p>
            The product is computed in <code>int</code> before it is added to the pointer. One layer is{' '}
            {n(KV.tokens)} × {KV.heads} × {KV.dim} = 2<sup>27</sup> elements, and an <code>int</code> holds at most 2
            <sup>31</sup> − 1. Layer 15 starts at 15 × 2<sup>27</sup>, which fits; layer {KV_FIRST_BAD} starts at exactly
            2<sup>31</sup>, which does not. Signed overflow is undefined in C++, and in practice the offset wraps negative
            and the pointer lands before the cache.
          </p>
          <p>
            The 64-bit pointer does not help, because the narrow part is the arithmetic. The whole array is 8 GiB, nowhere
            near any address limit of <a href={href('appC', 'C.4')}>section C.4</a>. Write{' '}
            <code>(size_t)layer * tokens * heads * dim</code> so that every multiplication happens in 64 bits.
          </p>
        </>
      </Translated>
    ),
    source: 'ours: C++ integer arithmetic; the address widths are §C.4, pp. 625–626; index width in ch. 3 §3.4',
    vi: { source: 'của chúng tôi: phép tính số nguyên C++; độ rộng địa chỉ ở §C.4, pp. 625–626; độ rộng chỉ số ở ch. 3 §3.4' },
  },
  'appC/wrap/q-private-logits': {
    kind: 'choice',
    id: 'appC/wrap/q-private-logits',
    prompt: (
      <Translated
        vi={
          <>
            Một vòng lặp decode Metal trên Apple silicon ghi logits vào một buffer có lưu trữ private, vì private dành cho dữ
            liệu chỉ GPU chạm tới. Việc lấy mẫu chạy trên CPU. CPU phải làm gì để đọc logits sau mỗi token?
          </>
        }
      >
        <>
          A Metal decode loop on Apple silicon writes its logits into a buffer with private storage, because private is
          for data only the GPU touches. Sampling runs on the CPU. What does the CPU have to do to read the logits after
          each token?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Đọc logits.contents sau waitUntilCompleted: CPU và GPU dùng chung một bộ nhớ, nên buffer nào CPU cũng đọc được.">
            {'Read logits.contents after waitUntilCompleted: the CPU and the GPU share one memory, so every buffer is CPU-readable.'}
          </Translated>
        ),
        why: (
          <Translated vi="Một bộ nhớ vật lý không làm cho mọi buffer đều hiện ra với CPU. contents là NULL với buffer private.">
            One physical memory does not make every buffer CPU-visible. contents is NULL for a private buffer.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Blit logits sang một buffer shared, chờ command buffer đó, đọc buffer shared; hoặc dùng buffer shared ngay từ đầu.">
            {'Blit the logits into a shared buffer, wait for that command buffer, read the shared one; or use a shared buffer from the start.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Gọi didModifyRange trên buffer, rồi đọc nó.">Call didModifyRange on the buffer, then read it.</Translated>,
        why: (
          <Translated vi="didModifyRange thuộc về lưu trữ managed, thứ không có tác dụng trên Apple silicon.">
            didModifyRange belongs to managed storage, which has no effect on Apple silicon.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không gì làm chúng đọc được: việc lấy mẫu phải chuyển sang GPU.">
            Nothing can make them readable: sampling has to move to the GPU.
          </Translated>
        ),
        why: (
          <Translated vi="Một lần sao chép blit giữa các buffer chính là cách dữ liệu đi vào hoặc ra khỏi một buffer private.">
            A blit copy between buffers is exactly how data gets into or out of a private buffer.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Apple: tài nguyên private không truy cập được từ CPU, và <code>contents</code> trả về NULL với chúng. Dữ liệu vào
            hoặc ra khỏi một buffer private qua một lần sao chép blit sang hoặc từ một buffer shared, đúng như{' '}
            <code>feedPrivateCopy</code> của lab C.1 làm ở cả hai đầu. Logits mà CPU đọc ở mỗi token không phải "dữ liệu chỉ
            GPU chạm tới", nên một buffer shared, đọc sau lần chờ, là lựa chọn đơn giản hơn. Trên một bộ nhớ chung, câu hỏi
            không bao giờ là byte có phải đi qua đường truyền hay không; mà là bạn đã chọn loại buffer nào.
          </p>
        }
      >
        <p>
          Apple: private resources are not CPU-accessible, and <code>contents</code> returns NULL for them. Data reaches
          or leaves a private buffer through a blit copy into or from a shared one, which is what lab C.1's{' '}
          <code>feedPrivateCopy</code> does at both ends. Logits that the CPU reads every token are not "data only the GPU
          touches", so a shared buffer, read after the wait, is the simpler choice. On one memory the question is never
          whether bytes must cross a link; it is which buffer type you chose.
        </p>
      </Translated>
    ),
    source: 'not in the book; ch02-facts C2, C3; ch05-facts C6; appC-facts C2',
    vi: { source: 'không có trong sách; ch02-facts C2, C3; ch05-facts C6; appC-facts C2' },
  },
  'appC/wrap/q-mapped-atomics': {
    kind: 'choice',
    id: 'appC/wrap/q-mapped-atomics',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel lấy mẫu đếm số token được chấp nhận bằng <code>atomicAdd</code> trên một bộ đếm trong bộ nhớ host ánh
            xạ, để host đọc được nó mà không cần sao chép. Trong khi kernel chạy, một luồng host cũng cộng vào cùng bộ đếm đó
            bằng <code>std::atomic</code>. Tài liệu của NVIDIA nói gì?
          </>
        }
      >
        <>
          A sampling kernel counts accepted tokens with <code>atomicAdd</code> on a counter in mapped host memory, so that
          the host can read it without a copy. While the kernel runs, a host thread also adds to the same counter with{' '}
          <code>std::atomic</code>. What does NVIDIA's documentation say?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="An toàn: cả hai bên đều dùng phép đọc-sửa-ghi nguyên tử trên cùng một bộ nhớ vật lý, nên không cập nhật nào bị mất.">
            {'It is safe: both sides use atomic read-modify-write operations on the same physical memory, so no update is lost.'}
          </Translated>
        ),
        why: (
          <Translated vi="NVIDIA nói các phép toán nguyên tử của GPU trên bộ nhớ ánh xạ không nguyên tử khi nhìn từ phía host.">
            {"NVIDIA says the GPU's atomics on mapped memory are not atomic from the host's point of view."}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Có thể mất cập nhật: các phép toán nguyên tử trên bộ nhớ ánh xạ không nguyên tử khi nhìn từ phía host.">
            {"Updates can be lost: atomics on mapped memory are not atomic from the host's point of view."}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="An toàn miễn là bộ đếm được ghim.">It is safe as long as the counter is pinned.</Translated>
        ),
        why: (
          <Translated vi="Bộ nhớ ánh xạ vốn đã được ghim. Việc ghim giữ cho các trang không bị đẩy ra khỏi RAM; nó không nói gì về tính nguyên tử.">
            Mapped memory is pinned already. Pinning keeps the pages from being paged out; it says nothing about atomicity.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không biên dịch được: atomicAdd chỉ nhận con trỏ tới bộ nhớ device.">
            It does not compile: atomicAdd accepts only pointers to device memory.
          </Translated>
        ),
        why: (
          <Translated vi="Với unified addressing, kernel nhận một con trỏ thông thường, và atomicAdd biên dịch được. Vấn đề nằm ở lúc chạy.">
            Under unified addressing the kernel receives an ordinary pointer, and atomicAdd compiles. The problem is at run time.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Bộ nhớ ánh xạ làm cho bộ nhớ host truy cập được, chứ không phải được chia sẻ theo mọi nghĩa. NVIDIA cũng không bảo
            đảm rằng phép toán nguyên tử trên bộ nhớ ánh xạ được hỗ trợ trên mọi hệ thống. Hãy tách một bộ đếm mà cả hai bên
            cùng cập nhật thành hai phần: GPU đếm trong bộ nhớ device và host đọc nó sau khi kernel xong, hoặc mỗi bên giữ số
            đếm riêng và bạn cộng chúng lại ở cuối. Quy tắc riêng của sách từ <a href={href('appC', 'C.6')}>mục C.6</a> cũng
            áp dụng ở đây: sắp đặt thứ tự các truy cập đồng thời vẫn là việc của bạn.
          </p>
        }
      >
        <p>
          Mapped memory makes host memory reachable, not shared in every sense. NVIDIA also does not guarantee that atomics
          on mapped memory are supported on every system. Keep a counter that both sides update in
          two parts: the GPU counts in device memory and the host reads it after the kernel, or each side keeps its own
          count and you add them at the end. The book's own rule from <a href={href('appC', 'C.6')}>section C.6</a>{' '}
          applies here too: concurrent access is still yours to order.
        </p>
      </Translated>
    ),
    source: 'vendor documentation, not in the book: appC-facts A3; book §C.6, p. 626 for synchronising concurrent access',
    vi: {
      source: 'tài liệu của nhà cung cấp, không có trong sách: appC-facts A3; sách §C.6, p. 626 về việc đồng bộ hóa truy cập đồng thời',
    },
  },
  'appC/wrap/q-embed-mapped': {
    kind: 'numeric',
    id: 'appC/wrap/q-embed-mapped',
    prompt: (
      <Translated
        vi={
          <>
            Bảng token embedding của một mô hình có {n(VOCAB)} hàng và nằm trong bộ nhớ host ánh xạ. Mỗi token được sinh ra đọc
            một hàng, mỗi lần một hàng khác. Trong mô hình chỉ tính băng thông của mục C.2, với tỷ lệ đường truyền 10% của
            sách, sau bao nhiêu token được sinh ra thì việc sao chép cả bảng sang bộ nhớ device trước lẽ ra đã rẻ hơn?
          </>
        }
      >
        <>
          A model's token-embedding table has {n(VOCAB)} rows and sits in mapped host memory. Each generated token reads
          one row, a different one every time. In section C.2's bandwidth-only model, with the book's link ratio of 10%,
          after how many generated tokens would copying the whole table to device memory first have been cheaper?
        </>
      </Translated>
    ),
    answer: Math.round(EMBED_BREAK_EVEN),
    tol: 0.02,
    unit: 'tokens',
    explain: (
      <Translated
        vi={
          <>
            <p>
              Mỗi token đọc 1/{n(VOCAB)} của bảng, và đọc tại chỗ thắng khi các lần đọc trong suốt vòng đời của bảng cộng lại
              ít hơn 1 / (1 − 0.1) ≈ 1.11 bảng. Tức là {n(VOCAB)} / 0.9 ≈ {n(Math.round(EMBED_BREAK_EVEN))} token, nhiều hơn
              xa số token một lần sinh tạo ra. Bảng embedding chính là dữ liệu được đọc "thỉnh thoảng và thưa thớt" mà sách
              nghĩ tới: trong mô hình này nó có thể ở lại bộ nhớ host và giải phóng chỗ của nó trong bộ nhớ device.
            </p>
            <p>
              Mô hình bỏ qua độ trễ, thứ mà một hàng duy nhất mỗi token phải trả đủ, và prompt, nơi nhiều token đọc nhiều hàng
              cùng lúc. Workload <code>sparse1</code> của lab C.1 là phép tra này với một hàng trong 64; hãy đo nó trước khi
              quyết định.
            </p>
          </>
        }
      >
        <>
          <p>
            Each token reads 1/{n(VOCAB)} of the table, and reading in place wins while the reads over the table's life
            add up to less than 1 / (1 − 0.1) ≈ 1.11 tables. That is {n(VOCAB)} / 0.9 ≈ {n(Math.round(EMBED_BREAK_EVEN))}{' '}
            tokens, far more than one generation produces. The embedding table is the "occasionally and sparsely" read
            data the book has in mind: in this model it can stay in host memory and free its space in device memory.
          </p>
          <p>
            The model leaves out latency, which a single row per token pays in full, and the prompt, whose many tokens
            read many rows at once. Lab C.1's <code>sparse1</code> workload is this lookup at one row in 64; measure it
            before you decide.
          </p>
        </>
      </Translated>
    ),
    source: 'book p. 625 ("occasionally and sparsely", the 10%); the break-even rule is ours (§C.2 page); the lookup is ch. 20, p. 480',
    vi: {
      unit: 'token',
      source:
        'sách p. 625 ("thỉnh thoảng và thưa thớt", con số 10%); quy tắc hòa vốn là của chúng tôi (trang §C.2); phép tra ở ch. 20, p. 480',
    },
  },
};

const LAB = 'appC/lab1_feeding';
// Backends that are not a GPU: the emulator and its race-detection flavour.
const EMULATED = new Set(['emu', 'race']);

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const devices = Object.keys(reports).filter((b) => !EMULATED.has(b));
  const mine = devices.filter((b) => labStatus(reports[b], LAB, 'work') === 'pass');
  const mineEmu = labStatus(reports.emu, LAB, 'work') === 'pass';
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('appC/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>
        <Translated vi="Những gì cần mang theo">What to carry forward</Translated>
      </h2>
      <Translated
        vi={
          <ul>
            <li>
              Mọi câu hỏi về con trỏ đều có hai nửa. Giá trị đó có gọi tên được vùng nhớ không, và đoạn mã này có được
              dereference nó ở đó không? Lịch sử của phụ lục phần lớn là nửa đầu đến trước (một không gian địa chỉ trên GPU,
              rồi một không gian cho host và các device, rồi địa chỉ đủ rộng cho tất cả) còn nửa sau theo sau, muộn hơn, hoặc
              không bao giờ.
            </li>
            <li>
              Trong CUDA và HIP, một hàm trợ giúp với con trỏ thường phục vụ được bộ nhớ chia sẻ (shared memory), bộ nhớ cục
              bộ (local memory) và bộ nhớ toàn cục (global memory): địa chỉ quyết định một lần nạp chạm tới bộ nhớ nào, và tốc
              độ phụ thuộc vào bộ nhớ đó. MSL theo đặc tả buộc mọi con trỏ ghi rõ không gian địa chỉ của nó, nên một hàm trợ
              giúp được viết một lần cho mỗi bộ nhớ mà nó được gọi trên đó.
            </li>
            <li>
              Đọc bộ nhớ host tại chỗ phải trả giá kênh kết nối ở mọi lần truy cập. Nó chỉ thắng khi số byte kernel đọc trong
              suốt vòng đời còn dưới khoảng một bảng dữ liệu; vượt qua đó, hãy sao chép trước. Trên Apple silicon không có
              đường truyền nào để trả giá, và lựa chọn trở thành dùng loại buffer nào.
            </li>
            <li>
              Một không gian địa chỉ ảo cho runtime biết một con trỏ nằm ở đâu, đó là lý do <code>cudaMemcpyDefault</code> hoạt
              động. Nó không nói ai được dùng con trỏ. Trong CUDA, mã host không bao giờ dereference bộ nhớ{' '}
              <code>cudaMalloc</code>, trên bất kỳ hệ thống nào; HIP cho phép điều đó ở tốc độ đường truyền.
            </li>
            <li>
              Unified Memory, viết hoa chữ U, là một vùng trang mà runtime CUDA di chuyển giữa hai bộ nhớ. Ba thuộc tính của
              device cho biết bạn có loại nào: hạn chế (bộ nhớ managed di chuyển hàng loạt lúc khởi chạy, và CPU tránh xa khi
              GPU chạy), đầy đủ (trang di chuyển khi có lỗi trang, và bạn có thể cấp phát vượt dung lượng), và đầy đủ cho toàn
              bộ bộ nhớ host (HMM hoặc ATS). Unified memory của Apple là một bộ nhớ vật lý, không có gì để di chuyển.
            </li>
            <li>
              Một lỗi trang GPU cho phép runtime di chuyển một trang hoặc ánh xạ nó. Di chuyển trả giá cho mỗi trang được
              chuyển, với lượng công việc tỷ lệ với kích thước trang; ánh xạ trả giá ở mỗi lần truy cập. Gợi ý thay đổi chi phí,
              không bao giờ thay đổi kết quả.
            </li>
            <li>
              Truy cập bộ nhớ của GPU khác, và bộ nhớ do bạn tự ánh xạ, cần một lần cấp quyền tường minh:{' '}
              <code>cudaDeviceEnablePeerAccess</code>, mỗi lần một chiều, và <code>cuMemSetAccess</code> sau{' '}
              <code>cuMemMap</code>. Đặt trước địa chỉ cho phép một buffer như KV cache lớn dần mà không phải dời đi.
            </li>
            <li>
              Không điều nào ở trên sắp đặt thứ tự. Một con trỏ, các trang nhất quán hay một bộ nhớ vật lý, mỗi thứ đều bỏ được
              một lần sao chép, và thứ nào cũng vẫn để bạn tự làm cho CPU và GPU thay phiên nhau.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            Every pointer question has two halves. Can the value name the memory, and may this code dereference it there?
            The appendix's history is mostly the first half arriving (one address space on the GPU, then one for host and
            devices, then addresses wide enough for all of it) with the second following later, or not at all.
          </li>
          <li>
            In CUDA and HIP one helper with a plain pointer serves shared, local and global memory: the address decides
            which memory a load touches, and the speed depends on which. MSL as specified makes every pointer name its
            address space, so a helper is written once per memory it is called on.
          </li>
          <li>
            Reading host memory in place pays the interconnect on every access. It wins only while the bytes a kernel
            reads over its whole life stay below about one table's worth; past that, copy first. On Apple silicon there is
            no link to pay, and the choice becomes which buffer type you use.
          </li>
          <li>
            One virtual address space tells the runtime where a pointer lives, which is why <code>cudaMemcpyDefault</code>{' '}
            works. It does not say who may use the pointer. In CUDA, host code never dereferences{' '}
            <code>cudaMalloc</code> memory, on any system; HIP allows it at link speed.
          </li>
          <li>
            Unified Memory, with a capital U, is a pool of pages the CUDA runtime moves between two memories. Three device
            attributes say which kind you have: limited (managed memory moves in bulk at a launch, and the CPU keeps out
            while the GPU runs), full (pages move on a fault, and you may oversubscribe), and full for all host memory (HMM or ATS).
            Apple's unified memory is one physical memory with nothing to move.
          </li>
          <li>
            A GPU page fault lets the runtime migrate a page or map it. Migrating pays for every page moved, in work
            proportional to the page size; mapping pays on every access. Hints change the cost, never the result.
          </li>
          <li>
            Reaching another GPU's memory, and memory you mapped yourself, takes an explicit grant:{' '}
            <code>cudaDeviceEnablePeerAccess</code>, one direction at a time, and <code>cuMemSetAccess</code> after{' '}
            <code>cuMemMap</code>. Reserving addresses first lets a buffer such as a KV cache grow without moving.
          </li>
          <li>
            None of this orders anything. One pointer, coherent pages or one physical memory each remove a copy, and every
            one of them still leaves you to make the CPU and the GPU take turns.
          </li>
        </ul>
      </Translated>
      <Source section="C.1–C.9" pages="623–629" />
      <Translated
        vi={
          <p>
            Bản đồ từ <a href={href('appC', 'start')}>trang mở đầu</a>, dạng thu gọn, mở ở bước vượt ra ngoài sách. Chuyển sang
            HIP hoặc Metal để xem cùng các hàng đó ngày nay, và nhấp vào một ô để xem quy tắc và nguồn của nó.
          </p>
        }
      >
        <p>
          The map from the <a href={href('appC', 'start')}>start page</a>, compact, opened at the step beyond the book.
          Switch to HIP or Metal to see the same rows today, and click a cell for its rule and source.
        </p>
      </Translated>
      <DerefMap compact initialEra="hmm" /* i18n-ignore: era id */ />

      <h2>
        <Translated vi="Bài kiểm tra tổng hợp">A mixed check</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Các câu này dựa trên toàn bộ phụ lục và trên những tình huống mà các mục chưa cho thấy.
            {asked ? ` Đến lúc này trong phụ lục bạn đã trả lời đúng ${right} trên ${asked} câu hỏi, tính theo câu trả lời gần nhất của bạn cho mỗi câu.` : ''}
          </p>
        }
      >
        <p>
          These draw on the whole appendix and on situations the sections did not show.
          {asked ? ` So far in this appendix you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
        </p>
      </Translated>
      <Code
        tabs={[
          {
            label: tr('CUDA (not a lab file: for this question only)', 'CUDA (không phải tệp lab: chỉ cho câu hỏi này)'),
            code: LAYER_TABLE,
            tiers: { layers: 'global', out: 'global' },
            status: tr('front-end-checked; has a defect on purpose', 'front-end-checked; có một lỗi cài sẵn'),
          },
        ]}
        caption={
          <Translated
            vi={
              <>
                Của chúng tôi, không phải của sách. Bản đã kiểm tra là <code>labs/appC/snippets/wrap_layer_table.cu</code>. Các
                phép kiểm tra lỗi được bỏ đi để câu hỏi ngắn gọn.
              </>
            }
          >
            <>
              Ours, not the book's. The checked copy is <code>labs/appC/snippets/wrap_layer_table.cu</code>. Error checks are
              left out to keep the question short.
            </>
          </Translated>
        }
      />
      <Ask q={questionsWrap['appC/wrap/q-layer-table']} label={tr('Find the defect', 'Tìm lỗi')} />
      <Ask q={questionsWrap['appC/wrap/q-kv-offset']} />
      <Ask q={questionsWrap['appC/wrap/q-private-logits']} />
      <Ask q={questionsWrap['appC/wrap/q-mapped-atomics']} />
      <Ask q={questionsWrap['appC/wrap/q-embed-mapped']} />

      <h2>
        <Translated vi="Bài tập của sách">The book's exercises</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Phụ lục không có bài tập nào: bảy trang của nó kết thúc bằng một tài liệu tham khảo duy nhất. Thay vào đó, đây là
            nơi mỗi phần của nó được thực hành trong khóa học này, để bạn kiểm tra rằng mình đã gặp đủ cả.
          </p>
        }
      >
        <p>
          The appendix has none: its seven pages end with a single reference. In their place, here is where each part of
          it is worked in this course, so that you can check you have met all of it.
        </p>
      </Translated>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="exercises">
        <thead>
          <tr>
            <th>
              <Translated vi="Sách">Book</Translated>
            </th>
            <th>
              <Translated vi="Trang">Pages</Translated>
            </th>
            <th>
              <Translated vi="Được thực hành ở đâu trong khóa học">Where it is worked here</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <Translated vi="Mô hình sao chép và hai vấn đề của nó">The copy model and its two problems</Translated>
            </td>
            <td>623</td>
            <td>
              <Translated vi={<>bài dự đoán về trọng số và bản đồ trên <a href={href('appC', 'start')}>trang mở đầu</a></>}>
                <>the weights prediction and the map on the <a href={href('appC', 'start')}>start page</a></>
              </Translated>
            </td>
          </tr>
          <tr>
            <td>C.1</td>
            <td>624</td>
            <td>
              <Translated vi={<>trình khám phá cửa sổ địa chỉ và các thẻ biên dịch trong <a href={href('appC', 'C.1')}>C.1</a></>}>
                <>the address-window explorer and the compile cards in <a href={href('appC', 'C.1')}>C.1</a></>
              </Translated>
            </td>
          </tr>
          <tr>
            <td>C.2</td>
            <td>624–625</td>
            <td>
              <Translated vi={<>trình khám phá chi phí trong <a href={href('appC', 'C.2')}>C.2</a>; đường mapped của lab C.1</>}>
                <>the cost explorer in <a href={href('appC', 'C.2')}>C.2</a>; the mapped path of lab C.1</>
              </Translated>
            </td>
          </tr>
          <tr>
            <td>C.3</td>
            <td>625</td>
            <td>
              <Translated
                vi={<>trình khám phá chiều sao chép và trình chạy từng bước danh sách trong <a href={href('appC', 'C.3')}>C.3</a></>}
              >
                <>the copy-direction explorer and the list stepper in <a href={href('appC', 'C.3')}>C.3</a></>
              </Translated>
            </td>
          </tr>
          <tr>
            <td>C.4</td>
            <td>625–626</td>
            <td>
              <Translated vi={<>trình khám phá từ bit sang byte trong <a href={href('appC', 'C.4')}>C.4</a>; câu hỏi về offset KV ở trên</>}>
                <>the bits-to-bytes explorer in <a href={href('appC', 'C.4')}>C.4</a>; the KV offset question above</>
              </Translated>
            </td>
          </tr>
          <tr>
            <td>C.5</td>
            <td>626</td>
            <td>
              <Translated vi={<>ma trận peer access trong <a href={href('appC', 'C.5')}>C.5</a></>}>
                <>the peer access matrix in <a href={href('appC', 'C.5')}>C.5</a></>
              </Translated>
            </td>
          </tr>
          <tr>
            <td>C.6, Fig. C.1{/* i18n-ignore: section and figure numbers */}</td>
            <td>626–627</td>
            <td>
              <Translated
                vi={
                  <>
                    trình chạy từng bước Fig. C.1 và hoạt hình về hai loại unified memory trong <a href={href('appC', 'C.6')}>C.6</a>;
                    đường managed của lab C.1
                  </>
                }
              >
                <>the Fig. C.1 stepper and the animation of the two unified memories in <a href={href('appC', 'C.6')}>C.6</a>; the managed path of lab C.1</>
              </Translated>
            </td>
          </tr>
          <tr>
            <td>C.7</td>
            <td>627–628</td>
            <td>
              <Translated vi={<>bài kiểm tra độ vừa và bảng mô hình trong <a href={href('appC', 'C.7')}>C.7</a></>}>
                <>the fit check and the paradigm table in <a href={href('appC', 'C.7')}>C.7</a></>
              </Translated>
            </td>
          </tr>
          <tr>
            <td>C.8</td>
            <td>628–629</td>
            <td>
              <Translated
                vi={
                  <>
                    trình khám phá tính nhất quán, hoạt hình lỗi trang và trình khám phá cấp phát vượt dung lượng trong{' '}
                    <a href={href('appC', 'C.8')}>C.8</a>
                  </>
                }
              >
                <>the coherence explorer, the page-fault animation and the oversubscription explorer in <a href={href('appC', 'C.8')}>C.8</a></>
              </Translated>
            </td>
          </tr>
          <tr>
            <td>C.9</td>
            <td>629</td>
            <td>
              <Translated
                vi={
                  <>
                    trình chạy từng bước đặt trước-ánh xạ-cấp quyền và hoạt hình buffer lớn dần trong{' '}
                    <a href={href('appC', 'C.9')}>C.9</a>
                  </>
                }
              >
                <>the reserve-map-grant stepper and the growing-buffer animation in <a href={href('appC', 'C.9')}>C.9</a></>
              </Translated>
            </td>
          </tr>
        </tbody>
      </table>

      <h2>
        <Translated vi="Những chỗ bản PDF có vẻ sai">Where the PDF looks wrong</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trước hết là những lỗi nhỏ trên trang in của lần xuất bản mới này, được liệt kê để bạn không bị bối rối khi đọc
            sách song song.
          </p>
        }
      >
        <p>
          First the slips in the printed page of this new edition, listed so that you are not thrown by them when you read
          the book alongside.
        </p>
      </Translated>
      <Translated
        vi={
          <ul data-list="errata">
            <li>
              Fig. C.1 (p. 627): <code>cudaDeviceSynchronize()</code> không có dấu chấm phẩy, nên khung bên phải không biên dịch
              được như đã in; cấu hình khởi chạy bị lược thành <code>&lt;&lt;&lt;...&gt;&gt;&gt;</code>; văn bản nói kernel
              thay cho <code>qsort()</code>, trong khi hình gọi nó là <code>qsort_char</code>; tiêu đề có hai dấu cách giữa
              "CUDA 6" và "code". Các bản đã kiểm tra nằm trong <code>labs/appC/snippets/</code>, và{' '}
              <a href={href('appC', 'C.6')}>mục C.6</a> chạy từng bước qua chúng.
            </li>
            <li>
              p. 624: "We mentioned that the other values of the flag argument are for more advanced usage." Chúng tôi không
              tìm thấy lần nhắc nào trước đó trong lần xuất bản này (tìm trong văn bản chỉ thấy <code>cudaHostAlloc</code> ở
              Phụ lục C). Chương 23 (p. 562) thay vào đó cấp phát bộ nhớ ghim bằng <code>cudaMallocHost</code>, và sách không
              bao giờ liên hệ hai lời gọi này. Chúng là cùng một kiểu cấp phát, có cờ và không có cờ.
            </li>
            <li>p. 624: "shared, memory, local memory, and global memory" có một dấu phẩy thừa.</li>
            <li>p. 623 gọi phụ lục là "this section".</li>
            <li>
              Phụ lục viết "Unified Virtual Address Space (UVAS)" (p. 625), chú thích của chương 2 viết "Unified Virtual
              Address (UVA)" (p. 35). Cùng một tính năng; NVIDIA dùng cả hai tên.
            </li>
            <li>
              Tài liệu tham khảo [1] (p. 629) liệt kê các tác giả theo thứ tự khác với bài báo đã xuất bản, vốn là Gelado,
              Cabezas, Navarro, Stone, Patel, Hwu.
            </li>
          </ul>
        }
      >
        <ul data-list="errata">
          <li>
            Fig. C.1 (p. 627): <code>cudaDeviceSynchronize()</code> has no semicolon, so the right-hand box does not
            compile as printed; the launch configuration is elided as <code>&lt;&lt;&lt;...&gt;&gt;&gt;</code>; the prose
            says the kernel replaces <code>qsort()</code>, while the figure calls it <code>qsort_char</code>; the heading
            has two spaces between "CUDA 6" and "code". The checked copies are in <code>labs/appC/snippets/</code>, and{' '}
            <a href={href('appC', 'C.6')}>section C.6</a> steps through them.
          </li>
          <li>
            p. 624: "We mentioned that the other values of the flag argument are for more advanced usage." We found no
            earlier mention in this edition (a search of the text finds <code>cudaHostAlloc</code> only in Appendix C).
            Chapter 23 (p. 562) allocates pinned memory with <code>cudaMallocHost</code>{' '}
            instead, and the book never relates the two. They are one allocation, with flags and without.
          </li>
          <li>p. 624: "shared, memory, local memory, and global memory" has a stray comma.</li>
          <li>p. 623 calls the appendix "this section".</li>
          <li>
            The appendix says "Unified Virtual Address Space (UVAS)" (p. 625), chapter 2's footnote says "Unified Virtual
            Address (UVA)" (p. 35). Same feature; NVIDIA uses both names.
          </li>
          <li>
            Reference [1] (p. 629) lists the authors in a different order from the published paper, which is Gelado,
            Cabezas, Navarro, Stone, Patel, Hwu.
          </li>
        </ul>
      </Translated>
      <Aside>
        <Translated
          vi={<p>Đây là cách đọc của chúng tôi trên một bản PDF. Nếu bản của bạn khác, hãy tin bản của bạn.</p>}
        >
          <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Tiếp theo là những khẳng định mà chính tài liệu của NVIDIA, kiểm tra ngày 2026-11-20, sửa lại hoặc để ngỏ. Các mục
            dạy dạng đã được sửa; đây là danh sách gom vào một chỗ. Mỗi mục nêu tên mục tương ứng trong{' '}
            <code>notes/today/appC-facts.md</code>, nơi trích dẫn nguyên văn lời của nhà cung cấp.
          </p>
        }
      >
        <p>
          Then the statements that NVIDIA's own documents, checked on 2026-11-20, correct or leave open. The sections
          teach the corrected form; this is the list in one place. Each item names its entry in{' '}
          <code>notes/today/appC-facts.md</code>, where the vendor's own words are quoted.
        </p>
      </Translated>
      <Translated
        vi={
          <ul data-list="corrections">
            <li>
              Fermi "ra mắt năm 2009" (p. 624): whitepaper là từ năm 2009; NVIDIA nói Fermi có mặt trên thị trường vào tháng 3
              năm 2010 (A2).
            </li>
            <li>
              Kepler "ra mắt năm 2013" (p. 626): những GPU Kepler đầu tiên ra mắt vào tháng 3 năm 2012. Kepler cũng không phải
              đầu tiên: Fermi đã có định địa chỉ nạp và ghi 64-bit, một không gian địa chỉ hợp nhất 40-bit, hơn 4 GiB bộ nhớ và
              unified addressing. Địa chỉ vật lý "ít nhất 40-bit" của Kepler không kiểm chứng được (A6).
            </li>
            <li>
              Peer access "trước Kepler" cần các lần sao chép do host điều khiển (p. 626): theo lời kể của chính NVIDIA,
              unified addressing đã chạm tới bộ nhớ của GPU khác từ CUDA 4, và không nguồn nào chúng tôi đọc nêu tên kiến trúc
              đầu tiên có peer load. Peer access cũng phải được bật, mỗi lần một chiều, với tối đa tám peer mỗi GPU khi không
              có NVSwitch (A8).
            </li>
            <li>CUDA 6 và Unified Memory "2013" (p. 626): công bố tháng 11 năm 2013, phát hành tháng 4 năm 2014 (A9).</li>
            <li>
              Các lời gọi đặt trước-và-ánh xạ "CUDA 11" (p. 629): chúng xuất hiện trong CUDA 10.2, tháng 11 năm 2019, và riêng{' '}
              <code>cuMemMap</code> không làm cho bộ nhớ dùng được: <code>cuMemSetAccess</code> phải cấp quyền truy cập (A15).
            </li>
            <li>
              Bộ nhớ ánh xạ cần một con trỏ thứ hai từ <code>cudaHostGetDevicePointer()</code> (p. 624): đúng vào năm 2009.
              Với unified addressing, bộ nhớ <code>cudaHostAlloc</code> và <code>cudaMallocHost</code> dùng được trong kernel
              qua chính con trỏ của host; chỉ bộ nhớ được đăng ký và bộ nhớ write-combined vẫn có địa chỉ device riêng (A3).
            </li>
            <li>
              Đường truyền có băng thông "dưới 10%" băng thông device (p. 625): đúng với PCIe theo số liệu hiện nay của NVIDIA,
              không đúng với các hệ thống NVLink-C2C. NVIDIA không còn nêu một tỷ lệ nào (A4).
            </li>
            <li>
              "Mỗi địa chỉ bộ nhớ vật lý chỉ ánh xạ tới một vị trí ảo" (p. 625): bộ nhớ host được đăng ký và write-combined là
              các ngoại lệ được ghi nhận (A5).
            </li>
            <li>
              UVAS "bắt nguồn từ" GMAC (p. 625): không nguồn nào của NVIDIA mà chúng tôi đọc nhắc tới GMAC, nên đây vẫn là lời
              kể của sách (A5).
            </li>
            <li>
              Sách để ngỏ, NVIDIA đã trả lời: từ Pascal trở đi, kernel có thể đi theo các cấu trúc do host dựng trong bộ nhớ
              managed, nhưng chỉ chạm tới bộ nhớ <code>malloc</code> thông thường khi có HMM hoặc ATS (A12); và driver quyết
              định giữa di chuyển và ánh xạ bằng heuristic, mà <code>cudaMemAdvise</code> và <code>cudaMemPrefetchAsync</code>{' '}
              có thể điều khiển (A13).
            </li>
          </ul>
        }
      >
        <ul data-list="corrections">
          <li>
            Fermi "introduced 2009" (p. 624): the whitepaper is from 2009; NVIDIA says Fermi reached the market in March
            2010 (A2).
          </li>
          <li>
            Kepler "introduced in 2013" (p. 626): the first Kepler GPUs launched in March 2012. Nor was Kepler first:
            Fermi already had 64-bit load and store addressing, a 40-bit unified address space, more than 4 GiB of memory and
            unified addressing. Kepler's "at least 40-bit" physical addresses could not be checked (A6).
          </li>
          <li>
            Peer access "before Kepler" needed host-driven copies (p. 626): NVIDIA's own account has unified addressing
            reaching another GPU's memory since CUDA 4, and no source we read names the first architecture with peer
            loads. Peer access must also be enabled, one direction at a time, with at most eight peers per GPU without
            NVSwitch (A8).
          </li>
          <li>
            CUDA 6 and Unified Memory "2013" (p. 626): announced in November 2013, shipped in April 2014 (A9).
          </li>
          <li>
            The reserve-and-map calls "CUDA 11" (p. 629): they arrived in CUDA 10.2, in November 2019, and{' '}
            <code>cuMemMap</code> alone does not make memory usable: <code>cuMemSetAccess</code> must grant access (A15).
          </li>
          <li>
            Mapped memory needs a second pointer from <code>cudaHostGetDevicePointer()</code> (p. 624): true in 2009. Under
            unified addressing <code>cudaHostAlloc</code> and <code>cudaMallocHost</code> memory works in kernels through
            the host's own pointer; only registered and write-combined memory still has a distinct device address (A3).
          </li>
          <li>
            The link has "less than 10%" of device bandwidth (p. 625): true of PCIe on NVIDIA's current figures, not of
            NVLink-C2C systems. NVIDIA no longer states a ratio (A4).
          </li>
          <li>
            "Each physical memory address maps to only one virtual location" (p. 625): registered and write-combined host
            memory are documented exceptions (A5).
          </li>
          <li>
            UVAS "originated from" GMAC (p. 625): no NVIDIA source we read mentions GMAC, so this stays the book's account
            (A5).
          </li>
          <li>
            Left open by the book, settled by NVIDIA: from Pascal on a kernel can follow host-built structures in managed
            memory, but reaches plain <code>malloc</code> memory only with HMM or ATS (A12); and the driver decides between
            migrating and mapping by heuristics, which <code>cudaMemAdvise</code> and <code>cudaMemPrefetchAsync</code>{' '}
            can steer (A13).
          </li>
        </ul>
      </Translated>

      <h2>
        <Translated vi="Những gì phụ lục này bỏ qua">What this appendix left out</Translated>
      </h2>
      <Translated
        vi={
          <ul>
            <li>
              Vấn đề về ổ đĩa và card mạng ở phần mở đầu (p. 623) có một ghi chú trên trang mở đầu; sách không bao giờ quay lại
              nó.
            </li>
            <li>
              Lưới của chương 21 được cắt thành các lát 2D (p. 623) được nêu như ví dụ của sách về chia khối, không được làm lại
              ở đây; nó ở p. 517, và <a href={href('ch21', '21.2')}>mục 21.2</a> của chương 21 nói ngắn gọn về nó.
            </li>
            <li>
              Trao đổi vùng biên (halo) của chương 23 (p. 626) chỉ xuất hiện trong danh sách các bước của{' '}
              <a href={href('appC', 'C.5')}>mục C.5</a>. Stencil nhiều GPU, MPI, NCCL và NVSHMEM vẫn ở{' '}
              <a href={href('ch23')}>chương 23</a> (pp. 541–576).
            </li>
            <li>
              Các tập dữ liệu lớn trong bộ nhớ của CAD (p. 629) và các thư viện CUDA rẻ hơn (p. 624) mỗi thứ chỉ là một câu
              trong sách, và không được mở rộng.
            </li>
            <li>
              Bài báo GMAC, tài liệu tham khảo [1] của sách, được trích dẫn, với thứ tự tác giả như đã xuất bản, nhưng không
              được tóm tắt.
            </li>
            <li>
              Kernel sắp xếp của Fig. C.1 không có trong sách, và ở đây cũng không viết kernel nào: lab C.1 dùng một kernel gom
              (gather) thay thế.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            The opening's problem with disks and network cards (p. 623) gets one note on the start page; the book never
            returns to it.
          </li>
          <li>
            Chapter 21's grid cut into 2D slices (p. 623) is named as the book's example of chunking, not reworked here; it
            is on p. 517, and chapter 21's <a href={href('ch21', '21.2')}>section 21.2</a> covers it briefly.
          </li>
          <li>
            Chapter 23's halo exchange (p. 626) appears only in <a href={href('appC', 'C.5')}>section C.5</a>'s step list.
            The multi-GPU stencil, MPI, NCCL and NVSHMEM stay in <a href={href('ch23')}>chapter 23</a> (pp. 541–576).
          </li>
          <li>
            The large in-core data sets of CAD (p. 629) and the cheaper CUDA libraries (p. 624) are one sentence each in
            the book, and are not expanded.
          </li>
          <li>The GMAC paper, the book's reference [1], is cited, with its published author order, but not summarised.</li>
          <li>
            Fig. C.1's sort kernel is not in the book, and none is written here: lab C.1 uses a gather kernel instead.
          </li>
        </ul>
      </Translated>

      <h2>
        <Translated vi="Dự án đến lúc này">The project so far</Translated>
      </h2>
      <p>
        <Translated
          vi={
            <>
              Phụ lục không thêm mốc dự án nào. <code>gatherAddKernel</code> của lab C.1 theo các quy ước của dự án và có thể
              làm phép tra token embedding cho <code>nanoinfer</code>.
            </>
          }
        >
          <>
            The appendix adds no milestone. Lab C.1's <code>gatherAddKernel</code> follows the project's conventions and can
            serve as <code>nanoinfer</code>'s token-embedding lookup.
          </>
        </Translated>{' '}
        <span data-c1>
          {devices.length === 0 && !reports.emu
            ? tr('No lab results are recorded on this machine yet.', 'Chưa có kết quả lab nào được ghi lại trên máy này.')
            : mine.length
              ? tr(`Your own feed functions pass on ${mine.join(' and ')} on this machine.`, `Các hàm feed của chính bạn đã qua trên ${mine.join(' và ')} trên máy này.`)
              : mineEmu
                ? tr(
                    'Your own feed functions pass on the emulator; they have not passed on a device on this machine yet.',
                    'Các hàm feed của chính bạn đã qua trên trình giả lập; chúng chưa qua trên device nào trên máy này.',
                  )
                : tr('Your own feed functions have not passed on this machine yet.', 'Các hàm feed của chính bạn chưa qua trên máy này.')}
        </span>{' '}
        <Translated
          vi={
            <>
              Xem trang <a href={href('appC', 'labs')}>lab</a>. Điều phụ lục mang lại cho dự án là một cách quyết định mỗi
              tensor nằm ở đâu: trọng số thường trú trong bộ nhớ device, các bảng thưa như embedding được đọc tại chỗ nếu thiếu
              bộ nhớ, và logits nằm trong bộ nhớ mà cả hai bộ xử lý đều đọc được.
            </>
          }
        >
          <>
            See the <a href={href('appC', 'labs')}>labs</a>. What the appendix gives the project is a way to decide where each
            tensor lives: weights resident in device memory, sparse tables such as the embedding read in place if memory is
            short, and logits in memory both processors may read.
          </>
        </Translated>
      </p>
      <Translated
        vi={
          <p>
            Theo thứ tự của khóa học, <a href={href('x3')}>module X3, Đọc các backend thực tế</a>, là phần tiếp theo. Trên các
            backend GPU của llama.cpp, MLX và PyTorch, đó là nơi bạn gặp các cơ chế của phụ lục này trong mã production: pool
            bộ nhớ device được đặt trước và ánh xạ của llama.cpp cùng công tắc chuyển sang bộ nhớ managed, backend Metal của nó
            bọc trọng số được ánh xạ bộ nhớ mà không sao chép, và bộ nhớ host được ghim cùng expandable segments của PyTorch.
          </p>
        }
      >
        <p>
          In the course's order, <a href={href('x3')}>module X3, Reading real backends</a>, comes next. On the GPU
          backends of llama.cpp, MLX and PyTorch, it is where you meet this appendix's mechanisms in production code: llama.cpp's pool of reserved and mapped device memory and its
          switch to managed memory, its Metal backend wrapping memory-mapped weights without a copy, and PyTorch's pinned
          host memory and expandable segments.
        </p>
      </Translated>
    </>
  );
}
