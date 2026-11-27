import { Ask, type Question } from '../../components/Ask';
import { Beyond, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// Before you start. Prerequisite questions from the notes of chapters 4, 6, 10, 11, 12 and 13;
// vendor and project facts from notes/today/ch14-facts.md (A1, A7, A9, D3, D7), ch06-facts.md
// (A2) and ch10-facts.md (A1).
export const questionsS140: Record<string, Question> = {
  'ch14/start/q-exscan': {
    kind: 'numeric',
    id: 'ch14/start/q-exscan',
    prompt: (
      <Translated
        vi={
          <>
            Lấy scan loại trừ (với +) của các bit <code>[1, 0, 1, 1, 0, 0, 1, 1]</code>. Giá trị tại chỉ số 5 là bao
            nhiêu?
          </>
        }
      >
        Take the exclusive scan (with +) of the bits <code>[1, 0, 1, 1, 0, 0, 1, 1]</code>. What is the value at index
        5?
      </Translated>
    ),
    answer: 3,
    explain: (
      <Translated
        vi={
          <p>
            Scan loại trừ đặt tại mỗi chỉ số tổng của mọi thứ đứng hẳn trước nó: 1 + 0 + 1 + 1 + 0 = 3. Khi đầu vào là
            các bit, tổng đó chính là số bit 1 đứng trước mỗi vị trí. Mục{' '}
            <a href={href('ch14', '14.5')}>14.5</a> cần đúng con số này cho mọi khóa của một lượt radix, và gọi nó là
            "số bit 1 phía trước". Quay lại <a href={href('ch11', '11.1')}>mục 11.1</a>.
          </p>
        }
      >
        <p>
          An exclusive scan puts at each index the sum of everything strictly before it: 1 + 0 + 1 + 1 + 0 = 3. When the
          inputs are bits, that sum is the number of ones before each position. Section{' '}
          <a href={href('ch14', '14.5')}>14.5</a> needs exactly this number for every key of a radix pass, and calls it
          "ones before". Back to <a href={href('ch11', '11.1')}>section 11.1</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.1, p. 253',
  },
  'ch14/start/q-filterpos': {
    kind: 'numeric',
    id: 'ch14/start/q-filterpos',
    prompt: (
      <Translated
        vi={
          <>
            Một bộ lọc ổn định giữ lại các phần tử có cờ là <code>[0, 1, 1, 0, 1]</code>. Phần tử đầu vào 4 rơi vào vị
            trí đầu ra nào?
          </>
        }
      >
        A stable filter keeps the elements whose flags are <code>[0, 1, 1, 0, 1]</code>. At which output position does
        input element 4 land?
      </Translated>
    ),
    answer: 2,
    explain: (
      <Translated
        vi={
          <p>
            Scan loại trừ của các cờ là <code>[0, 0, 1, 2, 2]</code>: phần tử 4 có hai phần tử được giữ đứng trước nó,
            nên nó lấy ô 2, và các phần tử được giữ vẫn theo thứ tự đầu vào. Một lượt radix chính là bộ lọc này cộng
            thêm một danh sách đầu ra thứ hai cho các phần tử không qua, cũng theo thứ tự đầu vào (mục{' '}
            <a href={href('ch14', '14.4')}>14.4</a>). Quay lại <a href={href('ch12', '12.5')}>mục 12.5</a>.
          </p>
        }
      >
        <p>
          The exclusive scan of the flags is <code>[0, 0, 1, 2, 2]</code>: element 4 has two kept elements before it, so
          it takes slot 2, and the kept elements stay in input order. A radix pass is this filter with a second output
          list for the elements that fail, also in input order (section <a href={href('ch14', '14.4')}>14.4</a>). Back
          to <a href={href('ch12', '12.5')}>section 12.5</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 12 §12.5, pp. 297–298',
  },
  'ch14/start/q-stride-sectors': {
    kind: 'numeric',
    id: 'ch14/start/q-stride-sectors',
    prompt: (
      <Translated
        vi={
          <>
            32 lane của một warp ghi các giá trị 4 byte. Lane <i>k</i> ghi phần tử <code>2k</code> của một mảng có phần
            tử 0 nằm trên ranh giới 32 byte. Lệnh ghi chạm vào bao nhiêu sector 32 byte?
          </>
        }
      >
        The 32 lanes of a warp store 4-byte values. Lane <i>k</i> writes element <code>2k</code> of an array whose
        element 0 sits on a 32-byte boundary. How many 32-byte sectors does the store touch?
      </Translated>
    ),
    answer: 8,
    unit: 'sectors',
    explain: (
      <Translated
        vi={
          <p>
            Lane 31 ghi phần tử 62, các byte 248 đến 251, nên lệnh ghi trải từ byte 0 đến 251: tám sector 32 byte đã căn
            chỉnh, gấp đôi bốn sector mà 32 giá trị liền nhau cần, và sector nào cũng phí một nửa. NVIDIA phục vụ bộ nhớ
            toàn cục (global memory) theo các đơn vị 32 byte này. Mục <a href={href('ch14', '14.6')}>14.6</a> tính chi
            phí của các lần ghi rải rác trong một lượt radix theo cùng cách. Quay lại{' '}
            <a href={href('ch06', '6.1')}>mục 6.1</a>.
          </p>
        }
      >
        <p>
          Lane 31 writes element 62, bytes 248 to 251, so the store spans bytes 0 to 251: eight aligned 32-byte sectors,
          twice the four that 32 contiguous values need, and every sector is half wasted. NVIDIA serves global memory in
          these 32-byte units. Section <a href={href('ch14', '14.6')}>14.6</a> counts the cost of a radix pass's
          scattered writes the same way. Back to <a href={href('ch06', '6.1')}>section 6.1</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.1, pp. 124–132; the 32-byte sector: notes/today/ch06-facts.md A2',
    vi: { unit: 'sector', source: 'ch. 6 §6.1, pp. 124–132; sector 32 byte: notes/today/ch06-facts.md A2' },
  },
  'ch14/start/q-flagrace': {
    kind: 'choice',
    id: 'ch14/start/q-flagrace',
    prompt: (
      <Translated
        vi={
          <>
            Nhiều luồng của một kernel ghi giá trị 1 vào cùng một cờ trong bộ nhớ toàn cục, bằng lệnh ghi thường và
            không có gì sắp thứ tự chúng. Đây là gì?
          </>
        }
      >
        Many threads of a kernel write the value 1 to the same flag in global memory, with plain stores and nothing to
        order them. What is this?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không phải race: các lần ghi cùng một giá trị không thể xung đột, nên cờ đằng nào cũng thành 1.">
            Not a race: writes that all store the same value cannot conflict, so the flag ends up 1 either way.
          </Translated>
        ),
        why: (
          <Translated vi="Lũy đẳng không đồng nghĩa với không có race. Mô hình bộ nhớ định nghĩa race theo các lần truy cập, không theo giá trị.">
            Idempotent is not the same as race-free. The memory model defines a race by the accesses, not by the values.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ là vấn đề nếu các luồng ghi giá trị khác nhau; các lần ghi bằng nhau là lũy đẳng, nên kết quả được xác định.">
            Only a problem if the threads wrote different values; equal stores are idempotent, so the result is defined.
          </Translated>
        ),
        why: (
          <Translated vi="Với giá trị khác nhau, kết quả vừa không đoán trước được vừa không xác định. Với giá trị bằng nhau, nó vẫn không xác định.">
            With different values the result would be unpredictable as well as undefined. With equal values it is still
            undefined.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không xác định trên GPU AMD, nhưng mô hình bộ nhớ của NVIDIA coi các lần ghi cùng giá trị là vô hại.">
            Undefined on AMD GPUs, but NVIDIA's memory model treats equal-value stores as benign.
          </Translated>
        ),
        why: (
          <Translated vi="Quy tắc này là của chính NVIDIA: mọi data race trong mô hình bộ nhớ CUDA đều là hành vi không xác định.">
            The rule is NVIDIA's own: any data race in the CUDA memory model is undefined behaviour.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một race condition (tranh chấp dữ liệu): hành vi không xác định trong mô hình bộ nhớ CUDA dù mọi lần ghi đều ghi 1. Hãy dùng lệnh ghi nguyên tử.">
            A data race: undefined behaviour in the CUDA memory model although every write stores 1. Make the store
            atomic.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Sách gọi race này là vô hại trong thực tế, rồi cảnh báo rằng mô hình bộ nhớ C++ coi data race là không xác
            định ngay cả khi phép toán là lũy đẳng, và khuyên dùng phép toán nguyên tử (p. 333). Hướng dẫn hiện tại của
            NVIDIA nói điều tương tự về CUDA. Cờ <code>hasChanged</code> của odd-even sort chính là trường hợp này (mục{' '}
            <a href={href('ch14', '14.2')}>14.2</a>). Quay lại <a href={href('ch10', '10.3')}>mục 10.3</a>, nơi mô hình
            bộ nhớ xuất hiện lần đầu.
          </p>
        }
      >
        <p>
          The book calls this race benign in practice, then warns that the C++ memory model makes a data race undefined
          even when the operation is idempotent, and advises an atomic (p. 333). NVIDIA's current guide says the same of
          CUDA. Odd-even sort's <code>hasChanged</code> flag is this case (section{' '}
          <a href={href('ch14', '14.2')}>14.2</a>). Back to <a href={href('ch10', '10.3')}>section 10.3</a>, where the
          memory model first comes up.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.2, p. 333; ch. 10 §10.3, p. 230; memory model: notes/today/ch10-facts.md A1, ch14-facts A9',
    vi: {
      source: 'ch. 14 §14.2, p. 333; ch. 10 §10.3, p. 230; mô hình bộ nhớ: notes/today/ch10-facts.md A1, ch14-facts A9',
    },
  },
  'ch14/start/q-gridscan': {
    kind: 'choice',
    id: 'ch14/start/q-gridscan',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel được khởi chạy theo cách thông thường muốn chạy một phép scan trên toàn lưới ngay giữa thân, với
            mọi luồng chờ kết quả rồi mới đi tiếp. Tại sao nó không làm được?
          </>
        }
      >
        A kernel launched the ordinary way wants to run a scan over the whole grid in its middle, with every thread
        waiting for the result before it goes on. Why can it not?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Bộ nhớ chia sẻ là riêng của một khối, nên không khối nào thấy được các bit mà các khối khác đã trích ra.">
            Shared memory is private to a block, so no block can see the bits the other blocks extracted.
          </Translated>
        ),
        why: (
          <Translated vi="Bộ nhớ chia sẻ đúng là riêng, nhưng các bit có thể đi qua bộ nhớ toàn cục, nơi mọi khối đều thấy. Trở ngại là việc chờ những khối có thể chưa chạy.">
            Shared memory is private, but the bits can go through global memory, which every block sees. The obstacle
            is waiting for blocks that may not be running.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các khối không nhất thiết đều thường trú hay chạy theo thứ tự nào, nên một khối chờ các khối còn lại có thể chờ mãi.">
            Blocks need not all be resident or run in any order, so a block waiting for the rest may wait forever.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Được chứ: __syncthreads() chờ mọi luồng của lần khởi chạy, nên phép scan chỉ chậm hơn, không bao giờ sai.">
            It could: __syncthreads() waits for every thread of the launch, so the scan would only be slower, never
            wrong.
          </Translated>
        ),
        why: (
          <Translated vi="__syncthreads() chỉ trải trên một khối, và một kiểu chờ tự chế giữa các khối có thể không bao giờ kết thúc. Tốc độ không phải là vấn đề.">
            __syncthreads() spans one block only, and a hand-made wait across blocks may never end. Speed is not the
            issue.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Rào chắn (barrier) chỉ trải trên một khối, và môi trường chạy có thể chạy các khối theo bất kỳ thứ tự nào,
            mỗi lúc vài khối. Chương 11 vượt qua ranh giới khối bằng các lần khởi chạy riêng, hoặc bằng một lần khởi
            chạy cooperative giữ mọi khối thường trú (trên CUDA và HIP), hoặc bằng các cờ chỉ chờ những khối trước đó.
            Kernel radix của sách gọi một phép scan toàn lưới ngay giữa thân; mục{' '}
            <a href={href('ch14', '14.5')}>14.5</a> chỉ ra điều gì phải thay đổi. Quay lại{' '}
            <a href={href('ch04', '4.3')}>mục 4.3</a> và <a href={href('ch11', '11.9')}>mục 11.9</a>.
          </p>
        }
      >
        <p>
          A barrier only spans a block, and the run time may run blocks in any order, a few at a time. Chapter 11 crosses
          blocks either with separate launches, or with a cooperative launch that keeps every block resident (on CUDA
          and HIP), or with flags that only wait for earlier blocks. The book's radix kernel calls a grid-wide scan in
          its middle; section <a href={href('ch14', '14.5')}>14.5</a> shows what has to change. Back to{' '}
          <a href={href('ch04', '4.3')}>section 4.3</a> and <a href={href('ch11', '11.9')}>section 11.9</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.3, pp. 70–74; ch. 11 §11.9, pp. 273–277', // i18n-ignore: citation only
  },
  'ch14/start/q-corank': {
    kind: 'choice',
    id: 'ch14/start/q-corank',
    prompt: (
      <Translated
        vi={
          <>
            Một luồng phải ghi các đầu ra từ <code>k</code> đến <code>k + 7</code> của phép trộn hai danh sách đã sắp
            xếp, mà không trộn gì đứng trước <code>k</code>. Nó cần gì trước tiên?
          </>
        }
      >
        A thread must write outputs <code>k</code> to <code>k + 7</code> of the merge of two sorted lists, without
        merging anything before <code>k</code>. What does it need first?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Giá trị của đầu ra k − 1, để nó tiếp tục trộn từ giá trị đó trong cả hai danh sách.">
            The value of output k − 1, so that it can carry on merging from that value in both lists.
          </Translated>
        ),
        why: (
          <Translated vi="Biết một giá trị không cho biết phải bắt đầu từ đâu trong mỗi đầu vào. Nó còn cần cả phép trộn của mọi thứ trước nó.">
            Knowing a value does not say where in each input to start. It also needs the merge of everything before it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Phần tử k của mỗi danh sách đầu vào, vì k đầu ra đầu tiên là k phần tử nhỏ nhất của cả hai danh sách gộp lại.">
            Element k of each input list, since the first k outputs are the k smallest of the two lists together.
          </Translated>
        ),
        why: (
          <Translated vi="k đầu ra đầu tiên có thể đến từ hai danh sách theo bất kỳ tỉ lệ nào; lấy k từ mỗi bên chỉ đúng một cách tình cờ.">
            The first k outputs can come from the two lists in any proportion; k of each is right only by accident.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Cặp co-rank (i, j), i + j = k: mỗi danh sách cung cấp bao nhiêu trong k đầu ra đầu tiên, tìm bằng tìm kiếm nhị phân.">
            The co-rank pair (i, j), i + j = k: how many of the first k outputs each list supplies, by binary search.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hàm co-rank biến một vị trí đầu ra thành hai vị trí đầu vào nuôi nó; sau đó mỗi luồng chạy một phép trộn
            tuần tự thông thường trên các lát của mình. Merge sort lặp lại phép trộn của chương 13 ở mọi tầng của cây
            (mục <a href={href('ch14', '14.3')}>14.3</a>). Quay lại <a href={href('ch13', '13.3')}>mục 13.3</a>.
          </p>
        }
      >
        <p>
          The co-rank function turns an output position into the two input positions that feed it; after that each
          thread runs an ordinary sequential merge on its slices. Merge sort repeats chapter 13's merge at every level of
          its tree (section <a href={href('ch14', '14.3')}>14.3</a>). Back to{' '}
          <a href={href('ch13', '13.3')}>section 13.3</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 13 §13.3–13.4, pp. 306–313',
  },
};

const SECTIONS: { id: string; readonly what: string }[] = [
  {
    id: '14.1',
    get what() {
      return tr(
        'what a sort must deliver, stable and unstable sorts, comparison and non-comparison sorts',
        'một phép sắp xếp phải mang lại gì, sắp xếp ổn định và không ổn định, sắp xếp có so sánh và không so sánh',
      );
    },
  },
  {
    id: '14.2',
    get what() {
      return tr(
        'odd-even sort: neighbouring pairs in parallel, alternating; simple and slow',
        'odd-even sort: các cặp kề nhau chạy song song, luân phiên; đơn giản và chậm',
      );
    },
  },
  {
    id: '14.3',
    get what() {
      return tr(
        "merge sort: chapter 13's merge, run at every level of a tree",
        'merge sort: phép trộn của chương 13, chạy ở mọi tầng của một cây',
      );
    },
  },
  {
    id: '14.4',
    get what() {
      return tr(
        'radix sort one digit at a time, and why every pass must be stable; float logits as keys (ours)',
        'radix sort từng chữ số một, và vì sao mọi lượt phải ổn định; logits dạng float làm khóa (của khóa học này)',
      );
    },
  },
  {
    id: '14.5',
    get what() {
      return tr(
        'one radix pass in parallel: an exclusive scan of the bits, then a scatter',
        'một lượt radix song song: scan loại trừ trên các bit, rồi một lần phân tán',
      );
    },
  },
  {
    id: '14.6',
    get what() {
      return tr(
        'sort locally in shared memory first, so the writes to global memory coalesce',
        'sắp xếp cục bộ trong bộ nhớ chia sẻ trước, để các lần ghi ra bộ nhớ toàn cục được hợp nhất',
      );
    },
  },
  {
    id: '14.7',
    get what() {
      return tr(
        'wider digits: fewer passes, worse coalescing, a bigger scan',
        'chữ số rộng hơn: ít lượt hơn, hợp nhất truy cập kém hơn, phép scan lớn hơn',
      );
    },
  },
  {
    id: '14.8',
    get what() {
      return tr(
        'several keys per thread, for bigger local buckets and a smaller scan',
        'nhiều khóa cho mỗi luồng, để có các bucket cục bộ lớn hơn và phép scan nhỏ hơn',
      );
    },
  },
  {
    id: '14.9',
    get what() {
      return tr(
        'sorting networks, sample sort, and most-significant-digit radix sort',
        'mạng sắp xếp, sample sort, và radix sort bắt đầu từ chữ số có nghĩa lớn nhất',
      );
    },
  },
];

export function S140() {
  const q = questionsS140;
  return (
    <>
      <Translated
        vi={
          <p>
            Sắp xếp đưa một danh sách vào thứ tự. Lý do sách quan tâm mang tính tổng quát: dữ liệu có thứ tự thì rẻ để
            tìm kiếm, rẻ để so sánh với dữ liệu có thứ tự khác, và nhiều thuật toán chạy nhanh hơn trên nó. Các phép sắp
            xếp hiệu quả cũng khó song song hóa, nên chúng là một ca nghiên cứu tốt cho các kỹ thuật của vài chương vừa
            qua.
          </p>
        }
      >
        <p>
          Sorting puts a list in order. The book's reason to care is general: ordered data is cheap to search, cheap to
          compare with other ordered data, and many algorithms run faster on it. Efficient sorts are also hard to
          parallelise, which makes them a good case study in the techniques of the last few chapters.
        </p>
      </Translated>
      <Source pages="329" />

      <h2>
        <Translated vi="Nó nằm ở đâu bên dưới một bộ lấy mẫu">Where it sits under a sampler</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Có lẽ bạn đã chạy một phép sắp xếp trên GPU mà không phải tự viết. Một bộ lấy mẫu top-p (nucleus) giữ tập
            nhỏ nhất gồm các token có khả năng cao nhất mà tổng xác suất đạt ít nhất p. Cách thường dùng để tìm tập đó
            là xếp từ vựng theo xác suất, lấy tổng tích lũy, rồi cắt ở chỗ tổng vượt p: một phép sắp xếp, rồi phép scan
            của chương 11.
          </p>
        }
      >
        <p>
          You have probably run a GPU sort without writing one. A top-p (nucleus) sampler keeps the smallest set of most
          likely tokens whose probabilities add up to at least p. The usual way to find that set orders the vocabulary
          by probability, takes a cumulative sum, and cuts where the sum crosses p: a sort, then chapter 11's scan.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch14-facts.md, D3, D7)', '2026-11-20 (notes/today/ch14-facts.md, mục D3, D7)')}>
        <Translated
          vi={
            <p>
              <code>TopPLogitsWarper</code> của Hugging Face: <code>torch.sort</code>, softmax, <code>cumsum</code>.
              Top-p backend thử nghiệm của llama.cpp: <code>ggml_argsort</code> giảm dần, <code>ggml_soft_max</code>,{' '}
              <code>ggml_cumsum</code>.
            </p>
          }
        >
          <p>
            Hugging Face's <code>TopPLogitsWarper</code>: <code>torch.sort</code>, softmax, <code>cumsum</code>.
            llama.cpp's experimental backend top-p: <code>ggml_argsort</code> descending, <code>ggml_soft_max</code>,{' '}
            <code>ggml_cumsum</code>.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Hai điều cần lưu ý. Top-k không cần sắp xếp toàn bộ: tìm k phần tử lớn nhất là một phép chọn, đó là lý do
            PyTorch có <code>torch.topk</code> và CUB nay có <code>cub::DeviceTopK</code>; module X1 xây một cái trong{' '}
            <a href={href('x1', 'X1.4')}>X1.4, "Top-k là một phép chọn, không phải sắp xếp"</a>. Và top-p trên một
            danh sách đã sắp xếp là cách diễn đạt phổ biến, không phải duy nhất: nhánh Triton mặc định của vLLM tìm điểm
            cắt bằng một phép tìm pivot và không bao giờ sắp xếp.
          </p>
        }
      >
        <p>
          Two qualifications. Top-k needs no full sort: finding the k largest is a selection, which is why PyTorch has{' '}
          <code>torch.topk</code> and CUB now has <code>cub::DeviceTopK</code>; module X1 builds one in{' '}
          <a href={href('x1', 'X1.4')}>X1.4, "Top-k is a selection, not a sort"</a>. And top-p over
          a sorted list is the common formulation, not the only one: vLLM's default Triton path finds the cut with a pivot
          search and never sorts.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách không nhắc đến lấy mẫu trong chương này; cây cầu này là của khóa học này, từ các nguồn dự án trong
              ch14-facts A7 và D7.
            </p>
          }
        >
          <p>
            The book does not mention sampling in this chapter; this bridge is ours, from the project sources in
            ch14-facts A7 and D7.
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Chương này xây dựng gì">What the chapter builds</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Ba phép sắp xếp, với tỉ trọng không đều. Odd-even sort ngắn vì nó đơn giản. Merge sort ngắn vì phần khó của
            nó là phép trộn song song của chương 13. Radix sort chiếm phần còn lại của chương: nó không so sánh khóa nào
            cả, và các tối ưu của nó là dạng tổng quát hơn của bộ lọc ổn định ở chương 12.
          </p>
        }
      >
        <p>
          Three sorts, weighted unevenly. Odd-even sort is brief because it is simple. Merge sort is brief because its
          hard part is chapter 13's parallel merge. Radix sort takes the rest of the chapter: it compares no keys at
          all, and its optimisations are a more general form of chapter 12's stable filter.
        </p>
      </Translated>
      <Source pages="329" />
      <table className="data" data-part="sections">
        <thead>
          <tr>
            <th>
              <Translated vi="Mục">Section</Translated>
            </th>
            <th>
              <Translated vi="Bạn làm gì ở đó">What you do there</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          {SECTIONS.map((s) => (
            <tr key={s.id}>
              <td className="nowrap">
                <a href={href('ch14', s.id)}>{s.id}</a>
              </td>
              <td>{s.what}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Kernel mà chương này mổ xẻ là Hình 14.7 của nó: một lượt của radix sort, tức là một phép scan loại trừ trên
            một bit của mỗi khóa, tiếp theo là một lần phân tán mỗi khóa về ô riêng của nó. Các mục 14.6 đến 14.8 sau đó
            làm cho các lần ghi của nó được hợp nhất.
          </p>
        }
      >
        <p>
          The kernel the chapter takes apart is its Fig. 14.7: one pass of a radix sort, which is an exclusive scan of
          one bit per key followed by a scatter of every key to its own slot. Sections 14.6 to 14.8 then make its writes
          coalesce.
        </p>
      </Translated>
      <Source section="14.5" pages="339" figs={tr('Fig. 14.7', 'Hình 14.7')} />
      <Translated
        vi={
          <p>
            Phần của dự án là các ứng viên đã sắp xếp của mốc dự án M5: lab 14.3 sắp xếp mọi hàng của một batch xác suất
            theo thứ tự giảm dần, một cách ổn định, kèm theo các id token. <a href={href('ch14', 'labs')}>Trang lab</a>{' '}
            có nó và hai lab nhỏ hơn dẫn tới nó.
          </p>
        }
      >
        <p>
          The project piece is milestone M5's sorted candidates: lab 14.3 sorts every row of a batch of probabilities
          in descending order, stably, with the token ids carried along. The <a href={href('ch14', 'labs')}>labs page</a>{' '}
          has it and two smaller labs that lead up to it.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch14-facts.md, A1)', '2026-11-20 (notes/today/ch14-facts.md, mục A1)')}>
        <Translated
          vi={
            <p>
              Sách kết thúc bằng lời khuyên dùng Thrust hoặc CUB thay vì tự viết phép sắp xếp (p. 347), và lời khuyên
              đó vẫn đúng. Radix sort của CUB ngày nay là thuật toán OneSweep mà sách dẫn ở p. 336: mặc định cho khóa
              từ 4 byte trở lên kể từ compute capability 6.0, và cho mọi kích thước khóa kể từ 9.0.
            </p>
          }
        >
          <p>
            The book ends by advising Thrust or CUB over a sort of your own (p. 347), and that stands. CUB's radix sort
            today is the OneSweep algorithm the book cites on p. 336: its default for keys of 4 bytes or more from
            compute capability 6.0, and for every key size from 9.0.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Trước khi bắt đầu">Before you start</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sáu câu hỏi về những gì chương này giả định từ chương 4, 6 và 10 đến 13. Mỗi lời giải thích nêu mục nào của
            chương này dựa vào nó, và mỗi câu hỏi sẽ quay lại trong phần <a href={href('review')}>ôn tập</a>. Thứ tự
            của khóa học này (mục <a href={href('ch01', '1.7')}>1.7</a>) đọc chương 13 sau chương này, nên nếu bạn theo
            nó, hãy coi câu hỏi co-rank như một bản xem trước: mục 14.3 chỉ dùng kết quả của phép tìm kiếm, còn{' '}
            <a href={href('ch13', '13.3')}>mục 13.3</a> giải nó ra.
          </p>
        }
      >
        <p>
          Six questions on what this chapter assumes from chapters 4, 6 and 10 to 13. Each explanation names the section
          of this chapter that relies on it, and each question returns in <a href={href('review')}>review</a>. This
          course's order (section <a href={href('ch01', '1.7')}>1.7</a>) reads chapter 13 after this one, so if you follow
          it, take the co-rank question as a preview: section 14.3 uses only the search's result, and{' '}
          <a href={href('ch13', '13.3')}>section 13.3</a> works it out.
        </p>
      </Translated>
      <Ask q={q['ch14/start/q-exscan']} label={tr('From chapter 11', 'Từ chương 11')} />
      <Ask q={q['ch14/start/q-filterpos']} label={tr('From chapter 12', 'Từ chương 12')} />
      <Ask q={q['ch14/start/q-stride-sectors']} label={tr('From chapter 6', 'Từ chương 6')} />
      <Ask q={q['ch14/start/q-flagrace']} label={tr('From chapter 10', 'Từ chương 10')} />
      <Ask q={q['ch14/start/q-gridscan']} label={tr('From chapters 4 and 11', 'Từ chương 4 và 11')} />
      <Ask q={q['ch14/start/q-corank']} label={tr('From chapter 13', 'Từ chương 13')} />
    </>
  );
}
