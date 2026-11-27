import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// Before you start. Prerequisite questions from the notes of chapters 4, 6, 9, 10 and 11;
// vendor facts from notes/today/ch12-facts.md (D3, D4, D6) and ch05-facts.md (B5).
export const questionsS120: Record<string, Question> = {
  'ch12/start/q-lane64': {
    kind: 'numeric',
    id: 'ch12/start/q-lane64',
    prompt: (
      <Translated vi={<>Một khối một chiều chạy trên GPU AMD có warp (AMD gọi là wavefront) rộng 64 lane. Luồng 100 của khối nằm ở lane nào?</>}>
        A one-dimensional block runs on an AMD GPU whose warps (AMD says wavefronts) are 64 lanes wide. Thread 100 of
        the block sits in which lane?
      </Translated>
    ),
    answer: 36,
    unit: 'lane',
    explain: (
      <Translated
        vi={
          <>
            <p>
              Warp được cắt từ các chỉ số luồng liên tiếp, nên 100 = 64 + 36: warp 1, lane 36. Trên warp rộng 32, cùng luồng
              đó là lane 4 của warp 3 (100 = 96 + 4).
            </p>
            <p>
              Trong một lần bỏ phiếu của warp, lane 36 sở hữu bit 36 của mặt nạ. Một <code>unsigned int</code> không có bit 36, và{' '}
              <code>1 &lt;&lt; 36</code> trên số nguyên 32 bit là hành vi không xác định. Đó là lý do mã warp của chương này cho
              mặt nạ một kiểu riêng, 64 bit trên HIP, và đếm nó bằng các intrinsic 64 bit. Xem{' '}
              <a href={href('ch04', '4.4')}>mục 4.4</a> để biết một khối được cắt thành các warp ra sao.
            </p>
          </>
        }
      >
        <>
          <p>
            Warps are cut from consecutive thread indices, so 100 = 64 + 36: warp 1, lane 36. On a 32-wide warp the same
            thread is lane 4 of warp 3 (100 = 96 + 4).
          </p>
          <p>
            In a warp vote, lane 36 owns bit 36 of the mask. An <code>unsigned int</code> has no bit 36, and{' '}
            <code>1 &lt;&lt; 36</code> on a 32-bit integer is undefined. That is why this chapter's warp code gives the
            mask its own type, 64 bits on HIP, and counts it with the 64-bit intrinsics. See{' '}
            <a href={href('ch04', '4.4')}>section 4.4</a> for how a block is cut into warps.
          </p>
        </>
      </Translated>
    ),
    source: 'ch. 4 §4.4, p. 75; ch. 10 §10.7, p. 239 (laneIdx); AMD widths: notes/today/ch05-facts.md B5; 64-bit masks: ch12-facts B1',
    vi: {
      unit: 'lane',
      source: 'ch. 4 §4.4, p. 75; ch. 10 §10.7, p. 239 (laneIdx); độ rộng của AMD: notes/today/ch05-facts.md B5; mặt nạ 64 bit: ch12-facts B1',
    },
  },
  'ch12/start/q-onecounter': {
    kind: 'choice',
    id: 'ch12/start/q-onecounter',
    prompt: (
      <Translated vi={<>100,000 luồng, mỗi luồng cộng nguyên tử thêm 1. Ở phiên bản A tất cả cộng vào một bộ đếm; ở phiên bản B mỗi luồng cộng vào bộ đếm riêng của mình. Phiên bản nào chạy lâu hơn, và vì sao?</>}>
        100,000 threads each atomically add 1. In version A they all add to one counter; in version B each thread adds
        to a counter of its own. Which version takes longer, and why?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không phiên bản nào: một atomic là một lệnh đọc-sửa-ghi duy nhất, nên 100,000 atomic tốn cùng thời gian bất kể chúng nhắm tới địa chỉ nào.">
            Neither: an atomic is a single read-modify-write instruction, so 100,000 of them take the same time whatever addresses they name.
          </Translated>
        ),
        why: (
          <Translated vi="Phần cứng vẫn chỉ cho phép một atomic trên một địa chỉ tại một thời điểm. Những atomic khác xếp hàng phía sau.">
            The hardware still allows only one atomic on a given address at a time. The others queue behind it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="A. Các atomic trên một địa chỉ bị tuần tự hóa, mỗi cái tốn khoảng một độ trễ đọc cộng một độ trễ ghi; trên các địa chỉ khác nhau chúng chồng lên nhau.">
            A. Atomics on one address are serialised, each taking about a load plus a store latency; on different addresses they overlap.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="B: 100,000 bộ đếm nghĩa là 100,000 vị trí riêng phải nạp và ghi lại, trong khi A cứ dùng lại một vị trí.">
            B: 100,000 counters means 100,000 separate locations to fetch and write back, where A keeps reusing one.
          </Translated>
        ),
        why: (
          <Translated vi="Thứ giới hạn A không phải lưu lượng bộ nhớ. Mà là hàng đợi trên một địa chỉ.">
            Memory traffic is not what limits A. The queue on one address is.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Tốc độ atomic trên một vị trí là khoảng một cái cho mỗi (độ trễ đọc + độ trễ ghi). Bộ lọc của mục 12.2 chính là
            phiên bản A: mọi khóa được giữ đều tăng cùng một bộ đếm đầu ra, và phần còn lại của nửa không ổn định trong chương
            này giảm số lần bộ đếm đó bị chạm vào. Quay lại <a href={href('ch09', '9.3')}>mục 9.3</a>.
          </p>
        }
      >
        <p>
          The rate of atomics on one location is about one per (load latency + store latency). Section 12.2's filter is
          version A: every kept key bumps the same output counter, and the rest of the unstable half of this chapter
          reduces how many times that counter is touched. Back to <a href={href('ch09', '9.3')}>section 9.3</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.3, pp. 209–210',
  },
  'ch12/start/q-exclusive': {
    kind: 'numeric',
    id: 'ch12/start/q-exclusive',
    prompt: (
      <Translated vi={<>Lấy scan loại trừ (với +) của <code>[0, 1, 1, 0, 1, 0, 0, 1]</code>. Giá trị tại chỉ số 7 là bao nhiêu?</>}>
        Take the exclusive scan (with +) of <code>[0, 1, 1, 0, 1, 0, 0, 1]</code>. What is the value at index 7?
      </Translated>
    ),
    answer: 3,
    explain: (
      <Translated
        vi={
          <p>
            Loại trừ nghĩa là "tổng của mọi thứ đứng trước tôi": 0 + 1 + 1 + 0 + 1 + 0 + 0 = 3. Khi đầu vào là các cờ giữ lại
            0/1, tổng đó là số phần tử được giữ đứng trước phần tử này, chính là ô mà phần tử này chiếm trong một danh sách nén
            gọn giữ nguyên thứ tự ban đầu. Bộ lọc ổn định của mục 12.5 được xây trên điều đó. Quay lại{' '}
            <a href={href('ch11', '11.1')}>mục 11.1</a>.
          </p>
        }
      >
        <p>
          Exclusive means "the sum of everything before me": 0 + 1 + 1 + 0 + 1 + 0 + 0 = 3. When the inputs are 0/1 keep
          flags, that sum is the number of kept items before this one, which is exactly the slot this item takes in a
          compacted list that keeps the original order. The stable filter of section 12.5 is built on it. Back to{' '}
          <a href={href('ch11', '11.1')}>section 11.1</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.1, p. 253',
  },
  'ch12/start/q-privatise': {
    kind: 'choice',
    id: 'ch12/start/q-privatise',
    prompt: (
      <Translated vi={<>Cấp cho mỗi khối một bản sao riêng của một đầu ra dùng chung mang lại gì, và tốn gì?</>}>
        <>What does giving each block a private copy of a shared output buy, and what does it cost?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Nó bỏ được nhu cầu dùng atomic: mỗi khối sở hữu bản sao của mình, nên các luồng của nó có thể cập nhật bằng các lệnh đọc và ghi thường.">
            It removes the need for atomics: each block owns its copy, so its threads can update it with plain loads and stores.
          </Translated>
        ),
        why: (
          <Translated vi="Các luồng của một khối vẫn dùng chung bản sao riêng, nên cập nhật của chúng vẫn cần atomic: loại rẻ hơn, trong bộ nhớ chia sẻ.">
            The threads of one block still share the private copy, so their updates still need atomics: cheaper ones, in shared memory.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Tăng tốc trên mọi đầu vào, vì cập nhật trong bộ nhớ chia sẻ rẻ hơn trong bộ nhớ toàn cục và bước gộp cuối chỉ là một lượt.">
            A speedup on every input, since shared-memory updates are cheaper than global ones and the final merge is one pass.
          </Translated>
        ),
        why: (
          <Translated vi="Khởi tạo và gộp các bản sao tốn một chi phí cho mỗi khối. Khi tranh chấp vốn đã thấp, chi phí đó có thể lớn hơn phần lợi.">
            Initialising and merging the copies costs something per block. When contention was low to begin with, that cost can outweigh the gain.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Ít cập nhật vào bản sao chung hơn rất nhiều, trả bằng bộ nhớ trên chip cho mỗi khối và một bước gộp ở cuối.">
            Far fewer updates to the public copy, paid for with on-chip memory per block and a merge step at the end.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mục 12.4 cấp cho mỗi khối một danh sách đầu ra riêng và một bộ đếm riêng trong bộ nhớ chia sẻ. Sau đó mỗi khối có
            một luồng giành chỗ trong danh sách chung bằng một atomic duy nhất. Quay lại{' '}
            <a href={href('ch09', '9.4')}>mục 9.4</a>.
          </p>
        }
      >
        <p>
          Section 12.4 gives each block a private output list and a private counter in shared memory. One thread per
          block then reserves space in the public list with a single atomic. Back to{' '}
          <a href={href('ch09', '9.4')}>section 9.4</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.4, pp. 211–214',
  },
  'ch12/start/q-coalesce': {
    kind: 'choice',
    id: 'ch12/start/q-coalesce',
    prompt: (
      <Translated
        vi={
          <>
            32 lane của một warp ghi các giá trị 4 byte. Trong trường hợp A, lane <i>i</i> ghi <code>out[base + i]</code>;
            trong trường hợp B, nó ghi <code>out[base + 32*i]</code>. Trường hợp nào được phục vụ với ít giao dịch bộ nhớ
            hơn, và vì sao?
          </>
        }
      >
        The 32 lanes of a warp store 4-byte values. In case A lane <i>i</i> writes <code>out[base + i]</code>; in case
        B it writes <code>out[base + 32*i]</code>. Which case is served with fewer memory transactions, and why?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="A: 32 địa chỉ nằm liền nhau và rơi vào một vài đoạn được căn chỉnh. B đặt mỗi lane vào một đoạn khác nhau.">
            A: the 32 addresses are adjacent and fall into a few aligned segments. B puts every lane in a different segment.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="B: các lần ghi cách nhau 128 byte rơi vào các bank bộ nhớ khác nhau, nên cả 32 lane được phục vụ cùng lúc.">
            B: stores 128 bytes apart fall in different memory banks, so all 32 lanes are served at once.
          </Translated>
        ),
        why: (
          <Translated vi="Bộ nhớ toàn cục được phục vụ theo các đoạn gồm những byte liền kề. B dùng 4 byte của mỗi đoạn nó chạm vào và trả giá cho toàn bộ đoạn.">
            Global memory is served in segments of adjacent bytes. B uses 4 bytes of each segment it touches and pays for all of them.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không trường hợp nào: dù cách nào mỗi lane cũng ghi 4 byte, nên cả hai di chuyển cùng 128 byte với cùng số giao dịch.">
            Neither: each lane stores 4 bytes either way, so both cases move the same 128 bytes in the same number of transactions.
          </Translated>
        ),
        why: (
          <Translated vi="Số byte được ghi là như nhau. Số đoạn bị chạm thì không.">
            The bytes stored are the same. The number of segments touched is not.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            NVIDIA phục vụ bộ nhớ toàn cục theo các giao dịch 32 byte: trường hợp A cần bốn giao dịch nếu <code>base</code> được
            căn chỉnh, trường hợp B cần 32. Vì vậy, các khóa được giữ rơi vào đâu là một câu hỏi về hiệu năng trong chương này:
            các mục 12.3, 12.4 và 12.6 đều sắp xếp để các luồng kề nhau ghi vào các ô kề nhau. Quay lại{' '}
            <a href={href('ch06', '6.1')}>mục 6.1</a>.
          </p>
        }
      >
        <p>
          NVIDIA serves global memory in 32-byte transactions: case A needs four of them if <code>base</code> is aligned,
          case B needs 32. Where kept keys land is therefore a performance question in this chapter: sections 12.3, 12.4
          and 12.6 all arrange for neighbouring threads to write neighbouring slots. Back to{' '}
          <a href={href('ch06', '6.1')}>section 6.1</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.1, pp. 124–132; 32-byte transactions: notes/today/ch06-facts.md A2',
    vi: { source: 'ch. 6 §6.1, pp. 124–132; giao dịch 32 byte: notes/today/ch06-facts.md A2' },
  },
  'ch12/start/q-shflmask': {
    kind: 'choice',
    id: 'ch12/start/q-shflmask',
    prompt: (
      <Translated
        vi={
          <>
            Bên trong một nhánh mà chỉ một số lane của warp đi vào, một kernel CUDA gọi{' '}
            <code>__shfl_sync(0xffffffff, v, src)</code>. Điều gì sai?
          </>
        }
      >
        Inside a branch that only some lanes of a warp take, a CUDA kernel calls{' '}
        <code>__shfl_sync(0xffffffff, v, src)</code>. What is wrong?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không có gì: mặt nạ chỉ gợi ý những lane nào sẽ tham gia, và phần cứng bỏ qua lane nào vắng mặt.">
            Nothing: the mask only hints at which lanes will take part, and the hardware skips any lane that is absent.
          </Translated>
        ),
        why: (
          <Translated vi="Nó là một giao kèo. Mọi lane có tên trong đó đều phải thực hiện lời gọi.">
            It is a contract. Every lane named in it must make the call.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ là các lane ngoài nhánh nhận 0, điều này vô hại vì chúng không bao giờ đọc kết quả.">
            Only that the lanes outside the branch receive 0, which is harmless since they never read the result.
          </Translated>
        ),
        why: (
          <Translated vi="Lời gọi này hoàn toàn không có kết quả xác định. Hướng dẫn lập trình xếp nó vào loại không hợp lệ.">
            No result is defined for this call at all. The programming guide classes it as invalid.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Mặt nạ nêu tên những lane không bao giờ thực hiện lời gọi, mà CUDA yêu cầu mọi lane trong mặt nạ phải thực hiện nó với cùng mặt nạ: lời gọi không hợp lệ và có thể bị treo.">
            The mask names lanes that never make the call, and CUDA requires every lane in the mask to make it with the same mask: the call is invalid and may hang.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mặt nạ cho biết những lane nào tham gia lệnh shuffle. Ở mục 12.3, kernel của sách shuffle bên trong một nhánh, nên
            nó truyền mặt nạ của những lane thực sự có mặt, chứ không phải <code>0xffffffff</code>. Quay lại{' '}
            <a href={href('ch10', '10.7')}>mục 10.7</a>.
          </p>
        }
      >
        <p>
          The mask says which lanes take part in the shuffle. In section 12.3 the book's kernel shuffles inside a branch,
          so it passes the mask of the lanes that are actually there, not <code>0xffffffff</code>. Back to{' '}
          <a href={href('ch10', '10.7')}>section 10.7</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.7, p. 238; notes/today/ch12-facts.md A4 (CUDA programming guide section 5.4.6.6)',
    vi: { source: 'ch. 10 §10.7, p. 238; notes/today/ch12-facts.md A4 (hướng dẫn lập trình CUDA, mục 5.4.6.6)' },
  },
};

const FAMILIES: { family: string; secs: { id: string; label: string }[]; slot: string; work: string }[] = [
  {
    get family() {
      return tr('Unstable', 'Không ổn định');
    },
    secs: [
      { id: '12.2', label: '12.2' },
      { id: '12.3', label: '12.3' },
      { id: '12.4', label: '12.4' },
    ],
    get slot() {
      return tr('takes the next free slot from one shared counter, with an atomic', 'lấy ô trống kế tiếp từ một bộ đếm dùng chung, bằng một atomic');
    },
    get work() {
      return tr('touch the counter less often: once per kept key, then once per warp, then once per block', 'chạm vào bộ đếm ít hơn: mỗi khóa được giữ một lần, rồi mỗi warp một lần, rồi mỗi khối một lần');
    },
  },
  {
    get family() {
      return tr('Stable', 'Ổn định');
    },
    secs: [
      { id: '12.5', label: '12.5' },
      { id: '12.6', label: '12.6' },
      { id: '12.7', label: '12.7' },
    ],
    get slot() {
      return tr('counts the kept keys before it: an exclusive scan of the keep flags', 'đếm các khóa được giữ đứng trước nó: một scan loại trừ của các cờ giữ lại');
    },
    get work() {
      return tr('the scan inside the filter kernel, then staging through shared memory for coalesced writes, then the whole thing in place', 'phép scan bên trong kernel lọc, rồi gom tạm qua bộ nhớ chia sẻ để các lần ghi được hợp nhất, rồi toàn bộ việc đó tại chỗ');
    },
  },
  {
    get family() {
      return tr('Relatives', 'Họ hàng');
    },
    secs: [{ id: '12.8', label: '12.8' }],
    get slot() {
      return tr('from its index alone, or by a scan when its value decides', 'chỉ từ chỉ số của nó, hoặc bằng một scan khi giá trị của nó quyết định');
    },
    get work() {
      return tr('removing duplicates, removing or adding rows and columns', 'loại bỏ phần tử trùng lặp, xóa hoặc thêm hàng và cột');
    },
  },
];

const LABS: { num: string; title: string; where: string }[] = [
  {
    num: '12.1',
    get title() {
      return tr('The unstable filter three ways', 'Bộ lọc không ổn định theo ba cách');
    },
    where: 'emu, race, Metal, CUDA, HIP', // i18n-ignore: backend names
  },
  {
    num: '12.2',
    get title() {
      return tr('Filter hunt', 'Săn lỗi trong bộ lọc');
    },
    where: 'emu, race, Metal, CUDA, HIP', // i18n-ignore: backend names
  },
  {
    num: '12.3',
    get title() {
      return tr('A stable filter by scan', 'Một bộ lọc ổn định bằng scan');
    },
    where: 'emu, race, Metal, CUDA, HIP', // i18n-ignore: backend names
  },
  {
    num: '12.4',
    get title() {
      return tr('Candidate compaction (milestone M5 piece)', 'Nén gọn ứng viên (phần của mốc dự án M5)');
    },
    where: 'emu, race, Metal, CUDA, HIP', // i18n-ignore: backend names
  },
  {
    num: '12.5',
    get title() {
      return tr('In-place stable filter in a single pass (optional)', 'Bộ lọc ổn định tại chỗ trong một lượt (tùy chọn)');
    },
    get where() { return tr('CUDA family only: emu, race, CUDA, HIP', 'chỉ họ CUDA: emu, race, CUDA, HIP'); },
  },
];

export function S120() {
  const q = questionsS120;
  return (
    <>
      <Translated
        vi={
          <p>
            Chương này dồn các phần tử của một danh sách vượt qua một phép kiểm tra (sách gọi chúng là <b>khóa</b>) vào một danh
            sách riêng không có khoảng trống, trên GPU;{' '}
            <a href={href('ch12', '12.1')}>mục 12.1</a> định nghĩa việc lọc một cách đầy đủ.
          </p>
        }
      >
        <p>
          This chapter packs the items of a list that pass a test (the book calls them <b>keys</b>) into a gapless list
          of their own, on the GPU;{' '}
          <a href={href('ch12', '12.1')}>section 12.1</a> defines filtering properly.
        </p>
      </Translated>
      <Source pages="289" />
      <Translated
        vi={
          <p>
            Phụ đề của chương là "And warp voting" (và bỏ phiếu trong warp). Kỹ thuật mới là cho 32 luồng của một warp (hoặc bao
            nhiêu luồng mà phần cứng của bạn có) bỏ phiếu bằng một mặt nạ bit, để một luồng có thể hành động thay cho tất cả.
            Các công cụ còn lại bạn đã có: atomic và riêng tư hóa (privatization) từ chương 9, và scan từ chương 11.
          </p>
        }
      >
        <p>
          The chapter's subtitle is "And warp voting". The new technique is letting the 32 threads of a warp (or however
          many your hardware has) vote with a bit mask, so that one thread can act for all of them. The other tools are
          ones you already have: atomics and privatisation from chapter 9, and the scan from chapter 11.
        </p>
      </Translated>

      <h2><Translated vi="Nơi bạn đã gặp nó">Where you have already met it</Translated></h2>
      <Translated
        vi={
          <p>
            Một bộ lấy mẫu gặp phép lọc ngay sau ngưỡng của nó. Top-k, top-p và min-p mỗi cái quyết định những token nào của một
            hàng từ vựng còn ở lại cuộc đua, và những token sống sót, được dồn lại cùng id token của chúng, chính là đầu ra của
            một bộ lọc. Khóa học xây bước đó trên GPU qua bốn phần.
          </p>
        }
      >
        <p>
          A sampler meets a filter right after its threshold. Top-k, top-p and min-p each decide which tokens of a
          vocabulary row stay in the running, and the survivors, packed together with their token ids, are a filter's
          output. The course builds that step on the GPU in four pieces.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách không nhắc tới bộ lấy mẫu ở đây: pipeline này là của khóa học này. Ngưỡng min-p là ngưỡng phía host của
              llama.cpp, kiểm tra logits và không cần softmax (ch12-facts D3).
            </p>
          }
        >
          <p>
            The book does not mention samplers here: this pipeline is ours. The min-p threshold is llama.cpp's host
            one, which tests logits and needs no softmax (ch12-facts D3).
          </p>
        </Translated>
      </Beyond>
      <table className="data" data-part="pipeline">
        <thead>
          <tr>
            <th><Translated vi="Bước">Step</Translated></th>
            <th><Translated vi="Nó tính gì">What it computes</Translated></th>
            <th><Translated vi="Bạn xây nó ở đâu">Where you build it</Translated></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="Giá trị lớn nhất của hàng">Row maximum</Translated></td>
            <td><Translated vi="logit lớn nhất của mỗi hàng">the largest logit of each row</Translated></td>
            <td>
              <a href={href('ch10', 'labs')}><Translated vi="bài lab 10.3">lab 10.3</Translated></a> <Translated vi="(mốc dự án M4)">(milestone M4)</Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="Ngưỡng">Threshold</Translated></td>
            <td>
              <Translated vi="max + log p, để một token sống sót khi xác suất của nó ít nhất bằng p lần xác suất của token đứng đầu">
                max + log p, so that a token survives when its probability is at least p times the top one
              </Translated>
            </td>
            <td><Translated vi="một phép so sánh, không phải một kernel riêng">a comparison, not a kernel of its own</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Nén gọn">Compaction</Translated></td>
            <td><Translated vi="những token sống sót của mỗi hàng, cùng id token, và số lượng của chúng">the survivors of each row, with their token ids, and how many there are</Translated></td>
            <td>
              <a href={href('ch12', 'labs')}><Translated vi="bài lab 12.4">lab 12.4</Translated></a> <Translated vi="(một phần của mốc dự án M5)">(a piece of milestone M5)</Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="Lấy mẫu">Sampling</Translated></td>
            <td><Translated vi="một token được rút ra từ những token sống sót">one token drawn from the survivors</Translated></td>
            <td>
              <Translated vi={<>module X1, <a href={href('x1', 'X1.8')}>mục X1.8</a></>}>
                module X1, <a href={href('x1', 'X1.8')}>section X1.8</a>
              </Translated>
            </td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Bạn có thể thắc mắc vì sao framework không làm việc nén gọn thay bạn. Câu trả lời nằm ở shape: số token sống sót phụ
            thuộc vào dữ liệu, và một framework trả về tensor phải biết shape của nó trước. PyTorch trả giá cho điều đó bằng một
            lần chờ.
          </p>
        }
      >
        <p>
          You may wonder why a framework does not do the compaction for you. The answer is about shapes: the number of
          survivors depends on the data, and a framework that hands back a tensor must know its shape first. PyTorch
          pays for that with a wait.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch12-facts.md, D4, D5)', '2026-11-20 (notes/today/ch12-facts.md, mục D4, D5)')}>
        <Translated
          vi={
            <p>
              <code>torch.nonzero</code> chép số đếm về host và chờ nó, trên CUDA và trên MPS.{' '}
              <code>torch.nonzero_static</code> nhận kích thước đầu ra ngay từ đầu và không phải chờ.
            </p>
          }
        >
          <p>
            <code>torch.nonzero</code> copies the count to the host and waits for it, on CUDA and on MPS.{' '}
            <code>torch.nonzero_static</code> takes the output size up front and does not wait.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Những framework khác tránh hẳn việc nén gọn và giữ mọi token, với logits của các token bị loại đặt thành âm vô cùng:
            một đồ thị có shape cố định không có cách nào nói "dài bằng số token sống sót". Một kernel không gặp vấn đề đó. Nó
            ghi những token sống sót và một số đếm vào bộ nhớ của device, và kernel kế tiếp đọc số đếm ở đó, không cần một chuyến
            về host. Đó là điều chương này dạy bạn viết.
          </p>
        }
      >
        <p>
          Others avoid compaction altogether and keep every token, with the losers' logits set to minus infinity: a
          graph whose shapes are fixed has no way to say "as long as the number of survivors". A kernel has no such
          problem. It writes the survivors and a count to device memory, and the next kernel reads the count there,
          with no trip to the host. That is what this chapter teaches you to write.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch12-facts.md, D3, D6)', '2026-11-20 (notes/today/ch12-facts.md, mục D3, D6)')}>
        <Translated
          vi={
            <p>
              MLX không có <code>nonzero</code> và chỉ "hỗ trợ hạn chế" cho shape đầu ra phụ thuộc vào dữ liệu. Các bộ lấy mẫu GPU
              thử nghiệm của llama.cpp và mlx-lm dùng mặt nạ thay vì nén gọn.
            </p>
          }
        >
          <p>
            MLX has no <code>nonzero</code> and "limited support" for output shapes that depend on the data.
            llama.cpp's experimental GPU samplers and mlx-lm mask instead of compacting.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Chương này qua hai họ">The chapter in two families</Translated></h2>
      <Translated
        vi={
          <p>
            Mọi kernel lọc đều phải trả lời một câu hỏi cho mỗi khóa được giữ: ô đầu ra nào là của tôi? Có hai kiểu câu trả lời,
            và chúng chia chương này làm hai. Một bộ lọc <b>không ổn định</b> để các khóa được giữ ra theo thứ tự bất kỳ; một bộ
            lọc <b>ổn định</b> giữ thứ tự đầu vào của chúng, điều này tốn hơn và quan trọng khi đầu vào đã được sắp xếp.
          </p>
        }
      >
        <p>
          Every filter kernel has to answer one question for each kept key: which output slot is mine? There are two
          kinds of answer, and they split the chapter in two. An <b>unstable</b> filter lets the kept keys come out in any
          order; a <b>stable</b> one keeps their input order, which costs more and matters when the input was sorted.
        </p>
      </Translated>
      <Source section="12.1" pages="289–290, 297" figs="Fig. 12.1" />
      <table className="data" data-part="families">
        <thead>
          <tr>
            <th><Translated vi="Họ">Family</Translated></th>
            <th><Translated vi="Các mục">Sections</Translated></th>
            <th><Translated vi="Một luồng tìm ô của nó thế nào">How a thread finds its slot</Translated></th>
            <th><Translated vi="Các mục cải thiện điều gì">What the sections improve</Translated></th>
          </tr>
        </thead>
        <tbody>
          {FAMILIES.map((f) => (
            <tr key={f.family}>
              <td>{f.family}</td>
              <td className="nowrap">
                {f.secs.map((s, i) => (
                  <span key={s.id}>
                    {i ? ', ' : ''}
                    <a href={href('ch12', s.id)}>{s.label}</a>
                  </span>
                ))}
              </td>
              <td>{f.slot}</td>
              <td>{f.work}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Nửa không ổn định là một bài nghiên cứu về tranh chấp trên một bộ đếm, và cách chữa là cách chương 9 đã dùng cho
            histogram, cộng thêm việc bỏ phiếu trong warp. Nửa ổn định là một bài nghiên cứu về scan, và nó kết thúc bằng một bộ
            lọc làm việc ngay trong mảng nó đọc, điều hóa ra là một câu hỏi về thứ tự các sự kiện giữa các khối.
          </p>
        }
      >
        <p>
          The unstable half is a study in contention on one counter, and the remedy is the one chapter 9 used for
          histograms, plus the warp vote. The stable half is a study in scan, and it ends with a filter that works in the
          very array it reads, which turns out to be a question about the order of events between blocks.
        </p>
      </Translated>
      <Source pages="290–301" />

      <h2><Translated vi="Trước khi bắt đầu">Before you start</Translated></h2>
      <Translated
        vi={
          <p>
            Sáu câu hỏi về những gì chương này giả định từ các chương 4, 6, 9, 10 và 11. Mỗi lời giải thích dẫn về mục mà nó
            xuất phát, và mỗi câu hỏi sẽ quay lại trong phần <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Six questions on what this chapter assumes from chapters 4, 6, 9, 10 and 11. Each explanation links back to
          the section it comes from, and each question returns in <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={q['ch12/start/q-lane64']} label={tr('From chapter 4', 'Từ chương 4')} />
      <Ask q={q['ch12/start/q-onecounter']} label={tr('From chapter 9', 'Từ chương 9')} />
      <Ask q={q['ch12/start/q-exclusive']} label={tr('From chapter 11', 'Từ chương 11')} />
      <Ask q={q['ch12/start/q-privatise']} label={tr('From chapter 9', 'Từ chương 9')} />
      <Ask q={q['ch12/start/q-coalesce']} label={tr('From chapter 6', 'Từ chương 6')} />
      <Ask q={q['ch12/start/q-shflmask']} label={tr('From chapter 10', 'Từ chương 10')} />

      <h2><Translated vi="Cái gì chạy ở đâu">What runs where</Translated></h2>
      <Translated
        vi={
          <p>
            Bốn lab chạy trên mọi backend mà khóa học có; lab thứ năm là tùy chọn và cần một điều chỉ CUDA ghi trong tài liệu.{' '}
            <a href={href('ch12', 'labs')}>Trang lab</a> cho thấy những gì đã thực sự được kiểm tra trên từng backend.
          </p>
        }
      >
        <p>
          Four labs run on every backend the course has; the fifth is optional and needs something only CUDA documents.
          The <a href={href('ch12', 'labs')}>labs page</a> shows what has actually been checked on each.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              <b>emu</b> chạy từng luồng một và bắt được chỉ số sai và rào chắn bị thiếu; <b>race</b> chạy các luồng đồng thời và
              báo các truy cập không được đồng bộ hóa. Không cái nào mô hình hóa thời gian.
            </p>
          }
        >
          <p>
            <b>emu</b> runs one thread at a time and catches wrong indices and missing barriers; <b>race</b> runs threads
            concurrently and reports unsynchronised accesses. Neither models timing.
          </p>
        </Translated>
      </Aside>
      <table className="data" data-part="labs">
        <thead>
          <tr>
            <th><Translated vi="Lab">Lab</Translated></th>
            <th><Translated vi="Tiêu đề">Title</Translated></th>
            <th><Translated vi="Backend">Backends</Translated></th>
          </tr>
        </thead>
        <tbody>
          {LABS.map((l) => (
            <tr key={l.num}>
              <td className="nowrap">{l.num}</td>
              <td>{l.title}</td>
              <td>{l.where}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Lab 12.5 không có bản Metal vì nó để một khối chờ một khối khác, và Apple không ghi trong tài liệu bảo đảm nào rằng
            khối đứng trước của một threadgroup đang chờ rồi sẽ được chạy. Mục <a href={href('ch12', '12.7')}>12.7</a> giải thích
            vì sao, và thay vào đó nên làm gì trên Metal.
          </p>
        }
      >
        <p>
          Lab 12.5 has no Metal version because it lets one block wait for another, and Apple documents no guarantee that
          a waiting threadgroup's predecessor will ever run. Section <a href={href('ch12', '12.7')}>12.7</a> explains
          why, and what to do on Metal instead.
        </p>
      </Translated>
    </>
  );
}
