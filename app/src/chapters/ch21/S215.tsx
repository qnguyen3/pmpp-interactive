import { Ask, Predict, type Question } from '../../components/Ask';
import { Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { BinnedBlockScene } from './BinnedBlockScene';
import { PairsExplorer } from './PairsExplorer';

// Section 21.5 (pp. 523–527): cutoff summation, bins, per-block neighbourhood lists, fixed
// bin capacity and the host's overflow pass. Every number on the page is the book's, a vendor
// document's, or computed by src/sim/cutoff.ts (pinned in cutoff.test.ts).

export const questionsS215: Record<string, Question> = {
  'ch21/21.5/q-atom-centric': {
    kind: 'choice',
    id: 'ch21/21.5/q-atom-centric',
    prompt: (
      <Translated
        vi={
          <>
            Vòng lặp bán kính cắt tuần tự, với mỗi nguyên tử, chỉ ghé các điểm lưới nằm trong bán kính cắt: rẻ và đơn giản.
            Sao không giao cho mỗi luồng GPU một nguyên tử và làm y như vậy?
          </>
        }
      >
        <>
          The sequential cutoff loop visits, for each atom, only the grid points within the cutoff: cheap and simple. Why
          not give each GPU thread one atom and do the same?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Các hình cầu của những nguyên tử gần nhau chồng lấn, nên nhiều luồng cộng vào cùng các điểm: một scatter cần phép toán nguyên tử, đúng vấn đề của mục 21.2.">Nearby atoms' spheres overlap, so several threads add to the same points: a scatter that needs atomics, the problem of section 21.2.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Không thể đánh chỉ số nguyên tử theo tọa độ của chúng, nên một luồng không tìm được những điểm lưới nào nằm gần nguyên tử của nó.">Atoms cannot be indexed by their coordinates, so a thread cannot find which grid points lie near its atom.</Translated>, why: <Translated vi="Chính các điểm lưới mới được đánh chỉ số theo tọa độ, và đó đúng là lý do vòng lặp tuần tự dễ dàng.">It is the grid points that are indexed by coordinate, which is exactly why the sequential loop is easy.</Translated> },
      { label: <Translated vi="Một hệ thực tế có quá ít nguyên tử để giao việc cho mọi luồng của GPU.">There are too few atoms in a realistic system to give every thread of a GPU something to do.</Translated>, why: <Translated vi="Các hệ thực tế có hàng trăm nghìn nguyên tử; độ song song không phải vấn đề.">Realistic systems have hundreds of thousands of atoms; parallelism is not the problem.</Translated> },
      { label: <Translated vi="Một luồng GPU không thể lặp qua các điểm của một hình cầu, vì số lần lặp khác nhau giữa các luồng.">A GPU thread cannot loop over the points of a sphere, because the trip count differs between threads.</Translated>, why: <Translated vi="Nó có thể. Số lần lặp khác nhau tốn một ít phân kỳ, nhưng đó không phải lý do.">It can. Different trip counts cost some divergence, but they are not the reason.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Vòng lặp tuần tự lấy nguyên tử làm trung tâm, là một scatter, và các đầu ra của scatter va chạm nhau. Các ngăn là
            thứ làm cho gather lấy lưới làm trung tâm khả thi ở đây: chúng cho luồng của một điểm lưới biết những nguyên tử
            nào có thể có ảnh hưởng, mà không phải quét hết tất cả.
          </p>
        }
      >
        <p>
          The sequential loop is atom-centric, a scatter, and a scatter's outputs collide. Bins are what make the
          grid-centric gather possible here: they tell a thread for a grid point which atoms can matter, without scanning
          them all.
        </p>
      </Translated>
    ),
    source: 'ch. 21 §21.5, p. 524; the scale of real systems from ch. 22, p. 536',
    vi: { source: 'ch. 21 §21.5, p. 524; quy mô của các hệ thực tế lấy từ ch. 22, p. 536' },
  },
  'ch21/21.5/q-fully': {
    kind: 'choice',
    id: 'ch21/21.5/q-fully',
    prompt: (
      <Translated
        vi={
          <>
            Các ngăn hình vuông cạnh b, và một vòng tròn lớn (super circle) bán kính 2.4 b có tâm ở ngăn chính giữa. Danh
            sách chứa bao nhiêu ngăn, và bao nhiêu trong số đó nằm hoàn toàn trong vòng tròn?
          </>
        }
      >
        <>
          Square bins of side b, and a super circle of radius 2.4 b centred on the middle bin. How many bins does the list
          hold, and how many of them lie entirely inside the circle?
        </>
      </Translated>
    ),
    fixedOrder: true,
    options: [
      { label: <Translated vi="21, trong đó 9 ngăn nằm trọn bên trong, như Fig. 21.14 ghi nhãn danh sách của nó.">21, of which 9 are fully inside, as Fig. 21.14 labels its list.</Translated>, why: <Translated vi="Các ngăn ở góc của hình vuông 5 × 5 bắt đầu chạm vào ở khoảng 2.12 b, cùng bán kính mà tại đó khối 3 × 3 được phủ trọn. Vượt qua đó, danh sách có 25.">The corner bins of the 5 × 5 square start to touch at about 2.12 b, the same radius at which the 3 × 3 block becomes fully covered. Past it the list holds 25.</Translated> },
      { label: <Translated vi="25, trong đó 5 ngăn nằm trọn bên trong: ngăn trung tâm và bốn ngăn kề cạnh của nó.">25, of which 5 are fully inside: the centre bin and its four edge neighbours.</Translated>, why: <Translated vi="Các góc xa nhất của những ngăn kề chéo cách tâm khoảng 2.12 b, nằm trong 2.4 b, nên chúng cũng nằm trọn bên trong.">The diagonal neighbours' farthest corners are about 2.12 b from the centre, inside 2.4 b, so they are fully inside too.</Translated> },
      { label: <Translated vi="25, trong đó 9 ngăn nằm trọn bên trong: toàn bộ hình vuông 5 × 5 chạm vào vòng tròn.">25, of which 9 are fully inside: all of the 5 × 5 square touches the circle.</Translated>, correct: true },
      { label: <Translated vi="29, trong đó 9 ngăn nằm trọn bên trong: các ngăn cách ba bước dọc theo các trục được thêm vào danh sách.">29, of which 9 are fully inside: the bins three steps out along the axes join the list.</Translated>, why: <Translated vi="Điểm gần nhất của chúng cách tâm 2.5 b, vừa nằm ngoài vòng tròn.">Their nearest point is 2.5 b from the centre, just outside the circle.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một ngăn nằm trong danh sách khi điểm gần nhất của nó nằm trong vòng tròn, và nằm trọn bên trong khi góc xa nhất
            của nó nằm trong. Các ngăn góc của hình vuông 5 × 5 bắt đầu ở √(1.5² + 1.5²) ≈ 2.12 b, nên chúng được tính; ngăn
            cách ba bước dọc một trục bắt đầu ở 2.5 b, nên bị loại: 25 ngăn. Một ngăn kề chéo kết thúc ở 2.12 b, nằm trọn bên
            trong; một ngăn cách hai bước dọc một trục kết thúc ở √(2.5² + 0.5²) ≈ 2.55 b, chỉ nằm một phần: 9 ngăn nằm trọn
            bên trong. Đặt bán kính cắt thành 1.70 bề rộng ngăn (R ≈ 2.41) để kiểm tra. Tại R = 2 b, các con số là 21 và 5,
            nên con số 21 với 9 ngăn nằm trọn của hình trộn lẫn hai trường hợp.
          </p>
        }
      >
        <p>
          A bin is in the list when its nearest point is inside the circle, and fully inside when its farthest corner is. The
          corner bins of the 5 × 5 square start at √(1.5² + 1.5²) ≈ 2.12 b, so they are in; three steps out along an axis
          starts at 2.5 b, so that is out: 25 bins. A diagonal neighbour ends at 2.12 b, fully inside; a bin two steps out
          along an axis ends at √(2.5² + 0.5²) ≈ 2.55 b, only partly: 9 fully inside. Set the cutoff to 1.70 bin widths (R ≈
          2.41) to check. At R = 2 b the counts are 21 and 5, so the figure's 21 with 9 fully inside mixes the two cases.
        </p>
      </Translated>
    ),
    source: 'ch. 21 Fig. 21.14, p. 526; notes/today/ch21-facts.md E4', // i18n-ignore: a citation
  },
  'ch21/21.5/q-ex5': {
    kind: 'choice',
    id: 'ch21/21.5/q-ex5',
    prompt: (
      <Translated vi={<>Bài tập 5. Phân kỳ điều khiển nảy sinh thế nào khi các luồng của một khối xử lý một ngăn từ danh sách lân cận của chúng?</>}>
        <>Exercise 5. How does control divergence arise when the threads of a block process a bin from their neighbourhood list?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Một ngăn có thể chỉ nằm một phần trong bán kính cắt của một số điểm, nên với một nguyên tử, vài lane nhận còn các lane khác loại.">A bin can lie only partly inside some points' cutoffs, so for one atom some lanes accept and others reject.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Các luồng khác nhau của khối nạp các ngăn khác nhau của danh sách, nên chúng chạy số lần lặp khác nhau.">Different threads of the block load different bins of the list, so they run different numbers of loop iterations.</Translated>, why: <Translated vi="Mọi luồng của một khối đi qua cùng một danh sách và cùng nhau nạp từng ngăn.">All threads of a block walk the same list and load each bin together.</Translated> },
      { label: <Translated vi="Rào chắn sau lần nạp cộng tác khiến vài lane phải chờ các lane khác, điều này chia mỗi warp thành hai nhánh.">The barrier after the cooperative load makes some lanes wait for the others, which splits each warp into two paths.</Translated>, why: <Translated vi="Chờ ở rào chắn là đồng bộ hóa, không phải phân kỳ: mọi luồng đi cùng một đường tới đó.">Waiting at a barrier is synchronisation, not divergence: everyone takes the same path to it.</Translated> },
      { label: <Translated vi="Nguyên tử giả đi một đường mã khác với nguyên tử thật, nên một ngăn chứa cả hai loại sẽ phân kỳ.">Dummy atoms take a different code path from real atoms, so a bin that holds some of each diverges.</Translated>, why: <Translated vi="Nguyên tử giả đi cùng đường với nguyên tử thật; điện tích bằng không của chúng không cộng thêm gì.">Dummy atoms take the same path as real ones; their charge of zero adds nothing.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mọi luồng quét cùng các nguyên tử đã nạp nhưng kiểm tra từng nguyên tử với bán kính cắt quanh điểm của riêng
            nó, và danh sách được lập để phủ điểm xa nhất của khối. Một nguyên tử gần mép bán kính cắt của một số điểm được
            vài lane nhận và bị các lane khác loại. Trong trường hợp 3D của sách, chỉ khoảng một phần tư số phép kiểm tra
            qua được (ch21-facts E4); bao nhiêu nguyên tử đã nạp thật sự chia tách một warp phụ thuộc vào vị trí của chúng,
            và bảng mặt nạ đếm chúng cho khối và độ rộng nhóm đã chọn.
          </p>
        }
      >
        <p>
          Every thread scans the same staged atoms but tests each against the cutoff around its own point, and the list was
          built to cover the block's farthest point. An atom near the edge of some points' cutoffs is accepted by some lanes
          and rejected by others. In the book's 3D case only about a quarter of the tests pass (ch21-facts E4); how many
          staged atoms actually split a warp depends on where they sit, and the mask panel counts them for the chosen block
          and group width.
        </p>
      </Translated>
    ),
    source: 'ch. 21 Exercise 5, p. 528; pp. 524–526; notes/today/ch21-facts.md E4',
    vi: { source: 'ch. 21 Bài tập 5, p. 528; pp. 524–526; notes/today/ch21-facts.md E4' },
  },
  'ch21/21.5/q-global': {
    kind: 'choice',
    id: 'ch21/21.5/q-global',
    prompt: (
      <Translated vi={<>Vì sao trong kernel phân ngăn, các nguyên tử chuyển từ bộ nhớ hằng sang bộ nhớ toàn cục, được nạp qua bộ nhớ chia sẻ?</>}>
        <>Why do the atoms move from constant memory to global memory, staged through shared memory, in the binned kernel?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Các khối khác nhau cần các ngăn khác nhau, nhiều hơn lượng bộ nhớ hằng chứa được cùng lúc; nạp mỗi ngăn một lần cho mỗi khối giữ được sự tái sử dụng.">Different blocks need different bins, more than constant memory holds at once; staging each bin once per block keeps the reuse.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Bộ nhớ hằng chỉ chứa được số nguyên và con trỏ, nên tọa độ và điện tích kiểu float phải đặt ở nơi khác.">Constant memory can only hold integers and pointers, so the float coordinates and charges have to go somewhere else.</Translated>, why: <Translated vi="Các kernel DCS giữ nguyên tử kiểu float trong bộ nhớ hằng.">The DCS kernels keep float atoms in constant memory.</Translated> },
      { label: <Translated vi="Bộ nhớ toàn cục nhanh hơn bộ nhớ hằng một khi các nguyên tử đã được xếp vào ngăn, vì các lần đọc của mỗi ngăn được hợp nhất.">Global memory is faster than constant memory once the atoms are sorted into bins, because each bin's reads are coalesced.</Translated>, why: <Translated vi="Không phải vậy; thay đổi này là về dung lượng và về những nguyên tử mà mỗi khối cần.">It is not; the change is about capacity and about which atoms each block needs.</Translated> },
      { label: <Translated vi="Bộ nhớ chia sẻ được dùng chung bởi mọi khối của một lưới, nên một bản sao đã nạp của mỗi ngăn phục vụ mọi khối cần nó.">Shared memory is shared by all the blocks of a grid, so one staged copy of each bin serves every block that needs it.</Translated>, why: <Translated vi="Bộ nhớ chia sẻ là riêng cho từng khối: mỗi khối nạp bản sao của riêng nó.">Shared memory is per block: each block stages its own copy.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong tổng trực tiếp, mọi luồng đọc mọi nguyên tử của một chunk, nên một chunk nhỏ trong bộ nhớ hằng phục vụ mọi
            khối. Với các ngăn, các khối khác nhau cần các nguyên tử khác nhau, và mọi khối đang hoạt động gộp lại cần nhiều
            hơn hẳn lượng bộ nhớ hằng chứa được. Vì thế các nguyên tử chuyển sang bộ nhớ toàn cục, và mỗi khối chép một ngăn
            vào bộ nhớ chia sẻ một lần rồi để mọi luồng của nó đọc ở đó.
          </p>
        }
      >
        <p>
          In direct summation every thread reads every atom of a chunk, so one small chunk in constant memory serves all
          blocks. With bins, different blocks need different atoms, and all the active blocks together need far more than
          constant memory holds. So the atoms go to global memory, and each block copies a bin into shared memory once and lets
          all its threads read it there.
        </p>
      </Translated>
    ),
    source: 'ch. 21 §21.5, pp. 526–527',
  },
  'ch21/21.5/q-capacity': {
    kind: 'choice',
    id: 'ch21/21.5/q-capacity',
    prompt: (
      <Translated
        vi={
          <>
            Sức chứa ngăn của bạn để lại 2 % số nguyên tử trên danh sách tràn. Một đồng nghiệp đề xuất nâng nó lên cho đến
            khi không còn gì tràn. Việc đó tốn gì?
          </>
        }
      >
        <>
          Your bin capacity leaves 2 % of the atoms on the overflow list. A colleague proposes raising it until nothing
          overflows. What does that cost?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Mọi ngăn được đệm đến số lượng của ngăn đầy nhất, nên nguyên tử giả tốn bộ nhớ, truyền dữ liệu và thời gian quét chỉ để tiết kiệm một lượt xử lý trên host cho 2 %.">Every bin is padded to the fullest bin's count, so dummy atoms cost memory, transfers and scan time to save a host pass over 2 %.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Không gì cả: nguyên tử giả có điện tích bằng không, nên sức chứa lớn hơn không thêm việc vào lần quét của khối nào.">Nothing: dummy atoms have a charge of zero, so a larger capacity adds no work to any block's scan.</Translated>, why: <Translated vi="Chúng không cộng gì vào tổng, nhưng mỗi nguyên tử giả vẫn được nạp, đưa vào bộ nhớ chia sẻ và được mọi luồng kiểm tra.">They add nothing to the sum, but each one is still loaded, staged and tested by every thread.</Translated> },
      { label: <Translated vi="Lượt xử lý phần tràn trở nên thiếu chính xác, vì khi đó host phải cộng các đóng góp trường xa mà kernel bỏ qua.">The overflow pass becomes inaccurate, because the host then has to sum the far-field contributions the kernel left out.</Translated>, why: <Translated vi="Lượt xử lý trên host là chính xác: một tổng bán kính cắt tuần tự trên các nguyên tử tràn.">The host pass is exact: a sequential cutoff sum over the overflow atoms.</Translated> },
      { label: <Translated vi="Kernel cần phép toán nguyên tử khi các ngăn lớn, vì khi đó nhiều luồng của một khối cùng lúc nạp và cập nhật một ngăn dùng chung.">The kernel needs atomics once bins are large, because several threads of a block then load and update one shared bin at once.</Translated>, why: <Translated vi="Quyền sở hữu các điểm lưới không thay đổi; mỗi luồng vẫn chỉ ghi điểm của riêng nó.">Ownership of the grid points does not change; each thread still writes only its own point.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Sức chứa là một sự đánh đổi mà sách mô tả: chọn một mức vừa cho đại đa số các ngăn, thấp hơn nhiều so với ngăn
            lớn nhất, và đẩy phần còn lại sang một đường chậm hơn. Một lượng tràn nhỏ tốn một lượt xử lý ngắn trên host, thậm
            chí có thể chạy trong khi device làm thể tích con kế tiếp. Nâng sức chứa lên bằng ngăn tệ nhất là trả giá cho một
            ngăn đó ở mọi ngăn. Kéo sức chứa trong trình khám phá và xem hai con số đổi chỗ cho nhau.
          </p>
        }
      >
        <p>
          Capacity is a trade the book describes: pick one that fits the vast majority of bins, far below the largest, and
          send the rest down a slower path. A small overflow costs a short host pass that can even run while the device works
          on the next sub-volume. Raising the capacity to the worst bin pays for that one bin in every bin. Drag the capacity
          in the explorer and watch the two counts trade places.
        </p>
      </Translated>
    ),
    source: 'ch. 21 §21.5, p. 527',
  },
  'ch21/21.5/q-vllm-pad': {
    kind: 'numeric',
    id: 'ch21/21.5/q-vllm-pad',
    prompt: (
      <Translated
        vi={
          <>
            <code>moe_align_block_size</code> của vLLM định kích thước buffer đã đệm của nó là{' '}
            <code>topk_ids.numel() + num_experts * (block_size - 1)</code>. Với 512 token được định tuyến tới top 2 trong 8
            expert với <code>block_size</code> 64, con số đó là bao nhiêu ô?
          </>
        }
      >
        <>
          vLLM's <code>moe_align_block_size</code> sizes its padded buffer as{' '}
          <code>topk_ids.numel() + num_experts * (block_size - 1)</code>. For 512 tokens routed to the top 2 of 8 experts
          with <code>block_size</code> 64, how many slots is that?
        </>
      </Translated>
    ),
    answer: 1528,
    unit: 'slots',
    vi: { unit: 'ô', source: 'của khóa học này, không phải của sách: notes/today/ch21-facts.md D6; p. 527' },
    explain: (
      <Translated
        vi={
          <p>
            512 × 2 = 1,024 mục được định tuyến, cộng 8 × 63 = 504: 1,528. Danh sách của mỗi expert được đệm lên bội số của{' '}
            <code>block_size</code>, điều này thêm nhiều nhất <code>block_size − 1</code> mục giả cho mỗi expert. Những mục
            giả đó chính là nguyên tử giả của sách với id token thay cho điện tích, và chúng tốn bộ nhớ theo cùng cách.
          </p>
        }
      >
        <p>
          512 × 2 = 1,024 routed entries, plus 8 × 63 = 504: 1,528. Each expert's list is padded up to a multiple of{' '}
          <code>block_size</code>, which adds at most <code>block_size − 1</code> dummy entries per expert. Those dummies are
          the book's dummy atoms with token ids for charges, and they cost memory the same way.
        </p>
      </Translated>
    ),
    source: 'ours, not the book: notes/today/ch21-facts.md D6; p. 527',
  },
};

/** Threadgroup memory as the reader's Mac reports it, if it has been recorded. */
function ThreadgroupMemory() {
  const { reports } = useReports();
  const v = Number(reports.metal?.device?.maxThreadgroupMemoryLength);
  return Number.isFinite(v) && v > 0 ? (
    <Translated vi={<>máy Mac của bạn báo {v.toLocaleString('en-US')} byte cho mỗi threadgroup</>}>
      <>your Mac reports {v.toLocaleString('en-US')} bytes per threadgroup</>
    </Translated>
  ) : (
    <Translated vi={<><code>python3 labs/pmpp.py doctor</code> cho biết kích thước của bạn</>}>
      <><code>python3 labs/pmpp.py doctor</code> shows your size</>
    </Translated>
  );
}

export function S215() {
  return (
    <>
      <Translated
        vi={
          <p>
            Tổng trực tiếp chính xác và song song, nhưng nó không mở rộng được. Với một hệ phân tử lớn hơn, số nguyên tử tăng
            tỷ lệ với thể tích, và số điểm lưới cũng vậy. Khối lượng việc, nguyên tử nhân điểm, tăng theo bình phương thể tích.
          </p>
        }
      >
        <p>
          Direct summation is exact and parallel, and it does not scale. For a larger molecular system the number of atoms
          grows in proportion to the volume, and so does the number of grid points. The work, atoms times points, grows with
          the square of the volume.
        </p>
      </Translated>
      <Source section="21.5" pages="523, 528" />
      <Predict
        id="ch21/21.5/p-volume"
        prompt={
          <Translated vi="Nhân đôi cạnh của hộp mô phỏng, giữ nguyên mật độ nguyên tử và khoảng cách lưới. Khối lượng việc của tổng trực tiếp tăng lên bao nhiêu lần?">
            Double the side of the simulated box, keeping the atom density and the grid spacing. By what factor does the work of direct summation grow?
          </Translated>
        }
        unit={tr('times', 'lần')}
      >
        {(v) => (
          <p>
            <Translated
              vi={
                <>
                  <b>64.</b> Tám lần số nguyên tử nhân tám lần số điểm. Với bán kính cắt, vùng lân cận của mỗi điểm giữ
                  nguyên kích thước, nên khối lượng việc tăng tám lần, theo đúng thể tích.
                </>
              }
            >
              <b>64.</b> Eight times the atoms times eight times the points. With a cutoff, each point's neighbourhood keeps its
              size, so the work grows eight-fold, in line with the volume.
            </Translated>{' '}
            {v === 64 ? tr('That is what you had.', 'Đúng như bạn đã trả lời.') : v === 8 ? tr('Eight is the cutoff algorithm. Direct summation pays for the new atoms at every new point.', 'Tám là thuật toán bán kính cắt. Tổng trực tiếp phải trả cho các nguyên tử mới ở mọi điểm mới.') : v === 4 ? tr('Four is one side squared; the box is three-dimensional.', 'Bốn là một cạnh bình phương; hộp là ba chiều.') : ''}
          </p>
        )}
      </Predict>

      <h2>
        <Translated vi="Chọn một thuật toán khác">Choosing a different algorithm</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Các thuật toán cho một bài toán khác nhau ở số bước, độ song song chúng phơi ra, độ ổn định số học và băng thông bộ
            nhớ chúng tiêu thụ, và thường không thuật toán nào thắng cả bốn. Thông thường một phương án thay thế phải đạt cùng
            kết quả. Nếu chấp nhận được một thay đổi nhỏ trong kết quả, những lựa chọn mạnh tay hơn sẽ mở ra. <b>Tổng theo bán
            kính cắt</b> là một: đóng góp giảm dần theo khoảng cách, nên chỉ những nguyên tử nằm trong một bán kính cố định quanh
            một điểm lưới mới được cộng trực tiếp, còn nhiều đóng góp xa và nhỏ được xử lý chung bằng một phương pháp riêng, rẻ
            hơn. Khi đó khối lượng việc tăng tuyến tính theo thể tích. Sách không mô tả phương pháp rẻ hơn đó; chính trang GPU
            của VMD nêu một phương pháp, tổng đa mức (multilevel summation) (ch21-facts A1).
          </p>
        }
      >
        <p>
          Algorithms for one problem differ in the number of steps, the parallelism they expose, their numerical stability and
          the memory bandwidth they consume, and usually none wins on all four. Normally an alternative must reach the same
          result. If a slight change in the result is acceptable, more aggressive options open up. <b>Cutoff summation</b> is
          one: a contribution falls off with distance, so only the atoms within a fixed radius of a grid point are summed
          directly, and the many distant, small contributions are handled together by a separate, cheaper method. The work
          then grows linearly with the volume. The book leaves that cheaper method undescribed; VMD's own GPU page names
          one, multilevel summation (ch21-facts A1).
        </p>
      </Translated>
      <Source section="21.5" pages="523–524" figs="Fig. 21.11" />
      <Translated
        vi={
          <p>
            Khi chạy tuần tự, bán kính cắt rất dễ: lấy từng nguyên tử một và chỉ ghé các điểm lưới nằm trong bán kính quanh nó.
            Các điểm lưới nằm trong một mảng đánh chỉ số theo tọa độ, nên tìm chúng chỉ cần một chút số học.
          </p>
        }
      >
        <p>
          Sequentially, the cutoff is easy: take one atom at a time and visit only the grid points within the radius of it.
          Grid points live in an array indexed by coordinate, so finding them is a little arithmetic.
        </p>
      </Translated>
      <Source section="21.5" pages="524" />
      <Ask q={questionsS215['ch21/21.5/q-atom-centric']} />
      <Translated
        vi={
          <p>
            Vòng lặp đó lấy nguyên tử làm trung tâm, và nó không song song hóa tốt. Kernel song song phải lấy lưới làm trung tâm:
            mỗi điểm lưới một luồng, tự tìm những nguyên tử có ảnh hưởng đến nó.
          </p>
        }
      >
        <p>
          That loop is atom-centric, and it does not parallelise well. The parallel kernel has to be grid-centric: one thread
          per grid point, finding the atoms that matter to it.
        </p>
      </Translated>

      <h2>
        <Translated vi="Ngăn và vùng lân cận">Bins and neighbourhoods</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trước hết, xếp các nguyên tử vào các <b>ngăn (bin)</b>: những hộp trong không gian, mỗi hộp chứa các nguyên tử rơi
            vào trong nó. Vùng lân cận của một điểm lưới là tập các ngăn có thể chứa nguyên tử nằm trong bán kính cắt của nó
            (Fig. 21.12). Luồng của điểm đó quét mọi nguyên tử của các ngăn đó và kiểm tra từng nguyên tử với bán kính cắt quanh
            điểm của riêng nó, vì một số nguyên tử nằm trong ngăn chỉ chồng lấn một phần với bán kính cắt.
          </p>
        }
      >
        <p>
          First, sort the atoms into <b>bins</b>: boxes in space, each holding the atoms that fall inside it. A grid point's
          neighbourhood is the set of bins that can hold atoms within its cutoff (Fig. 21.12). Its thread scans every atom of
          those bins and tests each against the cutoff around its own point, because some of them lie in a bin that only
          partly overlaps the cutoff.
        </p>
      </Translated>
      <Source section="21.5" pages="524" figs="Fig. 21.12" />
      <Translated
        vi={
          <p>
            Tính chính xác vùng lân cận của mọi điểm là một bài toán hình học tốn kém, nên sách định nghĩa một vùng lân cận cho
            mỗi khối, chuẩn bị một lần trên host trước khi khởi chạy, và đơn giản hóa thêm: các điểm mà một khối tính đúng bằng
            một ngăn. Vẽ một "vòng tròn lớn" (super circle) quanh ngăn của khối với bán kính bằng bán kính cắt cộng nửa đường
            chéo của ngăn; nó chứa vòng tròn bán kính cắt của mọi điểm trong ngăn. Mọi ngăn mà vòng tròn lớn phủ trọn hoặc một
            phần được đưa vào danh sách, lưu dưới dạng độ lệch so với ngăn của chính khối, nên một danh sách phục vụ mọi khối
            (Fig. 21.14).
          </p>
        }
      >
        <p>
          Working out the exact neighbourhood of every point is an expensive geometric problem, so the book defines one
          neighbourhood per block, prepared once on the host before the launch, and simplifies further: the points a block
          computes are exactly one bin. Draw a "super circle" around the block's bin with radius equal to the cutoff plus half
          the bin's diagonal; it contains the cutoff circle of every point in the bin. Every bin the super circle covers fully
          or partly goes on the list, stored as offsets relative to the block's own bin, so one list serves every block
          (Fig. 21.14).
        </p>
      </Translated>
      <Source section="21.5" pages="525–526" figs="Figs. 21.13, 21.14" />
      <Predict
        id="ch21/21.5/p-bins"
        prompt={
          <Translated vi="Các ngăn hình vuông cạnh b, vòng tròn lớn bán kính 2b, có tâm ở ngăn chính giữa. Vùng lân cận 2D chứa bao nhiêu ngăn?">
            Square bins of side b, super-circle radius 2b, centred on the middle bin. How many bins does the 2D neighbourhood contain?
          </Translated>
        }
        unit={tr('bins', 'ngăn')}
      >
        {(v) => (
          <p>
            <Translated
              vi={
                <>
                  <b>21.</b> Khối 3 × 3, bốn ngăn cách hai bước dọc theo các trục, và tám ngăn bên cạnh chúng; bốn ngăn góc
                  của hình vuông 5 × 5 nằm ngoài. Con số là 21 với mọi bán kính trong khoảng từ chừng 1.58 b đến 2.12 b.
                </>
              }
            >
              <b>21.</b> The 3 × 3 block, the four bins two steps out along the axes, and the eight beside those; the four
              corner bins of the 5 × 5 square stay out. The count is 21 for any radius between about 1.58 b and 2.12 b.
            </Translated>{' '}
            {v === 21 ? tr('That is what you had.', 'Đúng như bạn đã trả lời.') : v === 25 ? tr('Twenty-five takes the whole 5 × 5 square: the corners only start to touch at 2.12 b.', 'Hai mươi lăm là lấy cả hình vuông 5 × 5: các góc chỉ bắt đầu chạm ở 2.12 b.') : v === 9 ? tr('Nine is the 3 × 3 block alone: the circle reaches past it along the axes.', 'Chín là riêng khối 3 × 3: vòng tròn vươn ra ngoài nó dọc theo các trục.') : ''}
          </p>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Kiểm tra con số của bạn trong trình khám phá: nhấn "R = 2 bin widths". Rồi nhấp vào các ngăn và điểm khác, chọn các
            nguyên tử đã nạp trong bảng mặt nạ và đổi độ rộng nhóm của nó, nhân đôi hộp, và kéo sức chứa ngăn.
          </p>
        }
      >
        <p>
          Check your count in the explorer: press "R = 2 bin widths". Then click other bins and points, pick staged atoms in
          the mask panel and change its group width, double the box, and drag the bin capacity.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              2D, các nguyên tử ngẫu nhiên phân bố đều ở một độ cao cố định trên mặt phẳng, các điểm của khối đúng bằng một
              ngăn. Trường xa bị bỏ đi. Các ngăn nằm ngoài hộp bị bỏ qua. Các lane được nhóm theo độ rộng bạn chọn. Đếm, không đo
              thời gian.
            </p>
          }
        >
          <p>
            2D, uniform random atoms a fixed height above the plane, the block's points exactly one bin. The far field is
            dropped. Bins outside the box are skipped. Lanes are grouped by the width you choose. Counted, not timed.
          </p>
        </Translated>
      </Simplification>
      <PairsExplorer
        title={tr('Cutoff and bins', 'Bán kính cắt và ngăn')}
        task={
          <Translated vi="Nhấn R = 2 bin widths và đọc các con số. Rồi nhấp vào các ngăn và điểm khác, chọn các nguyên tử đã nạp, nhân đôi hộp, và kéo sức chứa.">
            Press R = 2 bin widths and read the counts. Then click other bins and points, pick staged atoms, double the box, and drag the capacity.
          </Translated>
        }
        views={['cutoff']}
        foot={
          <Translated vi="Mô hình: src/sim/cutoff.ts. Các nguyên tử (màu hổ phách) nằm trong bộ nhớ toàn cục; các ngăn được tô là những ngăn mà khối này nạp vào bộ nhớ chia sẻ.">
            Model: src/sim/cutoff.ts. Atoms (amber) live in global memory; the shaded bins are the ones this block stages in shared memory.
          </Translated>
        }
      />
      <Ask q={questionsS215['ch21/21.5/q-fully']} label={tr('Check the figure', 'Kiểm tra hình')} />
      <Translated
        vi={
          <p>
            Hai hình cần đọc cẩn thận. Fig. 21.14 nói 9 ngăn được phủ trọn và 12 ngăn phủ một phần, tổng cộng 21. Con số 21 đúng
            trên cả một dải bán kính, nhưng trong dải đó chỉ 5 ngăn nằm trọn bên trong; con số 9 chỉ đúng tại một bán kính duy
            nhất, khoảng 2.12 bề rộng ngăn. Còn Fig. 21.13 ghi nhãn bán kính là "cutoff distance + length of bin diagonal", cả
            đường chéo, trong khi văn bản và Fig. 21.14 nói một nửa; trình khám phá cung cấp cả hai. Bản thân hình chỉ là phác
            họa, không vẽ theo tỷ lệ. Hãy xem cả hàng trường xa: với điện tích hai dấu ở các vị trí ngẫu nhiên, phần mà bán kính
            cắt bỏ đi tại một điểm không hề nhỏ, đó là lý do sách giao trường xa cho một phương pháp riêng thay vì bỏ qua nó.
            Với trường hợp 3D của chính sách (p. 525), ngăn 4 Å và bán kính cắt 12 Å, phần đọc 3D của trình khám phá tính ra
            danh sách: 461 ngăn, 667 với bán kính của Fig. 21.13 (ch21-facts E4).
          </p>
        }
      >
        <p>
          Two of the figures need care. Fig. 21.14 says 9 bins are fully covered and 12 partly, 21 in all. The 21 is right
          over a whole range of radii, but in that range only 5 bins are fully inside; 9 holds at a single radius, about 2.12
          bin widths. And Fig. 21.13 labels its radius "cutoff distance + length of bin diagonal", the whole diagonal, where
          the text and Fig. 21.14 say half; the explorer offers both. The figure itself is a sketch, not drawn to scale. Look
          at the far-field row as well: with charges of both signs at random positions, what the cutoff drops at a point is
          not small, which is why the book hands the far field to a separate method instead of ignoring it. For the book's
          own 3D case (p. 525), 4 Å bins and a 12 Å cutoff, the explorer's 3D readout computes the list: 461 bins, 667 with
          Fig. 21.13's radius (ch21-facts E4).
        </p>
      </Translated>
      <Source section="21.5" pages="525–526" figs="Figs. 21.13, 21.14" />

      <h2>
        <Translated vi="Kernel">The kernel</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Danh sách được truyền cho kernel, nhiều khả năng trong bộ nhớ hằng, và mọi khối đi qua nó. Với mỗi mục, các luồng
            của khối cộng độ lệch vào ngăn của chính khối, cùng nhau nạp các nguyên tử của ngăn đó vào bộ nhớ chia sẻ, rồi mỗi
            luồng kiểm tra từng nguyên tử đã nạp với bán kính cắt quanh điểm của riêng nó và cộng những nguyên tử qua được. Phép
            kiểm tra là nơi warp phân kỳ: vài lane nhận một nguyên tử mà các lane khác loại.
          </p>
        }
      >
        <p>
          The list is passed to the kernel, most likely in constant memory, and every block walks it. For each entry its
          threads add the offset to the block's own bin, load that bin's atoms into shared memory together, and then each
          thread tests each staged atom against the cutoff around its own point and adds the ones that pass. The test is where
          warps diverge: some lanes accept an atom that others reject.
        </p>
      </Translated>
      <Source section="21.5" pages="524, 526" />
      <Translated
        vi={
          <p>
            Phác thảo của sách đi thẳng từ "cùng nhau nạp" sang "mỗi luồng kiểm tra". Một bản sao dùng chung cần hai rào chắn
            (barrier) của kernel phân ô ở chương 5 (<a href={href('ch05', '5.4')}>mục 5.4</a>): một sau lần nạp, một trước khi
            ngăn kế tiếp ghi đè lên nó. Hoạt ảnh cho thấy chúng; chúng là của khóa học này, không phải của sách.
          </p>
        }
      >
        <p>
          The book's outline goes from "load together" straight to "each thread checks". A shared copy needs the two
          barriers of chapter 5's tiled kernel (<a href={href('ch05', '5.4')}>section 5.4</a>): one after the load, one
          before the next bin overwrites it. The animation shows them; they are ours, not the book's.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch05-facts.md A1, A7, B4, C1, C2; ch21-facts C1)">{/* i18n-ignore: a date and a citation */}
        <Translated
          vi={
            <p>
              Metal: bộ nhớ <code>threadgroup</code>, <code>threadgroup_barrier</code>; <ThreadgroupMemory />. Nguyên tử dưới
              dạng <code>const device float4*</code>: Apple khuyên dùng <code>device</code> cho dữ liệu có kích thước thay đổi.
            </p>
          }
        >
          <p>
            Metal: <code>threadgroup</code> memory, <code>threadgroup_barrier</code>; <ThreadgroupMemory />. Atoms as{' '}
            <code>const device float4*</code>: Apple advises <code>device</code> for variable-size data.
          </p>
        </Translated>
      </Today>
      <Simplification>
        <Translated
          vi={
            <p>
              Hoạt ảnh: 2D, một khối 16 luồng, vẽ như một nhóm (ít hơn 32 lane của NVIDIA hay 32 hoặc 64 của AMD; độ rộng của
              Metal được đọc từ pipeline), ngăn 4 ô chứa, vòng tròn lớn 2 bề rộng ngăn, 160 nguyên tử ngẫu nhiên. Đếm, không đo
              thời gian.
            </p>
          }
        >
          <p>
            The animation: 2D, one block of 16 threads, drawn as one group (fewer than NVIDIA's 32 lanes or AMD's 32 or 64; Metal's
            width is read from the pipeline), bins of 4 slots, a super circle
            of 2 bin widths, 160 random atoms. Counted, not timed.
          </p>
        </Translated>
      </Simplification>
      <BinnedBlockScene />
      <Ask q={questionsS215['ch21/21.5/q-ex5']} label={tr('Book exercise', 'Bài tập của sách')} />
      <Ask q={questionsS215['ch21/21.5/q-global']} />
      <Translated
        vi={
          <p>
            Hãy để ý nơi các nguyên tử nằm bây giờ. Trong tổng trực tiếp, mọi luồng đọc mọi nguyên tử của một chunk, nên một
            chunk nhỏ trong bộ nhớ hằng phục vụ cả lưới. Ở đây mỗi luồng xem xét ít nguyên tử hơn nhiều, và các khối khác nhau cần
            các ngăn khác nhau, nên bộ nhớ hằng không còn chứa được những gì mọi khối đang hoạt động cần. Các nguyên tử chuyển
            sang bộ nhớ toàn cục, và mỗi khối nạp một ngăn vào bộ nhớ chia sẻ một lần rồi để mọi luồng của nó đọc ở đó.
          </p>
        }
      >
        <p>
          Notice where the atoms live now. In direct summation every thread read every atom of a chunk, so one small chunk in
          constant memory served the whole grid. Here each thread examines far fewer atoms, and different blocks need different
          bins, so constant memory can no longer hold what all the active blocks need. The atoms go to global memory, and each
          block loads a bin into shared memory once and lets all its threads read it there.
        </p>
      </Translated>
      <Source section="21.5" pages="526–527" />

      <h2>
        <Translated vi="Ngăn cùng một kích thước, và một danh sách tràn">Bins of one size, and an overflow list</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Tự nhiên các ngăn chứa số nguyên tử khác nhau, có ngăn nhiều, có ngăn không có. Để giữ lần nạp của mỗi ngăn được căn
            chỉnh và hợp nhất, sách cho mọi ngăn cùng một sức chứa và đệm các ô trống bằng nguyên tử giả có điện tích bằng không.
            Việc đó tốn hai lần: nguyên tử giả chiếm bộ nhớ và băng thông truyền, và chúng kéo dài lần quét của mọi khối có ngăn
            thưa. Vì vậy sức chứa được chọn để vừa đại đa số các ngăn, thấp hơn nhiều so với ngăn lớn nhất; một nguyên tử mà ngăn
            của nó đã đầy được đưa vào một <b>danh sách tràn</b>. Sau kernel, lưới được chép về và host chạy một tổng bán kính cắt
            tuần tự trên các nguyên tử tràn. Với dưới 3 % số nguyên tử trên danh sách, việc đó thường mất ít thời gian hơn lần
            chạy trên device, và nếu mỗi lần khởi chạy xử lý một thể tích con, host có thể xử lý phần tràn của một thể tích con
            trong khi device làm thể tích con kế tiếp. Điều đó đúng trên mọi backend: các lần khởi chạy CUDA và HIP, và{' '}
            <code>commit</code> của Metal, đều trả về trước khi công việc xong (ch02-facts A5, B3, C4).
          </p>
        }
      >
        <p>
          Bins naturally hold different numbers of atoms, some many, some none. To keep each bin's load aligned and coalesced,
          the book gives every bin the same capacity and pads the empty slots with dummy atoms of charge zero. That costs twice:
          dummy atoms take memory and transfer bandwidth, and they lengthen the scan of every block whose bins are sparse. So
          the capacity is chosen to fit the vast majority of bins, far below the largest; an atom whose bin is already full
          goes on an <b>overflow list</b>. After the kernel the grid is copied back and the host runs a sequential cutoff sum
          over the overflow atoms. With under 3 % of atoms on the list, that typically takes less time than the device run, and
          if each launch handles a sub-volume, the host can process one sub-volume's overflow while the device works on the
          next. That works on every backend: CUDA and HIP launches, and Metal's <code>commit</code>, return before the work
          finishes (ch02-facts A5, B3, C4).
        </p>
      </Translated>
      <Source section="21.5" pages="527" />
      <Ask q={questionsS215['ch21/21.5/q-capacity']} />

      <h2>
        <Translated vi="Cùng ý tưởng đó trong một LLM">The same idea in an LLM</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Định tuyến token tới các expert của một lớp mixture-of-experts là phân ngăn với expert đóng vai ngăn, và các thư viện
            bạn dùng đưa ra đúng các lựa chọn của sách một cách công khai: đếm, đệm đến một kích thước cố định, và gather để kết
            hợp.
          </p>
        }
      >
        <p>
          Routing tokens to the experts of a mixture-of-experts layer is binning with experts as bins, and the libraries you use
          make the book's choices in the open: count, pad to a fixed size, and gather to combine.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              <code>kernel_mul_mm_id_map0</code> cho Metal của llama.cpp đếm token của mỗi expert, tính tổng tiền tố (prefix
              sum) các con số đó, xuất các ô 32 hàng kèm số hàng hợp lệ, và đệm phần của mỗi expert thêm một ô để lần lưu của mọi
              lane đều vô điều kiện (nhận xét rằng các lần lưu phân kỳ thì chậm là của các tác giả, không phải của Apple).{' '}
              <code>mm_ids_helper</code> cho CUDA và HIP của nó giao cho mỗi expert một khối. <code>moe_align_block_size</code>{' '}
              của vLLM đệm danh sách của mỗi expert lên bội số của <code>block_size</code> bằng các id token giả. mlx-lm sắp xếp
              token theo expert khi có từ 64 chỉ số trở lên (ngưỡng riêng của nó, không được giải thích) và gọi{' '}
              <code>gather_mm(sorted_indices=True)</code>. Bước kết hợp là một gather (<code>moe_weighted_reduction</code> của
              llama.cpp). ch21-facts D4 đến D6.
            </p>
          }
        >
          <p>
            llama.cpp's Metal <code>kernel_mul_mm_id_map0</code> counts each expert's tokens, prefix-sums the counts, emits
            32-row tiles with a valid-row count, and pads each expert's slice with one slot so every lane's store is
            unconditional (the comment that divergent stores are slow is the authors', not Apple's). Its CUDA and HIP{' '}
            <code>mm_ids_helper</code> gives each expert one block. vLLM's <code>moe_align_block_size</code> pads each
            expert's list to a multiple of <code>block_size</code> with dummy token ids. mlx-lm sorts tokens by expert at 64
            or more indices (its own, unexplained threshold) and calls <code>gather_mm(sorted_indices=True)</code>. The
            combine is a gather (llama.cpp's <code>moe_weighted_reduction</code>). ch21-facts D4 to D6.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS215['ch21/21.5/q-vllm-pad']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Đếm token cho mỗi expert, bước đầu tiên của tất cả những cách trên, chính là histogram của chương 9; trang tóm tắt của
            chương đó có một câu hỏi về đúng điều này (<a href={href('ch09', 'wrap')}>chương 9, tóm tắt</a>).
          </p>
        }
      >
        <p>
          Counting the tokens per expert, the first step of all of these, is the histogram of chapter 9; its summary page has a
          question on exactly that (<a href={href('ch09', 'wrap')}>chapter 9, summary</a>).
        </p>
      </Translated>

      <h2>
        <Translated vi="Tiếp theo sẽ đi đâu">Where this goes next</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương 22 lùi lại khỏi từng kernel riêng lẻ. <a href={href('ch22', '22.2')}>Mục 22.2</a> phân loại mọi mẫu trong sách
            theo quyết định mà chương này liên tục đưa ra, gather lấy đầu ra làm trung tâm so với scatter lấy đầu vào làm trung
            tâm, và <a href={href('ch22', '22.1')}>mục 22.1</a> quay lại phân ngăn theo bán kính cắt như một ví dụ về việc đổi một
            chút độ chính xác lấy một độ phức tạp khác.
          </p>
        }
      >
        <p>
          Chapter 22 steps back from single kernels. <a href={href('ch22', '22.2')}>Section 22.2</a> sorts every pattern in the
          book by the decision this chapter keeps making, output-centric gather against input-centric scatter, and{' '}
          <a href={href('ch22', '22.1')}>section 22.1</a> returns to cutoff binning as an example of trading a little accuracy
          for a different complexity.
        </p>
      </Translated>
    </>
  );
}
