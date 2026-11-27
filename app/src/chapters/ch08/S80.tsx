import { Ask, Predict, type Question } from '../../components/Ask';
import { Beyond, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { haloCount, haloFraction, haloFraction2D, threadsCube } from '../../sim/stencil3d';

// The start page of chapter 8: what the chapter adds, the two assumptions every kernel in
// it relies on, a map, and a check on the four ideas from chapters 4 to 7 it builds on.
// Every number below comes from src/sim/stencil3d.ts.

const T3 = 8;
const HALO3 = haloCount(T3); // 296
const PCT3 = 100 * haloFraction(T3); // 57.8
const PCT2 = 100 * haloFraction2D(32); // 12.1
const pct = (v: number) => (Math.round(v * 10) / 10).toString();

export const questionsS80: Record<string, Question> = {
  'ch08/start/q-warp-rows': {
    kind: 'choice',
    id: 'ch08/start/q-warp-rows',
    prompt: (
      <Translated
        vi={
          <>
            Một khối có 8 × 8 × 8 luồng. Trên GPU NVIDIA, những luồng nào tạo nên warp 32 luồng đầu tiên của nó?
          </>
        }
      >
        <>
          A block has 8 × 8 × 8 threads. On an NVIDIA GPU, which threads make up its first warp of 32?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated
            vi={
              <>
                <code>threadIdx.z = 0</code>, <code>threadIdx.y</code> từ 0 đến 3, <code>threadIdx.x</code> từ 0 đến 7:
                bốn hàng, mỗi hàng tám luồng.
              </>
            }
          >
            <>
              <code>threadIdx.z = 0</code>, <code>threadIdx.y</code> from 0 to 3, <code>threadIdx.x</code> from 0 to 7:
              four rows of eight.
            </>
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated
            vi={
              <>
                <code>threadIdx.x = 0</code>, <code>threadIdx.y</code> từ 0 đến 3, <code>threadIdx.z</code> từ 0 đến 7:
                một cột xuyên qua khối.
              </>
            }
          >
            <>
              <code>threadIdx.x = 0</code>, <code>threadIdx.y</code> from 0 to 3, <code>threadIdx.z</code> from 0 to 7: a
              column through the block.
            </>
          </Translated>
        ),
        why: (
          <Translated vi="Như vậy z sẽ là chỉ số thay đổi nhanh nhất. Thứ tự thực ra ngược lại.">
            That would make z the fastest index. The order is the other way round.
          </Translated>
        ),
      },
      {
        label: (
          <Translated
            vi={
              <>
                Cả 64 luồng có <code>threadIdx.z = 0</code>: trọn một mặt phẳng.
              </>
            }
          >
            <>
              All 64 threads with <code>threadIdx.z = 0</code>: one full plane.
            </>
          </Translated>
        ),
        why: (
          <Translated vi="Một mặt phẳng 8 × 8 là 64 luồng, tức hai warp 32 luồng. (Trên GPU AMD có wavefront rộng 64, đó là một wavefront; trang 8.4 sẽ quay lại điểm này.)">
            A plane of 8 × 8 is 64 threads, two warps of 32. (It is one wavefront on an AMD GPU with 64-wide wavefronts,
            which page 8.4 comes back to.)
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không có quy tắc nào: cách các luồng được gom nhóm là do hiện thực quyết định trên mọi GPU.">
            There is no rule: how threads are grouped is implementation-defined on every GPU.
          </Translated>
        ),
        why: (
          <Translated vi="NVIDIA có tài liệu hóa quy tắc này. Trong ba backend của khóa học, Metal là backend để ngỏ cách gom luồng thành SIMD-group.">
            NVIDIA documents the rule. Of the three backends in this course, Metal is the one that leaves the grouping
            into SIMD-groups undefined.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Các luồng được xếp thành một hàng với x thay đổi nhanh nhất, rồi đến y, rồi z, và hàng đó được cắt thành từng
            đoạn 32. Với 8 luồng mỗi hàng, 32 luồng liên tiếp là bốn hàng của cùng một mặt phẳng z. Hãy giữ lấy hình ảnh
            này: ở trang 8.4, bốn hàng đó là bốn đoạn rời nhau của bộ nhớ toàn cục (global memory), và đó chính là vấn đề
            hợp nhất truy cập bộ nhớ (coalescing) của một ô (tile) hình lập phương.
          </p>
        }
      >
        <p>
          Threads are put in a line with x fastest, then y, then z, and the line is cut into runs of 32. With 8 threads per
          row, 32 consecutive threads are four rows of the same z-plane. Hold on to the picture: on page 8.4 those four rows
          are four separate stretches of global memory, and that is the coalescing problem of a cube-shaped tile.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.4, pp. 75–76; notes/today/ch08-facts.md (the linear order is settled in ch04 A8; Metal in ch04 C2)',
    vi: {
      source:
        'ch. 4 §4.4, pp. 75–76; notes/today/ch08-facts.md (thứ tự tuyến tính đã được xác lập ở ch04 A8; Metal ở ch04 C2)',
    },
  },
  'ch08/start/q-false-dep': {
    kind: 'choice',
    id: 'ch08/start/q-false-dep',
    prompt: (
      <Translated
        vi={
          <>
            Một khối chạy một vòng lặp. Ở mỗi lần lặp, mọi luồng ghi một phần tử của một buffer chia sẻ, chờ ở một rào
            chắn (barrier), tính toán từ những phần tử do các luồng khác ghi, rồi chờ ở rào chắn thứ hai. Lần lặp kế tiếp
            ghi đè lên chính buffer đó. Nếu mỗi lần lặp có một buffer riêng, bạn có thể bỏ rào chắn nào?
          </>
        }
      >
        <>
          A block runs a loop. Each iteration, every thread writes one element of a shared buffer, waits at a barrier,
          computes from elements other threads wrote, and waits at a second barrier. The next iteration overwrites the same
          buffer. If every iteration had a buffer of its own, which barrier could you drop?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Rào chắn nằm giữa các lần ghi và phần tính toán.">
            The one between the writes and the compute.
          </Translated>
        ),
        why: (
          <Translated vi="Các luồng vẫn đọc những gì luồng khác đã ghi. Chúng không được đọc trước khi dữ liệu có mặt, dù dữ liệu nằm trong buffer nào.">
            Threads still read what other threads wrote. They must not read it before it is there, whatever buffer it is
            in.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Rào chắn sau phần tính toán.">The one after the compute.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Cả hai.">Both.</Translated>,
        why: (
          <Translated vi="Một trong hai canh giữ một phụ thuộc mà thêm bộ nhớ cũng không xóa được.">
            One of them guards a dependence that more memory does not remove.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Không cái nào.">Neither.</Translated>,
        why: (
          <Translated vi="Một trong hai tồn tại chỉ vì buffer được dùng lại.">
            One of them exists only because the buffer is reused.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Rào chắn thứ nhất canh giữ một phụ thuộc thật: luồng đọc cần thứ luồng ghi tạo ra (đọc sau ghi). Rào chắn thứ
            hai canh giữ một phụ thuộc giả: các luồng ghi của lần lặp kế tiếp không cần gì từ các luồng đọc của lần lặp
            này; chúng chỉ không được ghi đè lên thứ vẫn đang được đọc (ghi sau đọc). Trang 8.5 có một vòng lặp giống hệt,
            với ba buffer có nội dung dịch đi một mặt phẳng sau mỗi lần lặp (bằng cách sao chép), và hai rào chắn trong
            mỗi lần lặp.
          </p>
        }
      >
        <p>
          The first barrier guards a true dependence: readers need what writers produce (read after write). The second
          guards a false one: the next iteration's writers need nothing from this iteration's readers; they only must not
          overwrite what is still being read (write after read). Page 8.5 has a loop like this one, with three buffers
          whose contents shift by one plane each iteration (by copying), and two barriers in every iteration.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.4, pp. 110–111',
  },
  'ch08/start/q-unroll-regs': {
    kind: 'choice',
    id: 'ch08/start/q-unroll-regs',
    prompt: (
      <Translated
        vi={
          <>
            Sau khi gộp việc cho luồng (thread coarsening), mỗi luồng giữ bốn kết quả từng phần trong{' '}
            <code>float acc[4]</code> và cập nhật <code>acc[c]</code> trong một vòng lặp theo <code>c</code> từ 0 đến 3.
            Vì sao sách muốn vòng lặp đó được unroll hoàn toàn?
          </>
        }
      >
        <>
          After coarsening, each thread keeps four partial results in <code>float acc[4]</code> and updates{' '}
          <code>acc[c]</code> in a loop over <code>c</code> from 0 to 3. Why does the book want that loop fully unrolled?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated
            vi={
              <>
                Với chỉ số là hằng, <code>acc[0]</code> đến <code>acc[3]</code> có thể được giữ trong bốn thanh ghi
                (register).
              </>
            }
          >
            <>
              With constant indices, <code>acc[0]</code> to <code>acc[3]</code> can be kept in four registers.
            </>
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Unroll loại bỏ rào chắn mà vòng lặp cần giữa lần cập nhật acc này và lần kế tiếp.">
            Unrolling removes the barrier the loop needs between one update of acc and the next.
          </Translated>
        ),
        why: (
          <Translated vi="Vòng lặp này không có rào chắn nào; unroll sao chép thân vòng lặp, nó không thay đổi những gì được đồng bộ hóa.">
            There is no barrier in this loop; unrolling copies the body, it does not change what synchronises.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Unroll làm tăng FLOP/B của kernel, vì phép tính chỉ số của vòng lặp không còn chạy nữa.">
            Unrolling raises the kernel's FLOP/B, because the loop's index arithmetic no longer runs.
          </Translated>
        ),
        why: (
          <Translated vi="Phép tính trên biến đếm vòng lặp là việc tính số nguyên, và FLOP/B chưa bao giờ tính nó. Số phép toán dấu phẩy động và số byte toàn cục không đổi. Thứ thay đổi là nơi acc nằm, không phải lưu lượng toàn cục.">
            Loop-counter arithmetic is integer work, and FLOP/B never counted it. The floating-point operations and the
            global bytes are unchanged. Where acc lives changes, not the global traffic.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Cần unroll thì các lần nạp trong vòng lặp mới được hợp nhất trên toàn warp.">
            Unrolling is needed for the loads inside the loop to coalesce across the warp.
          </Translated>
        ),
        why: (
          <Translated vi="Việc hợp nhất phụ thuộc vào những địa chỉ mà các lane của một warp chạm tới cùng lúc, không phụ thuộc vào cấu trúc vòng lặp bên trong một luồng.">
            Coalescing depends on which addresses the lanes of a warp touch at once, not on the loop structure inside one
            thread.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một mảng được đánh chỉ số bằng giá trị chỉ biết lúc chạy phải nằm ở nơi nào đó có địa chỉ. Sách nói các mảng
            cục bộ như vậy mặc định nằm ở bộ nhớ toàn cục; tài liệu của NVIDIA gọi đó là bộ nhớ cục bộ (local memory) và
            nói nó có độ trễ và băng thông của bộ nhớ thiết bị. Khi chỉ được đánh chỉ số bằng hằng, bốn phần tử có thể là
            bốn thanh ghi. Trang 8.5 dựa vào điều này: kernel phân ô thanh ghi của nó giữ các giá trị của mỗi luồng từ ba
            mặt phẳng liên tiếp trong ba biến có tên, không phải trong một mảng.
          </p>
        }
      >
        <p>
          An array indexed by a value known only at run time has to live somewhere addressable. The book says such local
          arrays go to global memory by default; NVIDIA's documentation calls it local memory and says it has the latency
          and bandwidth of device memory. Indexed only by constants, the four elements can be four registers. Page 8.5
          relies on this: its register-tiled kernel keeps each thread's values from three consecutive planes in three named
          variables, not in an array.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.6, pp. 145–146; NVIDIA\'s rule for local memory in notes/today/ch08-facts.md A5',
    vi: { source: 'ch. 6 §6.6, pp. 145–146; quy tắc của NVIDIA về bộ nhớ cục bộ trong notes/today/ch08-facts.md A5' },
  },
  'ch08/start/q-halo2d': {
    kind: 'numeric',
    id: 'ch08/start/q-halo2d',
    prompt: (
      <Translated
        vi={
          <>
            Một khối nạp một ô đầu vào 32 × 32 cho phép tích chập (convolution) 3 × 3 (bán kính 1). Trong các phần tử được
            nạp, bao nhiêu phần tử là vùng biên (halo), tức là được nạp nhưng không là tâm của đầu ra nào mà khối tính?
          </>
        }
      >
        <>
          A block loads a 32 × 32 input tile for a 3 × 3 convolution (radius 1). How many of the loaded elements are halo,
          that is, loaded but not the centre of any output the block computes?
        </>
      </Translated>
    ),
    answer: Math.round(1024 * haloFraction2D(32)),
    unit: 'elements',
    explain: (
      <Translated
        vi={
          <p>
            Ô đầu ra là ô đầu vào bớt đi một lớp ở mỗi phía: 30 × 30 = 900. Vậy 1024 − 900 = 124 phần tử là vùng biên,
            khoảng {Math.round(PCT2)}% của ô. Sách trích con số này ở chương 8 để so sánh với một ô 3D.
          </p>
        }
      >
        <p>
          The output tile is the input tile less one layer on each side: 30 × 30 = 900. So 1024 − 900 = 124 elements are
          halo, about {Math.round(PCT2)}% of the tile. The book quotes this figure in chapter 8 to compare it with a 3D tile.
        </p>
      </Translated>
    ),
    source: 'ch. 7 §7.5, pp. 172–176; the numbers as the book gives them in ch. 8 §8.4, p. 193',
    vi: {
      unit: 'phần tử',
      source: 'ch. 7 §7.5, pp. 172–176; các con số theo cách sách đưa ra ở ch. 8 §8.4, p. 193',
    },
  },
};

/** The reveal after the halo prediction, written for what the reader committed. */
function HaloReveal({ v }: { v: number }) {
  const close = Math.abs(v - PCT3) <= 4;
  const twoD = Math.abs(v - (100 - 100 * (6 / 8) ** 2)) <= 2.5; // 43.75: the 2D share at this width
  const oneD = Math.abs(v - 25) <= 2.5; // 1 − 6/8
  const carried = v <= PCT2 + 6; // the 2D figure carried over
  return (
    <>
      <Translated
        vi={
          <p>
            <b>
              {HALO3} trong số {threadsCube(T3)} phần tử được nạp là vùng biên: {pct(PCT3)}%.
            </b>{' '}
            Ô đầu ra là 6 × 6 × 6 = 216, và phần còn lại của ô 8 × 8 × 8 có mặt chỉ để cung cấp dữ liệu cho nó.
          </p>
        }
      >
        <p>
          <b>
            {HALO3} of the {threadsCube(T3)} loaded elements are halo: {pct(PCT3)}%.
          </b>{' '}
          The output tile is 6 × 6 × 6 = 216, and the rest of the 8 × 8 × 8 tile is there only to feed it.
        </p>
      </Translated>
      {close ? (
        <Translated
          vi={
            <p>
              Đúng như bạn dự đoán. Hơn một nửa những gì khối nạp không phải là tâm của đầu ra nào.
            </p>
          }
        >
          <p>That is what you predicted. More than half of what the block loads is not the centre of any output.</p>
        </Translated>
      ) : twoD ? (
        <Translated
          vi={
            <p>
              Bạn đang dùng tỉ lệ của một hình vuông 8 × 8: 1 − (6/8)² = 43.75%. Trục thứ ba nhân phần bên trong thêm
              một lần 6/8 nữa, nên phần bên trong là (6/8)³ = 42% của khối lập phương và vùng biên là 58% còn lại.
            </p>
          }
        >
          <p>
            You have the share for an 8 × 8 square: 1 − (6/8)² = 43.75%. The third axis multiplies the interior share by
            another 6/8, so the interior is (6/8)³ = 42% of the cube and the halo the other 58%.
          </p>
        </Translated>
      ) : oneD ? (
        <Translated
          vi={
            <p>
              Đó là tỉ lệ dọc theo một trục, 2 trên 8. Phần bên trong là 6/8 của ô dọc theo mỗi trục trong ba trục, nên nó
              là (6/8)³ = 42% của khối lập phương, và vùng biên là phần còn lại.
            </p>
          }
        >
          <p>
            That is the share along one axis, 2 of 8. The interior is 6/8 of the tile along each of three axes, so it is
            (6/8)³ = 42% of the cube, and the halo is the rest.
          </p>
        </Translated>
      ) : carried ? (
        <Translated
          vi={
            <p>
              Con số 2D không áp dụng sang được. Một ô 32 × 32 là rộng; một ô 8 × 8 × 8 thì không, vì giới hạn số luồng
              cho phép 32 × 32 = 1024 trong hai chiều chỉ cho phép khoảng 10 mỗi cạnh trong ba chiều. Một ô mỏng chủ yếu
              là bề mặt: (6/8)³ = 42% là phần bên trong.
            </p>
          }
        >
          <p>
            The 2D figure does not carry over. A 32 × 32 tile is wide; an 8 × 8 × 8 tile is not, because the thread limit
            that allows 32 × 32 = 1024 in two dimensions allows only about 10 per side in three. A thin tile is mostly
            surface: (6/8)³ = 42% interior.
          </p>
        </Translated>
      ) : (
        <Translated
          vi={
            <p>
              Phần bên trong là 6/8 của ô dọc theo mỗi trục trong ba trục, nên nó là (6/8)³ = 42% của khối lập phương, và
              vùng biên là 58% còn lại.
            </p>
          }
        >
          <p>
            The interior is 6/8 of the tile along each of three axes, so it is (6/8)³ = 42% of the cube, and the halo is
            the other 58%.
          </p>
        </Translated>
      )}
      <Translated
        vi={
          <p>
            So với {pct(PCT2)}% của ô 2D ở trên, đó là vấn đề mà <a href={href('ch08', '8.4')}>trang 8.4</a>{' '}
            mổ xẻ. Trình khám phá "Một ô hình lập phương" của trang đó cho bạn đổi cạnh và xem tỉ lệ giảm xuống, và chỉ ra
            điều gì ngăn bạn làm cạnh lớn hơn.
          </p>
        }
      >
        <p>
          Against {pct(PCT2)}% for the 2D tile above, that is the problem <a href={href('ch08', '8.4')}>page 8.4</a>{' '}
          takes apart. Its explorer "A cube-shaped tile" lets you change the side and watch the share fall, and shows what
          stops you making the side larger.
        </p>
      </Translated>
    </>
  );
}

export function S80() {
  return (
    <>
      <Translated
        vi={
          <p>
            Một transformer không có lưới ba chiều nào cần quét, nên phép tính mà chương này mang tên ít quan trọng với
            bạn. Hãy đọc nó vì hai kỹ thuật mà sách phát triển trên nền phép tính đó, cả hai đều chuyển sang được mọi
            kernel bạn sẽ viết:
          </p>
        }
      >
        <p>
          A transformer has no three-dimensional grids to sweep, so the computation this chapter is named after matters
          little to you. Read it for the two techniques the book develops on it, both of which transfer to any kernel you
          will write:
        </p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>
              <b>gộp việc cho luồng (thread coarsening) dọc theo một chiều</b>: một khối luồng đi qua dữ liệu từng lát
              một và chỉ giữ trên chip một cửa sổ trượt nhỏ gồm vài lát, nhờ đó ô mà nó phủ có thể lớn lên mà không cần
              thêm luồng;
            </li>
            <li>
              <b>phân ô thanh ghi quyết định theo việc ai đọc một giá trị</b>: một giá trị mà đúng một luồng đọc thuộc về
              các thanh ghi (register) của luồng đó; chỉ những giá trị mà nhiều luồng cùng đọc mới cần bộ nhớ chia sẻ
              (shared memory).
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            <b>thread coarsening along one dimension</b>: a block of threads walks through the data one slice at a time and
            keeps only a small rolling window of slices on chip, so the tile it covers can grow without more threads;
          </li>
          <li>
            <b>register tiling decided by who reads a value</b>: a value that exactly one thread reads belongs in that
            thread's registers; only values that several threads read need shared memory.
          </li>
        </ul>
      </Translated>
      <Source section="8 (opening), 8.1, 8.2" pages="183–188" />
      <Translated
        vi={
          <p>
            Trên đường đi, chương này còn cho một điều thứ ba, nhỏ hơn: cách tính sổ sách FLOP/B cho một ô có vùng biên
            (halo) trong ba chiều, nơi vùng biên hóa ra chiếm phần lớn ô. Chính sách gắn kỹ thuật gộp việc của chương với
            phép nhân ma trận, ở chương 6 (p. 153) và lần nữa ở chương 15 (p. 352). Đó là nơi bạn sẽ dùng nó: GEMM đã gộp
            việc và phân ô thanh ghi của <a href={href('ch15', 'start')}>chương 15</a> là mốc dự án M3.
          </p>
        }
      >
        <p>
          Along the way it gives a third, smaller thing: the FLOP/B bookkeeping for a tile with a halo in three
          dimensions, where the halo turns out to be most of the tile. The book itself ties the chapter's coarsening to
          matrix multiplication, in chapter 6 (p. 153) and again in chapter 15 (p. 352). That is where you will use it: the
          coarsened, register-tiled GEMM of <a href={href('ch15', 'start')}>chapter 15</a> is milestone M3.
        </p>
      </Translated>

      <h2>
        <Translated vi="Nền tảng trong hai câu">The background in two sentences</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một <b>stencil</b> tính mỗi điểm của một lưới đều từ chính điểm đó và vài điểm lân cận dọc theo các trục, với
            trọng số đến từ việc xấp xỉ đạo hàm bằng sai phân hữu hạn. Một bộ giải áp dụng nó lên mọi điểm của lưới, hết{' '}
            <b>lượt quét (sweep)</b> này đến lượt khác. Số điểm lân cận ở mỗi phía của tâm là <b>bậc (order)</b>{' '}
            của stencil: stencil 3D của chương này có bậc 1, nên nó đọc điểm đó và sáu điểm lân cận dọc trục, tổng cộng 7
            điểm.
          </p>
        }
      >
        <p>
          A <b>stencil</b> computes each point of a regular grid from that point and a few neighbours along the axes, with
          weights that come from approximating derivatives by finite differences. A solver applies it to every point of
          the grid, one <b>sweep</b> after another. The number of neighbours on each side of the centre is the stencil's{' '}
          <b>order</b>: the 3D stencil of this chapter has order 1, so it reads the point and its six axis neighbours, 7
          points in all.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Mọi kernel trong chương dựa trên hai giả định đơn giản hóa. Không đầu ra nào của một lượt quét phụ thuộc vào
            đầu ra khác của cùng lượt quét, nên mọi đầu ra có thể được tính song song từ lưới đầu vào. Và các điểm trên
            biên của lưới giữ điều kiện biên: chúng giữ nguyên giá trị đầu vào, và chỉ các điểm bên trong được tính.
          </p>
        }
      >
        <p>
          Every kernel in the chapter rests on two simplifying assumptions. No output of a sweep depends on another output
          of the same sweep, so all outputs can be computed in parallel from the input grid. And the points on the
          grid's boundary hold boundary conditions: they keep their input values, and only interior points are computed.
        </p>
      </Translated>
      <Source section="8.2" pages="187–188" figs={tr('Fig. 8.5', 'Hình 8.5')} />

      <h2>
        <Translated vi="Vị trí trong một hệ thống suy luận">Where this sits in an inference stack</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Có một chỗ trong suy luận LLM thực sự chạy một stencil. Các lớp kiểu Mamba áp dụng một phép tích chập
            (convolution) nhân quả ngắn dọc theo trục token, và một phép tích chập dọc theo một trục là một stencil 1D.
            Trong llama.cpp, kernel CUDA cho nó,{' '}
            <code>ssm_conv</code>, dùng chính các bước đi của chương này. Nhánh cho chuỗi ngắn của nó gộp việc dọc theo
            các token: mỗi luồng đi qua một kênh với một cửa sổ riêng cho từng luồng chứa vài đầu vào gần nhất (cửa sổ đó
            có nằm lại trong thanh ghi hay không thì chưa được kiểm tra). Nhánh cho chuỗi dài của nó là ô có vùng biên của
            trang 8.4: một khối nạp một ô token cùng vùng biên của nó vào bộ nhớ chia sẻ. Trang 8.5 quay lại nó bằng mã.
          </p>
        }
      >
        <p>
          One place in LLM inference does run a stencil. Mamba-style layers apply a short causal convolution along the
          token axis, and a convolution along one axis is a 1D stencil. In llama.cpp the CUDA kernel for it,{' '}
          <code>ssm_conv</code>, uses this chapter's moves. Its short-sequence path coarsens along the tokens: each thread
          marches through one channel with a per-thread window of the last few inputs (whether that window stays in
          registers was not checked). Its long-sequence path is the halo tile of page 8.4: a block loads a tile of tokens
          plus its halo into shared memory. Page 8.5 comes back to it in code.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Mối liên hệ với các lớp kiểu Mamba và llama.cpp là của khóa học này, không phải của sách. Cấu trúc của
              kernel được đọc từ mã nguồn llama.cpp (notes/today/ch08-facts.md, D1).
            </p>
          }
        >
          <p>
            The link to Mamba-style layers and llama.cpp is ours, not the book's. The kernel's structure was read from the
            llama.cpp source (notes/today/ch08-facts.md, D1).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Kernel nằm ở <code>ggml/src/ggml-cuda/ssm-conv.cu</code>, và cùng tệp đó dựng backend HIP của llama.cpp. Cửa
            sổ của nó được đánh chỉ số bằng biến đếm token, chính là chỉ số lúc chạy mà câu hỏi thứ ba bên dưới nói đến.
          </p>
        }
      >
        <p>
          The kernel is <code>ggml/src/ggml-cuda/ssm-conv.cu</code>, and the same file builds llama.cpp's HIP backend. Its
          window is indexed by the token counter, the run-time index the third question below is about.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch08-facts.md, D1)', '2026-11-20 (notes/today/ch08-facts.md, mục D1)')}>
        <Translated
          vi={
            <p>
              Đọc trên nhánh mặc định của llama.cpp; commit cuối cùng của tệp là ngày 2026-11-18. Kernel cho chuỗi ngắn là{' '}
              <code>ssm_conv_f32</code>, kernel cho chuỗi dài là <code>ssm_conv_long_token_f32</code>.
            </p>
          }
        >
          <p>
            Read on the llama.cpp default branch; the file's last commit was on 2026-11-18. The short-sequence kernel is{' '}
            <code>ssm_conv_f32</code>, the long-sequence one <code>ssm_conv_long_token_f32</code>.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Chương này trong khóa học">The chapter in this course</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hai trang. <a href={href('ch08', '8.4')}>Trang 8.4</a> đếm những gì kernel một-luồng-một-điểm và kernel phân
            ô lập phương đòi hỏi ở bộ nhớ, và cho thấy vì sao một ô hình lập phương không đi xa được: giới hạn số luồng
            chặn cạnh của nó, vùng biên chiếm hơn một nửa ô, và lần nạp của mỗi warp bị chia ra trên nhiều hàng (sách pp.
            187–193).{' '}
            <a href={href('ch08', '8.5')}>Trang 8.5</a> cắt đứt mối liên hệ giữa kích thước ô và số luồng bằng cách cho
            một mặt phẳng luồng tiến dần theo z, rồi chuyển những giá trị chỉ một luồng đọc vào thanh ghi (pp. 193–198).
            Sau đó là một <a href={href('ch08', 'labs')}>lab</a> tùy chọn, dựng cả hai kernel và đo chúng, và phần{' '}
            <a href={href('ch08', 'wrap')}>tổng kết</a> với các bài tập của sách.
          </p>
        }
      >
        <p>
          Two pages. <a href={href('ch08', '8.4')}>Page 8.4</a> counts what the one-thread-per-point kernel and the
          cube-tiled kernel ask of memory, and shows why a cube-shaped tile cannot get far: the thread limit caps its side,
          the halo takes more than half of it, and each warp's load is split across rows (book pp. 187–193).{' '}
          <a href={href('ch08', '8.5')}>Page 8.5</a> breaks the link between tile size and thread count by marching a
          plane of threads through z, then moves the values only one thread reads into registers (pp. 193–198). Then one
          optional <a href={href('ch08', 'labs')}>lab</a>, which builds both kernels and measures them, and the{' '}
          <a href={href('ch08', 'wrap')}>summary</a> with the book's exercises.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Phần được rút gọn: nền tảng sai phân hữu hạn của §8.1 (pp. 184–187), với đường sin được lấy mẫu của Hình 8.1,
            phép suy ra stencil 3 điểm và các hình vẽ stencil 1D, 2D và 3D trong Hình 8.2 đến 8.4, được thu gọn thành hai
            đoạn văn ở trên. Những nhận xét của sách về độ chính xác (pp. 184–185) chỉ xuất hiện ở trang 8.4 dưới dạng tác
            động của một giá trị 8 byte lên các tỉ số. Hãy đọc những trang đó nếu bạn muốn khía cạnh phương pháp số; không
            phần nào sau này của khóa học phụ thuộc vào chúng.
          </p>
        }
      >
        <p>
          What is condensed: the finite-difference background of §8.1 (pp. 184–187), with the sampled sine curve of Fig.
          8.1, the derivation of the 3-point stencil and the drawings of 1D, 2D and 3D stencils in Figs. 8.2 to 8.4, is
          reduced to the two paragraphs above. The book's remarks on precision (pp. 184–185) appear on page 8.4 only as
          what an 8-byte value does to the ratios. Read those pages if you want the numerical-methods side; nothing later in
          this course depends on them.
        </p>
      </Translated>

      <h2>
        <Translated vi="Trước khi bắt đầu">Before you start</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bốn câu hỏi về những gì chương này lấy từ chương 4 đến 7. Mỗi câu được dùng ở một trang sau, và mỗi câu sẽ
            quay lại trong phần <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Four questions on what this chapter takes from chapters 4 to 7. Each one is used on a later page, and each one
          comes back in <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={questionsS80['ch08/start/q-warp-rows']} label={tr('From chapter 4', 'Từ chương 4')} />
      <Ask q={questionsS80['ch08/start/q-false-dep']} label={tr('From chapter 5', 'Từ chương 5')} />
      <Ask q={questionsS80['ch08/start/q-unroll-regs']} label={tr('From chapter 6', 'Từ chương 6')} />
      <Ask q={questionsS80['ch08/start/q-halo2d']} label={tr('From chapter 7', 'Từ chương 7')} />
      <Translated
        vi={
          <p>
            Giờ là cùng phép đếm đó trong ba chiều. Một ô 3D cho stencil bậc 1 phải nhỏ, vì mỗi luồng nạp một phần tử và
            một khối có số luồng giới hạn: sách dùng 8 × 8 × 8.
          </p>
        }
      >
        <p>
          Now the same count in three dimensions. A 3D tile for a stencil of order 1 has to be small, because one thread
          loads one element and a block has a limited number of threads: the book uses 8 × 8 × 8.
        </p>
      </Translated>
      <Predict
        id="ch08/start/p-halo3d"
        prompt={
          <Translated
            vi={
              <>
                Một khối nạp một ô đầu vào 8 × 8 × 8 cho stencil 7 điểm (một điểm lân cận dọc mỗi trục). Bao nhiêu phần
                trăm số phần tử được nạp là vùng biên? Hãy chốt một con số phần trăm.
              </>
            }
          >
            <>
              A block loads an 8 × 8 × 8 input tile for the 7-point stencil (one neighbour along each axis). What share of
              the loaded elements is halo? Commit a percentage.
            </>
          </Translated>
        }
        unit="%"
      >
        {(v) => <HaloReveal v={v} />}
      </Predict>
    </>
  );
}
