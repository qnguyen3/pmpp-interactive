import type { Question } from '../../components/Ask';
import { Beyond, Source, Today, type Tier } from '../../components/notes';
import {
  BackendsAndEmulator,
  LabFiles,
  LabPrediction,
  LabResults,
  LabsIntro,
  LabsPage,
  RaceStatus,
  ReferenceSolution,
  RunningLabs,
  VerificationStatus,
  benchRow,
  defineLabs,
  measure,
  msCol,
  numCol,
  rawCol,
  useLabs,
  type BenchSet,
} from '../../components/LabKit';
import { benchRows, fmt, labStatus, num, realBackends, type BenchRow, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 13's own content.
// No question on this page: the lab is the exercise. The two predictions are stored, not
// graded, under ch13/labs/p-coarsen and ch13/labs/p-tiled.
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch13', [{ num: '13.1', dir: 'lab1_merge', name: 'merge', vi: { name: 'trộn' } }]);
const [l1] = LABS;

/** Identifier colours in the reference solution: where each value lives. */
const TIERS: Record<string, Tier> = {
  A: 'global', B: 'global', C: 'global', bounds: 'global', out: 'global',
  tile: 'shared', A_S: 'shared', B_S: 'shared', As: 'shared', Bs: 'shared',
  i: 'reg', j: 'reg', k: 'reg', i_low: 'reg', j_low: 'reg', delta: 'reg', active: 'reg',
  tid: 'reg', k_curr: 'reg', k_next: 'reg', i_curr: 'reg', i_next: 'reg', j_curr: 'reg', j_next: 'reg',
  b: 'reg', tileSize: 'reg', numTiles: 'reg', C_curr: 'reg', C_next: 'reg', A_curr: 'reg', A_next: 'reg',
  B_curr: 'reg', B_next: 'reg', A_length: 'reg', B_length: 'reg', C_length: 'reg', c_curr: 'reg', c_next: 'reg',
  a_curr: 'reg', a_next: 'reg', b_curr: 'reg', b_next: 'reg', iTooHigh: 'reg', jTooHigh: 'reg',
};

// ---------------------------------------------------------------- what each prediction is measured against
const ms = (r: BenchRow | undefined) => {
  const v = num(r?.seconds);
  return v ? `${fmt(v * 1000)} ms` : '?'; // i18n-ignore: number and unit
};
const size = (r: BenchRow | undefined) => (num(r?.total) ? tr(`${num(r?.total)!.toLocaleString('en-US')} outputs`, `${num(r?.total)!.toLocaleString('en-US')} đầu ra`) : '');

/** Basic kernel: time at 1 output per thread over time at 8. Above 1, coarsening was faster. */
const coarsen = (reports: Reports) =>
  measure(reports, l1, (s: BenchSet) => {
    const one = benchRow(s, 'basic', { per_thread: 1 });
    const eight = benchRow(s, 'basic', { per_thread: 8 });
    const a = num(one?.seconds);
    const b = num(eight?.seconds);
    if (!a || !b) return null;
    return { value: a / b, detail: tr(`${ms(eight)} at 8 outputs per thread against ${ms(one)} at 1, ${size(eight)}`, `${ms(eight)} với 8 đầu ra mỗi luồng so với ${ms(one)} với 1, ${size(eight)}`) };
  });

/** At 8 outputs per thread: basic time over tiled time (the tiled time includes its partition launch). */
const tiled = (reports: Reports) =>
  measure(reports, l1, (s: BenchSet) => {
    const basic = benchRow(s, 'basic', { per_thread: 8 });
    const tile = benchRow(s, 'tiled', { per_thread: 8 });
    const a = num(basic?.seconds);
    const b = num(tile?.seconds);
    if (!a || !b) return null;
    return { value: a / b, detail: tr(`${ms(tile)} tiled, partition launch included, against ${ms(basic)} basic, ${size(tile)}`, `${ms(tile)} phân ô, tính cả lần khởi chạy phân chia, so với ${ms(basic)} cơ bản, ${size(tile)}`) };
  });

const MISSING = (
  <Translated
    vi={
      <>
        Phần này cần các hàng benchmark của lab 13.1 từ một thiết bị thật. Chạy{' '}
        <code>python3 labs/pmpp.py run 13.1 --bench</code> khi các kernel của bạn đã qua (hoặc với <code>--solution</code>{' '}
        để đo lời giải tham chiếu). Trình giả lập (emulator) không bao giờ được đo hiệu năng.
      </>
    }
  >
    <>
      This needs the benchmark rows of lab 13.1 from a real device. Run <code>python3 labs/pmpp.py run 13.1 --bench</code>{' '}
      once your kernels pass (or with <code>--solution</code> to time the reference). The emulator is never benchmarked.
    </>
  </Translated>
);

// ---------------------------------------------------------------- results table
const kernelName = (r: BenchRow): string =>
  r.name === 'basic' ? tr('basic, searches global memory', 'cơ bản, tìm trên bộ nhớ toàn cục') : r.name === 'tiled' ? tr('tiled, through shared memory', 'phân ô, qua bộ nhớ chia sẻ') : r.name;

/** The rows a backend's comparison uses: the reader's own where they passed, else the reference's. */
function preferredRows(reports: Reports, b: string, lab: string) {
  const run = reports[b];
  const mine = labStatus(run, lab, 'work') === 'pass' ? benchRows(run, lab, 'work') : [];
  return mine.length ? mine : benchRows(run, lab, 'solution');
}

/** The tiled kernel's minimum-traffic rate beside lab 5.1's measured bandwidth, where both exist. */
function AgainstBandwidth() {
  const { reports, loading } = useLabs();
  if (loading) return null;
  const lines = realBackends(reports).flatMap((b) => {
    const rows = preferredRows(reports, b, l1.id).filter((r) => r.name === 'tiled' && num(r.gb_per_s));
    const best = rows.reduce<BenchRow | undefined>((x, r) => (!x || num(r.gb_per_s)! > num(x.gb_per_s)! ? r : x), undefined);
    const roof = preferredRows(reports, b, 'ch05/lab1_roofline').find((r) => r.name === 'roofline');
    const bw = num(roof?.bandwidth_gb_per_s);
    const g = num(best?.gb_per_s);
    if (!best || !g || !bw) return [];
    return [{ b, device: String(reports[b].device.name ?? b), g, bw, per: best.per_thread }];
  });
  return lines.length ? (
    <>
      {lines.map((l) => (
        <Translated
          key={l.b}
          vi={
            <p data-bandwidth={l.b} data-share={fmt(l.g / l.bw)}>
              Trên {l.device} ({l.b}), hàng tốt nhất của kernel phân ô, với {String(l.per)} đầu ra mỗi luồng, chuyển dữ liệu
              ở mức <b>{fmt(l.g)} GB/s</b> tính theo lưu lượng tối thiểu, và phép đo băng thông của lab 5.1 đo được{' '}
              <b>{fmt(l.bw)} GB/s</b>: tỉ lệ {fmt(l.g / l.bw)}.
            </p>
          }
        >
          <p key={l.b} data-bandwidth={l.b} data-share={fmt(l.g / l.bw)}>
            On {l.device} ({l.b}) the tiled kernel's best row, at {String(l.per)} outputs per thread, moved data at{' '}
            <b>{fmt(l.g)} GB/s</b> counted at the minimum traffic, and lab 5.1's bandwidth probe measured{' '}
            <b>{fmt(l.bw)} GB/s</b>: a ratio of {fmt(l.g / l.bw)}.
          </p>
        </Translated>
      ))}
    </>
  ) : (
    <Translated
      vi={
        <p className="muted" data-bandwidth="none">
          Để đặt phép trộn cạnh băng thông của thiết bị, bạn cần các hàng benchmark từ lab 13.1 và từ lab 5.1 trên cùng một
          backend: <code>python3 labs/pmpp.py run 13.1 --bench</code> và <code>python3 labs/pmpp.py run 5.1 --bench</code>.
        </p>
      }
    >
      <p className="muted" data-bandwidth="none">
        To set the merge beside your device's bandwidth you need benchmark rows from lab 13.1 and from lab 5.1 on the same
        backend: <code>python3 labs/pmpp.py run 13.1 --bench</code> and <code>python3 labs/pmpp.py run 5.1 --bench</code>.
      </p>
    </Translated>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS} tiers={TIERS}>
      <Translated
        vi={
          <LabsIntro>
            Một lab, trong <code>labs/ch13/lab1_merge/</code>. Lab này tùy chọn, vì chương này được rút gọn, và nó không bổ
            sung gì cho dự án xuyên suốt. Nó xây hai kernel trộn của chương, kernel thứ hai theo dạng mà các thư viện phát
            hành, và nó kiểm tra tính ổn định, điều mà một phép trộn số nguyên thuần không thể cho thấy.
          </LabsIntro>
        }
      >
        <LabsIntro>
          One lab, in <code>labs/ch13/lab1_merge/</code>. It is optional, as this chapter is condensed, and it adds nothing
          to the through-line project. It builds the chapter's two merge kernels, the second in the shape the libraries
          ship, and it tests stability, which a merge of plain integers cannot show.
        </LabsIntro>
      </Translated>
      <RunningLabs
        example="13.1"
        alsoShownIn={
          <Translated
            vi={
              <>
                đoạn mã của kernel phân ô ở <a href={href('ch13', '13.3')}>13.3</a>
              </>
            }
          >
            <>
              the tiled kernel's listing on <a href={href('ch13', '13.3')}>13.3</a>
            </>
          </Translated>
        }
      />
      <BackendsAndEmulator />
      <VerificationStatus />

      {/* ------------------------------------------------------------ 13.1 */}
      <h2>
        <Translated vi="Lab 13.1: trộn hai mảng đã sắp xếp bằng cách phân chia theo co-rank">
          Lab 13.1: merge two sorted arrays with co-rank partitioning
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            A và B là các mảng đã sắp xếp gồm các cặp khóa–giá trị, <code>struct KV {'{'} int key; int val; {'}'}</code>, và
            bạn ghi <code>total</code> đầu ra đầu tiên của phép trộn ổn định của chúng vào C, chỉ so sánh khóa. Khi hòa thì về
            A, và các phần tử từ cùng một mảng giữ thứ tự của chúng: quy tắc của sách. <code>total = m + n</code> là toàn bộ
            phép trộn. Bạn viết nó hai lần, bằng CUDA và bằng Metal.
          </p>
        }
      >
        <p>
          A and B are sorted arrays of key–value pairs, <code>struct KV {'{'} int key; int val; {'}'}</code>, and you write the
          first <code>total</code> outputs of their stable merge to C, comparing keys only. Ties go to A, and elements from
          the same array keep their order: the book's rule. <code>total = m + n</code> is the whole merge. You write it
          twice, in CUDA and in Metal.
        </p>
      </Translated>
      <Source section="13.2 to 13.6" pages="305–320" figs="Figs. 13.2, 13.5, 13.9; Exercise 3" />
      <Translated
        vi={
          <p>
            Phần (a) là kernel cơ bản của trang <a href={href('ch13', '13.3')}>13.3</a>, đã sửa: <code>co_rank</code>,{' '}
            <code>merge_sequential</code> với <code>&lt;=</code>, và <code>mergeBasicKernel</code>, trong đó mỗi luồng sở
            hữu <code>perThread</code> đầu ra liên tiếp, giới hạn đầu ra đầu tiên của nó và của luồng kế tiếp ở{' '}
            <code>total</code>, trả về nếu nó không sở hữu đầu ra nào, rồi thực hiện hai lời gọi <code>co_rank</code> và một
            phép trộn. Hàm khởi chạy của nó dùng các khối 128 luồng (<code>MERGE_BLOCK</code>), đủ số khối cho{' '}
            <code>total</code> đầu ra bằng phép chia nguyên lấy trần, và không khởi chạy gì khi <code>total</code> bằng 0.
          </p>
        }
      >
        <p>
          Part (a) is the basic kernel of page <a href={href('ch13', '13.3')}>13.3</a>, corrected: <code>co_rank</code>,{' '}
          <code>merge_sequential</code> with <code>&lt;=</code>, and <code>mergeBasicKernel</code>, where each thread owns{' '}
          <code>perThread</code> consecutive outputs, limits its first and the next thread's first output to{' '}
          <code>total</code>, returns if it owns none, and then makes two <code>co_rank</code> calls and one merge. Its
          launcher uses blocks of 128 threads (<code>MERGE_BLOCK</code>), enough of them for <code>total</code> outputs by
          integer ceiling division, and launches nothing when <code>total</code> is 0.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Phần (b) đưa đầu vào vào bộ nhớ chia sẻ (shared memory) với nạp chính xác, thiết kế của mục "Nạp đúng những gì
            một ô dùng" ở trang 13.3. <code>partitionKernel</code> lưu co-rank của mọi ranh giới ô vào <code>bounds</code>,
            một ô (tile) là <code>128 × perThread</code> đầu ra của một khối. <code>mergeTiledKernel</code> cho khối b khoảng{' '}
            <code>A[bounds[b] .. bounds[b+1])</code> và khoảng B tạo nên phần còn lại của các đầu ra của nó, chép cả hai vào
            một <code>extern __shared__ KV tile[]</code> với các luồng liên tiếp trên các phần tử liên tiếp, chờ ở một{' '}
            <code>__syncthreads()</code>, rồi mỗi luồng tìm trên các bản sao chia sẻ và trộn vào C. Hàm khởi chạy thực hiện
            hai lần khởi chạy, cấp cho lần thứ hai <code>tile × sizeof(KV)</code> byte bộ nhớ chia sẻ động.
          </p>
        }
      >
        <p>
          Part (b) stages the inputs in shared memory with exact loads, the design of page 13.3's "Load exactly what a tile
          uses". <code>partitionKernel</code> stores the co-rank of every tile boundary in <code>bounds</code>, a tile being
          the <code>128 × perThread</code> outputs of one block. <code>mergeTiledKernel</code> gives block b the range{' '}
          <code>A[bounds[b] .. bounds[b+1])</code> and the B range that makes up the rest of its outputs, copies both into
          one <code>extern __shared__ KV tile[]</code> with consecutive threads on consecutive elements, waits at one{' '}
          <code>__syncthreads()</code>, and then each thread searches the shared copies and merges into C. The launcher
          makes the two launches, giving the second <code>tile × sizeof(KV)</code> bytes of dynamic shared memory.
        </p>
      </Translated>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <Translated
              vi={
                <>
                  <code>co_rank</code>, <code>merge_sequential</code>, <code>mergeBasicKernel</code>,{' '}
                  <code>partitionKernel</code>, <code>mergeTiledKernel</code>, <code>launchMergeBasic</code> và{' '}
                  <code>launchMergeTiled</code>; <code>coRankKernel</code> và <code>launchCoRank</code> được cho sẵn để các
                  bài kiểm tra có thể gọi riêng <code>co_rank</code> của bạn
                </>
              }
            >
              <>
                <code>co_rank</code>, <code>merge_sequential</code>, <code>mergeBasicKernel</code>,{' '}
                <code>partitionKernel</code>, <code>mergeTiledKernel</code>, <code>launchMergeBasic</code> and{' '}
                <code>launchMergeTiled</code>; <code>coRankKernel</code> and <code>launchCoRank</code> are provided so the
                tests can call your <code>co_rank</code> on its own
              </>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated
              vi={
                <>
                  <code>co_rank</code>, <code>merge_sequential</code>, <code>merge_basic</code>, <code>merge_partition</code>{' '}
                  và <code>merge_tiled</code>; <code>co_rank_test</code> được cho sẵn
                </>
              }
            >
              <>
                <code>co_rank</code>, <code>merge_sequential</code>, <code>merge_basic</code>, <code>merge_partition</code>{' '}
                and <code>merge_tiled</code>; <code>co_rank_test</code> is provided
              </>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated
              vi={
                <>
                  <code>encodeMergeBasic</code> và <code>encodeMergeTiled</code>; <code>encodeCoRank</code> được cho sẵn
                </>
              }
            >
              <>
                <code>encodeMergeBasic</code> and <code>encodeMergeTiled</code>; <code>encodeCoRank</code> is provided
              </>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            <code>merge.h</code> (struct, kích thước khối và kích thước của <code>bounds</code>) và <code>main.cpp</code>{' '}
            là chỉ đọc.
          </p>
        }
      >
        <p>
          <code>merge.h</code> (the struct, the block size and the size of <code>bounds</code>) and <code>main.cpp</code>{' '}
          are read-only.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Một <code>total</code> nhỏ hơn m + n là của chúng tôi, không phải của sách. Đó là phép trộn hai danh sách top-k,
            khi chỉ cần k đầu ra đầu tiên: một luồng có đầu ra đầu tiên tại hoặc quá <code>total</code> thì không có việc gì.
            Đó là cách backend Metal của llama.cpp kết hợp các danh sách top-k theo từng khối. Module X1 đi một đường khác
            tới top-k, một radix select (<a href={href('x1', 'X1.4')}>X1.4, "Top-k là một phép chọn, không phải sắp xếp"</a>
            ), nên không gì ở đó phụ thuộc vào chế độ này.
          </p>
        }
      >
        <p>
          A <code>total</code> below m + n is ours, not the book's. It is the merge of two top-k lists, where only the
          first k outputs are wanted: a thread whose first output is at or past <code>total</code> has nothing to do. That
          is how llama.cpp's Metal backend combines per-block top-k lists. Module X1 takes another route to top-k, a radix
          select (<a href={href('x1', 'X1.4')}>X1.4, "Top-k is a selection, not a sort"</a>), so nothing there depends on
          this mode.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Chế độ <code>total</code> và cách dùng cho top-k là của chúng tôi; kernel của llama.cpp nằm trong
              notes/today/ch13-facts.md, D3. Một danh sách top-k được sắp xếp lớn nhất trước, nên nó được trộn với khóa đổi
              dấu hoặc mọi phép so sánh đảo ngược.
            </p>
          }
        >
          <p>
            The <code>total</code> mode and the top-k use are ours; the llama.cpp kernel is in notes/today/ch13-facts.md,
            D3. A top-k list is sorted largest first, so it merges with the keys negated or every comparison reversed.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Trong Metal có hai điều thay đổi. Phép tìm và phép trộn chạy trên bộ nhớ thiết bị trong <code>merge_basic</code>{' '}
            và trên bộ nhớ threadgroup trong <code>merge_tiled</code>, và vì mọi con trỏ Metal gắn với một không gian địa chỉ,
            hàm trợ giúp <code>const KV*</code> duy nhất của tệp CUDA không thể phục vụ cả hai. Mã khởi đầu (starter) biến mỗi
            hàm trợ giúp thành một template trên kiểu con trỏ, như merge sort MPS của PyTorch làm; MLX thì viết hai overload.
            Và vì đặc tả Metal không nhắc lại quy tắc của C++ rằng <code>&amp;&amp;</code> bỏ qua toán hạng bên phải, hãy
            kiểm tra biên trước rồi đọc bên trong <code>if</code>. <code>encodeMergeTiled</code> đặt phép phân chia và phép
            trộn vào một encoder, encoder chạy chúng theo thứ tự, và bộ nhớ threadgroup của nó, <code>tile × 8</code> byte,
            luôn là bội của 16 như Metal yêu cầu.
          </p>
        }
      >
        <p>
          In Metal two things change. The search and the merge run on device memory in <code>merge_basic</code> and on
          threadgroup memory in <code>merge_tiled</code>, and since every Metal pointer names one address space, the single{' '}
          <code>const KV*</code> helper of the CUDA file cannot serve both. The starter makes each helper a template on the
          pointer type, as PyTorch's MPS merge sort does; MLX writes two overloads instead. And because the Metal
          specification does not restate C++'s rule that <code>&amp;&amp;</code> skips its right operand, test the bounds
          first and read inside the <code>if</code>. <code>encodeMergeTiled</code> puts the partition and the merge in one
          encoder, which runs them in order, and its threadgroup memory, <code>tile × 8</code> bytes, is always the multiple
          of 16 that Metal requires.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch13-facts.md C2, C3; ch02-facts C9; ch05-facts C8)" /* i18n-ignore: date and citation */>
        <Translated
          vi={
            <p>
              Các không gian địa chỉ của MSL tách rời nhau, và overload mở rộng tới chúng. Đặc tả không nói gì về thứ tự đánh
              giá của <code>&amp;&amp;</code>. Hai dispatch trong một compute encoder mặc định chạy theo thứ tự.
            </p>
          }
        >
          <p>
            MSL address spaces are disjoint, and overloading extends to them. The spec says nothing of the order in which{' '}
            <code>&amp;&amp;</code> evaluates. Two dispatches in one default compute encoder run in order.
          </p>
        </Translated>
      </Today>

      <h3>
        <Translated vi="Lab khác trang lý thuyết ở đâu">Where the lab differs from the page</Translated>
      </h3>
      <ul>
        <Translated
          vi={
            <li>
              Phép chia nguyên lấy trần. Fig. 13.9 dòng 03 lấy <code>ceil</code> của một thương số nguyên đã bị cắt cụt. Ở
              đây số đầu ra mỗi luồng là một đối số và hàm khởi chạy định kích thước lưới bằng{' '}
              <code>(total + per·128 − 1) / (per·128)</code>.
            </li>
          }
        >
          <li>
            Integer ceiling division. Fig. 13.9 line 03 takes <code>ceil</code> of an integer quotient, which has already
            been truncated. Here the per-thread count is an argument and the launcher sizes the grid with{' '}
            <code>(total + per·128 − 1) / (per·128)</code>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <code>__device__</code> trên <code>co_rank</code> và <code>merge_sequential</code>: các đoạn mã không ghi gì, và
              trên CUDA và HIP một hàm không có từ khóa chỉ chạy trên host.
            </li>
          }
        >
          <li>
            <code>__device__</code> on <code>co_rank</code> and <code>merge_sequential</code>: the listings show none, and on
            CUDA and HIP an unqualified function is host-only.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Một luồng có đầu ra đầu tiên tại hoặc quá <code>total</code> sẽ trả về. Sách chỉ kẹp <code>k_next</code> và dựa
              vào cách khởi chạy.
            </li>
          }
        >
          <li>
            A thread whose first output is at or past <code>total</code> returns. The book clamps only{' '}
            <code>k_next</code> and relies on the launch.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Các cặp khóa–giá trị thay vì <code>int</code>, để có thể kiểm tra tính ổn định, và đầu vào <code>const</code>.
            </li>
          }
        >
          <li>Key–value pairs instead of <code>int</code>, so that stability can be tested, and <code>const</code> inputs.</li>
        </Translated>
        <Translated
          vi={
            <li>
              Nạp chính xác, không phải một ô đầy cho mỗi đầu vào như §13.6: co-rank của mọi khối được tính trước, trong lần
              khởi chạy riêng của chúng, trong khi Fig. 13.11 để luồng 0 tính chúng giữa hai rào chắn (barrier). Mỗi khối một
              ô, nên không có vòng while, không có <code>A_consumed</code>, và không còn gì thừa để một buffer vòng giữ lại.
            </li>
          }
        >
          <li>
            Exact loading, not §13.6's full tile of each input: every block's co-ranks are computed first, in their own
            launch, where Fig. 13.11 has thread 0 compute them between two barriers. One tile per block, so no while-loop,
            no <code>A_consumed</code>, and nothing left over for a circular buffer to keep.
          </li>
        </Translated>
      </ul>

      <h3>
        <Translated vi="Các bài kiểm tra kiểm tra gì">What the tests check</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 13.1 --backend emu</code> trước. Mọi phép kiểm tra đều là so bằng chính xác:
            một phép trộn chép phần tử và không làm phép tính số học nào, nên không áp dụng dung sai. Đầu tiên là riêng{' '}
            <code>co_rank</code>, qua kernel cho sẵn: trên ví dụ xuyên suốt tại mọi k từ 0 đến 9, rồi trên các mảng ngẫu nhiên
            có nhiều khóa bằng nhau tại mọi k, mỗi giá trị kỳ vọng được đếm từ <code>std::merge</code>. Rồi cả hai phép trộn
            trên đầu vào rỗng, một đầu vào rỗng, mọi khóa bằng nhau, một mảng nằm hoàn toàn dưới mảng kia, độ dài rất chênh
            lệch và đầu vào ngẫu nhiên, với 1, 3 và 8 đầu ra mỗi luồng (trên thiết bị thêm 16 và hai kích thước lớn), và ở
            nhiều giá trị <code>total</code>.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 13.1 --backend emu</code> first. Every check is exact equality: a merge copies
          elements and does no arithmetic, so no tolerance applies. First <code>co_rank</code> alone, through the provided
          kernel: on the running example at every k from 0 to 9, then on random arrays with many equal keys at every k,
          each expected value counted from <code>std::merge</code>. Then both merges on empty inputs, one empty input,
          all keys equal, one array entirely below the other, very unequal lengths and random inputs, at 1, 3 and 8 outputs
          per thread (devices add 16 and two large sizes), and at several values of <code>total</code>.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Phần tử <code>A[i]</code> mang giá trị i và <code>B[j]</code> mang giá trị 1,000,000,000 + j, nên mỗi đầu ra cho
            biết nó đến từ đâu. Tham chiếu là <code>std::merge</code> chỉ trên khóa, mà chuẩn C++ bảo đảm ổn định với dãy thứ
            nhất đi trước, nên khớp chính xác <code>C[0 .. total)</code> cũng chứng minh tính ổn định. Một lỗi trông như{' '}
            <i>C[6] came from B[1]; a stable merge takes A[4] first (both have key 10)</i>. C bắt đầu đầy giá trị canh
            (sentinel), nên một đầu ra bạn chưa bao giờ ghi sẽ được báo, và mọi thứ được ghi tại hoặc quá{' '}
            <code>C[total]</code> cũng vậy.
          </p>
        }
      >
        <p>
          Element <code>A[i]</code> carries the value i and <code>B[j]</code> the value 1,000,000,000 + j, so every output
          says where it came from. The reference is <code>std::merge</code> on keys only, which the C++ standard guarantees
          to be stable with the first range first, so an exact match of <code>C[0 .. total)</code> also proves stability.
          A failure reads like <i>C[6] came from B[1]; a stable merge takes A[4] first (both have key 10)</i>. C starts full
          of a sentinel, so an output you never wrote is reported, and so is anything written at or past{' '}
          <code>C[total]</code>.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch13-facts.md A6)" /* i18n-ignore: date and citation */>
        <Translated
          vi={
            <p>
              Bản nháp C++: <code>std::merge</code> là "Stable", và với các phần tử tương đương, những phần tử từ dãy thứ nhất
              đứng trước những phần tử từ dãy thứ hai.
            </p>
          }
        >
          <p>
            The C++ draft: <code>std::merge</code> is "Stable", and for equivalent elements those from the first range
            precede those from the second.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Kernel phân ô chia sẻ ô của nó giữa các luồng sau một rào chắn, nên lab cũng được build cho biến thể race của
            trình giả lập: <code>python3 labs/pmpp.py run 13.1 --backend race</code> chạy đồng thời các luồng của kernel phân
            ô dưới ThreadSanitizer, trên ví dụ xuyên suốt và trên các mảng ngẫu nhiên 300 và 257 phần tử với 4 và 2 đầu ra mỗi
            luồng. Khi bỏ rào chắn khỏi lời giải tham chiếu, nó đã báo các lần ghi vào ô xung đột với các lần đọc ô đó của
            phép tìm và phép trộn. Một lần chạy sạch là bằng chứng, không phải chứng minh: nó chỉ thấy những lịch chạy đã xảy
            ra.
          </p>
        }
      >
        <p>
          The tiled kernel shares its tile between threads behind one barrier, so the lab is also built for the emulator's
          race flavour: <code>python3 labs/pmpp.py run 13.1 --backend race</code> runs the tiled kernel's threads
          concurrently under ThreadSanitizer, on the running example and on random arrays of 300 and 257 elements at 4 and
          2 outputs per thread. With the barrier removed from the reference solution, it reported the tile's stores
          against the searches' and merges' reads of it. A clean run is evidence, not proof: it sees only the
          schedules that happened.
        </p>
      </Translated>
      <RaceStatus lab={l1} />
      <Translated
        vi={
          <p>
            Đừng trả về sớm từ <code>mergeTiledKernel</code>. Mọi luồng phải tới được rào chắn, kể cả một luồng của khối cuối
            không sở hữu đầu ra nào: nó có thể vẫn có phần tử cần nạp. Thay vào đó hãy kẹp khoảng đầu ra của nó vào độ dài của
            khối, để nó trộn không phần tử nào.
          </p>
        }
      >
        <p>
          Do not return early from <code>mergeTiledKernel</code>. Every thread must reach the barrier, including a thread
          of the last block that owns no outputs: it may still have elements to load. Clamp its output range to the
          block's length instead, so that it merges zero elements.
        </p>
      </Translated>

      <h3>
        <Translated vi="Thí nghiệm">The experiment</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Làm việc này khi các kernel của bạn đã qua, và trước khi chạy <code>--bench</code>. Trình khám phá warp của trang
            13.3 nói rằng với một đầu ra mỗi luồng, mỗi đầu ra trả cho hai phép tìm nhưng các lần ghi được hợp nhất, và với tám
            đầu ra thì các phép tìm được chia sẻ trong khi mỗi lane ghi vào đoạn của riêng nó. Nó cũng nói kernel phân ô chuyển
            các lần đọc không đều vào bộ nhớ chia sẻ. Cả hai đều là số đếm theo từng lệnh, không có bộ nhớ đệm. Hãy ghi lại hai
            con số.
          </p>
        }
      >
        <p>
          Do this once your kernels pass, and before you run <code>--bench</code>. Page 13.3's warp explorer says that at
          one output per thread every output pays for two searches but the stores coalesce, and at eight the searches are
          shared while each lane stores into its own segment. It also says the tiled kernel moves the irregular reads into
          shared memory. Both are counts per instruction, with no caches. Write down two numbers.
        </p>
      </Translated>
      <Source section="13.5, 13.6, 13.8" pages="314, 320, 327" />
      <LabPrediction
        id="ch13/labs/p-coarsen"
        prompt={
          <Translated vi={<>Với kernel cơ bản, 8 đầu ra mỗi luồng sẽ chạy nhanh hơn 1 đầu ra bao nhiêu lần trên GPU của bạn? (Dưới 1 nghĩa là chậm hơn.)</>}>
            <>For the basic kernel, how many times faster will 8 outputs per thread run than 1 on your GPU? (Below 1 means slower.)</>
          </Translated>
        }
        unit={tr('× the speed at 1 output per thread', '× tốc độ với 1 đầu ra mỗi luồng')}
        measure={coarsen}
        missing={MISSING}
        reading={
          <Translated
            vi={
              <>
                Mô hình không hứa hẹn theo hướng nào: ít phép tìm hơn kéo về một phía, các lần ghi rải rác kéo về phía kia. Mỗi
                luồng của kernel cơ bản đi qua lát của riêng nó theo thứ tự, nên bộ nhớ đệm (cache) có thể phục vụ các bước
                sau của nó từ những dòng mà các bước trước đã nạp, điều mà mô hình bỏ qua.
              </>
            }
          >
            <>
              The model makes no promise either way: fewer searches pull one way, scattered stores the other. Each thread
              of the basic kernel walks its own slice in order, so a cache may serve its later steps from lines its earlier
              steps fetched, which the model leaves out.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch13/labs/p-tiled"
        prompt={
          <Translated vi={<>Với 8 đầu ra mỗi luồng, kernel phân ô sẽ chạy nhanh hơn kernel cơ bản bao nhiêu lần, tính cả lần khởi chạy phân chia?</>}>
            <>At 8 outputs per thread, how many times faster than the basic kernel will the tiled kernel run, partition launch included?</>
          </Translated>
        }
        unit={tr('× the basic kernel', '× kernel cơ bản')}
        measure={tiled}
        missing={MISSING}
        reading={
          <Translated
            vi={
              <>
                Kernel phân ô thêm một lần khởi chạy và một rào chắn, và vẫn tìm hai lần mỗi luồng, trong bộ nhớ chia sẻ. Thứ nó
                loại bỏ là lưu lượng toàn cục rải rác, nên tỉ lệ này cho biết bao nhiêu phần lưu lượng đó thực sự chạm tới bộ
                nhớ trên thiết bị của bạn.
              </>
            }
          >
            <>
              The tiled kernel adds a launch and a barrier, and still searches twice per thread, in shared memory. What it
              removes is the scattered global traffic, so the ratio says how much of that traffic was really reaching
              memory on your device.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l1}
        cols={[
          { head: tr('What ran', 'Đã chạy gì'), cell: kernelName },
          rawCol(tr('Outputs per thread', 'Số đầu ra mỗi luồng'), 'per_thread'),
          { head: tr('Outputs', 'Số đầu ra'), num: true, cell: (r) => (num(r.total) ? num(r.total)!.toLocaleString('en-US') : '') },
          msCol,
          numCol(tr('GB/s at minimum traffic', 'GB/s theo lưu lượng tối thiểu'), 'gb_per_s'),
        ]}
      />
      <Translated
        vi={
          <p>
            Cách benchmark hoạt động: khóa là các số nguyên 31-bit ngẫu nhiên phân bố đều, đã sắp xếp, với m = n. Nó tăng đầu
            vào từ 2^20 phần tử mỗi mảng cho đến khi kernel cơ bản với 8 đầu ra mỗi luồng mất ít nhất 20 ms, hoặc mỗi đầu vào
            đạt 2^23 phần tử, với <code>total = m + n</code>. Mỗi hàng là kết quả tốt nhất của nhiều lần chạy, và mọi kết quả
            đều được kiểm tra với CPU. "GB/s theo lưu lượng tối thiểu" là 2 × total × 8 byte chia cho thời gian: mỗi phần tử
            được đọc một lần và ghi một lần, mức ít nhất mà mọi phép trộn phải chuyển, không phải lượng kernel thực sự đã
            chuyển. Một hàng mà thiết bị không chạy được sẽ được in là đã bỏ qua, kèm lý do.
          </p>
        }
      >
        <p>
          How the benchmark works: keys are uniform random 31-bit integers, sorted, with m = n. It grows the inputs from
          2^20 elements each until the basic kernel at 8 outputs per thread takes at least 20 ms, or each input reaches
          2^23 elements, with <code>total = m + n</code>. Each row is the best of several runs, and every result is checked
          against the CPU. "GB/s at minimum traffic" is 2 × total × 8 bytes over the time: every element read once and
          written once, the least any merge can move, not what the kernel actually moved. A row the device cannot run is
          printed as skipped, with the reason.
        </p>
      </Translated>
      <AgainstBandwidth />
      <ReferenceSolution lab={l1} />
    </LabsPage>
  );
}
