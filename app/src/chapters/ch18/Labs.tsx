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
  defineLabs,
  measure,
  msCol,
  numCol,
  rawCol,
  type BenchSet,
  type Col,
} from '../../components/LabKit';
import { Translated, tr } from '../../lib/i18n';
import { fmt, num, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 18's own content.
// No question on this page: the lab is the exercise. The three predictions are stored, not
// graded, under ch18/labs/p-lab1-floor, ch18/labs/p-lab1-rmat and ch18/labs/p-lab1-coop.
// The benchmark rows are a contract with labs/ch18/lab1_frontier/main.cpp: names push,
// frontier, privatised, cooperative and floor; graph 'grid' or 'rmat'; seconds, levels,
// us_per_level and medges_per_s.
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch18', [{ num: '18.1', dir: 'lab1_frontier', name: 'frontier BFS', vi: { name: 'BFS dùng frontier' } }]);
const [l1] = LABS;

/** Identifier colours in the reference solution: the graph and frontiers in global memory, the private queue in shared. */
const TIERS: Record<string, Tier> = {
  srcPtrs: 'global', dst: 'global', level: 'global', prevFrontier: 'global', currFrontier: 'global', numCurrFrontier: 'global',
  currFrontier_s: 'shared', numCurrFrontier_s: 'shared', currFrontierStartIdx: 'shared',
  i: 'reg', vertex: 'reg', vertexId: 'reg', edge: 'reg', neighbor: 'reg', currFrontierIdx_s: 'reg', currFrontierIdx: 'reg',
  unvisited: 'reg', expected: 'reg', visitSuccess: 'reg', count: 'reg', j: 'reg', gridRank: 'reg', gridThreads: 'reg',
};

/** The benchmark's two graphs, in the driver's order. */
export const GRAPHS = ['grid', 'rmat'] as const;
/** The benchmark's searches, in the driver's order. */
export const SEARCHES = ['push', 'frontier', 'privatised', 'cooperative', 'floor'] as const;

// ---------------------------------------------------------------- reading the benchmark rows
const row = (s: BenchSet, name: string, graph: string): BenchRow | undefined =>
  s.rows.find((r) => r.name === name && r.graph === graph);
const secs = (r: BenchRow | undefined) => num(r?.seconds);
const ms = (r: BenchRow | undefined) => {
  const v = secs(r);
  return v ? `${fmt(v * 1000)} ms` : '?'; // i18n-ignore: a unit
};
const levelsOf = (r: BenchRow | undefined) => (num(r?.levels) != null ? tr(`${num(r?.levels)!.toLocaleString('en-US')} levels`, `${num(r?.levels)!.toLocaleString('en-US')} mức`) : '');

/** Prediction 1: on 'grid', floor time over privatised time. */
const floorShare = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const fl = row(s, 'floor', 'grid');
    const pr = row(s, 'privatised', 'grid');
    const a = secs(fl);
    const b = secs(pr);
    if (!a || !b) return null;
    return {
      value: a / b,
      detail: tr(
        `${ms(fl)} for the empty levels against ${ms(pr)} for the privatised search, ${levelsOf(pr)}`,
        `${ms(fl)} cho các mức rỗng so với ${ms(pr)} cho phép tìm kiếm riêng tư hóa, ${levelsOf(pr)}`,
      ),
    };
  });

/** Prediction 2: on 'rmat', frontier time over privatised time. */
const rmatRelief = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const fr = row(s, 'frontier', 'rmat');
    const pr = row(s, 'privatised', 'rmat');
    const a = secs(fr);
    const b = secs(pr);
    if (!a || !b) return null;
    return {
      value: a / b,
      detail: tr(
        `${ms(fr)} frontier against ${ms(pr)} privatised, ${levelsOf(pr)}`,
        `${ms(fr)} cho frontier so với ${ms(pr)} cho bản riêng tư hóa, ${levelsOf(pr)}`,
      ),
    };
  });

/** Prediction 3, grid half: privatised time over cooperative time. Only CUDA and HIP have a cooperative row. */
const coopGain = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const pr = row(s, 'privatised', 'grid');
    const co = row(s, 'cooperative', 'grid');
    const a = secs(pr);
    const b = secs(co);
    if (!a || !b) return null;
    const pr2 = secs(row(s, 'privatised', 'rmat'));
    const co2 = secs(row(s, 'cooperative', 'rmat'));
    const rmat = pr2 && co2
      ? tr(`; on rmat the same ratio is ${fmt(pr2 / co2)}`, `; trên rmat cùng tỉ số đó là ${fmt(pr2 / co2)}`)
      : tr('; no rmat pair in this run', '; lần chạy này không có cặp rmat');
    return {
      value: a / b,
      detail: tr(
        `${ms(pr)} privatised against ${ms(co)} cooperative on grid${rmat}`,
        `${ms(pr)} cho bản riêng tư hóa so với ${ms(co)} cho bản cooperative trên grid${rmat}`,
      ),
    };
  });

const MISSING = (
  <Translated
    vi={
      <>
        Phần này cần các hàng benchmark của lab 18.1 từ một thiết bị thật. Hãy chạy{' '}
        <code>python3 labs/pmpp.py run 18.1 --bench</code> khi các kernel của bạn đã đạt (hoặc với <code>--solution</code>{' '}
        để đo lời giải tham chiếu). Các trình giả lập không bao giờ được đo hiệu năng.
      </>
    }
  >
    <>
      This needs the benchmark rows of lab 18.1 from a real device. Run <code>python3 labs/pmpp.py run 18.1 --bench</code>{' '}
      once your kernels pass (or with <code>--solution</code> to time the reference). The emulators are never benchmarked.
    </>
  </Translated>
);
const MISSING_COOP = (
  <Translated
    vi={
      <>
        Phần này cần một hàng <code>cooperative</code>, mà chỉ một thiết bị CUDA hoặc HIP hỗ trợ cooperative launch mới cho
        được, khi phần B đã đạt: chạy <code>python3 labs/pmpp.py run 18.1 --bench</code> ở đó. Metal không có phần B, và một
        lần chạy chỉ trên Metal để trống phần này.
      </>
    }
  >
    <>
      This needs a <code>cooperative</code> row, which only a CUDA or HIP device that supports cooperative launch can give,
      once part B passes: <code>python3 labs/pmpp.py run 18.1 --bench</code> there. Metal has no part B, and a run on
      Metal alone leaves this empty.
    </>
  </Translated>
);

// ---------------------------------------------------------------- results table
const COLS: Col[] = [
  {
    get head() {
      return tr('Search', 'Phép tìm kiếm');
    },
    cell: (r) => String(r.name ?? ''),
  },
  {
    get head() {
      return tr('Graph', 'Đồ thị');
    },
    cell: (r) => String(r.graph ?? ''),
  },
  numCol('Vertices', 'vertices'), // i18n-ignore: Vietnamese head supplied where LabResults renders
  numCol('Edges', 'edges'), // i18n-ignore: Vietnamese head supplied where LabResults renders
  rawCol('Levels', 'levels'), // i18n-ignore: Vietnamese head supplied where LabResults renders
  msCol,
  numCol('µs per level', 'us_per_level'), // i18n-ignore: Vietnamese head supplied where LabResults renders
  numCol('Million edges/s', 'medges_per_s'), // i18n-ignore: Vietnamese head supplied where LabResults renders
];

/** What each backend calls the pieces of the lab (notes/today/ch18-facts.md, "Where the backends really differ"). */
function BackendsTable() {
  return (
    <table className="data wide" style={{ maxWidth: '58rem' }} data-table="backends">
      <thead>
        <tr><th><Translated vi="Trong lab này">In this lab</Translated></th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
      </thead>
      <tbody>
        <tr>
          <td><Translated vi="Thăm đúng một lần">Visit exactly once</Translated></td>
          <td>
            <Translated vi={<><code>atomic_ref::compare_exchange_strong</code>, một giá trị expected mới cho mỗi lần gọi</>}>
              <code>atomic_ref::compare_exchange_strong</code>, a fresh expected value per call
            </Translated>
          </td>
          <td>
            <Translated vi={<><code>atomicCAS(&amp;level[v], UNVISITED, currLevel) == UNVISITED</code> kiểu cũ (legacy), vốn là dạng mạnh</>}>
              legacy <code>atomicCAS(&amp;level[v], UNVISITED, currLevel) == UNVISITED</code>, which is strong
            </Translated>
          </td>
          <td>
            <Translated vi={<><code>atomic_compare_exchange_weak_explicit</code> trong một vòng lặp thử lại khi <code>expected</code> vẫn giữ <code>UNVISITED</code></>}>
              <code>atomic_compare_exchange_weak_explicit</code> in a loop that retries while <code>expected</code> still holds <code>UNVISITED</code>
            </Translated>
          </td>
        </tr>
        <tr>
          <td><Translated vi="Bộ đếm riêng">Private counter</Translated></td>
          <td>
            <Translated vi={<><code>atomic_ref&lt;unsigned, thread_scope_block&gt;</code> trên một biến <code>__shared__</code></>}>
              <code>atomic_ref&lt;unsigned, thread_scope_block&gt;</code> on a <code>__shared__</code> variable
            </Translated>
          </td>
          <td>
            <Translated vi={<><code>atomicAdd</code> kiểu cũ trên cùng biến đó</>}>
              legacy <code>atomicAdd</code> on the same variable
            </Translated>
          </td>
          <td>
            <Translated vi={<><code>threadgroup atomic_uint</code>, đặt về 0 bằng <code>atomic_store_explicit</code></>}>
              <code>threadgroup atomic_uint</code>, zeroed with <code>atomic_store_explicit</code>
            </Translated>
          </td>
        </tr>
        <tr>
          <td><Translated vi="Đặt lại khi tràn (dòng 29)">Reset on overflow (line 29)</Translated></td>
          <td>
            <Translated vi={<><code>store</code> qua cùng <code>atomic_ref</code> đó, hoặc kẹp giá trị sau rào chắn</>}>
              <code>store</code> through the same <code>atomic_ref</code>, or a clamp after the barrier
            </Translated>
          </td>
          <td><code>atomicExch</code></td>
          <td>
            <Translated vi={<><code>atomic_store_explicit</code>: không thể viết lần ghi thường</>}>
              <code>atomic_store_explicit</code>: the plain store cannot be written
            </Translated>
          </td>
        </tr>
        <tr>
          <td><Translated vi="Giá trị canh">Sentinel</Translated></td>
          <td colSpan={2}>
            <Translated vi={<><code>UINT_MAX</code> từ <code>&lt;climits&gt;</code> (các trình giả lập không có <code>&lt;cuda/std/limits&gt;</code>)</>}>
              <code>UINT_MAX</code> from <code>&lt;climits&gt;</code> (the emulators have no <code>&lt;cuda/std/limits&gt;</code>)
            </Translated>
          </td>
          <td><code>numeric_limits&lt;uint&gt;::max()</code></td>
        </tr>
        <tr>
          <td><Translated vi="Phần B: rào chắn lưới">Part B: grid barrier</Translated></td>
          <td>
            <Translated vi={<><code>this_grid().sync()</code>, <code>&lt;cooperative_groups.h&gt;</code>, rank 64-bit</>}>
              <code>this_grid().sync()</code>, <code>&lt;cooperative_groups.h&gt;</code>, 64-bit ranks
            </Translated>
          </td>
          <td>
            <Translated vi={<>cùng tên, <code>&lt;hip/hip_cooperative_groups.h&gt;</code>, rank 32-bit</>}>
              same names, <code>&lt;hip/hip_cooperative_groups.h&gt;</code>, 32-bit ranks
            </Translated>
          </td>
          <td><Translated vi="không có: mỗi mức một dispatch">none: one dispatch per level</Translated></td>
        </tr>
        <tr>
          <td><Translated vi="Phần B: khởi chạy">Part B: launch</Translated></td>
          <td>
            <Translated vi={<><code>cudaLaunchCooperativeKernel</code>, sáu đối số</>}>
              <code>cudaLaunchCooperativeKernel</code>, six arguments
            </Translated>
          </td>
          <td>
            <Translated vi={<>cùng lời gọi đó qua <code>pmpp/vendors/hip.h</code>; bắt buộc đủ sáu</>}>
              the same call through <code>pmpp/vendors/hip.h</code>; six are required
            </Translated>
          </td>
          <td><Translated vi="không có">none</Translated></td>
        </tr>
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS} tiers={TIERS}>
      <LabsIntro>
        <Translated
          vi={
            <>
              Một lab, trong <code>labs/ch18/lab1_frontier/</code>. Nó là tùy chọn, vì chương này được rút gọn, và không thêm gì
              vào dự án xuyên suốt. Nó biến các kernel đã sửa của trang{' '}
              <a href={href('ch18', '18.6')}>18.6</a> thành mã bạn chạy trên mọi backend, và benchmark của nó đặt ra trên GPU của
              bạn câu hỏi mà trang đó để ngỏ: trong một phép tìm kiếm có nhiều mức nhỏ, bao nhiêu phần là chi phí cố định của một
              mức.
            </>
          }
        >
          One lab, in <code>labs/ch18/lab1_frontier/</code>. It is optional, as this chapter is condensed, and it adds
          nothing to the through-line project. It turns the corrected kernels of page{' '}
          <a href={href('ch18', '18.6')}>18.6</a> into code you run on every backend, and its benchmark asks the question
          that page leaves open on your own GPU: how much of a search with many small levels is the fixed cost of a level.
        </Translated>
      </LabsIntro>
      <RunningLabs example="18.1" />
      <BackendsAndEmulator />
      <VerificationStatus />

      {/* ------------------------------------------------------------ 18.1 */}
      <h2>
        <Translated vi="Lab 18.1: BFS dùng frontier với một hàng đợi riêng tư hóa">Lab 18.1: frontier BFS with a privatised queue</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Phần A, trên mọi backend, là phép tìm kiếm của trang 18.6 với mỗi mức một lần khởi chạy. Bạn viết phép thăm, một
            compare-and-swap biến <code>UNVISITED</code> thành mức hiện tại cho đúng một luồng, và kernel theo mức: mỗi đỉnh của
            frontier trước một luồng, một frontier và bộ đếm riêng cho mỗi khối trong bộ nhớ chia sẻ (shared memory), một nhánh
            tràn ra frontier chung khi frontier riêng đầy, một atomic phạm vi thiết bị cho mỗi khối để đặt chỗ khoảng của nó, và
            một lần commit trong đó các luồng cạnh nhau ghi vào các địa chỉ cạnh nhau.
          </p>
        }
      >
        <p>
          Part A, on every backend, is the search of page 18.6 with one launch per level. You write the visit, a
          compare-and-swap that turns <code>UNVISITED</code> into the current level for exactly one thread, and the
          level kernel: one thread per vertex of the previous frontier, a private frontier and counter per block in shared
          memory, an overflow path to the public frontier when the private one is full, one device-scope atomic per block
          to reserve its range, and a commit in which neighbouring threads write neighbouring addresses.
        </p>
      </Translated>
      <Source section="18.5, 18.6" pages="440–444" figs={tr('Figs. 18.14, 18.15', 'Hình 18.14, 18.15')} />
      <Translated
        vi={
          <p>
            Phần B, tùy chọn, chạy trên CUDA, HIP và các trình giả lập. Vòng lặp mức được đưa vào trong kernel: một lưới có kích
            thước sao cho mọi khối đều thường trú, mỗi luồng đi theo bước nhảy qua frontier trước, và ba rào chắn lưới ở cuối
            mỗi mức. Bạn cũng viết các dòng định kích thước của launcher, một truy vấn occupancy (mức chiếm dụng) nhân với số
            SM, và cooperative launch.
          </p>
        }
      >
        <p>
          Part B, optional, runs on CUDA, HIP and the emulators. The level loop moves into the kernel: a grid sized so that
          every block is resident, each thread striding through the previous frontier, and three grid barriers at the end
          of each level. You also write the sizing lines of the launcher, an occupancy query times the number of SMs, and
          the cooperative launch.
        </p>
      </Translated>
      <Source section="18.7" pages="445–448" figs={tr('Fig. 18.17', 'Hình 18.17')} />
      <Translated
        vi={
          <p>
            Metal không có phần B: nó không ghi nhận rào chắn lưới, cooperative launch hay truy vấn occupancy nào để định kích
            thước một lưới thường trú. Trên Mac, phép tìm kiếm giữ nguyên mỗi mức một dispatch.
          </p>
        }
      >
        <p>
          Metal has no part B: it documents no grid barrier, no cooperative launch and no occupancy query from which to size
          a resident grid. On a Mac the search stays at one dispatch per level.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch18-facts.md C1; ch04-facts C3, C5)', '2026-11-20 (notes/today/ch18-facts.md, mục C1; ch04-facts, mục C3, C5)')}>
        <Translated
          vi={<p>Không có rào chắn trên toàn lưới, cooperative launch, số lõi hay truy vấn occupancy nào được tài liệu hóa cho Metal.</p>}
        >
          <p>No grid-wide barrier, no cooperative launch and no core count or occupancy query are documented for Metal.</p>
        </Translated>
      </Today>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <Translated
              vi={
                <>
                  <code>visitVertexAtomically</code> và <code>bfsPrivatisedKernel</code> (phần A); <code>bfsCooperativeKernel</code>{' '}
                  và các dòng được đánh dấu của <code>launchBfsCooperative</code> (phần B). Launcher theo từng mức{' '}
                  <code>launchBfsPrivatised</code> đã được viết sẵn cho bạn
                </>
              }
            >
              <>
                <code>visitVertexAtomically</code> and <code>bfsPrivatisedKernel</code> (part A); <code>bfsCooperativeKernel</code>{' '}
                and the marked lines of <code>launchBfsCooperative</code> (part B). The per-level launcher{' '}
                <code>launchBfsPrivatised</code> is written for you
              </>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated
              vi={
                <>
                  <code>visit_vertex</code> và <code>bfs_privatised</code>, với <code>level</code> là{' '}
                  <code>device atomic_uint*</code> và bộ đếm riêng là một <code>threadgroup atomic_uint</code>
                </>
              }
            >
              <>
                <code>visit_vertex</code> and <code>bfs_privatised</code>, with <code>level</code> as{' '}
                <code>device atomic_uint*</code> and the private counter a <code>threadgroup atomic_uint</code>
              </>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated
              vi={
                <>
                  được cung cấp sẵn: mỗi mức một command buffer, <code>waitUntilCompleted</code>, rồi đọc số đếm. Bạn không cần
                  sửa nó
                </>
              }
            >
              <>
                provided: one command buffer per level, <code>waitUntilCompleted</code>, then a read of the count. You need not
                edit it
              </>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Chỉ đọc: <code>provided/ref_kernels.cu</code>, <code>provided/ref_kernels.metal</code> và{' '}
            <code>provided/ref_launch.mm</code>, các mốc so sánh của benchmark: <code>push</code> (Hình 18.6, mọi đỉnh ở mọi
            mức, với atomic relaxed thay cho race vô hại của trang), <code>frontier</code> (Hình 18.12 và 18.14, một atomic chung
            cho mỗi đỉnh tìm thấy) và <code>floor</code>, một kernel rỗng. Và <code>main.cpp</code>: các đồ thị, BFS tuần tự,
            các bài kiểm tra và benchmark. <code>BLOCK_DIM</code> là 128 và <code>PRIVATE_FRONTIER_CAPACITY</code> là 256, lớn
            hơn khối, nên vòng lặp commit phải chạy nhiều hơn một vòng khi một khối tìm thấy nhiều đỉnh.
          </p>
        }
      >
        <p>
          Read-only: <code>provided/ref_kernels.cu</code>, <code>provided/ref_kernels.metal</code> and{' '}
          <code>provided/ref_launch.mm</code>, the benchmark's baselines: <code>push</code> (Fig. 18.6, every vertex at
          every level, with relaxed atomics in place of the page's benign race), <code>frontier</code> (Figs. 18.12 and
          18.14, one public atomic per vertex found) and <code>floor</code>, an empty kernel. And <code>main.cpp</code>: the
          graphs, the sequential BFS, the tests and the benchmark. <code>BLOCK_DIM</code> is 128 and{' '}
          <code>PRIVATE_FRONTIER_CAPACITY</code> is 256, larger than the block, so the commit loop has to go round more than
          once when a block finds many vertices.
        </p>
      </Translated>

      <h3><Translated vi="Lab khác trang ở đâu">Where the lab differs from the page</Translated></h3>
      <Translated
        vi={
          <p>
            <b>Dòng 29 là một lần ghi atomic.</b> Khi tràn, Hình 18.15 đặt bộ đếm riêng trở lại dung lượng bằng một lần ghi
            thường trong khi các luồng khác của khối có thể đang thực hiện <code>fetch_add</code> trên nó (dòng 35 của Hình
            18.17). Đó là một race condition (tranh chấp dữ liệu), nên là hành vi không xác định, bất kể phần cứng làm gì. Hãy ghi
            qua cùng <code>atomic_ref</code> đó, với relaxed, hoặc bỏ lần ghi và kẹp số đếm bằng <code>min</code> sau rào chắn.
          </p>
        }
      >
        <p>
          <b>Line 29 is an atomic store.</b> On overflow, Fig. 18.15 puts the private counter back to the capacity with a
          plain store while other threads of the block may be doing <code>fetch_add</code> on it (line 35 of Fig. 18.17).
          That is a data race, so undefined behaviour, whatever the hardware does. Store through the same{' '}
          <code>atomic_ref</code>, relaxed, or drop the store and clamp the count with <code>min</code> after the barrier.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch18-facts.md A7, E1)', '2026-11-20 (notes/today/ch18-facts.md, mục A7, E1)')}>
        <Translated
          vi={
            <p>
              Một truy cập thường tranh chấp với một truy cập atomic là một race condition theo chính định nghĩa của NVIDIA (ch09
              A2). Việc lần ghi atomic giữ số đếm đúng là kết quả chúng tôi liệt kê mọi cách xen kẽ cho các trường hợp nhỏ, không
              phải một tuyên bố của nhà cung cấp.
            </p>
          }
        >
          <p>
            A plain access racing an atomic one is a data race by NVIDIA's own definition (ch09 A2). That the atomic store
            keeps the counts right is our enumeration of every interleaving for small cases, not a vendor statement.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            <b>UINT_MAX cho giá trị canh.</b> <code>cuda::std::numeric_limits&lt;unsigned&gt;</code> của trang gọi tên trait mà
            thiếu <code>::max()</code>. Các file <code>.cu</code> dùng <code>UINT_MAX</code> từ <code>&lt;climits&gt;</code>,
            cùng giá trị, vì các trình giả lập của khóa học không kèm <code>&lt;cuda/std/limits&gt;</code>.
          </p>
        }
      >
        <p>
          <b>UINT_MAX for the sentinel.</b> The page's <code>cuda::std::numeric_limits&lt;unsigned&gt;</code> names the
          trait without <code>::max()</code>. The <code>.cu</code> files use <code>UINT_MAX</code> from{' '}
          <code>&lt;climits&gt;</code>, the same value, because the course's emulators ship no{' '}
          <code>&lt;cuda/std/limits&gt;</code>.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>Biến chia sẻ ở đầu.</b> Hình 18.17 khai báo các biến <code>__shared__</code> bên trong vòng lặp mức. Dù sao chúng
            cũng sống bằng thời gian sống của khối; hãy khai báo chúng một lần, ở đầu, và để lần đặt lại ở đầu mỗi mức làm trống
            frontier riêng.
          </p>
        }
      >
        <p>
          <b>Shared variables at the top.</b> Fig. 18.17 declares its <code>__shared__</code> variables inside the level
          loop. They live as long as the block either way; declare them once, at the top, and let the reset at the start of
          each level empty the private frontier.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>Sáu đối số, rank 32-bit.</b> Truyền đủ sáu đối số cho <code>cudaLaunchCooperativeKernel</code>, kết thúc bằng{' '}
            <code>kernelArgs, 0, 0</code>: CUDA đặt mặc định cho hai đối số cuối, HIP thì không. Và ép kiểu{' '}
            <code>grid.thread_rank()</code> và <code>grid.num_threads()</code> sang <code>unsigned int</code> một lần, vì chúng
            là 64-bit trên CUDA và 32-bit trên HIP.
          </p>
        }
      >
        <p>
          <b>Six arguments, 32-bit ranks.</b> Pass all six arguments to <code>cudaLaunchCooperativeKernel</code>, ending{' '}
          <code>kernelArgs, 0, 0</code>: CUDA defaults the last two, HIP does not. And cast{' '}
          <code>grid.thread_rank()</code> and <code>grid.num_threads()</code> to <code>unsigned int</code> once, since they
          are 64-bit on CUDA and 32-bit on HIP.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch18-facts.md A1, A4, B1, B2)', '2026-11-20 (notes/today/ch18-facts.md, mục A1, A4, B1, B2)')}>
        <Translated
          vi={
            <p>
              Lời gọi bốn đối số chỉ biên dịch được cho CUDA. Chính ví dụ của AMD truyền <code>0</code> làm mảng đối số cho một
              kernel có tham số: đừng sao chép nó.
            </p>
          }
        >
          <p>
            The four-argument call compiles for CUDA only. AMD's own example passes <code>0</code> as the argument array of a
            kernel with parameters: do not copy it.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            <b>Word canh trên frontier chung.</b> Không đoạn mã nào kiểm tra dung lượng của frontier chung, và trang không trình
            bày lần cấp phát nào. Ở đây mỗi frontier chung chứa <code>numVertices</code> phần tử, đủ khi mỗi đỉnh được thêm một
            lần, và chương trình điều khiển (driver) đặt các word canh sau cả hai buffer và kiểm tra chúng sau mỗi phép tìm kiếm.
          </p>
        }
      >
        <p>
          <b>Guard words on the public frontier.</b> No listing checks the public frontier's capacity, and the page shows
          no allocation. Here each public frontier holds <code>numVertices</code> entries, enough when each vertex is added
          once, and the driver puts guard words after both buffers and checks them after every search.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>Con trỏ thô.</b> <code>srcPtrs</code>, <code>dst</code> và <code>numVertices</code> là các đối số riêng; struct{' '}
            <code>CSRGraph</code> của sách không bao giờ được định nghĩa. Mọi đoạn mã trong chương đều tên là{' '}
            <code>bfs_kernel</code>; các tên ở đây là của chúng tôi.
          </p>
        }
      >
        <p>
          <b>Raw pointers.</b> <code>srcPtrs</code>, <code>dst</code> and <code>numVertices</code> are separate arguments;
          the book's <code>CSRGraph</code> struct is never defined. Every listing in the chapter is called{' '}
          <code>bfs_kernel</code>; the names here are ours.
        </p>
      </Translated>

      <h3><Translated vi="Cùng các kernel đó trên các backend khác">The same kernels on other backends</Translated></h3>
      <Translated
        vi={
          <p>
            Bản thân phép tìm kiếm giống nhau trên cả ba. Điều khác là độ mạnh của compare-and-swap, cách khai báo một bộ đếm, và
            phần B có tồn tại hay không.
          </p>
        }
      >
        <p>
          The search itself is the same on all three. What differs is the compare-and-swap's strength, how a counter is
          declared, and whether part B exists at all.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ dùng CUDA. Mọi điều trong bảng này về HIP và Metal lấy từ notes/today/ch18-facts.md, kiểm tra ngày
              2026-11-20. Các nhánh HIP của lab chưa được biên dịch cho HIP hay chạy trên phần cứng AMD.
            </p>
          }
        >
          <p>
            The book uses CUDA only. Everything in this table about HIP and Metal is from notes/today/ch18-facts.md, checked
            2026-11-20. The HIP branches of the lab have not been compiled for HIP or run on AMD hardware.
          </p>
        </Translated>
      </Beyond>
      <BackendsTable />
      <Translated
        vi={
          <p>
            Trên HIP, <code>cuda::atomic_ref</code> biên dịch được qua libhipcxx, nhưng <code>compare_exchange_strong</code> của
            nó được cài đặt bằng built-in dạng yếu ở phạm vi khối và thiết bị. Đó là lý do các file <code>.cu</code> viết các lời
            gọi kiểu cũ dưới <code>#if defined(__HIPCC__)</code>, như lab 12.1. Việc GPU AMD hay Apple có bao giờ làm một
            compare-and-swap yếu thất bại giả hay không thì không được tài liệu hóa: unverified theo cả hai hướng, và đó là lý do
            phép thăm trên Metal có vòng lặp.
          </p>
        }
      >
        <p>
          On HIP, <code>cuda::atomic_ref</code> compiles through libhipcxx, but its <code>compare_exchange_strong</code> is
          implemented with the weak built-in at block and device scope. That is why the <code>.cu</code> files write the
          legacy calls under <code>#if defined(__HIPCC__)</code>, as lab 12.1 does. Whether AMD or Apple GPUs ever fail a
          weak compare-and-swap spuriously is not documented: unverified either way, which is why the Metal visit loops.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch18-facts.md B4, C4; infra-facts B1, B2)', '2026-11-20 (notes/today/ch18-facts.md, mục B4, C4; infra-facts B1, B2)')}>
        <Translated
          vi={
            <p>
              libhipcxx hạ dạng mạnh xuống <code>__hip_atomic_compare_exchange_weak</code>; <code>atomicCAS</code> của HIP là
              dạng mạnh. MSL không có dạng mạnh.
            </p>
          }
        >
          <p>
            libhipcxx lowers the strong form to <code>__hip_atomic_compare_exchange_weak</code>; HIP's{' '}
            <code>atomicCAS</code> is strong. MSL has no strong form.
          </p>
        </Translated>
      </Today>

      <h3><Translated vi="Các bài kiểm tra kiểm tra gì">What the tests check</Translated></h3>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 18.1 --backend emu</code> trước. Mọi trường hợp so sánh chính xác{' '}
            <code>level</code> với một BFS tuần tự trong <code>main.cpp</code>; các đỉnh không tới được phải giữ nguyên{' '}
            <code>UNVISITED</code>. Với phép tìm kiếm theo từng mức, kích thước frontier của mỗi mức phải bằng số đỉnh ở mức đó
            theo cách đếm tuần tự, điều này bắt được một đỉnh bị thêm hai lần và một đỉnh bị mất, và số lần khởi chạy phải khớp.
            Các word canh sau cả hai frontier chung phải còn nguyên.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 18.1 --backend emu</code> first. Every case compares <code>level</code> with a
          sequential BFS in <code>main.cpp</code> exactly; unreached vertices must stay <code>UNVISITED</code>. For the
          per-level search, each level's frontier size must equal the sequential count of vertices at that level, which
          catches a vertex added twice and a vertex lost, and the number of launches must match. The guard words after both
          public frontiers must be untouched.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Các đồ thị: đồ thị của sách từ gốc 0 và 2, phải cho ra các mức của Hình 18.4; đồ thị Bài tập 1 từ 0, theo cách chúng
            tôi đọc hình vẽ ở p. 450; một đỉnh đơn; một gốc không có cạnh ra; hai thành phần liên thông; cạnh tự vòng và cạnh
            lặp lại; một chuỗi; một lưới với 4 đỉnh kề mỗi đỉnh. Rồi đến các trường hợp được dựng để làm hỏng từng bước một.
            "Fan-out" có 128 con tìm ra 512 cháu trong một khối so với 256 ô riêng, nên nhiều luồng tràn cùng lúc và phần commit
            có nhiều phần tử hơn số luồng của khối. "Fan-out shared" khiến các con tranh nhau cùng các cháu. Một hub bậc 5,000
            tự tràn một mình. "Stale-expected trap", 0 tới 1 và 2, 1 tới 3, 2 tới 3 và 4, khiến luồng của đỉnh 2 thua đỉnh 3 mà
            vẫn phải thắng đỉnh 4. Hai đồ thị R-MAT từ seed cố định khép lại danh sách. Phần B chạy cùng các đồ thị qua{' '}
            <code>launchBfsCooperative</code>; trên trình giả lập đó là 8 khối 128 luồng, giới hạn 1,024 luồng của nó cho một
            lưới cooperative.
          </p>
        }
      >
        <p>
          The graphs: the book's graph from roots 0 and 2, which must give Fig. 18.4's levels; the Exercise 1 graph from 0,
          in our reading of the drawing on p. 450; a single vertex; a root with no out-edges; two components; self-loops
          and repeated edges; a chain; a grid with 4 neighbours per vertex. Then the cases built to break one step each.
          "Fan-out" has 128 children find 512 grandchildren in one block against 256 private slots, so many threads
          overflow at once and the commit has more entries than the block has threads. "Fan-out shared" makes children
          compete for the same grandchildren. A hub of degree 5,000 overflows alone. The "stale-expected trap", 0 to 1 and
          2, 1 to 3, 2 to 3 and 4, makes vertex 2's thread lose vertex 3 and still have to win vertex 4. Two R-MAT graphs
          from fixed seeds close the list. Part B runs the same graphs through <code>launchBfsCooperative</code>; on the
          emulator that is 8 blocks of 128 threads, its limit of 1,024 threads for a cooperative grid.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Mỗi backend thấy những lỗi khác nhau. Trình giả lập chạy từng luồng một theo ba thứ tự dưới AddressSanitizer: nó bắt
            được một giá trị expected bị dùng lại giữa các lần gọi, một phép kiểm tra dung lượng bị thiếu, một lần commit viết
            bằng <code>if</code>, và phần lớn các rào chắn bị thiếu. Nó không thấy được một compare-and-swap bị thiếu hay lần ghi
            thường của trang ở dòng 29, vì không ai bị ngắt giữa một lần kiểm tra và một lần ghi, và một số rào chắn bị thiếu tình
            cờ vẫn cho đáp án đúng dưới lịch chạy của nó. Biến thể <code>race</code> chạy các luồng đồng thời dưới ThreadSanitizer
            và nêu tên cả hai dòng mã nguồn của mỗi cặp không đồng bộ: hãy chạy{' '}
            <code>python3 labs/pmpp.py run 18.1 --backend race</code> trước khi bạn tin một lần đạt. Một GPU chỉ để lộ bản trùng
            hay đỉnh bị mất khi lịch chạy của nó tạo ra chúng, và một rào chắn bị thiếu có thể vẫn đạt ở đó.
          </p>
        }
      >
        <p>
          Each backend sees different mistakes. The emulator runs one thread at a time in three orders under
          AddressSanitizer: it catches an expected value reused across calls, a missing capacity check, a commit written as
          an <code>if</code>, and most missing barriers. It cannot see a missing compare-and-swap or the page's plain store
          on line 29, because nobody is interrupted between a check and a store, and some missing barriers happen to give
          the right answer under its schedule. The <code>race</code> flavour runs the threads concurrently under
          ThreadSanitizer and names both source lines of each unsynchronised pair: run{' '}
          <code>python3 labs/pmpp.py run 18.1 --backend race</code> before you trust a pass. A GPU shows duplicates or lost
          vertices only when its schedule produces them, and a missing barrier can pass there.
        </p>
      </Translated>
      <RaceStatus lab={l1} />
      <Translated
        vi={
          <p>
            Trên Metal, phần A chạy dưới dạng mỗi mức một compute dispatch, mỗi dispatch trong command buffer riêng, được chờ
            trước khi đọc số đếm về. Các dispatch chạy theo thứ tự và mỗi dispatch thấy các lần ghi của dispatch trước, vì
            runtime của khóa học dùng một <code>MTLCommandQueue</code> với các buffer được theo dõi và một compute encoder thường.
            Các bài kiểm tra của phần B bị bỏ qua ở đó kèm một thông báo, như trên một thiết bị CUDA hay HIP không báo hỗ trợ
            cooperative launch, và khi <code>launchBfsCooperative</code> vẫn trả về <code>BFS_COOP_NOT_WRITTEN</code>.
          </p>
        }
      >
        <p>
          On Metal, part A runs as one compute dispatch per level, each in its own command buffer, waited on before the
          count is read back. The dispatches run in order and each sees the previous one's writes, because the course's
          runtime uses an <code>MTLCommandQueue</code> with tracked buffers and a plain compute encoder. Part B's tests are
          skipped there with a message, as they are on a CUDA or HIP device that reports no cooperative launch, and while{' '}
          <code>launchBfsCooperative</code> still returns <code>BFS_COOP_NOT_WRITTEN</code>.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch18-facts.md C2)', '2026-11-20 (notes/today/ch18-facts.md, mục C2)')}>
        <Translated
          vi={
            <p>
              Thứ tự dispatch tuần tự và việc theo dõi hazard được bảo đảm trên một <code>MTLCommandQueue</code>; một{' '}
              <code>MTL4CommandQueue</code> không theo dõi hazard nào.
            </p>
          }
        >
          <p>
            Serial dispatch order and hazard tracking hold on an <code>MTLCommandQueue</code>; an{' '}
            <code>MTL4CommandQueue</code> tracks no hazards.
          </p>
        </Translated>
      </Today>

      <h3><Translated vi="Thí nghiệm">The experiment</Translated></h3>
      <Translated
        vi={
          <p>
            Hãy làm điều này khi các kernel của bạn đã đạt, và trước khi chạy <code>--bench</code>. Benchmark dựng hai đồ thị với
            số cạnh xấp xỉ nhau. <code>grid</code> là lưới 1024 × 1024 với 4 đỉnh kề mỗi đỉnh, tìm kiếm từ một góc: 2,046 mức,
            không mức nào lớn hơn 1,024 đỉnh, một mạng đường bộ thu nhỏ. <code>rmat</code> là một đồ thị R-MAT từ một seed cố
            định: một vài mức với frontier lớn, một mạng xã hội thu nhỏ. Với mỗi đồ thị, nó đo thời gian của cả phép tìm kiếm,
            lấy kết quả tốt nhất trong 5 lần sau một lần chạy không đo, với <code>level</code> được đặt lại bên ngoài phần đo,
            và kiểm tra mọi kết quả trước khi báo một hàng: <code>push</code>, <code>frontier</code>, <code>privatised</code>{' '}
            của bạn, <code>cooperative</code> của bạn (chỉ CUDA và HIP) và <code>floor</code>.
          </p>
        }
      >
        <p>
          Do this once your kernels pass, and before you run <code>--bench</code>. The benchmark builds two graphs with
          about the same number of edges. <code>grid</code> is 1024 × 1024 with 4 neighbours per vertex, searched from a
          corner: 2,046 levels, none larger than 1,024 vertices, a road network in miniature. <code>rmat</code> is an R-MAT
          graph from a fixed seed: a handful of levels with large frontiers, a social network in miniature. For each it
          times whole searches, best of 5 after an untimed run, with <code>level</code> reset outside the timing, and checks
          every result before it reports a row: <code>push</code>, <code>frontier</code>, your <code>privatised</code>, your{' '}
          <code>cooperative</code> (CUDA and HIP only) and <code>floor</code>.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Bộ sinh R-MAT và các tham số của nó là của chúng tôi và nằm trong main.cpp; floor cũng vậy. Sách không đưa ra phép
              đo nào cho bất kỳ điều gì trong số này.
            </p>
          }
        >
          <p>
            The R-MAT generator and its parameters are ours and are in main.cpp; so is the floor. The book gives no
            measurement for any of this.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            <code>floor</code> là vòng lặp host của phép tìm kiếm theo từng mức bọc quanh một kernel rỗng, mỗi mức của đồ thị đó
            một lần: đặt bộ đếm về 0, khởi chạy, đọc 4 byte về; trên Metal là một dispatch rỗng, một lần chờ và một lần đọc. Đó
            là chi phí cố định của một mức không có phép tìm kiếm nào bên trong. Sách nói rằng với frontier nhỏ chi phí này có
            thể chiếm phần lớn, còn với frontier lớn thì không đáng kể.
          </p>
        }
      >
        <p>
          <code>floor</code> is the host loop of the per-level search wrapped around an empty kernel, once per level of
          that graph: zero the counter, launch, read 4 bytes back; on Metal, an empty dispatch, a wait and a read. It is the
          fixed cost of a level with no search in it. The book says that with small frontiers this cost may dominate, and
          with large ones it is negligible.
        </p>
      </Translated>
      <Source section="18.7" pages="445" />
      <Translated
        vi={
          <p>
            Riêng tư hóa (privatization) nhằm gỡ tranh chấp khỏi bộ đếm chung duy nhất, nơi mọi luồng nối vào của kernel chưa
            riêng tư hóa đều chạm tới. Trình biên dịch có thể đã gộp các atomic đó trong một warp, điều sẽ làm giảm phần lợi mà
            riêng tư hóa mang lại. Hãy ghi lại ba con số.
          </p>
        }
      >
        <p>
          Privatisation is meant to take contention off the one public counter, which every appending thread of the
          unprivatised kernel hits. The compiler may already combine those atomics within a warp, which would shrink what
          privatisation gains. Write down three numbers.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch18-facts.md A8)', '2026-11-20 (notes/today/ch18-facts.md, mục A8)')}>
        <Translated
          vi={
            <p>
              NVIDIA chỉ nêu việc tự động gộp atomic theo warp trong một bài blog năm 2017; trình biên dịch của AMD làm điều đó
              cho các địa chỉ đồng nhất; Apple không ghi nhận gì.
            </p>
          }
        >
          <p>
            NVIDIA states automatic warp aggregation only in a 2017 blog post; AMD's compiler does it for uniform addresses;
            Apple documents nothing.
          </p>
        </Translated>
      </Today>
      <LabPrediction
        id="ch18/labs/p-lab1-floor"
        prompt={
          <Translated
            vi={
              <>
                Trên <code>grid</code>, tỉ lệ thời gian phép tìm kiếm riêng tư hóa của bạn dành cho các mức rỗng: thời gian{' '}
                <code>floor</code> chia cho thời gian <code>privatised</code>.
              </>
            }
          >
            <>
              On <code>grid</code>, the share of your privatised search's time that the empty levels account for:{' '}
              <code>floor</code> time divided by <code>privatised</code> time.
            </>
          </Translated>
        }
        unit={tr('(floor time / privatised time)', '(thời gian floor / thời gian privatised)')}
        measure={floorShare}
        missing={MISSING}
        reading={
          <Translated
            vi={
              <>
                Gần 1 nghĩa là phép tìm kiếm chủ yếu là chi phí của việc có một mức, đúng như nhận định ở p. 445 về mạng đường
                bộ. Cả hai hàng chạy cùng số mức, nên khác biệt giữa chúng là công việc riêng của phép tìm kiếm.
              </>
            }
          >
            <>
              Close to 1 means the search is mostly the cost of having a level at all, which is the p. 445 claim about
              road networks. Both rows run the same number of levels, so the difference between them is the search's own
              work.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch18/labs/p-lab1-rmat"
        prompt={
          <Translated
            vi={
              <>
                Trên <code>rmat</code>, <code>privatised</code> nhanh hơn <code>frontier</code> bao nhiêu lần: thời gian{' '}
                <code>frontier</code> chia cho thời gian <code>privatised</code>.
              </>
            }
          >
            <>
              On <code>rmat</code>, how many times faster <code>privatised</code> is than <code>frontier</code>:{' '}
              <code>frontier</code> time divided by <code>privatised</code> time.
            </>
          </Translated>
        }
        unit={tr('(frontier time / privatised time)', '(thời gian frontier / thời gian privatised)')}
        measure={rmatRelief}
        missing={MISSING}
        reading={
          <Translated
            vi={
              <>
                Trên 1, các hàng đợi riêng đã bù được chi phí rào chắn và lưu lượng bộ nhớ chia sẻ của chúng bằng cách gỡ tranh
                chấp khỏi bộ đếm chung duy nhất. Dưới 1, chúng không bù được, trên đồ thị này và thiết bị này.
              </>
            }
          >
            <>
              Above 1, the private queues paid for their barriers and shared-memory traffic by taking contention off the one
              public counter. Below 1, they did not, on this graph and this device.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch18/labs/p-lab1-coop"
        prompt={
          <Translated
            vi={
              <>
                Trên một thiết bị CUDA hoặc HIP, trên <code>grid</code>, <code>cooperative</code> nhanh hơn{' '}
                <code>privatised</code> bao nhiêu lần. Rồi nói, không cần ghi lại, bạn dự đoán gì trên <code>rmat</code>.
              </>
            }
          >
            <>
              On a CUDA or HIP device, on <code>grid</code>, how many times faster <code>cooperative</code> is than{' '}
              <code>privatised</code>. Then say, without recording it, what you expect on <code>rmat</code>.
            </>
          </Translated>
        }
        unit={tr('(privatised time / cooperative time)', '(thời gian privatised / thời gian cooperative)')}
        measure={coopGain}
        missing={MISSING_COOP}
        reading={
          <Translated
            vi={
              <>
                Lưới cooperative bị giới hạn ở mức thường trú, nên trên các frontier lớn của <code>rmat</code> mỗi luồng phải đi
                theo bước nhảy qua nhiều đỉnh, và ba rào chắn lưới mỗi mức thay cho một lần khởi chạy. Chi phí nào thắng là điều
                mà hai đồ thị tách bạch.
              </>
            }
          >
            <>
              The cooperative grid is capped at what is resident, so on <code>rmat</code>'s large frontiers each thread
              strides through many vertices, and three grid barriers per level replace one launch. Which of those costs
              wins is what the two graphs separate.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <LabResults
            lab={l1}
            cols={COLS.map((c) => ({
              ...c,
              head:
                ({
                  Vertices: 'Số đỉnh',
                  Edges: 'Số cạnh',
                  Levels: 'Số mức',
                  'µs per level': 'µs mỗi mức',
                  'Million edges/s': 'Triệu cạnh/s',
                } as Record<string, string>)[c.head] ?? c.head,
            }))}
          />
        }
      >
        <LabResults lab={l1} cols={COLS} />
      </Translated>
      <Translated
        vi={
          <p>
            Cách đọc bảng. "µs mỗi mức" chia mỗi thời gian cho số lần khởi chạy, tức số mức sau gốc cộng lần khởi chạy cuối
            không tìm thấy gì. "Triệu cạnh/s" đếm các cạnh ra của mọi đỉnh tới được, thứ mà mọi phép tìm kiếm ở đây đều xem xét.
            Thời gian của cùng một phép tìm kiếm thay đổi từ lần chạy này sang lần khác, nên hãy chạy benchmark hai lần trước khi
            tin một khác biệt nhỏ giữa hai phép tìm kiếm.
          </p>
        }
      >
        <p>
          How to read the table. "µs per level" divides each time by the number of launches, the levels after the root
          plus the last launch, which finds nothing. "Million edges/s" counts the out-edges of every reached vertex, which
          every search here examines. Times for the same search vary from one run to the next, so run the benchmark twice
          before you trust a small difference between two searches.
        </p>
      </Translated>

      <h3><Translated vi="Điều này nằm ở đâu dưới những gì bạn đã biết">Where this sits under what you know</Translated></h3>
      <Translated
        vi={
          <p>
            Sinh token có hình dạng của đồ thị lưới: nhiều bước ngắn, mỗi bước do host khởi chạy, mỗi bước làm ít việc. Hàng
            floor là ba chi phí cho mỗi mức: một lần khởi chạy, một lần memset bộ đếm, và một lần đọc chặn số đếm về host. Một
            CUDA graph trả phần khởi chạy một lần cho cả graph thay vì mỗi lần khởi chạy; nó không loại bỏ lần đọc về, chính là
            thứ ngăn một vòng lặp host như thế này được capture. Loại bỏ nó cần vòng lặp mức nằm trên device: một node điều kiện
            WHILE (CUDA 12.3 trở lên) hoặc cooperative kernel của lab 18.1. Và phép nén gọn (compaction) trong kernel mức của
            bạn, chiếm một ô bằng một atomic rồi lưu, chính là hình dạng của việc gom các token sống sót sau một lần cắt top-p.
          </p>
        }
      >
        <p>
          Token generation is the grid graph's shape: many short steps, each launched by the host, each doing little work.
          The floor row is three costs per level: a launch, a memset of the counter, and a blocking read of the count back
          to the host. A CUDA graph pays the launch part once for the whole graph instead of per launch; it does not
          remove the read-back, which is what stops a host loop like this from being captured at all. Taking that out
          needs the level loop on the device: a conditional WHILE node (CUDA 12.3 and later) or the cooperative kernel
          of lab 18.1. And the compaction in your level kernel, claim a slot with an atomic and store, is the shape of
          collecting the survivors of a top-p cut.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cách liên hệ với decode và lấy mẫu là của chúng tôi, không phải của sách; trang 18.6 đưa ra câu trả lời của chính
              llama.cpp và PyTorch cho chi phí khởi chạy. Graph và node WHILE làm gì là theo mô tả của NVIDIA, còn ảnh hưởng của
              lần đọc về lên việc capture là theo PyTorch (notes/today/ch18-facts.md A10, D5, kiểm tra ngày 2026-11-20).
            </p>
          }
        >
          <p>
            The decoding and sampling readings are ours, not the book's; page 18.6 gives llama.cpp's and PyTorch's own
            answers to launch overhead. What graphs and WHILE nodes do is NVIDIA's account, and the read-back's effect
            on capture PyTorch's (notes/today/ch18-facts.md A10, D5, checked 2026-11-20).
          </p>
        </Translated>
      </Beyond>

      <h3><Translated vi="Đi xa hơn">Go further</Translated></h3>
      <Translated
        vi={
          <p>
            Bài tập 3 yêu cầu Hình 18.17 với một rào chắn lưới mỗi mức: mỗi mức một bộ đếm, tất cả được đặt về 0 trước lần khởi
            chạy. Trang 18.6 đưa ra một đáp án, đã kiểm tra trên trình giả lập. Hãy viết nó vào phần B của bạn và đo thời gian so
            với phiên bản ba rào chắn của bạn trên <code>grid</code>. Và đổi lần ghi ở dòng 29 của bạn thành phép kẹp giá trị sau
            rào chắn, chạy <code>race</code> trên cả hai, rồi đặt lại lần ghi thường của sách và đọc xem ThreadSanitizer nói gì.
          </p>
        }
      >
        <p>
          Exercise 3 asks for Fig. 18.17 with one grid barrier per level: a counter per level, all zeroed before the launch.
          Page 18.6 shows one answer, checked on the emulator. Write it into your part B and time it against your
          three-barrier version on <code>grid</code>. And swap your line-29 store for the clamp after the barrier, run{' '}
          <code>race</code> on both, then put the book's plain store back and read what ThreadSanitizer says.
        </p>
      </Translated>
      <Source section="18.10" pages="450" />
      <ReferenceSolution lab={l1} />
    </LabsPage>
  );
}
