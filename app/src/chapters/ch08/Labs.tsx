import type { Question } from '../../components/Ask';
import { Aside, Source, Today, type Tier } from '../../components/notes';
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
  preferredBench,
  rawCol,
  useLabs,
  type BenchSet,
} from '../../components/LabKit';
import { Translated, tr } from '../../lib/i18n';
import { fmt, num, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 8's own content.
// No question on this page: the lab is the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch08', [{ num: '8.1', dir: 'lab1_stencil3d', name: '3D stencil', vi: { name: 'stencil 3D' } }]);
const [l1] = LABS;

/** Identifier colours in the reference solution: where each value lives. */
const TIERS: Record<string, Tier> = {
  in: 'global',
  out: 'global',
  inCurr_s: 'shared',
  inPrev: 'reg',
  inCurr: 'reg',
  inNext: 'reg',
  iStart: 'reg',
  i: 'reg',
  j: 'reg',
  k: 'reg',
};

// ---------------------------------------------------------------- what each prediction is measured against
/** The book's configuration: 32 × 32 blocks, a slab of 30 output planes. */
const BOOK_TILE = 32;
const BOOK_DEPTH = BOOK_TILE - 2;
const bookRegtile = (s: BenchSet) => benchRow(s, 'regtile', { tile: BOOK_TILE, depth: BOOK_DEPTH });
const ms = (r: BenchRow | undefined) => {
  const v = num(r?.seconds);
  return v ? `${fmt(v * 1000)} ms` : '?'; // i18n-ignore: unit
};

/** Register-tiled at the book's slab over the basic kernel. */
const overBasic = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const r = bookRegtile(s);
    const v = num(r?.speedup_vs_basic);
    if (!r || !v) return null;
    return {
      value: v,
      detail: tr(
        `${ms(r)} against ${ms(benchRow(s, 'basic'))} for the basic kernel, grid side ${String(r.n)}`,
        `${ms(r)} so với ${ms(benchRow(s, 'basic'))} của kernel cơ bản, cạnh lưới ${String(r.n)}`,
      ),
    };
  });

/** Register-tiled at the book's slab over the provided three-shared-plane kernel, same tile. */
const overPlanes = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const r = bookRegtile(s);
    const v = num(r?.speedup_vs_planes);
    if (!r || !v) return null;
    return {
      value: v,
      detail: tr(
        `${ms(r)} against ${ms(benchRow(s, 'planes', { tile: BOOK_TILE }))} for the planes kernel, grid side ${String(r.n)}`,
        `${ms(r)} so với ${ms(benchRow(s, 'planes', { tile: BOOK_TILE }))} của kernel ba mặt phẳng, cạnh lưới ${String(r.n)}`,
      ),
    };
  });

/** The slab depth whose register-tiled run at T = 32 was fastest. */
const fastestDepth = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const rows = s.rows
      .filter((r) => r.name === 'regtile' && num(r.tile) === BOOK_TILE && num(r.seconds) && num(r.depth))
      .sort((a, b) => num(a.depth)! - num(b.depth)!);
    if (rows.length < 2) return null;
    const best = rows.reduce((a, b) => (num(b.seconds)! < num(a.seconds)! ? b : a));
    return {
      value: num(best.depth)!,
      detail: rows
        .map((r) =>
          tr(
            `depth ${String(r.depth)}${r.depth === r.n ? ' (one slab)' : ''}: ${ms(r)}`,
            `độ sâu ${String(r.depth)}${r.depth === r.n ? ' (một tấm)' : ''}: ${ms(r)}`,
          ),
        )
        .join(', '),
    };
  });

const MISSING = (
  <Translated
    vi={
      <>
        Phần này cần hàng của kernel phân ô thanh ghi ở T = 32 và độ sâu 30 của sách. Hãy chạy{' '}
        <code>python3 labs/pmpp.py run 8.1 --bench</code> trên một máy có GPU khi kernel của bạn đã qua (hoặc với{' '}
        <code>--solution</code> để đo thời gian bản tham chiếu). Trên Metal, T = 32 bị bỏ qua ở nơi pipeline cho phép ít hơn
        1024 luồng; các giới hạn bên dưới cho biết điều đó có xảy ra hay không.
      </>
    }
  >
    <>
      This needs the register-tiled kernel's row at T = 32 and the book's depth of 30. Run{' '}
      <code>python3 labs/pmpp.py run 8.1 --bench</code> on a machine with a GPU once your kernel passes (or with{' '}
      <code>--solution</code> to time the reference). On Metal, T = 32 is skipped where the pipeline allows fewer than
      1024 threads; the limits below say whether that happened.
    </>
  </Translated>
);

// ---------------------------------------------------------------- results table
const variant = (r: BenchRow): string =>
  r.name === 'basic'
    ? tr('basic, 8 × 8 × 8 blocks', 'cơ bản, khối 8 × 8 × 8')
    : r.name === 'cube'
      ? tr(`cube, T = ${String(r.tile)} (provided)`, `lập phương, T = ${String(r.tile)} (cung cấp sẵn)`)
      : r.name === 'planes'
        ? tr(`three shared planes, T = ${String(r.tile)} (provided)`, `ba mặt phẳng chia sẻ, T = ${String(r.tile)} (cung cấp sẵn)`)
        : r.name === 'regtile'
          ? tr(`register-tiled, T = ${String(r.tile)}`, `phân ô thanh ghi, T = ${String(r.tile)}`)
          : r.name;
const depthCell = (r: BenchRow): string => {
  const d = num(r.depth);
  if (d == null) return '';
  if (d === num(r.n)) return tr(`${d} (one slab)`, `${d} (một tấm)`);
  if (d === (num(r.tile) ?? 0) - 2) return tr(`${d} (book's)`, `${d} (của sách)`);
  return String(d);
};

/** The thread limits lab 8.1 recorded: on Metal each belongs to a compiled pipeline. */
function Limits() {
  const { reports, loading } = useLabs();
  const sets = preferredBench(reports, l1).filter((s) => s.rows.some((r) => r.name === 'limits'));
  if (loading)
    return (
      <p className="muted">
        <Translated vi="Đang đọc báo cáo.">Reading reports.</Translated>
      </p>
    );
  if (!sets.length)
    return (
      <p className="muted" data-limits="none">
        <Translated vi="Máy này chưa ghi giới hạn nào. Benchmark ghi chúng, mỗi kernel và mỗi độ rộng ô một giới hạn.">
          No limits recorded on this machine yet. The benchmark writes them, one per kernel and tile width.
        </Translated>
      </p>
    );
  return (
    <ul data-limits="list">
      {sets.map((s) => {
        const rows = s.rows.filter((r) => r.name === 'limits');
        const cube = rows.find((r) => r.kernel === 'cube');
        const reg = rows.filter((r) => r.kernel !== 'cube').sort((a, b) => (num(a.tile) ?? 0) - (num(b.tile) ?? 0));
        const parts = [
          cube
            ? tr(`cube kernel (512 threads): ${String(cube.max_threads)}`, `kernel lập phương (512 luồng): ${String(cube.max_threads)}`)
            : '',
          ...reg.map((r) =>
            tr(
              `register-tiled at T = ${String(r.tile)} (${(num(r.tile) ?? 0) ** 2} threads): ${String(r.max_threads)}`,
              `phân ô thanh ghi ở T = ${String(r.tile)} (${(num(r.tile) ?? 0) ** 2} luồng): ${String(r.max_threads)}`,
            ),
          ),
        ].filter(Boolean);
        return (
          <li key={s.backend} data-limits={s.backend}>
            <Translated
              vi={
                <>
                  {s.backend} ({s.device}), lần chạy ngày {s.date}: {parts.join('; ')}.
                </>
              }
            >
              {s.backend} ({s.device}), run of {s.date}: {parts.join('; ')}.
            </Translated>
          </li>
        );
      })}
    </ul>
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
              Một lab, trong <code>labs/ch08/lab1_stencil3d/</code>. Lab này là tùy chọn, vì chương này được rút gọn, và nó
              không thêm gì vào dự án xuyên suốt: nó xây kernel đầu tiên và kernel cuối cùng của chương và đo mô hình mà{' '}
              <a href={href('ch08', '8.4')}>8.4</a> và <a href={href('ch08', '8.5')}>8.5</a> dùng để lập luận.
            </>
          }
        >
          One lab, in <code>labs/ch08/lab1_stencil3d/</code>. It is optional, as this chapter is condensed, and it adds
          nothing to the through-line project: it builds the first and the last kernel of the chapter and measures the
          model that <a href={href('ch08', '8.4')}>8.4</a> and <a href={href('ch08', '8.5')}>8.5</a> argue from.
        </Translated>
      </LabsIntro>
      <RunningLabs
        example="8.1"
        alsoShownIn={
          <Translated
            vi={
              <>
                các tùy chọn giới hạn số luồng của các trình khám phá ở <a href={href('ch08', '8.4')}>8.4</a> và{' '}
                <a href={href('ch08', '8.5')}>8.5</a>
              </>
            }
          >
            <>
              the thread-limit options of the explorers on <a href={href('ch08', '8.4')}>8.4</a> and{' '}
              <a href={href('ch08', '8.5')}>8.5</a>
            </>
          </Translated>
        }
      />
      <BackendsAndEmulator />
      <VerificationStatus />

      {/* ------------------------------------------------------------ 8.1 */}
      <h2>
        <Translated vi="Lab 8.1: một luồng cho mỗi điểm, rồi tiến theo z với thanh ghi">
          Lab 8.1: one thread per point, then marching in z with registers
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một lượt quét stencil 7 điểm trên một lưới N × N × N số float, theo thứ tự hàng (row-major) với x thay đổi nhanh
            nhất. Mỗi điểm bên trong của <code>out</code> trở thành tổng có trọng số của cùng điểm đó trong <code>in</code> và
            sáu láng giềng theo trục của nó; lớp biên giữ nguyên giá trị đầu vào. Bạn viết hai kernel, bằng CUDA và bằng Metal.
            Kernel thứ nhất là kernel cơ bản của Hình 8.6: một luồng (thread) cho mỗi điểm, mọi toán hạng lấy từ bộ nhớ toàn cục
            (global memory), khối 8 × 8 × 8 như trong bài tập 1b. Kernel thứ hai là Hình 8.12, đã sửa: một khối (block){' '}
            <code>TILE_WIDTH</code> × <code>TILE_WIDTH</code> luồng, tức một mặt phẳng x-y của ô đầu vào, tiến qua một tấm các
            mặt phẳng đầu ra, giữ giá trị trước, hiện tại và kế tiếp của mỗi luồng trong thanh ghi (register), và chỉ sao lại
            mặt phẳng hiện tại trong bộ nhớ chia sẻ (shared memory).
          </p>
        }
      >
        <p>
          A 7-point stencil sweep over an N × N × N grid of floats, row-major with x fastest. Every interior point of{' '}
          <code>out</code> becomes a weighted sum of the same point of <code>in</code> and its six axis neighbours; the
          boundary layer keeps its input values. You write two kernels, in CUDA and in Metal. The first is the basic kernel
          of Fig. 8.6: one thread per point, every operand from global memory, 8 × 8 × 8 blocks as in exercise 1b. The
          second is Fig. 8.12, corrected: a block of <code>TILE_WIDTH</code> × <code>TILE_WIDTH</code> threads, one x-y
          plane of the input tile, marches through a slab of output planes, keeps each thread's previous, current and next
          values in registers, and mirrors only the current plane in shared memory.
        </p>
      </Translated>
      <Source section="8.2 to 8.6" pages="187–198" figs={tr('Figs. 8.6, 8.12; exercise 1b', 'Hình 8.6, 8.12; bài tập 1b')} />
      <Translated
        vi={
          <p>
            Hai kernel đối chứng được cung cấp sẵn, đã sửa và được kiểm thử bởi cùng chương trình điều khiển (driver): kernel
            phân ô lập phương của Hình 8.8 (khối 8 × 8 × 8) và kernel ba mặt phẳng chia sẻ của Hình 8.10, trong{' '}
            <code>labs/ch08/shared/</code>. Bạn không sửa chúng; <a href={href('ch08', '8.4')}>8.4</a> và{' '}
            <a href={href('ch08', '8.5')}>8.5</a> trình bày chúng. Có chúng, benchmark đo thời gian cả bốn kernel của chương trên
            cùng một lưới.
          </p>
        }
      >
        <p>
          Two comparison kernels are provided, corrected and tested by the same driver: the cube-tiled kernel of Fig. 8.8
          (8 × 8 × 8 blocks) and the three-shared-plane kernel of Fig. 8.10, in <code>labs/ch08/shared/</code>. You do
          not edit them; <a href={href('ch08', '8.4')}>8.4</a> and <a href={href('ch08', '8.5')}>8.5</a> show them. With
          them the benchmark times all four of the chapter's kernels on one grid.
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
                  <code>stencilBasicKernel</code>, <code>stencilRegTiledKernel</code> và phần khởi chạy của chúng,{' '}
                  <code>launchStencilBasic</code> và <code>launchStencilRegTiled</code>
                </>
              }
            >
              <>
                <code>stencilBasicKernel</code>, <code>stencilRegTiledKernel</code> and their launches,{' '}
                <code>launchStencilBasic</code> and <code>launchStencilRegTiled</code>
              </>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated
              vi={
                <>
                  <code>stencil_basic</code> và <code>stencil_regtile</code>; mọi thuộc tính vị trí đều là một{' '}
                  <code>uint3</code>
                </>
              }
            >
              <>
                <code>stencil_basic</code> and <code>stencil_regtile</code>; every position attribute is a{' '}
                <code>uint3</code>
              </>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated
              vi={
                <>
                  hai lệnh dispatch, trong <code>encodeStencilBasic</code> và <code>encodeStencilRegTiled</code>
                </>
              }
            >
              <>
                the two dispatches, in <code>encodeStencilBasic</code> and <code>encodeStencilRegTiled</code>
              </>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Độ sâu của tấm là một tham số lúc chạy, <code>depth</code>, trong khi sách cố định nó ở{' '}
            <code>OUT_TILE_DIM</code>: với <code>depth = TILE_WIDTH − 2</code> kernel của bạn chính là kernel của sách, và
            benchmark có thể thử các giá trị khác. Lưới gồm <code>ceil(N / (TILE_WIDTH − 2))</code> khối theo x và y và{' '}
            <code>ceil(N / depth)</code> theo z, và tấm z bắt đầu ở mặt phẳng <code>z · depth</code>. Bản build định nghĩa{' '}
            <code>TILE_WIDTH</code>: mặc định là 32, và thêm 16 và 8 cho benchmark, nên không chỗ nào được giả định 32.
          </p>
        }
      >
        <p>
          The slab depth is a run-time argument, <code>depth</code>, where the book fixes it at{' '}
          <code>OUT_TILE_DIM</code>: with <code>depth = TILE_WIDTH − 2</code> your kernel is the book's, and the benchmark
          can try others. The grid is <code>ceil(N / (TILE_WIDTH − 2))</code> blocks in x and y and{' '}
          <code>ceil(N / depth)</code> in z, and slab z starts at plane <code>z · depth</code>. The build defines{' '}
          <code>TILE_WIDTH</code>: 32 by default, and 16 and 8 as well for the benchmark, so nothing may assume 32.
        </p>
      </Translated>
      <Translated vi={<p>Ba chỗ lab khác với các listing in trong sách, chỗ nào cũng được kiểm thử:</p>}>
        <p>Three places where the lab differs from the printed listings, each tested:</p>
      </Translated>
      <ul>
        <li>
          <Translated
            vi={
              <>
                <code>c3</code> nhân với láng giềng y − 1 và <code>c4</code> với láng giềng y + 1, như trong Hình 8.6 và 8.8.
                Hình 8.10 và 8.12 in cặp này theo chiều ngược lại. Các bài kiểm thử dùng bảy trọng số khác nhau, nên cách ghép
                cặp như bản in sẽ thất bại.
              </>
            }
          >
            <code>c3</code> multiplies the y − 1 neighbour and <code>c4</code> the y + 1 neighbour, as in Figs. 8.6 and
            8.8. Figs. 8.10 and 8.12 print the pair the other way round. The tests use seven different weights, so the
            printed pairing fails.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Dòng cuối trong vòng lặp của Hình 8.12 lưu <code>inNext_s</code>, thứ mà kernel đó không có. Giá trị cần lưu là
                thanh ghi <code>inNext</code>.
              </>
            }
          >
            The last line of Fig. 8.12's loop stores <code>inNext_s</code>, which that kernel does not have. The value to
            store is the register <code>inNext</code>.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Các listing dùng <code>c0</code> đến <code>c6</code> mà không khai báo chúng. Ở đây chúng được truyền vào như
                một struct theo giá trị, <code>StencilCoeffs</code> trong <code>stencil.h</code>.
              </>
            }
          >
            The listings use <code>c0</code> to <code>c6</code> without declaring them. Here they arrive as one
            by-value struct, <code>StencilCoeffs</code> in <code>stencil.h</code>.
          </Translated>
        </li>
      </ul>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 8.1</code>, bắt đầu với <code>--backend emu</code>. Trước mỗi lần khởi chạy,
            chương trình điều khiển chép <code>in</code> vào <code>out</code> và ghi một dấu mốc sau phần cuối của{' '}
            <code>out</code>, rồi so từng điểm với một bản tham chiếu trên CPU trong một dung sai tỉ lệ với độ lớn bảy số hạng
            của nó. Vì vậy một điểm bên trong chưa được ghi, một điểm biên bị thay đổi và một lần ghi vượt quá phần cuối đều
            được báo cáo đích danh. Trên trình giả lập (emulator), các cạnh lưới là 3, 4, 5, 33, 34 và 63, ở độ rộng ô T = 32,
            với độ sâu tấm 1, 7, T − 2 và cả cột, theo ba thứ tự luồng; mất vài phút, lưới nhỏ nhất chạy trước. Trên thiết bị,
            các cạnh là 3, 34, 65, 130 và 257 với cùng các độ sâu, và Metal còn chạy thêm độ rộng ô 8 và 16. Các kernel cung
            cấp sẵn được kiểm thử bằng cùng các trường hợp.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 8.1</code>, starting with <code>--backend emu</code>. Before every launch the
          driver copies <code>in</code> into <code>out</code> and writes a marker after the end of <code>out</code>, then
          compares every point with a CPU reference within a tolerance scaled to the size of its seven terms. So an
          unwritten interior point, a changed boundary point and a write past the end are each reported by name. On the
          emulator the grid sides are 3, 4, 5, 33, 34 and 63, at tile width T = 32, with slab depths of 1, 7, T − 2 and the
          whole column, under three thread orders; it takes a few minutes, smallest grids first. On a device the sides are 3,
          34, 65, 130 and 257 with the same depths, and Metal also runs tile widths 8 and 16. The provided kernels are
          tested by the same cases.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Kernel của bạn chia sẻ <code>inCurr_s</code> giữa các luồng và bảo vệ nó bằng hai rào chắn (barrier), nên lab cũng
            được build cho biến thể race của trình giả lập: <code>python3 labs/pmpp.py run 8.1 --backend race</code> chạy đồng
            thời các luồng của một khối dưới ThreadSanitizer, trên các cạnh lưới 3, 4, 5 và 34 ở T = 32. Trong khi trình giả
            lập thường chỉ cho thấy một rào chắn bị thiếu qua các giá trị sai, biến thể race chỉ ra đích danh hai dòng không
            được sắp thứ tự: khi bỏ rào chắn 2 khỏi lời giải tham chiếu, nó báo lệnh lưu mặt phẳng hiện tại mới vào{' '}
            <code>inCurr_s</code> xung đột với các lần đọc nó của láng giềng. Một lần chạy sạch là bằng chứng, không phải chứng
            minh: nó chỉ thấy những lịch thực thi đã xảy ra.
          </p>
        }
      >
        <p>
          Your kernel shares <code>inCurr_s</code> between threads and guards it with two barriers, so the lab is also
          built for the emulator's race flavour: <code>python3 labs/pmpp.py run 8.1 --backend race</code> runs the threads
          of a block concurrently under ThreadSanitizer, on grid sides 3, 4, 5 and 34 at T = 32. Where the plain emulator
          shows a missing barrier only as wrong values, the race flavour names the two lines that are not ordered: with
          barrier 2 removed from the reference solution it reported the store of the new current plane into{' '}
          <code>inCurr_s</code> against the neighbours' reads of it. A clean run is evidence, not proof: it sees only the
          schedules that happened.
        </p>
      </Translated>
      <RaceStatus lab={l1} />
      <Translated
        vi={
          <p>
            Đừng return sớm. Mọi luồng của khối phải đến được cả hai rào chắn trong mọi vòng lặp của vòng z: một luồng ở rìa
            khối hoặc ở biên lưới không tính gì cả nhưng vẫn nạp phần tử vùng biên (halo) của nó. Hãy đặt điều kiện quanh các
            lần nạp và lệnh lưu, đừng bao giờ đặt một <code>return</code> trước các rào chắn của vòng lặp. Trình giả lập không
            tự bắt được một lần return sớm; nó bắt được thiệt hại, một giá trị vùng biên không ai nạp.
          </p>
        }
      >
        <p>
          Do not return early. Every thread of the block must reach both barriers in every iteration of the z loop: a
          thread on the block's edge or the grid's boundary computes nothing but still loads its element of the halo.
          Put the conditions around the loads and the store, never a <code>return</code> ahead of the loop's barriers. The
          emulator cannot catch an early return by itself; it catches the damage, a halo value nobody loaded.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A6, B3, C3)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Rào chắn của NVIDIA chờ mọi luồng chưa thoát, và Apple nói điều tương tự cho Apple silicon; trình giả lập mô hình
              hóa đúng như vậy. AMD không nêu quy tắc nào. Quy tắc khả chuyển là quy tắc của sách: mọi luồng đều đến được mọi
              rào chắn.
            </p>
          }
        >
          <p>
            NVIDIA's barrier waits for all non-exited threads, and Apple says the same for Apple silicon; the emulator
            models that. AMD states no rule. The portable rule is the book's: every thread reaches every barrier.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Có một cái bẫy mà các bài kiểm thử không thể cho thấy ở đây. Trên CUDA, một khối 1024 luồng chỉ để lại tối đa 64
            thanh ghi cho mỗi luồng; một kernel được biên dịch dùng nhiều hơn sẽ thất bại lúc khởi chạy với "too many resources
            requested". Nếu bản build T = 32 của bạn gặp điều đó, <code>nvcc --resource-usage</code> cho thấy con số, và{' '}
            <a href={href('ch08', '8.5')}>8.5</a> giải thích cách sửa, <code>__launch_bounds__</code>. Các kernel của lab không
            mang nó, và bạn có thể thêm <code>__launch_bounds__(TILE_WIDTH * TILE_WIDTH)</code> vào kernel phân ô thanh ghi
            của mình (không phải kernel cơ bản, vì khối của nó cố định 8 × 8 × 8, nhiều luồng hơn mức giới hạn đó cho phép ở T
            = 16 hoặc 8): trình giả lập chấp nhận và bỏ qua nó, và phép kiểm tra front-end CUDA chỉ phân tích cú pháp nó, nên
            không cái nào cho bạn biết một bản build cho thiết bị có cần nó hay không. Trên Metal, chương trình điều khiển đọc
            giới hạn số luồng của từng pipeline và bỏ qua T = 32, kèm thông báo, ở nơi giới hạn đó dưới 1024.
          </p>
        }
      >
        <p>
          One trap the tests cannot show here. On CUDA a 1024-thread block leaves at most 64 registers per thread; a kernel
          compiled to more fails at launch with "too many resources requested". If your T = 32 build does that,{' '}
          <code>nvcc --resource-usage</code> shows the count, and <a href={href('ch08', '8.5')}>8.5</a> explains the fix,{' '}
          <code>__launch_bounds__</code>. The lab's kernels do not carry it, and you may add{' '}
          <code>__launch_bounds__(TILE_WIDTH * TILE_WIDTH)</code> to your register-tiled kernel (not the basic one, whose
          block is a fixed 8 × 8 × 8, more threads than that bound allows at T = 16 or 8): the emulator accepts it and
          ignores it, and
          the CUDA front-end check only parses it, so neither can tell you whether a device build needs it. On Metal the
          driver reads each pipeline's thread cap and skips T = 32, with a message, where it is below 1024.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch08-facts.md A4, B4, C1)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              NVIDIA ghi nhận <code>__launch_bounds__</code> là cách sửa cho lỗi khởi chạy đó. HIP biên dịch cho khối lớn nhất
              của thiết bị trừ khi được chỉ định khác, nên nó giới hạn số thanh ghi, và có thể tràn (spill), thay vì thất bại.
              Metal có thể hạ giới hạn số luồng của một pipeline cho một kernel cần nhiều tài nguyên hơn.
            </p>
          }
        >
          <p>
            NVIDIA documents <code>__launch_bounds__</code> as the fix for that launch failure. HIP compiles for the
            device's largest block unless told otherwise, so it limits registers, and may spill, instead of failing. Metal
            may lower a pipeline's thread cap for a kernel that needs more resources.
          </p>
        </Translated>
      </Today>

      <h3>
        <Translated vi="Thí nghiệm">The experiment</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Hãy làm phần này khi các kernel của bạn đã qua. Mô hình nói kernel cơ bản yêu cầu 32 byte cho mỗi điểm đầu ra, 0.41
            FLOP/B, còn kernel phân ô thanh ghi ở tấm của sách, T = 32 và 30 mặt phẳng, nạp 32 × 32 × 32 giá trị và lưu 30 × 30
            × 30 giá trị mỗi khối: 1.47 FLOP/B. Tức là số byte được yêu cầu ít hơn khoảng 3.6 lần. Hãy ghi lại ba dự đoán trước
            khi bạn chạy <code>--bench</code>.
          </p>
        }
      >
        <p>
          Do this once your kernels pass. The model says the basic kernel asks for 32 bytes per output point, 0.41 FLOP/B,
          and the register-tiled kernel at the book's slab, T = 32 and 30 planes, loads 32 × 32 × 32 values and stores 30 ×
          30 × 30 per block: 1.47 FLOP/B. That is about 3.6 times fewer bytes requested. Write down three predictions
          before you run <code>--bench</code>.
        </p>
      </Translated>
      <Source section="8.3, 8.5" pages="189, 195" />
      <LabPrediction
        id="ch08/labs/p-lab1-speedup"
        prompt={
          <Translated
            vi={
              <>
                Ở T = 32 và độ sâu tấm 30 của sách, kernel phân ô thanh ghi của bạn sẽ chạy nhanh hơn kernel cơ bản bao nhiêu
                lần trên GPU của bạn?
              </>
            }
          >
            <>
              At T = 32 and the book's slab depth of 30, how many times faster than the basic kernel will your
              register-tiled kernel run on your GPU?
            </>
          </Translated>
        }
        unit={tr('× the basic kernel', '× kernel cơ bản')}
        measure={overBasic}
        missing={MISSING}
        reading={
          <Translated
            vi={
              <>
                3.6 là tỉ lệ số byte được yêu cầu. Bộ nhớ đệm (cache) phục vụ một phần các lần đọc láng giềng của kernel cơ bản
                (sách nói vậy ở p. 189), và hệ số đo được nhỏ hơn ở bất cứ đâu cache vốn đã làm việc đó. Cột "GB/s ở 8
                B/điểm" bên dưới cho thấy mỗi kernel tiến gần đến mức chỉ di chuyển các byte bắt buộc đến đâu.
              </>
            }
          >
            <>
              3.6 is a ratio of requested bytes. Caches serve some of the basic kernel's neighbour reads (the book says
              so on p. 189), and the measured factor is smaller wherever they were already doing that work. The "GB/s at
              8 B/point" column below shows how close each kernel comes to moving only the compulsory bytes.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch08/labs/p-lab1-planes"
        prompt={
          <Translated
            vi={
              <>
                Kernel ba mặt phẳng chia sẻ được cung cấp sẵn di chuyển đúng cùng lượng dữ liệu toàn cục như kernel của bạn
                (p. 198); kernel của bạn đổi hai mặt phẳng chia sẻ lấy thanh ghi. Kernel của bạn sẽ chạy nhanh hơn nó bao nhiêu
                lần, ở cùng T = 32 và cùng độ sâu?
              </>
            }
          >
            <>
              The provided three-shared-plane kernel moves exactly the same global data as yours (p. 198); yours swaps two
              shared planes for registers. How many times faster than it will yours run, at the same T = 32 and depth?
            </>
          </Translated>
        }
        unit={tr('× the planes kernel', '× kernel ba mặt phẳng')}
        measure={overPlanes}
        missing={MISSING}
        reading={
          <Translated
            vi={
              <>
                Cùng lưu lượng toàn cục, nên mọi khác biệt đều nằm trên chip: ít lần đọc bộ nhớ chia sẻ hơn và không chép mặt
                phẳng, đổi lại thêm một lần ghi giá trị hiện tại. Không nhà cung cấp nào ghi nhận một truy cập thanh ghi tiết
                kiệm được bao nhiêu so với một truy cập bộ nhớ chia sẻ (<a href={href('ch08', '8.5')}>8.5</a>, "Những gì các
                nhà cung cấp ghi nhận về thanh ghi").
              </>
            }
          >
            <>
              Same global traffic, so any difference is on chip: fewer shared-memory reads and no plane copies, against
              one more write of the current value. No vendor documents how much a register access saves over a shared
              one (<a href={href('ch08', '8.5')}>8.5</a>, "What vendors document about registers").
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch08/labs/p-lab1-depth"
        prompt={
          <Translated
            vi={
              <>
                Ở T = 32, benchmark thử các độ sâu tấm 8, 16, 30 (của sách), 64 và cả cột. Độ sâu nào, tính bằng số mặt
                phẳng, sẽ nhanh nhất trên GPU của bạn?
              </>
            }
          >
            <>
              At T = 32 the benchmark tries slab depths of 8, 16, 30 (the book's), 64 and the whole column. Which depth, in
              planes, will be fastest on your GPU?
            </>
          </Translated>
        }
        unit={tr('planes', 'mặt phẳng')}
        measure={fastestDepth}
        missing={MISSING}
        reading={
          <Translated
            vi={
              <>
                Tỉ lệ của mô hình tăng theo độ sâu, tiến về 1.52, trong khi lưới có ít khối hơn và mỗi khối làm nhiều việc tuần
                tự hơn (<a href={href('ch08', '8.5')}>8.5</a>, "Các con số"). Các độ sâu kề nhau có thể chênh nhau vài phần
                trăm giữa các lần chạy; hãy chạy benchmark hai lần trước khi suy diễn nhiều từ một khoảng chênh nhỏ.
              </>
            }
          >
            <>
              The model's ratio rises with depth towards 1.52, while the grid has fewer blocks and each does more work in
              sequence (<a href={href('ch08', '8.5')}>8.5</a>, "The numbers"). Neighbouring depths can differ by a few
              per cent from one run to the next; run the benchmark twice before you read much into a small gap.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l1}
        filter={(r) => r.name !== 'limits'}
        cols={[
          { head: tr('What ran', 'Đã chạy gì'), cell: variant },
          { head: tr('Slab depth', 'Độ sâu tấm'), num: true, cell: depthCell },
          rawCol('N', 'n'),
          msCol,
          numCol(tr('Gpoints/s', 'Gđiểm/s'), 'gpts'),
          numCol(tr('GB/s at 8 B/point', 'GB/s ở 8 B/điểm'), 'gbps_compulsory'),
          numCol(tr('× basic', '× cơ bản'), 'speedup_vs_basic'),
          numCol(tr('× planes', '× ba mặt phẳng'), 'speedup_vs_planes'),
        ]}
      />
      <Translated
        vi={
          <p>
            Cách benchmark hoạt động: nó tăng N cho đến khi kernel cơ bản chạy đủ lâu để đo thời gian, không để{' '}
            <code>in</code> và <code>out</code> cộng lại vượt quá 2 GiB hoặc một phần tám bộ nhớ mà thiết bị báo cáo, rồi đo
            thời gian mọi kernel trên lưới đó, lấy lần tốt nhất trong vài lần chạy, và sau đó kiểm tra từng kết quả với CPU.
            Một kernel có đầu ra sai sẽ được báo cáo, không được đo thời gian. "GB/s ở 8 B/điểm" tính một lần nạp 4 byte và một
            lần lưu 4 byte cho mỗi điểm bên trong, tức lưu lượng của mức lý tưởng 13/8, không phải lượng kernel đã yêu cầu.
          </p>
        }
      >
        <p>
          How the benchmark works: it grows N until the basic kernel takes long enough to time, without letting{' '}
          <code>in</code> and <code>out</code> together pass 2 GiB or an eighth of the memory the device reports, then
          times every kernel on that grid, best of several runs, and checks each result against the CPU afterwards. A
          kernel whose output is wrong is reported, not timed. "GB/s at 8 B/point" counts one 4-byte load and one 4-byte
          store per interior point, the traffic of the 13/8 ideal, not what the kernel asked for.
        </p>
      </Translated>
      <Translated vi={<p>Các giới hạn số luồng mà benchmark đọc được, theo kernel và độ rộng ô, tính bằng số luồng mỗi khối:</p>}>
        <p>The thread limits the benchmark read, per kernel and tile width, in threads per block:</p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Trên Metal, giới hạn thuộc về một pipeline đã biên dịch (<code>maxTotalThreadsPerThreadgroup</code>), nên mỗi
              kernel có giới hạn riêng; trên CUDA và HIP, đó là <code>maxThreadsPerBlock</code> của thiết bị.
            </p>
          }
        >
          <p>
            On Metal the limit belongs to a compiled pipeline (<code>maxTotalThreadsPerThreadgroup</code>), so each kernel
            has its own; on CUDA and HIP it is the device's <code>maxThreadsPerBlock</code>.
          </p>
        </Translated>
      </Aside>
      <Limits />
      <ReferenceSolution lab={l1} />
    </LabsPage>
  );
}
