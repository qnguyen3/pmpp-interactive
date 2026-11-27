import type { Question } from '../../components/Ask';
import { Beyond, Source, Today, type Tier } from '../../components/notes';
import {
  BackendsAndEmulator,
  LabFiles,
  LabPrediction,
  LabResults,
  LabsIntro,
  LabsPage,
  ReferenceSolution,
  RunningLabs,
  VerificationStatus,
  benchRow,
  defineLabs,
  measure,
  msCol,
  numCol,
  preferredBench,
  useLabs,
  whose,
  type BenchSet,
} from '../../components/LabKit';
import { fmt, labStatus, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 16's own content.
// No question on this page: the lab is the exercise. The three predictions are stored, not
// graded, under ch16/labs/p-lab1-baseline, p-lab1-hyper and p-lab1-pad.
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch16', [{ num: '16.1', dir: 'lab1_wavefront', name: 'square and sheared tiles', vi: { name: 'ô vuông và ô đã trượt' } }]);
const [l1] = LABS;

/** Identifier colours in the reference solution: where each value lives (as on page 16.6). */
const TIERS: Record<string, Tier> = {
  H: 'global', a: 'global', b: 'global', swTile: 'shared',
  r: 'reg', q: 'reg', r_tile: 'reg', q_tile: 'reg', d_tile: 'reg', tile_row: 'reg', tile_col: 'reg',
  n: 'reg', w: 'reg', nw: 'reg', s: 'reg', row: 'reg', firstRow: 'reg', stride: 'reg',
};

// ---------------------------------------------------------------- what each prediction is measured against
const ms = (r: BenchRow | undefined) => {
  const v = num(r?.seconds);
  return v ? `${fmt(v * 1000)} ms` : '?'; // i18n-ignore: unit
};
const count = (v: unknown) => (num(v) != null ? num(v)!.toLocaleString('en-US') : '?');
const at32 = (s: BenchSet, name: 'square' | 'hyper', pad: number) => benchRow(s, name, { tile: 32, pad });

/** (1) Baseline time over square-tile time at W = 32, unpadded: how many times faster the tiles are. */
const baselineSpeedup = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const base = benchRow(s, 'baseline');
    const sq = at32(s, 'square', 0);
    const a = num(base?.seconds);
    const b = num(sq?.seconds);
    if (!a || !b) return null;
    return {
      value: a / b,
      detail: tr(
        `baseline ${ms(base)} in ${count(base?.launches)} launches, square tiles ${ms(sq)} in ${count(sq?.launches)}, L_seq ${count(sq?.lseq)}`,
        `phương án cơ sở ${ms(base)} trong ${count(base?.launches)} lần khởi chạy, ô vuông ${ms(sq)} trong ${count(sq?.launches)}, L_seq ${count(sq?.lseq)}`,
      ),
    };
  });

/** (2) Sheared time over square time, both at W = 32, unpadded. */
const shearedOverSquare = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const sq = at32(s, 'square', 0);
    const hy = at32(s, 'hyper', 0);
    const a = num(hy?.seconds);
    const b = num(sq?.seconds);
    if (!a || !b) return null;
    return {
      value: a / b,
      detail: tr(
        `sheared ${ms(hy)} in ${count(hy?.launches)} launches, square ${ms(sq)} in ${count(sq?.launches)}, L_seq ${count(sq?.lseq)}`,
        `đã trượt ${ms(hy)} trong ${count(hy?.launches)} lần khởi chạy, vuông ${ms(sq)} trong ${count(sq?.launches)}, L_seq ${count(sq?.lseq)}`,
      ),
    };
  });

/** (3) Sheared time with one gap per tile row over sheared time without, at W = 32. */
const paddedOverUnpadded = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const p0 = at32(s, 'hyper', 0);
    const p1 = at32(s, 'hyper', 1);
    const a = num(p1?.seconds);
    const b = num(p0?.seconds);
    if (!a || !b) return null;
    return {
      value: a / b,
      detail: tr(
        `${ms(p1)} with one gap per row, ${ms(p0)} without, L_seq ${count(p0?.lseq)}`,
        `${ms(p1)} với một khoảng trống mỗi hàng, ${ms(p0)} khi không có, L_seq ${count(p0?.lseq)}`,
      ),
    };
  });

const MISSING = (
  <Translated
    vi={
      <>
        Mục này cần các hàng benchmark của lab 16.1 từ một thiết bị thật. Chạy <code>python3 labs/pmpp.py run 16.1 --bench</code>{' '}
        khi các kernel của bạn đã đạt (hoặc thêm <code>--solution</code> để đo lời giải tham chiếu). Các trình giả lập không bao
        giờ được đo hiệu năng.
      </>
    }
  >
  <>
    This needs the benchmark rows of lab 16.1 from a real device. Run <code>python3 labs/pmpp.py run 16.1 --bench</code>{' '}
    once your kernels pass (or add <code>--solution</code> to time the reference). The emulators are never benchmarked.
  </>
  </Translated>
);

/** The fourth question of the README: which way padding moves the square kernel, measured per backend. */
function SquarePadding() {
  const { reports, loading } = useLabs();
  if (loading) return null;
  const lines = preferredBench(reports, l1).flatMap((s) => {
    const p0 = at32(s, 'square', 0);
    const p1 = at32(s, 'square', 1);
    const a = num(p1?.seconds);
    const b = num(p0?.seconds);
    return a && b ? [{ s, ratio: a / b, p0, p1 }] : [];
  });
  return lines.length ? (
    <>
      {lines.map(({ s, ratio, p0, p1 }) => (
        <p key={s.backend} data-square-pad={s.backend} data-ratio={fmt(ratio)}>
          <Translated
            vi={
              <>
                Kernel vuông trên {s.device} ({s.backend}), {whose(s)}, ở W = 32: {ms(p1)} với một khoảng trống mỗi hàng so với{' '}
                {ms(p0)} khi không có, tỉ lệ <b>{fmt(ratio)}</b>. Trên 1 nghĩa là đệm đã làm nó chậm đi.
              </>
            }
          >
          The square kernel on {s.device} ({s.backend}), {whose(s)}, at W = 32: {ms(p1)} with one gap per row against{' '}
          {ms(p0)} without, a ratio of <b>{fmt(ratio)}</b>. Above 1, padding slowed it.
          </Translated>
        </p>
      ))}
    </>
  ) : (
    <p className="muted" data-square-pad="none">
      <Translated
        vi={
          <>
            Chưa có hàng ô vuông nào ở W = 32 có và không có đệm trên máy này. Chạy{' '}
            <code>python3 labs/pmpp.py run 16.1 --bench</code> trên một thiết bị thật.
          </>
        }
      >
      No square-tile rows at W = 32 with and without padding on this machine yet. Run{' '}
      <code>python3 labs/pmpp.py run 16.1 --bench</code> on a real device.
      </Translated>
    </p>
  );
}

// ---------------------------------------------------------------- results table
const variant = (r: BenchRow): string =>
  r.name === 'baseline' ? tr('baseline (provided)', 'phương án cơ sở (được cung cấp)') : r.name === 'square' ? tr('square tiles', 'ô vuông') : r.name === 'hyper' ? tr('sheared tiles', 'ô đã trượt') : r.name;
const ifTiled = (r: BenchRow, key: string) => (r.name === 'baseline' ? '–' : String(r[key] ?? ''));

/** One line on lab 16.1 on the race backend, which the status table does not show. */
function RaceStatus() {
  const { reports, loading } = useLabs();
  const run = reports.race;
  const line = (v: 'solution' | 'work') => {
    const rs = (run?.results.filter((x) => x.lab === l1.id && x.variant === v) ?? []) as (RunResult & { tsan?: unknown[] })[];
    const tests = rs.flatMap((x) => x.tests);
    const found = rs.reduce((n, x) => n + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0);
    return { s: labStatus(run, l1.id, v), tests, passed: tests.filter((t) => t.pass).length, found };
  };
  const sol = line('solution');
  const mine = line('work');
  return (
    <p data-race-status={loading ? 'loading' : sol.s} data-race-work={loading ? 'loading' : mine.s}>
      {loading ? (
        tr('Reading reports.', 'Đang đọc báo cáo.')
      ) : !run ? (
        tr('No run on the race backend on this machine yet.', 'Chưa có lần chạy nào trên backend race ở máy này.')
      ) : (
        <>
          {sol.s === 'none'
            ? tr('On this machine the reference solution has no run on the race backend yet.', 'Trên máy này, lời giải tham chiếu chưa có lần chạy nào trên backend race.')
            : sol.s === 'pass'
              ? tr(`On this machine the reference solution passed all ${sol.tests.length} tests on the race backend with ${sol.found ? `${sol.found} ThreadSanitizer report(s)` : 'no race reported'}: emulator-checked on this flavour.`, `Trên máy này, lời giải tham chiếu đã vượt qua cả ${sol.tests.length} phép kiểm thử trên backend race với ${sol.found ? `${sol.found} báo cáo ThreadSanitizer` : 'không race nào được báo cáo'}: emulator-checked trên biến thể này.`)
              : tr(`On this machine the reference solution failed on the race backend (${sol.passed} of ${sol.tests.length} tests): unverified there.`, `Trên máy này, lời giải tham chiếu thất bại trên backend race (${sol.passed} trên ${sol.tests.length} phép kiểm thử): unverified ở đó.`)}{' '}
          {mine.s === 'none'
            ? tr('Your own kernels have no recorded run there, or are still the untouched starter.', 'Các kernel của bạn chưa có lần chạy nào được ghi ở đó, hoặc vẫn là mã khởi đầu (starter) chưa sửa.')
            : mine.s === 'pass'
              ? tr(`Your own kernels passed all ${mine.tests.length} tests there with ${mine.found ? `${mine.found} ThreadSanitizer report(s): read them` : 'no race reported'}.`, `Các kernel của bạn đã vượt qua cả ${mine.tests.length} phép kiểm thử ở đó với ${mine.found ? `${mine.found} báo cáo ThreadSanitizer: hãy đọc chúng` : 'không race nào được báo cáo'}.`)
              : tr(`Your own kernels passed ${mine.passed} of ${mine.tests.length} tests there, with ${mine.found} ThreadSanitizer report(s) in the run.`, `Các kernel của bạn vượt qua ${mine.passed} trên ${mine.tests.length} phép kiểm thử ở đó, với ${mine.found} báo cáo ThreadSanitizer trong lần chạy.`)}
        </>
      )}
    </p>
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
              Một bài thực hành, trong <code>labs/ch16/lab1_wavefront/</code>. Nó là tùy chọn, vì chương này được rút gọn, và nó
              không thêm gì cho dự án xuyên suốt. Nó dựng cả hai kernel phân ô của trang{' '}
              <a href={href('ch16', '16.6')}>16.6</a> và đặt các số đếm lần khởi chạy, luồng rảnh và xung đột bank (bank conflict)
              của trang đó cạnh một chiếc đồng hồ trên GPU của chính bạn.
            </>
          }
        >
        One lab, in <code>labs/ch16/lab1_wavefront/</code>. It is optional, as this chapter is condensed, and it adds
        nothing to the through-line project. It builds both tiled kernels of page{' '}
        <a href={href('ch16', '16.6')}>16.6</a> and puts that page's counts of launches, idle threads and bank conflicts
        against a clock on your own GPU.
        </Translated>
      </LabsIntro>
      <RunningLabs
        example="16.1"
        alsoShownIn={
          <>
            <Translated
              vi={
                <>
                  trạng thái của mã ở trang <a href={href('ch16', '16.6')}>16.6</a>
                </>
              }
            >
            the status of the listing on page <a href={href('ch16', '16.6')}>16.6</a>
            </Translated>
          </>
        }
      />
      <BackendsAndEmulator />
      <VerificationStatus />

      {/* ------------------------------------------------------------ 16.1 */}
      <h2><Translated vi="Lab 16.1: một wavefront phân ô, ô vuông rồi ô đã trượt">Lab 16.1: a tiled wavefront, square tiles then sheared tiles</Translated></h2>
      <Translated
        vi={
          <p>
            Bảng là bảng Smith-Waterman của trang 16.6. <code>H</code> có L_seq + 1 hàng và cột kiểu{' '}
            <code>int</code>, thứ tự hàng (row-major), và hàng 0 cùng cột 0 chứa số 0. Chuỗi <code>a</code> đánh chỉ số các hàng và{' '}
            <code>b</code> các cột, và mỗi ô còn lại là giá trị lớn nhất trong 0, láng giềng tây bắc cộng điểm khớp hoặc không khớp,
            và các láng giềng tây và bắc cộng một điểm khoảng trống (gap), vốn âm. Ba hằng số là các giá trị ví dụ của sách, trong{' '}
            <code>sw.h</code>, và không gì trong bài thực hành phụ thuộc vào chúng.
          </p>
        }
      >
      <p>
        The table is Smith-Waterman's from page 16.6. <code>H</code> has L_seq + 1 rows and columns of{' '}
        <code>int</code>, row-major, and row 0 and column 0 hold zeros. Sequence <code>a</code> indexes rows and{' '}
        <code>b</code> columns, and every other cell is the largest of 0, its north-west neighbour plus a match or
        mismatch score, and its west and north neighbours plus a gap, which is negative. The three constants are the
        book's example values, in <code>sw.h</code>, and nothing in the lab depends on them.
      </p>
      </Translated>
      <Source section="16.5" pages="382–383" figs={tr('Eq. 16.2', 'Phương trình 16.2')} />
      <Translated
        vi={
          <p>
            Một phương án cơ sở được cung cấp: một lần khởi chạy mỗi đường chéo phụ của bảng, một luồng mỗi ô, mọi đầu vào đọc từ
            bộ nhớ toàn cục (global memory), nên có 2·L_seq − 1 lần khởi chạy. Sách để nó làm bài tập; bản của chúng tôi nằm trong{' '}
            <code>labs/ch16/shared/</code>, là <code>baseline.cu</code>, <code>baseline.metal</code> và{' '}
            <code>baseline_launch.mm</code>. Bạn viết hai phiên bản cần ít lần khởi chạy hơn nhiều.
          </p>
        }
      >
      <p>
        A baseline is provided: one launch per anti-diagonal of the table, one thread per cell, every input read from
        global memory, so 2·L_seq − 1 launches. The book leaves it as an exercise; ours is in{' '}
        <code>labs/ch16/shared/</code>, as <code>baseline.cu</code>, <code>baseline.metal</code> and{' '}
        <code>baseline_launch.mm</code>. You write two versions that need far fewer launches.
      </p>
      </Translated>
      <Source section="16.6" pages="384" figs={tr('Fig. 16.7(a)', 'Hình 16.7(a)')} />
      <Translated
        vi={
          <p>
            Phần (a), ô vuông. Một ô (tile) W × W cho mỗi khối W luồng, giữ trong bộ nhớ chia sẻ (shared memory), và một lần khởi
            chạy mỗi mặt sóng của ô: 2T − 1 lần khởi chạy với T = ⌈L_seq / W⌉ ô mỗi cạnh. <code>swSquareKernel</code> lấy hàng ô
            của nó là <code>firstRow + blockIdx.x</code> và cột ô là <code>d</code> trừ đi số đó. Trong vòng lặp tính toán, chỉ số
            luồng là hàng của ô; có 2W − 1 bước, một luồng chỉ tính khi ô con của nó nằm trong ô và trong bảng, và mọi luồng đi
            qua một rào chắn (barrier) sau mỗi bước, bên ngoài phép kiểm tra đó. Trong <code>sqStoreTile</code>, chỉ số luồng là
            cột của ô, nên mỗi hàng của ô đi ra thành một mảnh được hợp nhất.{' '}
            <code>launchSwSquare</code> khởi chạy đúng các ô của mỗi mặt sóng, bài tập sách đặt ra ở p. 386, với{' '}
            <code>W × stride</code> int bộ nhớ chia sẻ động, trong đó <code>stride = W + pad</code>.
          </p>
        }
      >
      <p>
        Part (a), square tiles. One W × W tile per block of W threads, held in shared memory, and one launch per front
        of tiles: 2T − 1 launches for T = ⌈L_seq / W⌉ tiles per side. <code>swSquareKernel</code> takes its tile row as{' '}
        <code>firstRow + blockIdx.x</code> and its tile column as <code>d</code> minus that. In the compute loop the
        thread index is the tile row; there are 2W − 1 steps, a thread computes only when its cell is inside the tile and
        inside the table, and every thread passes one barrier after every step, outside that test. In{' '}
        <code>sqStoreTile</code> the thread index is the tile column, so each tile row leaves as one coalesced piece.{' '}
        <code>launchSwSquare</code> launches exactly the tiles of each front, the exercise the book sets on p. 386, with{' '}
        <code>W × stride</code> ints of dynamic shared memory, where <code>stride = W + pad</code>.
      </p>
      </Translated>
      <Source section="16.6" pages="384–389" figs={tr('Figs. 16.8 to 16.12', 'Hình 16.8 đến 16.12')} />
      <Translated
        vi={
          <p>
            Phần (b), ô đã trượt. Mỗi hàng của ô trượt sang trái thêm một cột so với hàng phía trên, nên mọi bước bên trong một ô
            là một đường chéo phụ đầy đủ gồm W ô con và một ô cần W bước. <code>hyInitTile</code> đặt ô về 0 sau một rào chắn;{' '}
            <code>hyLoadN</code>, <code>hyLoadW</code> và <code>hyLoadNW</code> đọc các láng giềng từ ô, được lưu dưới dạng hình
            vuông (hãy tự tính là những cột lưu trữ nào: sách để hai trong số đó cho bạn ở p. 394, và câu hỏi ở trang 16.6 đi qua
            một cột);{' '}
            <code>swHyperKernel</code> đặt mỗi hàng ô sau hàng phía trên hai cột ô, với{' '}
            <code>tile_col = d − 2·tile_row</code> và có tính cột ô cuối cùng; <code>hyStoreTile</code> bỏ qua các ô con nằm
            ngoài bảng ở mọi phía. <code>launchSwHyper</code> khởi chạy đúng các ô của mỗi mặt sóng trong 3T − 1 mặt sóng.
          </p>
        }
      >
      <p>
        Part (b), sheared tiles. Each tile row slides one column further left than the row above, so every step inside a
        tile is a full anti-diagonal of W cells and a tile takes W steps. <code>hyInitTile</code> zeroes the tile behind a
        barrier; <code>hyLoadN</code>, <code>hyLoadW</code> and <code>hyLoadNW</code> read the neighbours from the tile,
        stored as a square (work out which stored columns: the book leaves two of them to you on p. 394, and the question
        on page 16.6 walks through one);{' '}
        <code>swHyperKernel</code> puts each tile row two tile columns behind the row above, with{' '}
        <code>tile_col = d − 2·tile_row</code> and the last tile column included; <code>hyStoreTile</code> skips cells
        outside the table on every side. <code>launchSwHyper</code> launches exactly the tiles of each of the 3T − 1
        fronts.
      </p>
      </Translated>
      <Source section="16.7" pages="389–396" figs={tr('Figs. 16.14 to 16.20', 'Hình 16.14 đến 16.20')} />
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <Translated
              vi={
                <>
                  <code>max4</code>, <code>sqLoadN</code>, <code>sqLoadW</code>, <code>sqLoadNW</code>,{' '}
                  <code>sqStoreTile</code>, <code>swSquareKernel</code> và <code>launchSwSquare</code>; rồi{' '}
                  <code>hyInitTile</code>, <code>hyLoadN</code>, <code>hyLoadW</code>, <code>hyLoadNW</code>,{' '}
                  <code>hyStoreTile</code>, <code>swHyperKernel</code> và <code>launchSwHyper</code>
                </>
              }
            >
            <>
              <code>max4</code>, <code>sqLoadN</code>, <code>sqLoadW</code>, <code>sqLoadNW</code>,{' '}
              <code>sqStoreTile</code>, <code>swSquareKernel</code> and <code>launchSwSquare</code>; then{' '}
              <code>hyInitTile</code>, <code>hyLoadN</code>, <code>hyLoadW</code>, <code>hyLoadNW</code>,{' '}
              <code>hyStoreTile</code>, <code>swHyperKernel</code> and <code>launchSwHyper</code>
            </>
            </Translated>,
          ],
          ['kernel.metal', <Translated vi={<><code>sw_square</code>, <code>sw_hyper</code> và các hàm phụ như trên, cùng tên</>}><><code>sw_square</code>, <code>sw_hyper</code> and the same helpers, by the same names</></Translated>],
          ['launch.mm', <Translated vi={<><code>encodeSwSquare</code> và <code>encodeSwHyper</code></>}><><code>encodeSwSquare</code> and <code>encodeSwHyper</code></></Translated>],
        ]}
      />
      <Translated
        vi={
          <p>
            <code>main.cpp</code> (các phép kiểm thử, benchmark và bản tham chiếu CPU), <code>sw.h</code> và phương án cơ sở trong{' '}
            <code>../shared/</code> là chỉ đọc. Hãy viết mọi phép tính chỉ số bằng <code>int</code>.
          </p>
        }
      >
      <p>
        <code>main.cpp</code> (the tests, the benchmark and the CPU reference), <code>sw.h</code> and the baseline in{' '}
        <code>../shared/</code> are read-only. Write all index arithmetic in <code>int</code>.
      </p>
      </Translated>
      <Translated
        vi={
          <p>
            Trong Metal, ô là một tham số kernel, <code>threadgroup int* swTile [[threadgroup(0)]]</code>, được truyền cho các hàm
            phụ <code>static inline</code>, vì một hàm phụ không thể tự khai báo bộ nhớ threadgroup. Rào chắn là{' '}
            <code>threadgroup_barrier(mem_flags::mem_threadgroup)</code>: các đầu vào mà một ô đọc từ bộ nhớ device được các dispatch
            trước ghi, nên rào chắn chỉ cần sắp thứ tự ô. <code>encodeSwSquare</code> và{' '}
            <code>encodeSwHyper</code> ghi một dispatch mỗi mặt sóng vào encoder được giao: tham số của mặt sóng đó bằng{' '}
            <code>setBytes</code>, rồi <code>setThreadgroupMemoryLength</code> làm tròn lên bội số của 16 byte, rồi các
            threadgroup. Encoder chạy các dispatch của nó theo thứ tự, đó là dạng của Metal cho việc khởi chạy lại giữa các mặt sóng.
          </p>
        }
      >
      <p>
        In Metal the tile is a kernel parameter, <code>threadgroup int* swTile [[threadgroup(0)]]</code>, handed to{' '}
        <code>static inline</code> helpers, because a helper cannot declare threadgroup memory itself. The barrier is{' '}
        <code>threadgroup_barrier(mem_flags::mem_threadgroup)</code>: the inputs a tile reads from device memory were
        written by earlier dispatches, so the barrier only has to order the tile. <code>encodeSwSquare</code> and{' '}
        <code>encodeSwHyper</code> write one dispatch per front into the encoder they are given: that front's parameters
        with <code>setBytes</code>, then <code>setThreadgroupMemoryLength</code> rounded up to a multiple of 16 bytes, then
        the threadgroups. The encoder runs its dispatches in order, which is Metal's form of the relaunch between fronts.
      </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch16-facts C3, C4; ch05-facts C1, C8; ch02-facts C9)', 'ngày 2026-11-20 (ch16-facts C3, C4; ch05-facts C1, C8; ch02-facts C9)')}>
        <Translated
          vi={
            <p>
              Một hàm phụ MSL không thể khai báo một mảng threadgroup; một mảng được truyền vào thì được. Độ dài bộ nhớ threadgroup
              là bội số của 16 byte, tổng cộng 32 KB trên Apple4 trở đi. Các dispatch trong một compute encoder thông thường chạy
              theo thứ tự và thấy được những gì nhau đã ghi.
            </p>
          }
        >
        <p>
          An MSL helper cannot declare a threadgroup array; one passed in works. Threadgroup memory lengths are multiples
          of 16 bytes, 32 KB in all on Apple4 and later. Dispatches in one plain compute encoder run in order and see each
          other's writes.
        </p>
        </Translated>
      </Today>

      <h3><Translated vi="Những chỗ bài thực hành khác trang sách">Where the lab differs from the page</Translated></h3>
      <ul>
        <li>
          <Translated
            vi={
              <>
                Mọi phép tính chỉ số đều là <code>int</code>. Sách tính <code>tile_col = d - blockIdx.x</code> từ các toán hạng
                unsigned và trông vào việc chuyển đổi sang <code>int</code> cho ra một số âm, điều mà C++17, chuẩn các bài thực hành
                biên dịch theo, để cho trình biên dịch quyết định.
              </>
            }
          >
          All index arithmetic is <code>int</code>. The book computes <code>tile_col = d - blockIdx.x</code> from unsigned
          operands and counts on the conversion to <code>int</code> giving a negative number, which C++17, the standard
          the labs compile as, leaves to the compiler.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Mỗi lần khởi chạy phủ đúng các ô của mặt sóng của nó, thông qua <code>firstRow</code>. Sách khởi chạy T khối mỗi lần
                và để các khối thừa thoát ra.
              </>
            }
          >
          Each launch covers exactly the tiles of its front, through <code>firstRow</code>. The book launches T blocks
          every time and lets the spare ones exit.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Đệm là một khoảng trống mỗi hàng của ô, thông qua <code>stride</code>. Macro <code>pad()</code> của sách chèn một
                khoảng trống sau mỗi số int bằng số bank, chỉ bằng một khoảng trống mỗi hàng khi W bằng số bank; các mã in của sách
                không bao giờ áp dụng nó.
              </>
            }
          >
          Padding is one gap per tile row, through <code>stride</code>. The book's <code>pad()</code> macro inserts a gap
          every bank-count ints, which is one per row only when W equals the number of banks; its listings never apply
          it.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Phép lưu của ô đã trượt kiểm tra <code>q &gt;= 1</code>. Fig. 16.18 chỉ kiểm tra <code>q &lt; L</code> với{' '}
                <code>L</code> unsigned, nên một q âm thất bại nhờ bị quay vòng.
              </>
            }
          >
          The sheared store tests <code>q &gt;= 1</code>. Fig. 16.18 tests only <code>q &lt; L</code> with an unsigned{' '}
          <code>L</code>, so a negative q fails by wrapping round.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Một hằng số khoảng trống, và các chuỗi kiểu <code>int</code>. Hai điểm phạt của sách có cùng giá trị và tên của chúng
                đổi chỗ giữa Eq. 16.2 và kernel; <code>MATCH</code>, <code>MISMATCH</code>, <code>DELETION</code>,{' '}
                <code>INSERTION</code> và kiểu của chuỗi không bao giờ được định nghĩa.
              </>
            }
          >
          One gap constant, and <code>int</code> sequences. The book's two penalties have the same value and their names
          swap between Eq. 16.2 and the kernel; <code>MATCH</code>, <code>MISMATCH</code>, <code>DELETION</code>,{' '}
          <code>INSERTION</code> and the sequence type are never defined.
          </Translated>
        </li>
      </ul>
      <Today checked={tr('2026-11-20 (ch16-facts A3)', 'ngày 2026-11-20 (ch16-facts A3)')}>
        <Translated
          vi={
            <p>
              Chuyển một giá trị unsigned ngoài phạm vi sang một kiểu có dấu là do cài đặt quyết định (implementation-defined) cho đến
              C++20 và theo modulo từ C++20; các bài thực hành dựng theo C++17.
            </p>
          }
        >
        <p>
          Converting an out-of-range unsigned value to a signed type is implementation-defined until C++20 and modular
          from C++20; the labs build as C++17.
        </p>
        </Translated>
      </Today>

      <h3><Translated vi="Các phép kiểm thử kiểm tra gì">What the tests check</Translated></h3>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 16.1 --backend emu</code> trước. Mọi ô của bảng, kể cả hàng 0 và cột 0, phải bằng
            đúng bản tham chiếu CPU, vì phép tính là trên số nguyên, và 256 int được cấp phát sau bảng phải còn nguyên. Trước mỗi
            lần chạy, phần trong của bảng chứa 1234567, nên một ô không ai ghi được báo cáo là chưa bao giờ được ghi. Cả hai kernel
            chạy với pad 0 và 1, và phương án cơ sở cũng được kiểm thử.
          </p>
        }
      >
      <p>
        Run <code>python3 labs/pmpp.py run 16.1 --backend emu</code> first. Every cell of the table, row 0 and column 0
        included, must equal the CPU reference exactly, since the arithmetic is on integers, and 256 ints allocated after
        the table must stay untouched. Before each run the interior of the table holds 1234567, so a cell nobody wrote is
        reported as never written. Both kernels run with pad 0 and 1, and the baseline is tested too.
      </p>
      </Translated>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-table="test-shapes">
          <thead>
            <tr>
              <th><Translated vi="Backend">Backend</Translated></th>
              <th>L_seq{/* i18n-ignore: symbol name */}</th>
              <th>W</th>
            </tr>
          </thead>
          <tbody>
            <tr><td><code>emu</code></td><td>1, 2, 7, 8, 9, 31, 33, 50</td><td>3, 4, 5, 8, 16</td></tr>
            <tr><td><code>race</code></td><td>9, 20</td><td>4, 8</td></tr>
            <tr><td><code>metal</code>, <code>cuda</code>, <code>hip</code></td><td>1, 31, 32, 33, 100, 1000, 2049</td><td>8, 16, 32, 64</td></tr>
          </tbody>
        </table>
      </div>
      <Translated
        vi={
          <p>
            Trên một thiết bị thật, một W cần nhiều luồng hơn, hoặc nhiều bộ nhớ chia sẻ hay threadgroup hơn, mỗi nhóm so với mức
            thiết bị báo cáo sẽ bị bỏ qua kèm một thông báo. Đầu vào ở mỗi độ dài là các chuỗi ngẫu nhiên trên 4 ký hiệu với ba
            seed, hai chuỗi giống hệt nhau, hai chuỗi không khớp chút nào (một bảng toàn số 0), và một cặp mà cột đầu tiên khớp
            thường xuyên. Trên trình giả lập, ba độ dài dài nhất chạy ba trong số này, vì trình giả lập phải trả giá cho mọi rào
            chắn của mọi luồng.
          </p>
        }
      >
      <p>
        On a real device a W that needs more threads, or more shared or threadgroup memory, per group than the device
        reports is skipped with a message. The inputs at each length are random sequences over 4 symbols with three seeds,
        two identical sequences, two with no match at all (an all-zero table), and a pair whose first column matches often.
        On the emulator the three longest lengths run three of these, because the emulator pays for every barrier of every
        thread.
      </p>
      </Translated>
      <Translated
        vi={
          <p>
            Cặp cuối cùng đó ở đó vì một lỗi. Ở rìa trái của bảng, các ô đã trượt chìa ra ngoài cột 0, và một ô con ở cột 1 đọc
            đầu vào tây và tây bắc từ các ô nhớ của ô mà không luồng nào tính. Khi đặt về 0, chúng được đọc như biên. Khi không đặt
            về 0, chúng chứa bất cứ thứ gì đã có ở đó. Trên trình giả lập, bộ nhớ chia sẻ không ai ghi chứa các byte{' '}
            <code>0xCD</code>, một <code>int</code> âm lớn, và phép lấy max với 0 che phần lớn điều đó: thiếu{' '}
            <code>hyInitTile</code> chỉ lộ ra ở chỗ điểm khớp của một ô con cột 1 lẽ ra phải thắng, điều mà cặp này cung cấp. Trên
            GPU, không gì bảo đảm một ô nhớ chưa ghi chứa gì, nên cùng lỗi đó có thể vẫn đạt ở đó, ở một số lần chạy hoặc mọi lần.
          </p>
        }
      >
      <p>
        That last pair is there for one defect. At the left edge of the table, sheared tiles hang over column 0, and a cell
        in column 1 reads its west and north-west inputs from tile slots that no thread computes. Zeroed, they read as the
        boundary. Not zeroed, they hold whatever was there. On the emulator, shared memory nobody wrote holds{' '}
        <code>0xCD</code> bytes, a large negative <code>int</code>, and the maximum with 0 hides most of it: a missing{' '}
        <code>hyInitTile</code> shows only where a column-1 cell's match should win, which this pair provides. On a GPU
        nothing promises what an unwritten slot holds, so the same defect may pass there, on some runs or on all of them.
      </p>
      </Translated>
      <Translated
        vi={
          <p>
            Mỗi lỗi sau đây, đứng riêng trong một lời giải vốn đúng, đều thất bại trên trình giả lập: bỏ rào chắn khỏi vòng lặp
            bước; <code>&lt;</code> thay cho <code>&lt;=</code> với cột cuối của ô đã trượt; bỏ lời gọi{' '}
            <code>hyInitTile</code>; bỏ <code>q &gt;= 1</code> khỏi phép kiểm tra tính toán của ô đã trượt;{' '}
            <code>sqLoadNW</code> thiếu trường hợp <code>q_tile == 0</code>; thiếu một ô trong một mặt sóng; bước nhảy W + 1 với
            chỉ W × W int bộ nhớ chia sẻ; và rào chắn bị dời vào trong phép kiểm tra biên. Ba trong số đó dừng lần chạy bằng một
            báo cáo AddressSanitizer chỉ ra dòng lỗi; số còn lại lộ ra thành các ô sai hoặc chưa bao giờ được ghi.
          </p>
        }
      >
      <p>
        Each of these mistakes, alone in an otherwise correct solution, fails on the emulator: the barrier removed from the
        step loop; <code>&lt;</code> instead of <code>&lt;=</code> for the sheared tiles' last column; the{' '}
        <code>hyInitTile</code> call removed; <code>q &gt;= 1</code> dropped from the sheared compute test;{' '}
        <code>sqLoadNW</code> without its <code>q_tile == 0</code> case; one tile short in a front; a stride of W + 1 with
        only W × W ints of shared memory; and the barrier moved inside the bounds test. Three of them stop the run with an
        AddressSanitizer report that names the line; the rest show as wrong or never-written cells.
      </p>
      </Translated>
      <Translated
        vi={
          <p>
            Bài thực hành cũng được dựng cho biến thể race, chạy các luồng của một khối cùng lúc dưới ThreadSanitizer:{' '}
            <code>python3 labs/pmpp.py run 16.1 --backend race</code>, ở L_seq 9 và 20 với W = 4 và 8. Một lần chạy sạch là bằng
            chứng, không phải chứng minh: nó chỉ thấy những lịch chạy đã xảy ra.
          </p>
        }
      >
      <p>
        The lab is also built for the race flavour, which runs a block's threads at the same time under ThreadSanitizer:{' '}
        <code>python3 labs/pmpp.py run 16.1 --backend race</code>, at L_seq 9 and 20 with W = 4 and 8. A clean run is
        evidence, not proof: it sees only the schedules that happened.
      </p>
      </Translated>
      <RaceStatus />

      <h3><Translated vi="Thí nghiệm">The experiment</Translated></h3>
      <Translated
        vi={
          <p>
            Làm việc này khi các kernel của bạn đã đạt, và trước khi chạy <code>--bench</code>. Trang 16.6 đã đếm số lần khởi chạy,
            số ô mỗi lần khởi chạy, số luồng bận mỗi bước và xung đột bank; không số đếm nào trong đó có đồng hồ. Hãy ghi ra ba
            con số, và chạy benchmark hai lần: một khác biệt không sống sót qua lần chạy thứ hai là nhiễu.
          </p>
        }
      >
      <p>
        Do this once your kernels pass, and before you run <code>--bench</code>. Page 16.6 counted launches, tiles per
        launch, busy threads per step and bank conflicts; none of those counts has a clock in it. Write down three numbers,
        and run the benchmark twice: a difference that does not survive a second run is noise.
      </p>
      </Translated>
      <Source section="16.6, 16.7" pages="384–396" figs={tr('Eqs. 16.3, 16.4; Fig. 16.20', 'Phương trình 16.3, 16.4; Hình 16.20')} />
      <LabPrediction
        id="ch16/labs/p-lab1-baseline"
        prompt={
          <Translated
            vi={
              <>
                Ô vuông ở W = 32, không đệm, sẽ lấp bảng trên GPU của bạn nhanh hơn phương án cơ sở bao nhiêu lần?
              </>
            }
          >
          <>
            How many times faster than the baseline will square tiles at W = 32, unpadded, fill the table on your GPU?
          </>
          </Translated>
        }
        unit={tr("× the baseline's speed", '× tốc độ của phương án cơ sở')}
        measure={baselineSpeedup}
        missing={MISSING}
        reading={
          <Translated
            vi={
              <>
                Có lợi cho các ô: 2T − 1 lần khởi chạy thay vì 2·L_seq − 1, và láng giềng lấy từ bộ nhớ chia sẻ. Bất lợi: một mặt sóng
                chứa tối đa T ô, mỗi ô một khối W luồng, và bên trong một ô, một bước có từ 1 đến W ô con, trung bình khoảng W/2
                (Eq. 16.3). Lần khởi chạy đầu và cuối của phương án cơ sở cũng chỉ có vài luồng.
              </>
            }
          >
          <>
            For the tiles: 2T − 1 launches instead of 2·L_seq − 1, and neighbours from shared memory. Against them: a front
            holds at most T tiles, one block of W threads each, and inside a tile a step has between 1 and W cells, about
            W/2 on average (Eq. 16.3). The baseline's first and last launches have only a few threads too.
          </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch16/labs/p-lab1-hyper"
        prompt={<Translated vi={<>Ở W = 32, không đệm, thời gian của kernel đã trượt chia cho thời gian của kernel vuông sẽ là bao nhiêu? (Dưới 1 nghĩa là phép trượt nhanh hơn.)</>}><>At W = 32, unpadded, the sheared kernel's time divided by the square kernel's time will be what? (Below 1 means the shear was faster.)</></Translated>}
        unit={tr("× the square kernel's time", '× thời gian của kernel vuông')}
        measure={shearedOverSquare}
        missing={MISSING}
        reading={
          <Translated
            vi={
              <>
                Có lợi cho phép trượt: W bước mỗi ô thay vì 2W − 1, mọi luồng đều bận ở mọi bước (Eq. 16.4). Bất lợi: 3T − 1 lần khởi
                chạy thay vì 2T − 1, tối đa khoảng T/2 + 1 ô mỗi mặt sóng thay vì T, các ô không trọn ở cả hai rìa, và mọi bước đi dọc
                một cột của hình vuông được lưu, các luồng liên tiếp cách nhau W int.
              </>
            }
          >
          <>
            For the shear: W steps per tile instead of 2W − 1, every thread busy at every step (Eq. 16.4). Against it: 3T − 1
            launches instead of 2T − 1, at most about T/2 + 1 tiles per front instead of T, part tiles at both edges, and
            every step walks down a column of the stored square, W ints apart from one thread to the next.
          </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch16/labs/p-lab1-pad"
        prompt={<Translated vi={<>Ở W = 32, thời gian của kernel đã trượt khi có một khoảng trống mỗi hàng của ô chia cho thời gian khi không có sẽ là bao nhiêu?</>}><>At W = 32, the sheared kernel's time with one gap per tile row divided by its time without will be what?</></Translated>}
        unit={tr('× the unpadded sheared time', '× thời gian của kernel đã trượt không đệm')}
        measure={paddedOverUnpadded}
        missing={MISSING}
        reading={
          <Translated
            vi={
              <>
                Theo mô hình bank của NVIDIA, cột không đệm là một xung đột 31 đến 32 chiều ở W = 32, và một khoảng trống mỗi hàng
                loại bỏ nó. AMD có 32 hoặc 64 bank tùy kiến trúc. Apple không ghi nhận bank nào cho bộ nhớ threadgroup, nên trên Metal
                con số này là một phép đo không có mô hình nào của nhà cung cấp đứng sau, và một tỉ lệ gần 1 chỉ nói rằng truy cập này
                không phải là thứ giới hạn kernel này trên thiết bị này.
              </>
            }
          >
          <>
            Under NVIDIA's bank model the unpadded column is a 31- to 32-way conflict at W = 32, and one gap per row removes
            it. AMD has 32 or 64 banks depending on the architecture. Apple documents no threadgroup-memory banks, so on
            Metal this number is a measurement with no vendor model behind it, and a ratio near 1 says only that this
            access was not what limited this kernel on this device.
          </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Rồi nói xem đệm đẩy kernel <b>vuông</b> theo hướng nào. Các truy cập của nó chạy dọc một đường chéo phụ của ô, các luồng
            liên tiếp cách nhau W − 1 int, mà cùng mô hình NVIDIA thấy là không xung đột khi không đệm; một khoảng trống mỗi hàng
            biến khoảng cách đó thành W int, tới xung đột 32 chiều ở W = 32. Sách chỉ đệm ô đã trượt. Phép đo, từ cùng các hàng
            với bảng bên dưới:
          </p>
        }
      >
      <p>
        Then say which way padding moves the <b>square</b> kernel. Its accesses run along an anti-diagonal of the tile, W −
        1 ints apart from one thread to the next, which the same NVIDIA model finds conflict-free unpadded; one gap per row
        makes that W ints, up to a 32-way conflict at W = 32. The book pads only the sheared tile. The measurement, from the
        same rows as the table below:
      </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Số xung đột là phép tính của chúng tôi dựa trên quy tắc được NVIDIA ghi nhận (ch16-facts E1); sách không đưa ra con số
              nào. Các số 64 bank của AMD là một mô hình (ch16-facts B3, E1); Apple không ghi nhận bank nào (infra-facts C13).
            </p>
          }
        >
        <p>
          The conflict counts are our arithmetic on NVIDIA's documented rule (ch16-facts E1); the book gives none. AMD's
          64-bank figures are a model (ch16-facts B3, E1); Apple documents no banks (infra-facts C13).
        </p>
        </Translated>
      </Beyond>
      <SquarePadding />
      <LabResults
        lab={l1}
        cols={[
          { head: tr('Variant', 'Biến thể'), cell: variant },
          { head: 'W', num: true, cell: (r) => ifTiled(r, 'tile') },
          { head: tr('Pad', 'Đệm'), num: true, cell: (r) => ifTiled(r, 'pad') },
          { head: tr('Launches', 'Số lần khởi chạy'), num: true, cell: (r) => count(r.launches) },
          { head: tr('L_seq', 'L_seq'), num: true, cell: (r) => count(r.lseq) },
          msCol,
          numCol(tr('GCUPS', 'GCUPS'), 'gcups'),
          numCol(tr('Over baseline', 'So với phương án cơ sở'), 'speedup_vs_baseline'),
        ]}
      />
      <Translated
        vi={
          <p>
            Cách benchmark hoạt động: hai chuỗi ngẫu nhiên trên 4 ký hiệu. L_seq bắt đầu ở 1024 và nhân đôi cho đến khi kernel
            vuông ở W = 32 mất ít nhất 20 ms hoặc L_seq đạt 8192; mọi hàng sau đó dùng chính bảng đó. Nó đo phương án cơ sở, và cả
            hai kernel ở W = 8, 16, 32 và 64 (khi thiết bị cho phép) với pad 0 và 1. Mỗi hàng là kết quả tốt nhất của vài lần
            chạy. GCUPS là L_seq² lần cập nhật ô mỗi giây, chia cho 10⁹; "so với phương án cơ sở" là thời gian của phương án cơ sở
            chia cho thời gian của hàng. Một W mà thiết bị không chạy được sẽ được in là bị bỏ qua, kèm lý do.
          </p>
        }
      >
      <p>
        How the benchmark works: two random sequences over 4 symbols. L_seq starts at 1024 and doubles until the square
        kernel at W = 32 takes at least 20 ms or L_seq reaches 8192; every row then uses that one table. It times the
        baseline, and both kernels at W = 8, 16, 32 and 64 (where the device allows) with pad 0 and 1. Each row is the best
        of several runs. GCUPS is L_seq² cell updates per second, divided by 10⁹; "over baseline" is the baseline's time
        divided by the row's. A W the device cannot run is printed as skipped, with the reason.
      </p>
      </Translated>
      <Translated
        vi={
          <p>
            Không có trong bài thực hành này: lệnh DPX của bài tập 6, vốn chỉ có trên CUDA (trang{' '}
            <a href={href('ch16', '16.6')}>16.6</a> có câu hỏi), và các kernel dùng cờ của bài tập 4 và 5: chỉ CUDA ghi nhận một
            dạng an toàn, dạng của HIP là unverified và Metal không có (trang <a href={href('ch16', '16.3')}>16.3</a>).{' '}
            <code>README.md</code> của bài thực hành nói cách thử lời gọi DPX so với <code>max4</code> trên một GPU NVIDIA.
          </p>
        }
      >
      <p>
        Not in this lab: exercise 6's DPX instruction, which is CUDA only (page <a href={href('ch16', '16.6')}>16.6</a> has
        the question), and the flag-based kernels of exercises 4 and 5: only CUDA documents a safe form, HIP's is unverified
        and Metal has none (page <a href={href('ch16', '16.3')}>16.3</a>). The lab's <code>README.md</code> says how to try
        the DPX call against <code>max4</code> on an NVIDIA GPU.
      </p>
      </Translated>
      <ReferenceSolution lab={l1} />
    </LabsPage>
  );
}
