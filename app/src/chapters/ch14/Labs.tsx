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
  type BenchSet,
} from '../../components/LabKit';
import { fmt, num, type BenchRow, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 14's own content.
// No question on this page: the labs are the exercise. Its four predictions are stored, not
// graded, under ch14/labs/p-lab1-oddeven, p-lab1-cpu, p-lab2-cpu and p-lab3-batch1. Lab 14.2
// has two more on the pages they belong to: ch14/14.6/p-local-gain and ch14/14.7/p-best-r.
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch14', [
  { num: '14.1', dir: 'lab1_twosorts', name: 'odd-even and one-bit radix', vi: { name: 'odd-even và radix một bit' } },
  { num: '14.2', dir: 'lab2_multibit', name: 'multi-bit, local sort', vi: { name: 'nhiều bit, sắp xếp cục bộ' } },
  { num: '14.3', dir: 'lab3_sortrows', name: 'sorted rows (M5)', vi: { name: 'các hàng đã sắp xếp (M5)' } },
]);
const [l1, l2, l3] = LABS;

/** Identifier colours in the reference solutions: where each value lives. */
const GLOBAL_1 = ['data', 'hasChanged', 'flag', 'keys', 'vals', 'tmpKeys', 'tmpVals', 'bits', 'scanScratch', 'keysIn', 'valsIn', 'keysOut', 'valsOut', 'onesBefore'];
const TIERS_1: Record<string, Tier> = {
  ...Object.fromEntries(GLOBAL_1.map((k) => [k, 'global' as Tier])),
  i: 'reg', tmp: 'reg', key: 'reg', bit: 'reg', dst: 'reg',
};
const TIERS_2: Record<string, Tier> = {
  ...Object.fromEntries([...GLOBAL_1, 'keysLocal', 'valsLocal', 'table', 'tableScanned', 'localStarts'].map((k) => [k, 'global' as Tier])),
  tile_s: 'shared', scan_s: 'shared', start_s: 'shared', offset_s: 'shared', keys_s: 'shared', vals_s: 'shared', sgTotals: 'shared',
  t: 'reg', c: 'reg', p: 'reg', tile: 'reg', base: 'reg', count: 'reg', digit: 'reg', dst: 'reg', myKeys: 'reg', myVals: 'reg', myOnes: 'reg',
};
const TIERS_3: Record<string, Tier> = {
  probs: 'global', probsBits: 'global', sortedProbs: 'global', sortedIdx: 'global', scratch: 'global',
  keys: 'global', idx: 'global', keysIn: 'global', idxIn: 'global', keysOut: 'global', idxOut: 'global', idxSorted: 'global',
  offset_s: 'shared', tileStart_s: 'shared', keys_s: 'shared', idx_s: 'shared', warpSums_s: 'shared',
  hist: 'shared', offset: 'shared', tile_start: 'shared', group_sums: 'shared',
  t: 'reg', u: 'reg', key: 'reg', digit: 'reg', dst: 'reg',
};

// ---------------------------------------------------------------- reading the reports
const N22 = 4194304;
const VOCAB = 151936;
const ms = (s: number | undefined) => (s ? `${fmt(s * 1000)} ms` : '?'); // i18n-ignore: a number and its unit
const big = (v: unknown) => (num(v) != null ? num(v)!.toLocaleString('en-US') : '');

/** Lab 14.1: odd-even sort's wall time over the one-bit radix sort's wall time, n = 4,096. */
const oddEvenSlowdown = (reports: Reports) =>
  measure(reports, l1, (s: BenchSet) => {
    const oe = benchRow(s, 'oddeven', { n: 4096 });
    const rx = benchRow(s, 'radix1', { n: 4096 });
    const a = num(oe?.seconds);
    const b = num(rx?.wall_seconds) ?? num(rx?.seconds);
    if (!a || !b) return null;
    return {
      value: a / b,
      detail: tr(
        `odd-even ${ms(a)} in ${big(oe!.steps)} steps against ${ms(b)} for the radix sort, both host wall time, n = 4,096`,
        `odd-even ${ms(a)} trong ${big(oe!.steps)} bước so với ${ms(b)} cho radix sort, cả hai là thời gian thực đo trên host, n = 4,096`,
      ),
    };
  });

/** Lab 14.1: std::stable_sort's time over the one-bit radix sort's, 2^22 pairs. */
const radixOverCpu = (reports: Reports) =>
  measure(reports, l1, (s: BenchSet) => {
    const cpu = num(benchRow(s, 'cpu_stable_sort', { n: N22 })?.seconds);
    const rx = num(benchRow(s, 'radix1', { n: N22 })?.seconds);
    if (!cpu || !rx) return null;
    return {
      value: cpu / rx,
      detail: tr(
        `${ms(rx)} for the radix sort against ${ms(cpu)} for std::stable_sort on one CPU thread, ${N22.toLocaleString('en-US')} pairs`,
        `${ms(rx)} cho radix sort so với ${ms(cpu)} cho std::stable_sort trên một luồng CPU, ${N22.toLocaleString('en-US')} cặp`,
      ),
    };
  });

/** Lab 14.2: the fastest local-sort row against std::stable_sort, at the size the CPU row was timed. */
function bestLocal(rows: BenchRow[]): { best: BenchRow; cpu: BenchRow } | null {
  const cpu = rows.find((r) => r.name === 'cpu_stable_sort' && num(r.seconds));
  if (!cpu) return null;
  const local = rows.filter((r) => r.name === 'local' && num(r.seconds) && r.n === cpu.n);
  const best = local.reduce<BenchRow | undefined>((x, r) => (!x || num(r.seconds)! < num(x.seconds)! ? r : x), undefined);
  return best ? { best, cpu } : null;
}
const multiOverCpu = (reports: Reports) =>
  measure(reports, l2, (s: BenchSet) => {
    const m = bestLocal(s.rows);
    if (!m) return null;
    const a = num(m.cpu.seconds)!;
    const b = num(m.best.seconds)!;
    return {
      value: a / b,
      detail: tr(
        `fastest row r = ${String(m.best.radix_bits)}, c = ${String(m.best.keys_per_thread)} at ${ms(b)}, against ${ms(a)} for std::stable_sort on one CPU thread, ${big(m.cpu.n)} pairs`,
        `hàng nhanh nhất r = ${String(m.best.radix_bits)}, c = ${String(m.best.keys_per_thread)} với ${ms(b)}, so với ${ms(a)} cho std::stable_sort trên một luồng CPU, ${big(m.cpu.n)} cặp`,
      ),
    };
  });

/** Lab 14.3: std::stable_sort per row over the GPU sort, one row of 151,936; batch 64 alongside. */
const ratioAt = (s: BenchSet, rows: number) => {
  const gpu = num(benchRow(s, 'sort_rows', { rows, cols: VOCAB })?.seconds);
  const cpu = num(benchRow(s, 'cpu_stable_sort_rows', { rows, cols: VOCAB })?.seconds);
  return gpu && cpu ? { gpu, cpu, ratio: cpu / gpu } : null;
};
const rowsOverCpu = (reports: Reports) =>
  measure(reports, l3, (s: BenchSet) => {
    const one = ratioAt(s, 1);
    if (!one) return null;
    const many = ratioAt(s, 64);
    return {
      value: one.ratio,
      detail:
        tr(
          `${ms(one.gpu)} per call on the device against ${ms(one.cpu)} on one CPU thread, 1 × 151,936`,
          `${ms(one.gpu)} mỗi lần gọi trên device so với ${ms(one.cpu)} trên một luồng CPU, 1 × 151,936`,
        ) +
        (many
          ? tr(
              `; at 64 × 151,936 the ratio is ${fmt(many.ratio)} (${ms(many.gpu)} against ${ms(many.cpu)})`,
              `; ở 64 × 151,936 tỉ số là ${fmt(many.ratio)} (${ms(many.gpu)} so với ${ms(many.cpu)})`,
            )
          : tr('; no batch-64 row', '; không có hàng batch 64')),
    };
  });

const missingBench = (labNum: string) => (
  <Translated
    vi={
      <>
        Phần này cần các hàng benchmark của lab {labNum} từ một thiết bị thật. Hãy chạy{' '}
        <code>python3 labs/pmpp.py run {labNum} --bench</code> khi các kernel của bạn đã chạy đạt (hoặc với{' '}
        <code>--solution</code> để đo lời giải tham chiếu). Trình giả lập không bao giờ được đo hiệu năng.
      </>
    }
  >
    <>
      This needs the benchmark rows of lab {labNum} from a real device. Run{' '}
      <code>python3 labs/pmpp.py run {labNum} --bench</code> once your kernels pass (or with <code>--solution</code> to time
      the reference). The emulator is never benchmarked.
    </>
  </Translated>
);

// ---------------------------------------------------------------- results tables
const WHAT_1: Record<string, string> = {
  get oddeven() { return tr('odd-even sort, whole host loop', 'odd-even sort, cả vòng lặp host'); },
  get radix1() { return tr('one-bit radix sort', 'radix sort một bit'); },
  get cpu_stable_sort() { return tr('std::stable_sort, one CPU thread', 'std::stable_sort, một luồng CPU'); },
};
const WHAT_2: Record<string, string> = {
  get onebit_global() { return tr('one bit, direct scatter (provided)', 'một bit, phân tán trực tiếp (được cung cấp)'); },
  get local() { return tr('local sort', 'sắp xếp cục bộ'); },
  get cpu_stable_sort() { return tr('std::stable_sort, one CPU thread', 'std::stable_sort, một luồng CPU'); },
};
const WHAT_3: Record<string, string> = {
  get sort_rows() { return tr('GPU sort', 'sắp xếp trên GPU'); },
  get cpu_stable_sort_rows() { return tr('std::stable_sort, one CPU thread', 'std::stable_sort, một luồng CPU'); },
};
const what = (names: Record<string, string>) => ({ get head() { return tr('What ran', 'Cái gì đã chạy'); }, cell: (r: BenchRow) => names[r.name] ?? r.name });
const nCol = { head: 'n', num: true, cell: (r: BenchRow) => big(r.n) };
const wallCol = {
  get head() {
    return tr('Wall, ms', 'Thời gian thực, ms');
  },
  num: true,
  cell: (r: BenchRow) => (num(r.wall_seconds) != null ? fmt(num(r.wall_seconds)! * 1000) : r.name === 'oddeven' ? fmt(num(r.seconds)! * 1000) : ''),
};

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS}>
      <LabsIntro>
        <Translated
          vi={
            <>
              Ba lab, mỗi lab trong thư mục riêng dưới <code>labs/ch14/</code>. Lab đầu viết hai kernel được liệt kê trong chương,
              odd-even sort và một lượt radix một bit, rồi đo chúng với nhau và với một luồng CPU. Lab thứ hai làm các bài tập 1 đến 3
              của sách trong một radix sort nhiều bit có sắp xếp cục bộ. Lab thứ ba là mốc dự án M5 của dự án xuyên suốt: mọi hàng của
              một batch xác suất được sắp xếp lớn nhất trước, ổn định, kèm theo các id token.
            </>
          }
        >
          Three labs, each in its own directory under <code>labs/ch14/</code>. The first writes the chapter's two listed
          kernels, odd-even sort and a one-bit radix pass, and times them against each other and against one CPU thread. The
          second does the book's exercises 1 to 3 in one multi-bit radix sort with local sorting. The third is the
          through-line project's milestone M5: every row of a batch of probabilities sorted largest first, stably, with the
          token ids carried along.
        </Translated>
      </LabsIntro>
      <Translated
        vi={
          <p>
            Lab 14.1 là tùy chọn. Các kernel của nó được giải trong mục <a href={href('ch14', '14.2')}>14.2</a> và{' '}
            <a href={href('ch14', '14.5')}>14.5</a>, và lab 14.2 cung cấp sẵn phép sắp xếp một bit của nó làm baseline để so sánh.
          </p>
        }
      >
        <p>
          Lab 14.1 is optional. Its kernels are worked through in sections <a href={href('ch14', '14.2')}>14.2</a> and{' '}
          <a href={href('ch14', '14.5')}>14.5</a>, and lab 14.2 ships its one-bit sort ready-made as the baseline it
          measures against.
        </p>
      </Translated>
      <Beyond title={tr('Why three labs', 'Vì sao ba lab')}>
        <Translated
          vi={
            <p>
              PLAN.md xếp chương này ở mức Standard, một hoặc hai lab. Lab 14.3 là mốc dự án M5 và lab 14.2 là các bài tập 1 đến 3 của
              sách, nên hai lab đó là các lab của chương; lab 14.1 được giữ lại như một bước đầu tùy chọn, vì nó là baseline đo được
              mà lab 14.2 so sánh với.
            </p>
          }
        >
          <p>
            PLAN.md gives this chapter the Standard tier, one or two labs. Lab 14.3 is milestone M5 and lab 14.2 is the
            book's exercises 1 to 3, so those two are the chapter's labs; lab 14.1 is kept, as an optional first step, because
            it is the measured baseline that lab 14.2 compares against.
          </p>
        </Translated>
      </Beyond>
      <RunningLabs
        example="14.1"
        alsoShownIn={
          <Translated
            vi={
              <>
                các cột đo được và các dự đoán của <a href={href('ch14', '14.6')}>14.6</a>, <a href={href('ch14', '14.7')}>14.7</a> và{' '}
                <a href={href('ch14', '14.8')}>14.8</a>
              </>
            }
          >
            <>
              the measured columns and predictions of <a href={href('ch14', '14.6')}>14.6</a>,{' '}
              <a href={href('ch14', '14.7')}>14.7</a> and <a href={href('ch14', '14.8')}>14.8</a>
            </>
          </Translated>
        }
      />
      <BackendsAndEmulator />
      <Translated
        vi={
          <p>
            Mọi lab của chương này cũng được build cho biến thể race của trình giả lập, vì lab nào cũng có một cập nhật dùng chung
            cần làm đúng: một cờ ở lab 14.1, một ô (tile) và các phép scan của nó ở lab 14.2, một histogram, một ô và các offset
            chạy ở lab 14.3. <code>python3 labs/pmpp.py run 14.1 --backend race</code> chạy các luồng của một khối cùng lúc dưới
            ThreadSanitizer, công cụ báo hai lần truy cập vào một vị trí mà không có gì sắp thứ tự chúng, kèm cả hai dòng mã nguồn.
            Một lần chạy sạch là bằng chứng, không phải chứng minh: nó chỉ thấy những lịch chạy đã xảy ra (các bài kiểm thử của lab
            14.2 cho thấy điều đó nghĩa là gì với một rào chắn bị thiếu).
          </p>
        }
      >
        <p>
          Every lab of this chapter is also built for the emulator's race flavour, because each has a shared update to get
          right: a flag in lab 14.1, a tile and its scans in lab 14.2, a histogram, a tile and running offsets in lab 14.3.{' '}
          <code>python3 labs/pmpp.py run 14.1 --backend race</code> runs a block's threads at once under ThreadSanitizer,
          which reports two accesses to one location with nothing ordering them, with both source lines. A clean run is
          evidence, not proof: it sees only the schedules that happened (lab 14.2's tests show what that means for a
          missing barrier).
        </p>
      </Translated>
      <VerificationStatus />
      <Translated
        vi={
          <p>
            Hai lab dùng mã được cung cấp mà bạn không sửa, trong <code>labs/ch14/shared-g11/</code>: một phép scan loại trừ trên số
            nguyên không dấu cho số lượng bất kỳ, và baseline của lab 14.2. Phép scan được xây theo cách mục 11.9 xây: scan trong
            từng khối, scan các tổng của khối, rồi một phép cộng, nên không khối nào phải chờ khối khác. Mỗi lab kiểm thử riêng mã
            được cung cấp trước, nên một tệp được cung cấp bị hỏng sẽ lộ ra đúng là nó.
          </p>
        }
      >
        <p>
          Two labs use provided code that you do not edit, in <code>labs/ch14/shared-g11/</code>: an exclusive scan of
          unsigned integers for any count, and lab 14.2's baseline. The scan is built the way section 11.9 builds one:
          block scans, a scan of the block totals, then an add, so no block ever waits for another. Each lab tests the
          provided code on its own first, so a broken provided file shows up as itself.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Phép scan và baseline được cung cấp là của khóa học này; chương 14 của sách gọi một phép scan toàn lưới mà nó không bao giờ định nghĩa.</p>}>
          <p>The provided scan and baseline are ours; the book's chapter 14 calls a grid-wide scan it never defines.</p>
        </Translated>
      </Beyond>

      {/* ------------------------------------------------------------ 14.1 */}
      <h2>
        <Translated vi="Lab 14.1: hai phép sắp xếp đầu tiên, odd-even và radix một bit">Lab 14.1: two first sorts, odd-even and one-bit radix</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hai phép sắp xếp khóa không dấu 32 bit, viết theo cách chương này xây chúng. Phần A là odd-even sort: kernel của Hình
            14.2 và vòng lặp host mà sách mô tả nhưng không đưa ra. Phần B là một radix sort LSD một bit trên các cặp (khóa, giá
            trị): Hình 14.7, được tách tại chỗ nó gọi một phép scan trên toàn lưới.
          </p>
        }
      >
        <p>
          Two sorts of 32-bit unsigned keys, written the way the chapter builds them. Part A is odd-even sort: the kernel of
          Fig. 14.2 and the host loop the book describes but does not show. Part B is a one-bit LSD radix sort of (key, value)
          pairs: Fig. 14.7, split where it calls a scan across the whole grid.
        </p>
      </Translated>
      <Source section="14.2, 14.5" pages="331–333, 336–339" figs={tr('Figs. 14.2, 14.6 and 14.7', 'Hình 14.2, 14.6 và 14.7')} />
      <Translated
        vi={
          <p>
            <code>oddEvenStep(data, hasChanged, n, isOddStep)</code> giao cho luồng t cặp (i, i + 1), với i = 2t ở bước chẵn và 2t +
            1 ở bước lẻ, sắp cặp đó đúng thứ tự và, nếu có đổi chỗ, đặt cờ bằng một lệnh ghi nguyên tử. Nó trả về ngay khi n nhỏ hơn
            2 hoặc cặp của nó vượt quá cuối. <code>launchOddEvenSort(data, flag, n)</code> luân phiên bước chẵn và lẻ, bắt đầu bằng
            bước chẵn: xóa cờ, khởi chạy một bước chẵn và bước lẻ sau nó, rồi đọc cờ, và chỉ dừng khi cặp bước đó không đổi chỗ gì.
            Nó trả về số bước đã khởi chạy.
          </p>
        }
      >
        <p>
          <code>oddEvenStep(data, hasChanged, n, isOddStep)</code> gives thread t the pair (i, i + 1), with i = 2t on an
          even step and 2t + 1 on an odd one, puts the pair in order and, if it swapped, sets the flag with an atomic store.
          It returns at once when n is below 2 or its pair runs past the end. <code>launchOddEvenSort(data, flag, n)</code>{' '}
          alternates even and odd steps, starting even: clear the flag, launch an even step and the odd step after it, then
          read the flag, and stop only when that pair of steps made no swap. It returns the number of steps launched.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Với phần B, Hình 14.7 trích các bit, gọi <code>gridExclusiveScan</code>, rồi phân tán. Sách không bao giờ định nghĩa hàm
            đó, và một phép scan xuyên các khối bên trong một kernel cần mọi khối chờ nhau: một lần khởi chạy cooperative trên CUDA
            và HIP, không có gì được tài liệu hóa trên Metal. Nên một lượt ở đây là ba lần khởi chạy. <code>radixExtractBits</code>{' '}
            ghi <code>bits[i]</code> và <code>bits[n] = 0</code>; <code>launchExclusiveScan(bits, n + 1, scratch)</code> được cung
            cấp biến chúng thành số bit 1 phía trước, với tổng ở <code>bits[n]</code>; <code>radixScatter</code> áp dụng hai công
            thức đích đến và di chuyển mỗi giá trị cùng khóa của nó. <code>launchRadixSortOneBit</code> chạy 32 lượt, đọc một cặp
            buffer và ghi cặp kia, và để kết quả trong <code>keys</code> và <code>vals</code>.
          </p>
        }
      >
        <p>
          For part B, Fig. 14.7 extracts the bits, calls <code>gridExclusiveScan</code>, then scatters. The book never
          defines that function, and a scan across blocks inside one kernel needs every block to wait for the others: a
          cooperative launch on CUDA and HIP, nothing documented on Metal. So a pass here is three launches.{' '}
          <code>radixExtractBits</code> writes <code>bits[i]</code> and <code>bits[n] = 0</code>; the provided{' '}
          <code>launchExclusiveScan(bits, n + 1, scratch)</code> turns that into ones-before, with the total at{' '}
          <code>bits[n]</code>; <code>radixScatter</code> applies the two destination formulas and moves each value with its
          key. <code>launchRadixSortOneBit</code> runs 32 passes, reading one pair of buffers and writing the other, and
          leaves the result in <code>keys</code> and <code>vals</code>.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch04-facts.md A7, C3; ch14-facts.md C4)', '2026-11-20 (notes/today/ch04-facts.md mục A7, C3; ch14-facts.md mục C4)')}>
        <Translated
          vi={
            <p>
              Rào chắn toàn lưới chỉ tồn tại trên CUDA và HIP dưới một lần khởi chạy cooperative với mọi khối thường trú. Metal không có,
              và không ghi nhận trong tài liệu bảo đảm tiến triển nào giữa các threadgroup.
            </p>
          }
        >
          <p>
            A grid-wide barrier exists on CUDA and HIP only under a cooperative launch with every block resident. Metal has
            none, and documents no forward progress between threadgroups.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Giá trị là vị trí đầu vào của các khóa; trong lab 14.3 chúng là id token. Với vị trí làm giá trị, bài kiểm thử thấy được
            tính ổn định trực tiếp: giữa các khóa bằng nhau, các vị trí phải tăng dần.
          </p>
        }
      >
        <p>
          The values are the keys' input positions; in lab 14.3 they are token ids. With positions as values the test sees
          stability directly: among equal keys, the positions must increase.
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
                  <code>oddEvenStep</code>, <code>launchOddEvenSort</code>, <code>radixExtractBits</code>, <code>radixScatter</code> và{' '}
                  <code>launchRadixSortOneBit</code>
                </>
              }
            >
              <>
                <code>oddEvenStep</code>, <code>launchOddEvenSort</code>, <code>radixExtractBits</code>,{' '}
                <code>radixScatter</code> and <code>launchRadixSortOneBit</code>
              </>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated
              vi={
                <>
                  <code>odd_even_step</code>, <code>radix_extract_bits</code> và <code>radix_scatter</code>; cờ là một{' '}
                  <code>device atomic_uint*</code> được đặt bằng <code>atomic_store_explicit</code>
                </>
              }
            >
              <>
                <code>odd_even_step</code>, <code>radix_extract_bits</code> and <code>radix_scatter</code>; the flag is a{' '}
                <code>device atomic_uint*</code> set with <code>atomic_store_explicit</code>
              </>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated
              vi={
                <>
                  <code>runOddEvenSort</code>, hàm commit một command buffer cho mỗi cặp bước chẵn-lẻ, chờ nó và đọc cờ, và{' '}
                  <code>encodeRadixSortOneBit</code>, hàm đưa cả 32 lượt, kể cả các dispatch của phép scan được cung cấp, vào một compute
                  encoder mà nó được giao
                </>
              }
            >
              <>
                <code>runOddEvenSort</code>, which commits one command buffer per even-odd pair of steps, waits for it and
                reads the flag, and <code>encodeRadixSortOneBit</code>, which puts all 32 passes, the provided scan's
                dispatches included, into the one compute encoder it is given
              </>
            </Translated>,
          ],
        ]}
      />
      <Today checked={tr('2026-11-20 (infra-facts.md C1, C12; ch05-facts.md C6; ch02-facts.md C9)', '2026-11-20 (infra-facts.md mục C1, C12; ch05-facts.md mục C6; ch02-facts.md mục C9)')}>
        <Translated
          vi={
            <p>
              Các hàm nguyên tử của Metal chỉ nhận con trỏ kiểu atomic. CPU chỉ được đọc một shared buffer mà GPU đã ghi khi command
              buffer đã hoàn tất. Các dispatch trong một compute encoder thông thường chạy theo thứ tự và thấy được các lần ghi của
              nhau.
            </p>
          }
        >
          <p>
            Metal's atomic functions take only atomic-typed pointers. The CPU may read a shared buffer the GPU wrote only once
            the command buffer has completed. Dispatches in one plain compute encoder run in order and see each other's
            writes.
          </p>
        </Translated>
      </Today>

      <h3>
        <Translated vi="Năm cái bẫy trên trang sách">Five traps on the pages</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Mỗi cái biến một cách đọc theo nghĩa đen của chương thành một chương trình sai hoặc dễ vỡ. README của lab nêu tên chúng
            mà không kèm mã, và lời giải tham chiếu nói trong phần chú thích chỗ nào nó khác với trang sách.
          </p>
        }
      >
        <p>
          Each turns a literal reading of the chapter into a wrong or fragile program. The lab's README names them without
          the code, and the reference solution says in its comments where it differs from the page.
        </p>
      </Translated>
      <ul>
        <Translated
          vi={
            <li>
              <b>Phép scan N + 1.</b> Hình 14.7 đọc <code>bits[N]</code> làm tổng số bit 1; phần tử đó chỉ tồn tại nếu phép scan phủ N +
              1 phần tử. Hình 14.6 vẽ N.
            </li>
          }
        >
          <li>
            <b>The N + 1 scan.</b> Fig. 14.7 reads <code>bits[N]</code> as the total number of ones; that entry exists only
            if the scan covers N + 1 entries. Fig. 14.6 draws N.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Lỗi "zero bucket".</b> Ở p. 337, câu giới thiệu công thức thứ hai nói nó dành cho khóa của bucket 0. Công thức đó là
              của bucket 1.
            </li>
          }
        >
          <li>
            <b>The "zero bucket" slip.</b> On p. 337 the sentence that introduces the second formula says it is for keys of
            the zero bucket. The formula is the one bucket's.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Quay vòng số không dấu.</b> Hình 14.2 kiểm tra <code>i &lt; N - 1</code>; với N không dấu bằng 0, phép đó quay vòng và
              mọi luồng đều qua. Chương trình điều khiển (driver) không bao giờ khởi chạy với n = 0, nhưng hãy cứ viết phép kiểm tra
              sao cho nó không thể quay vòng.
            </li>
          }
        >
          <li>
            <b>The unsigned wrap.</b> Fig. 14.2 tests <code>i &lt; N - 1</code>; with an unsigned N of 0 that wraps and
            every thread passes. The driver never launches with n = 0, so write the guard so that it cannot wrap anyway.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Quy tắc dừng.</b> Một bước yên lặng không chứng minh gì: <code>[1, 3, 2, 4]</code> không đổi chỗ gì ở bước đầu tiên,
              bước chẵn. Hai bước liên tiếp, một chẵn và một lẻ theo thứ tự bất kỳ, đều yên lặng, mới cho thấy mọi cặp đúng thứ tự.
              Đọc cờ một lần mỗi cặp chẵn-lẻ, như lab này làm, chỉ thấy các cặp yên lặng bắt đầu ở bước chẵn, nên nó có thể khởi chạy
              thừa một bước, đổi lại số lần đọc giảm một nửa.
            </li>
          }
        >
          <li>
            <b>The stopping rule.</b> One quiet step proves nothing: <code>[1, 3, 2, 4]</code> makes no swap on its first,
            even, step. Two steps in a row, one even and one odd in either order, both quiet, show that every pair is in
            order. Reading the flag once per even-odd pair, as this lab does, sees only the quiet pairs that start on an
            even step, so it may launch one step more than necessary, in exchange for half the reads.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Tranh chấp trên cờ.</b> Nhiều luồng có thể đặt <code>hasChanged</code> trong một bước. Sách gọi nó là vô hại rồi cảnh
              báo rằng mô hình bộ nhớ C++ dù sao cũng coi nó là hành vi không xác định (p. 333). Hãy dùng lệnh ghi nguyên tử.
            </li>
          }
        >
          <li>
            <b>The flag race.</b> Many threads may set <code>hasChanged</code> in one step. The book calls it benign and then
            warns that the C++ memory model makes it undefined behaviour anyway (p. 333). Use an atomic store.
          </li>
        </Translated>
      </ul>
      <Source section="14.2, 14.5" pages="331–339" figs={tr('Figs. 14.2, 14.6 and 14.7', 'Hình 14.2, 14.6 và 14.7')} />

      <h3>
        <Translated vi="Các bài kiểm thử kiểm tra gì">What the tests check</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Hãy chạy <code>python3 labs/pmpp.py run 14.1 --backend emu</code> trước; nó mất vài phút, vì mọi luồng GPU đều chạy trên
            CPU. Phép scan được cung cấp được kiểm thử riêng. Odd-even sort chạy ở n = 1 (với 2), 3, 4, 8, 33, 256 và 1000 trên đầu
            vào ngẫu nhiên, đảo ngược, đã sắp xếp, toàn bằng nhau và lớn nhất đứng đầu, và trên cái bẫy ở trên; kết quả phải được sắp
            xếp và là một hoán vị của đầu vào, và số bước phải chẵn, ít nhất 2 khi n từ 2 trở lên, và nhiều nhất là n + 2 làm tròn
            lên thành số cặp nguyên. Radix sort chạy ở n = 1 (với 2), 31, 32, 33, 1000, 4097, 2^16 và 2^20 trên khóa ngẫu nhiên toàn
            dải (bit cao nhất được đặt khoảng nửa số lần), nhiều giá trị trùng, toàn bằng nhau, đã sắp xếp, đảo ngược và lũy thừa của
            hai; khóa phải bằng đúng của <code>std::stable_sort</code> và giá trị phải đúng hoán vị của nó. Các trình giả lập dừng ở
            kích thước nhỏ hơn.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 14.1 --backend emu</code> first; it takes a few minutes, because every GPU
          thread runs on the CPU. The provided scan is tested on its own. Odd-even sort runs at n = 1 (with 2), 3, 4, 8, 33,
          256 and 1000 on random, reversed, sorted, all-equal and largest-first input and on the trap above; the result must be
          sorted and a permutation of the input, and the step count even, at least 2 for n of 2 or more, and at most n + 2
          rounded up to whole pairs. The radix sort runs at n = 1 (with 2), 31, 32, 33, 1000, 4097, 2^16 and 2^20 on
          full-range random keys (the top bit set about half the time), heavy ties, all equal, sorted, reversed and powers of
          two; keys must equal <code>std::stable_sort</code>'s and values its permutation exactly. The emulators stop at
          smaller sizes.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Trước mỗi lần chạy, chương trình điều khiển ghi <code>0xCDCDCDCD</code> vào <code>bits</code>, vùng nháp của phép scan,
            các buffer tạm và 64 ô canh gác sau cuối mỗi mảng, nên một lượt ghi quá n sẽ bị bắt. Trên backend race, một lệnh ghi
            thường vào cờ được báo là data race kèm cả hai dòng mã nguồn; trình giả lập không thấy được, vì nó không bao giờ ngắt một
            luồng giữa các điểm đồng bộ hóa.
          </p>
        }
      >
        <p>
          Before every run the driver writes <code>0xCDCDCDCD</code> into <code>bits</code>, the scan's scratch, the
          temporary buffers and 64 guard slots past the end of every array, so a pass that writes past n is caught. On the
          race backend a plain store to the flag is reported as a data race with both source lines; the emulator cannot see
          it, because it never interrupts a thread between synchronisation points.
        </p>
      </Translated>
      <RaceStatus lab={l1} />

      <h3>
        <Translated vi="Trước khi đo hiệu năng">Before you benchmark</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Trước hết hãy viết ra lý do, từ hai phép đếm. Số lần khởi chạy và lượt đi về với host: odd-even sort khởi chạy một kernel
            mỗi bước, khoảng n lần, và chờ GPU một lần mỗi cặp bước để đọc cờ; radix sort khởi chạy 32 × (2 + số lần khởi chạy của
            phép scan) kernel và không bao giờ chờ ở giữa. Số byte mỗi lượt: một lượt radix đọc mọi cặp và ghi các bit, scan n + 1
            phần tử, rồi đọc khóa, giá trị và bit và ghi khóa và giá trị, lặp 32 lần, so với cỡ n log n phép so sánh và di chuyển
            của <code>std::stable_sort</code>.
          </p>
        }
      >
        <p>
          Write down why first, from two counts. Launches and host round trips: odd-even sort launches one kernel per step,
          about n of them, and waits for the GPU once per pair of steps to read the flag; the radix sort launches 32 × (2 +
          the scan's launches) kernels and never waits in the middle. Bytes per pass: one radix pass reads every pair and
          writes the bits, scans n + 1 entries, then reads keys, values and bits and writes keys and values, 32 times over,
          against <code>std::stable_sort</code>'s order of n log n comparisons and moves.
        </p>
      </Translated>
      <LabPrediction
        id="ch14/labs/p-lab1-oddeven"
        prompt={
          <Translated vi={<>Ở n = 4,096, odd-even sort của bạn sẽ mất thời gian gấp bao nhiêu lần radix sort một bit của bạn, cả hai được đo bằng đồng hồ của host?</>}>
            <>At n = 4,096, how many times longer than your one-bit radix sort will your odd-even sort take, both timed on the host's clock?</>
          </Translated>
        }
        unit={tr("× the radix sort's time", '× thời gian của radix sort')}
        measure={oddEvenSlowdown}
        missing={missingBench('14.1')}
        reading={
          <Translated
            vi={
              <>
                Ở đây cả hai phép sắp xếp được host đo thời gian, từ lần khởi chạy đầu đến khi có kết quả, nên odd-even sort trả giá cho
                mọi lần khởi chạy và mọi lần đọc cờ. Hãy so cột số bước với n trước khi giải thích con số.
              </>
            }
          >
            <>
              Both sorts are timed by the host here, from the first launch to the result, so odd-even sort pays for every
              launch and every read of the flag. Compare the steps column with n before you explain the number.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch14/labs/p-lab1-cpu"
        prompt={
          <Translated vi={<>Ở 2^22 cặp, radix sort một bit của bạn sẽ nhanh hơn <code>std::stable_sort</code> trên một luồng CPU bao nhiêu lần? (Dưới 1 nghĩa là chậm hơn.)</>}>
            <>At 2^22 pairs, how many times faster than one CPU thread's <code>std::stable_sort</code> will your one-bit radix sort be? (Below 1 means slower.)</>
          </Translated>
        }
        unit={tr('× the speed of std::stable_sort', '× tốc độ của std::stable_sort')}
        measure={radixOverCpu}
        missing={missingBench('14.1')}
        reading={
          <Translated
            vi={
              <>
                Thời gian radix là thời gian device của công việc trên Metal, và là thời gian của host bao quanh các lần khởi chạy và lần
                đồng bộ hóa cuối trên CUDA và HIP; thời gian CPU là một phép đo trên host với cùng các cặp. Một phép sắp xếp một bit thực
                hiện 32 lượt qua mọi cặp; các mục 14.6 đến 14.8 nói về việc làm ít lượt hơn và rẻ hơn.
              </>
            }
          >
            <>
              The radix time is the device time of the work on Metal, and the host's time around the launches and the final
              synchronisation on CUDA and HIP; the CPU time is a host measurement of the same pairs. A one-bit sort makes 32
              passes over every pair; sections 14.6 to 14.8 are about making fewer, cheaper ones.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l1}
        cols={[what(WHAT_1), nCol, { head: tr('Steps', 'Số bước'), num: true, cell: (r) => big(r.steps) }, msCol, wallCol, numCol('Mkeys/s', 'mkeys_per_s')]}
      />
      <Translated
        vi={
          <p>
            Benchmark hoạt động thế nào: odd-even sort ở n = 1,024 và 4,096, đo quanh cả vòng lặp host; radix sort ở 1,024, 4,096,
            2^16, 2^20 và 2^22, trên khóa ngẫu nhiên toàn dải; <code>std::stable_sort</code> trên cùng các cặp với một luồng CPU. Mỗi
            hàng là kết quả tốt nhất trong 5 lần chạy sau một lần khởi động, và mọi kết quả được đo đều được kiểm tra trước khi báo
            cáo.
          </p>
        }
      >
        <p>
          How the benchmark works: odd-even sort at n = 1,024 and 4,096, timed around the whole host loop; the radix sort at
          1,024, 4,096, 2^16, 2^20 and 2^22, on random full-range keys; <code>std::stable_sort</code> of the same pairs on one
          CPU thread. Each row is the best of 5 runs after a warm-up, and every timed result is checked before it is
          reported.
        </p>
      </Translated>
      <ReferenceSolution lab={l1} tiers={TIERS_1} />

      {/* ------------------------------------------------------------ 14.2 */}
      <h2>
        <Translated vi="Lab 14.2: radix sort nhiều bit với sắp xếp cục bộ">Lab 14.2: multi-bit radix sort with local sorting</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một radix sort LSD ổn định trên các cặp (khóa, giá trị) làm cùng lúc các bài tập 1, 2 và 3 của sách: sắp xếp cục bộ
            trong bộ nhớ chia sẻ (shared memory) của mục 14.6, chữ số r bit như ở 14.7, và c khóa mỗi luồng như ở 14.8. Cả r và c đều
            chạy từ 1 đến 8 và được chọn lúc chạy. Một khối có 256 luồng và sở hữu một ô (tile) gồm 256c cặp liên tiếp; ô cuối có thể
            không đầy, và phần đệm của nó không được đếm cũng không được ghi. Phần B của lab 14.1 là baseline của lab này, được cung
            cấp sẵn, nên bạn có thể bắt đầu từ đây.
          </p>
        }
      >
        <p>
          One stable LSD radix sort of (key, value) pairs that does the book's exercises 1, 2 and 3 at once: the local sort in
          shared memory of section 14.6, a digit of r bits as in 14.7, and c keys per thread as in 14.8. Both r and c run from 1
          to 8 and are chosen at run time. A block has 256 threads and owns a tile of 256c consecutive pairs; the last tile may
          be partial, and its padding is neither counted nor written. Lab 14.1's part B is this lab's baseline, provided
          ready-made, so you can start here.
        </p>
      </Translated>
      <Source section={tr('14.6 to 14.8', '14.6 đến 14.8')} pages="339–345, 347" figs={tr('Figs. 14.8, 14.9, 14.12, 14.13; exercises 1 to 3', 'Hình 14.8, 14.9, 14.12, 14.13; bài tập 1 đến 3')} />
      <Translated
        vi={
          <p>
            Một lượt trên chữ số <code>(key &gt;&gt; shift) &amp; (2^r − 1)</code> là ba lần khởi chạy, như mục 14.6 trình bày.{' '}
            <code>radixLocalSort</code> nạp ô vào bộ nhớ chia sẻ, sắp nó theo chữ số bằng r lần chia một bit ổn định (mỗi lần là một
            phép scan loại trừ cấp khối), ghi nó trở lại chỗ cũ, ghi 2^r kích thước bucket của khối vào bảng bố trí theo chữ số{' '}
            <code>table[d × numBlocks + block]</code> (số đếm của mọi khối cho chữ số 0, rồi chữ số 1, như trong Hình 14.9 và 14.12)
            và ghi vị trí bắt đầu của mỗi bucket bên trong ô vào <code>localStarts</code>. Phép scan được cung cấp biến bảng thành
            điểm bắt đầu toàn cục của mọi bucket cục bộ. Sau đó <code>radixScatterLocal</code> gửi cặp ở vị trí p của ô có chữ số d
            tới điểm bắt đầu đó cộng (p − điểm bắt đầu cục bộ của bucket), nên các luồng kề nhau ghi vào các địa chỉ kề nhau bên
            trong một bucket.
          </p>
        }
      >
        <p>
          One pass over the digit <code>(key &gt;&gt; shift) &amp; (2^r − 1)</code> is three launches, as section 14.6 lays it
          out. <code>radixLocalSort</code> loads the tile into shared memory, sorts it by the digit with r stable one-bit
          splits (each a block-level exclusive scan), writes it back to the same place, writes the block's 2^r bucket sizes
          into the digit-major table <code>table[d × numBlocks + block]</code> (all blocks' counts for digit 0, then digit 1,
          as in Figs. 14.9 and 14.12) and where each bucket starts inside the tile into <code>localStarts</code>. The provided
          scan turns the table into every local bucket's global start. <code>radixScatterLocal</code> then sends the pair at
          tile position p with digit d to that start plus (p − the bucket's local start), so neighbouring threads write
          neighbouring addresses within a bucket.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <code>launchRadixSortMulti</code> chạy ceil(32 / r) lượt. Khi r không chia hết 32, lượt cuối chỉ có các bit còn lại:
            phép dịch 32 bit trở lên là không xác định trong C++, nên chữ số phải thu nhỏ lại, không chỉ riêng mặt nạ.
          </p>
        }
      >
        <p>
          <code>launchRadixSortMulti</code> runs ceil(32 / r) passes. When r does not divide 32 the last pass has only the
          bits that remain: a shift by 32 or more is undefined in C++, so the digit must shrink, not the mask alone.
        </p>
      </Translated>
      <LabFiles
        lab={l2}
        files={[
          [
            'kernel.cu',
            <Translated
              vi={
                <>
                  <code>radixLocalSort</code>, <code>radixScatterLocal</code> và <code>launchRadixSortMulti</code>; cả hai kernel nhận{' '}
                  <code>keysPerThread</code> làm đối số cuối
                </>
              }
            >
              <>
                <code>radixLocalSort</code>, <code>radixScatterLocal</code> and <code>launchRadixSortMulti</code>; both
                kernels take <code>keysPerThread</code> as their last argument
              </>
            </Translated>,
          ],
          ['kernel.metal', <Translated vi={<><code>radix_local_sort</code> và <code>radix_scatter_local</code></>}><><code>radix_local_sort</code> and <code>radix_scatter_local</code></></Translated>],
          [
            'launch.mm',
            <Translated
              vi={
                <>
                  <code>encodeRadixSortMulti</code>: mọi lượt, kể cả phép scan được cung cấp, trong một compute encoder mà nó được giao,
                  với bộ nhớ threadgroup được định cỡ và kiểm tra với giới hạn của thiết bị trước khi encode bất cứ thứ gì
                </>
              }
            >
              <>
                <code>encodeRadixSortMulti</code>: every pass, the provided scan included, in the one compute encoder it is
                given, with the threadgroup memory sized and checked against the device's limit before anything is encoded
              </>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Hãy làm phép scan cấp khối trong bộ nhớ chia sẻ với rào chắn. Một phép scan bằng warp-shuffle cần một lane mask, mà trên
            HIP phải là số nguyên 64 bit trên mọi target; dạng bộ nhớ chia sẻ không có mask nào để sai. Trong Metal, phép scan có thể
            dùng <code>simd_prefix_exclusive_sum</code> bên trong mỗi SIMD group và một ô threadgroup cho mỗi SIMD group để chứa tổng
            của chúng, với độ rộng SIMD group đọc từ pipeline, không bao giờ giả định. Một phép scan như vậy cộng các tổng của luồng
            theo thứ tự (SIMD group, lane), nên trong các lần chia hãy giao cho mỗi luồng các vị trí trong ô theo hạng của nó trong thứ
            tự đó, <code>sg × width + lane</code>, không phải theo <code>thread_position_in_threadgroup</code> của nó; lời giải tham
            chiếu và phép scan được cung cấp làm vậy. Ô chiếm 256c × 8 byte bộ nhớ threadgroup; có vừa hay không là câu trả lời của
            thiết bị của bạn, từ <code>python3 labs/pmpp.py doctor</code>.
          </p>
        }
      >
        <p>
          Do the block scan in shared memory with barriers. A warp-shuffle scan needs a lane mask, which on HIP must be a
          64-bit integer on every target; the shared-memory form has no mask to get wrong. In Metal the scan may use{' '}
          <code>simd_prefix_exclusive_sum</code> inside each SIMD-group and one threadgroup slot per SIMD-group for their
          totals, with the SIMD-group width read from the pipeline, never assumed. Such a scan adds up the thread totals in
          (SIMD-group, lane) order, so in the splits give each thread the tile positions of its rank in that order,{' '}
          <code>sg × width + lane</code>, not of its <code>thread_position_in_threadgroup</code>; the reference and the
          provided scan do. The tile is 256c × 8 bytes of threadgroup memory; whether it fits is your device's answer, from{' '}
          <code>python3 labs/pmpp.py doctor</code>.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (infra-facts.md B3, C2, C5; ch04-facts.md C2; ch05-facts.md C1, C3, C8)', '2026-11-20 (infra-facts.md mục B3, C2, C5; ch04-facts.md mục C2; ch05-facts.md mục C1, C3, C8)')}>
        <Translated
          vi={
            <p>
              Các intrinsic warp của HIP nhận mask 64 bit. Các tổng tiền tố SIMD group của MSL cần GPU family Apple7. Độ rộng SIMD group
              không phải là một hằng số được tài liệu hóa, và cách một threadgroup được chia thành các SIMD group là
              "implementation-defined" (do hiện thực quyết định). Độ dài bộ nhớ threadgroup là bội của 16 byte và phải vừa{' '}
              <code>maxThreadgroupMemoryLength</code>.
            </p>
          }
        >
          <p>
            HIP's warp intrinsics take a 64-bit mask. MSL's SIMD-group prefix sums need GPU family Apple7. The SIMD-group width
            is not a documented constant, and how a threadgroup is split into SIMD-groups is "implementation-defined".
            Threadgroup memory lengths are multiples of 16 bytes and must fit <code>maxThreadgroupMemoryLength</code>.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Được cung cấp, không sửa: phép scan của lab 14.1, và <code>onebit_global.cu</code>, <code>.metal</code> và{' '}
            <code>_launch.mm</code>, vốn là thiết kế của lab 14.1, một bit mỗi lượt với mọi khóa được ghi thẳng vào vị trí toàn cục
            của nó. Đó là baseline cho khẳng định của mục 14.6 rằng sắp xếp cục bộ có lợi.
          </p>
        }
      >
        <p>
          Provided, not edited: the scan of lab 14.1, and <code>onebit_global.cu</code>, <code>.metal</code> and{' '}
          <code>_launch.mm</code>, which are lab 14.1's design, one bit per pass with every key written straight to its
          global place. That is the baseline for section 14.6's claim that local sorting pays.
        </p>
      </Translated>

      <h3>
        <Translated vi="Các bài kiểm thử kiểm tra gì">What the tests check</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Trên GPU, mọi r trong 1, 2, 3, 4, 5 và 8 (3 và 5 không chia hết 32) với mọi c trong 1, 2 và 4, ở n = 1, 255, 256, 257,
            1000, 4097, 65537 và 2^20, với tám mẫu khóa: sáu mẫu của lab 14.1, cộng "mọi khóa có chung byte thấp" (ở các lượt đầu, mỗi
            khối đặt mọi khóa của nó vào một bucket, nên bảng phần lớn là số 0) và "khóa nhỏ hơn 256" (mọi chữ số cao hơn đều bằng 0).
            Khóa phải bằng đúng của <code>std::stable_sort</code> và giá trị phải đúng hoán vị của nó. Chương trình điều khiển điền
            bảng, <code>localStarts</code>, vùng nháp và 64 ô canh gác bằng các byte <code>0xCD</code> trước mỗi lần chạy, nên một số
            đếm không được ghi cho lượt này sẽ thất bại. Một cấu hình mà thiết bị không chạy được sẽ được in là bị bỏ qua kèm lý do,
            không bao giờ được tính là đạt. Các trình giả lập phải trả giá cho mọi luồng của tối đa 96 lần khởi chạy mỗi phép sắp
            xếp, nên chúng chạy một ma trận nhỏ hơn với các mẫu khóa luân phiên giữa các cấu hình; README liệt kê nó.
          </p>
        }
      >
        <p>
          On a GPU, every r in 1, 2, 3, 4, 5 and 8 (3 and 5 do not divide 32) with every c in 1, 2 and 4, at n = 1, 255, 256,
          257, 1000, 4097, 65537 and 2^20, with eight key patterns: lab 14.1's six, plus "every key shares its low byte" (on
          the first passes each block puts all its keys in one bucket, so the table is mostly zeros) and "keys below 256"
          (every higher digit is zero). Keys must equal <code>std::stable_sort</code>'s and values its permutation exactly.
          The driver fills the table, <code>localStarts</code>, the scratch and 64 guard slots with <code>0xCD</code> bytes
          before every run, so a count that is not written for this pass fails. A configuration the device cannot run is
          printed as skipped with the reason, never counted as a pass. The emulators pay for every thread of up to 96 launches
          per sort, so they run a smaller matrix whose key patterns rotate across the configurations; the README lists it.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Một rào chắn bị thiếu ở cuối một lần chia cho phép một luồng đọc một ô đang ghi dở. Chúng tôi đã cài đúng lỗi đó vào một
            bản sao của lời giải tham chiếu (2026-11-20). Trình giả lập bắt được nó mọi lần: ở r = 2, thứ tự luồng đầu tiên đã cho ra
            thứ tự sai, và AddressSanitizer sau đó chỉ ra lần ghi vào ô bị vượt biên. Backend race cũng thất bại, và chỉ ra cặp mà
            rào chắn lẽ ra phải tách: lần ghi ô của một lần chia với lần đọc ô đó của lần chia kế tiếp. Rồi lần chạy dừng bằng một cú
            crash ở lần ghi ô khi một chỉ số bị hỏng được dùng. Ở r = 1 cả hai đều đạt, vì rào chắn của bước tiếp theo che được khoảng
            trống. Hãy vẫn chạy trình giả lập trước; một lần chạy race sạch không phải bằng chứng rằng mọi rào chắn đều có mặt.
          </p>
        }
      >
        <p>
          A missing barrier at the end of a split lets a thread read a half-written tile. We planted exactly that in a copy of
          the reference (2026-11-20). The emulator fails it every time: at r = 2 the first thread order already gives a
          wrong order, and AddressSanitizer then names the tile write that went out of bounds. The race backend also fails,
          and names the pair the barrier would have separated: a split's write of the tile against the next split's read of
          it. Then the run stops with a crash at the tile write once a corrupted index is used. At r = 1 both pass, because
          the next step's barrier covers the gap. Still run the emulator first; a clean race run is not evidence that every
          barrier is there.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Khi chúng tôi cài lỗi lần đầu, biến thể race dùng chung một token ThreadSanitizer cho mọi <code>__syncthreads()</code>{' '}
              của một khối, điều có thể che một tranh chấp giữa hai rào chắn, và nó chỉ báo một tranh chấp về sau trên các điểm bắt
              đầu bucket. Giờ nó luân phiên hai token mỗi rào chắn; chạy lại ngày 2026-11-21 (bốn lần chạy), nó chỉ ra lần ghi và đọc
              ô mọi lần (<code>labs/ch14/lab2_multibit/README.md</code>).
            </p>
          }
        >
          <p>
            When we first planted it, the race flavour shared one ThreadSanitizer token between all of a block's{' '}
            <code>__syncthreads()</code>, which could hide a race between two barriers, and it reported only a later race on
            the bucket starts. It now alternates two tokens per barrier; re-run on 2026-11-21 (four runs), it named the tile
            write and read every time (<code>labs/ch14/lab2_multibit/README.md</code>).
          </p>
        </Translated>
      </Beyond>
      <RaceStatus lab={l2} />

      <h3>
        <Translated vi="Trước khi đo hiệu năng">Before you benchmark</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Lab này có ba dự đoán. Hai cái nằm trên các trang có mô hình mà chúng kiểm chứng: <a href={href('ch14', '14.6')}>14.6</a>{' '}
            hỏi sắp xếp cục bộ ở r = 1, c = 1 có thắng phân tán trực tiếp được cung cấp không, và{' '}
            <a href={href('ch14', '14.7')}>14.7</a> hỏi r nào nhanh nhất ở c = 1; <a href={href('ch14', '14.8')}>14.8</a> thêm một
            câu về c. Hãy làm chúng ở đó. Cái thứ ba ở đây. Đếm trước: ceil(32 / r) lượt, mỗi lượt đọc và ghi mọi cặp hai lần; tối đa
            2^r đoạn địa chỉ mỗi warp khi phân tán, mỗi đoạn dài khoảng 256c / 2^r khóa trên khóa ngẫu nhiên; một bảng 2^r × ceil(n /
            256c) phần tử được ghi, scan và đọc ở mỗi lượt.
          </p>
        }
      >
        <p>
          This lab has three predictions. Two sit on the pages whose model they test:{' '}
          <a href={href('ch14', '14.6')}>14.6</a> asks whether local sorting at r = 1, c = 1 beats the provided direct
          scatter, and <a href={href('ch14', '14.7')}>14.7</a> which r is fastest at c = 1; <a href={href('ch14', '14.8')}>14.8</a>{' '}
          adds one on c. Make those there. The third is here. Count first: ceil(32 / r) passes, each reading and writing every
          pair twice; up to 2^r runs of addresses per warp in the scatter, each about 256c / 2^r keys long on random keys; a
          table of 2^r × ceil(n / 256c) entries written, scanned and read every pass.
        </p>
      </Translated>
      <LabPrediction
        id="ch14/labs/p-lab2-cpu"
        prompt={
          <Translated vi={<>Ở kích thước mà benchmark chọn (2^22 cặp trở lên), cấu hình nhanh nhất của bạn sẽ nhanh hơn <code>std::stable_sort</code> trên một luồng CPU bao nhiêu lần?</>}>
            <>At the size the benchmark settles on (2^22 pairs or more), how many times faster than one CPU thread's <code>std::stable_sort</code> will your fastest configuration be?</>
          </Translated>
        }
        unit={tr('× the speed of std::stable_sort', '× tốc độ của std::stable_sort')}
        measure={multiOverCpu}
        missing={missingBench('14.2')}
        reading={
          <Translated
            vi={
              <>
                Đặt nó cạnh dự đoán thứ hai của lab 14.1: các cặp ngẫu nhiên cùng loại, cùng baseline CPU, một phép sắp xếp ít lượt hơn
                và các lần ghi được hợp nhất. Phần chi tiết cho biết nó được đo ở kích thước nào. r và c nào thắng là một phần của câu
                trả lời; bảng hiển thị mọi hàng.
              </>
            }
          >
            <>
              Set it beside lab 14.1's second prediction: random pairs of the same kind, the same CPU baseline, a sort with
              fewer passes and coalesced writes. The detail says at which size it was timed. Which r and c won is part of
              the answer; the table shows every row.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l2}
        cols={[
          what(WHAT_2),
          nCol,
          rawCol('r', 'radix_bits'),
          rawCol('c', 'keys_per_thread'),
          rawCol(tr('Passes', 'Số lượt'), 'passes'),
          msCol,
          numCol('Mkeys/s', 'mkeys_per_s'),
          numCol(tr('Counted GB/s', 'GB/s đếm được'), 'counted_gbps'),
        ]}
      />
      <Translated
        vi={
          <p>
            Benchmark hoạt động thế nào: n bắt đầu ở 2^22 và nhân đôi khi baseline còn mất dưới 20 ms. Nó đo baseline, sắp xếp cục bộ
            ở r = 1, 2, 4, 6 và 8 với c = 1 và ở c = 1, 2, 4 và 8 với r = 4 và với r = 8, bỏ qua những gì thiết bị không chạy được,
            và <code>std::stable_sort</code> trên cùng các cặp với một luồng CPU. "GB/s đếm được" là lưu lượng được đếm, không phải
            đo: 32 byte mỗi cặp mỗi lượt chia cho thời gian, không tính bảng và phép scan.
          </p>
        }
      >
        <p>
          How the benchmark works: n starts at 2^22 and doubles while the baseline takes under 20 ms. It times the baseline,
          the local sort at r = 1, 2, 4, 6 and 8 with c = 1 and at c = 1, 2, 4 and 8 with r = 4 and with r = 8, skipping what
          the device cannot run, and <code>std::stable_sort</code> of the same pairs on one CPU thread. "Counted GB/s" is
          counted, not measured, traffic: 32 bytes per pair per pass divided by the time, with the table and the scan left
          out.
        </p>
      </Translated>
      <ReferenceSolution lab={l2} tiers={TIERS_2} />

      {/* ------------------------------------------------------------ 14.3 */}
      <h2>
        <Translated vi="Lab 14.3: ứng viên đã sắp xếp, mọi hàng giảm dần và ổn định (mốc dự án M5)">
          Lab 14.3: sorted candidates, every row descending and stable (milestone M5)
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một hàm host và các kernel đằng sau nó:{' '}
            <code>launchSortRowsDesc(probs, sortedProbs, sortedIdx, scratch, rows, cols)</code>. Với mỗi hàng của một tensor float32{' '}
            <code>[rows × cols]</code>, thứ tự hàng (row-major) với batch đứng đầu, <code>sortedIdx</code> nhận các chỉ số cột theo
            thứ tự giảm dần của giá trị, giá trị bằng nhau thì theo cột tăng dần, và <code>sortedProbs</code> nhận các giá trị theo
            thứ tự đó. <code>scratch</code> chứa 4 × rows × cols số nguyên không dấu để bạn dùng. Nó là đầu vào cho tổng tích lũy của
            top-p: sắp xếp, rồi scan.
          </p>
        }
      >
        <p>
          One host function and the kernels behind it:{' '}
          <code>launchSortRowsDesc(probs, sortedProbs, sortedIdx, scratch, rows, cols)</code>. For every row of a{' '}
          <code>[rows × cols]</code> float32 tensor, row-major with the batch first, <code>sortedIdx</code> gets the column
          indices in descending order of value, equal values by increasing column, and <code>sortedProbs</code> gets the values
          in that order. <code>scratch</code> holds 4 × rows × cols unsigned integers for you to use. It is the input to top-p's
          cumulative sum: sort, then scan.
        </p>
      </Translated>
      <Source section={tr('14.4 to 14.8', '14.4 đến 14.8')} pages="334–345" figs={tr('Figs. 14.6, 14.7; the multi-bit local sort of exercises 1 to 3', 'Hình 14.6, 14.7; sắp xếp cục bộ nhiều bit của bài tập 1 đến 3')} />
      <Translated
        vi={
          <p>
            Thứ tự là thứ tự của một khóa, sắp tăng dần. Lấy 32 bit của float; nếu chúng là −0.0 thì biến thành +0.0; đặt bit dấu của
            giá trị không âm và đảo mọi bit của giá trị âm; rồi đảo toàn bộ khóa, điều làm phép sắp xếp tăng dần thành giảm dần mà
            không lật ngược thứ tự của các giá trị bằng nhau. Trên các số, đó là thứ tự số giảm dần với hai số không bằng nhau. Trên
            NaN thì theo bit: +NaN trước mọi thứ và −NaN sau mọi thứ. Mục <a href={href('ch14', '14.4')}>14.4</a> xây phép biến đổi
            này từng bước.
          </p>
        }
      >
        <p>
          The order is that of one key, sorted ascending. Take the 32 bits of the float; if they are −0.0 make them +0.0; set
          the sign bit of a non-negative value and invert every bit of a negative one; then invert the whole key, which makes
          the ascending sort descending without reversing the order of ties. On numbers that is descending numeric order with
          the two zeros equal. On NaN it is the bits: +NaN before everything and −NaN after everything. Section{' '}
          <a href={href('ch14', '14.4')}>14.4</a> builds this transform step by step.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch14-facts.md A3, E1)', '2026-11-20 (notes/today/ch14-facts.md mục A3, E1)')}>
        <Translated
          vi={
            <p>
              Đây là phép biến đổi mà CUB ghi trong tài liệu cho khóa float của nó, gồm cả bước gộp hai số không, đảo khóa cho thứ tự
              giảm dần, và sắp xếp NaN theo bit của nó.
            </p>
          }
        >
          <p>
            This is the transform CUB documents for its float keys, including the step that merges the two zeros, inverting
            the key for descending order, and sorting NaN by its bits.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Các kernel không bao giờ thực hiện một phép toán float. Hàm khởi chạy trao cho chúng <code>probs</code> và{' '}
            <code>sortedProbs</code> dưới dạng góc nhìn không dấu (một phép ép kiểu trong CUDA; các buffer bind dưới dạng{' '}
            <code>device const uint*</code> và <code>device uint*</code> trong Metal), nên phép kiểm tra −0.0, kiểm tra dấu và phép
            sao chép đều là việc trên số nguyên. Trên Metal, cả hai chế độ toán mặc định có thể dùng đều cho phép "no signed zeros",
            nên một phép kiểm tra số không phía float có thể bị viết lại. Trên HIP, <code>__float_as_uint</code> có trong header
            nhưng không có trong tài liệu toán học của AMD. Góc nhìn không dấu hoàn toàn không cần intrinsic bit-cast nào, nên nó là
            dạng chạy được trên mọi backend.
          </p>
        }
      >
        <p>
          The kernels never perform a float operation. The launcher hands them <code>probs</code> and{' '}
          <code>sortedProbs</code> as unsigned views (a cast in CUDA; buffers bound as <code>device const uint*</code> and{' '}
          <code>device uint*</code> in Metal), so the −0.0 test, the sign test and the copy are integer work. On Metal both
          candidate default math modes allow "no signed zeros", so a float-side zero test may be rewritten. On HIP,{' '}
          <code>__float_as_uint</code> is in the header but not in AMD's math reference. The unsigned view needs no
          bit-cast intrinsic at all, which makes it the form that works on every backend.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch14-facts.md A8, B6, C3)', '2026-11-20 (notes/today/ch14-facts.md mục A8, B6, C3)')}>
        <Translated
          vi={
            <p>
              CUDA ghi <code>__float_as_uint</code> trong tài liệu. HIP định nghĩa nó trong một header device mà API toán học AMD công
              bố không liệt kê. Fast math của Metal để NaN và INF không xác định.
            </p>
          }
        >
          <p>
            CUDA documents <code>__float_as_uint</code>. HIP defines it in a device header that AMD's published math API
            does not list. Metal's fast math leaves NaN and INF undefined.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Thiết kế, của khóa học này: một khối 256 luồng mỗi hàng, dạng của lab 10.3. <code>descKeys</code> ghi khóa của mỗi giá trị
            và cột của nó. <code>sortRowsPass</code>, khởi chạy một lần cho mỗi chữ số r bit (gợi ý 8, nên bốn lượt), đếm các chữ số
            của hàng trong một histogram ở bộ nhớ chia sẻ bằng phép toán nguyên tử như ở chương 9, scan 2^r số đếm để tìm mỗi bucket
            bắt đầu ở đâu trong hàng đầu ra, rồi duyệt hàng theo các ô 256c khóa: nó sắp mỗi ô trong bộ nhớ chia sẻ theo chữ số bằng
            r lần chia một bit ổn định và ghi mỗi khóa vào offset chạy của bucket của nó, offset này sau đó tiến thêm bằng kích thước
            của bucket trong ô đó. <code>gatherRows</code> điền <code>sortedIdx</code> và sao chép bit của <code>probs</code> tại các
            chỉ số đó vào <code>sortedProbs</code>.
          </p>
        }
      >
        <p>
          The design, ours: one block of 256 threads per row, the shape of lab 10.3. <code>descKeys</code> writes each value's
          key and its column. <code>sortRowsPass</code>, launched once per digit of r bits (8 suggested, so four passes),
          counts the row's digits in a shared-memory histogram with atomics as in chapter 9, scans the 2^r counts to find where
          each bucket starts in the output row, then walks the row in tiles of 256c keys: it sorts each tile in shared memory
          by the digit with r stable one-bit splits and writes each key to its bucket's running offset, which then moves on by
          the bucket's size in that tile. <code>gatherRows</code> fills <code>sortedIdx</code> and copies the bits of{' '}
          <code>probs</code> at those indices into <code>sortedProbs</code>.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Vì sao nó ổn định qua các ô: một khối duyệt các ô theo thứ tự, và offset của một bucket chỉ tiến về phía trước, nên một
            khóa từ ô trước luôn rơi trước một khóa cùng chữ số từ ô sau. Offset chạy làm công việc của bảng kích thước bucket theo
            khối của sách: với một khối mỗi hàng thì chỉ có một khối cần tính, và không có phép scan xuyên khối giữa các lượt.
          </p>
        }
      >
        <p>
          Why it is stable across tiles: one block visits the tiles in order, and a bucket's offset only moves forward, so a
          key from an earlier tile always lands before an equal-digit key from a later tile. The running offset does the job
          of the book's table of bucket sizes per block: with one block per row there is one block to account for, and no
          scan across blocks between the passes.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Khóa float, một khối mỗi hàng, offset chạy cho mỗi bucket và bước gather là của khóa học này. Sách không có phép sắp xếp
              theo batch hay theo hàng.
            </p>
          }
        >
          <p>
            The float key, one block per row, the running offset per bucket and the gather are ours. The book has no batched
            or row-wise sort.
          </p>
        </Translated>
      </Beyond>
      <LabFiles
        lab={l3}
        files={[
          [
            'kernel.cu',
            <Translated
              vi={
                <>
                  <code>descKeys</code>, <code>sortRowsPass</code>, <code>gatherRows</code> và <code>launchSortRowsDesc</code>; phép scan
                  cấp khối được cung cấp sẵn trong tệp
                </>
              }
            >
              <>
                <code>descKeys</code>, <code>sortRowsPass</code>, <code>gatherRows</code> and{' '}
                <code>launchSortRowsDesc</code>; the block scan is provided in the file
              </>
            </Translated>,
          ],
          ['kernel.metal', <Translated vi={<><code>desc_keys</code>, <code>sort_rows_pass</code> và <code>gather_rows</code></>}><><code>desc_keys</code>, <code>sort_rows_pass</code> and <code>gather_rows</code></></Translated>],
          [
            'launch.mm',
            <Translated vi={<><code>encodeSortRowsDesc</code>, toàn bộ phép sắp xếp trong một compute encoder mà nó được giao</>}>
              <><code>encodeSortRowsDesc</code>, the whole sort in the one compute encoder it is given</>
            </Translated>,
          ],
        ]}
      />

      <h3>
        <Translated vi="Các bài kiểm thử kiểm tra gì">What the tests check</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Số hàng 1, 3, 4 và 7 nhân số cột 1, 2, 31, 255, 256, 257, 1000, 4096, 32000, 50257 và 151936 (hai số cuối là kích thước
            từ vựng của các mô hình thật), với dữ liệu trộn lẫn trong một batch để việc đọc sai hàng sẽ thất bại: xác suất softmax;
            cũng vậy nhưng có cài sẵn các giá trị bằng nhau tuyệt đối, kể cả các dãy 40 giá trị bằng nhau vắt qua mọi bội của 256 từ
            512 đến 8192, để với mọi c từ 1 đến 32 sẽ có một dãy vắt qua ranh giới ô 256c khóa khi hàng đủ dài; toàn bộ giá trị bằng
            nhau; các hàng one-hot (các số không phải giữ thứ tự cột); logits có −0.0 và +0.0 xen kẽ, vô cực,{' '}
            <code>±FLT_MAX</code> và số subnormal; và các hàng có NaN cả hai dấu.
          </p>
        }
      >
        <p>
          Rows 1, 3, 4 and 7 by columns 1, 2, 31, 255, 256, 257, 1000, 4096, 32000, 50257 and 151936 (the last two are
          vocabulary sizes of real models), with the data mixed within a batch so that reading the wrong row fails: softmax
          probabilities; the same with planted exact ties, including runs of 40 equal values across every multiple of 256
          from 512 to 8192, so that for any c from 1 to 32 a run crosses a tile boundary of 256c keys once the row is long
          enough; every value equal; one-hot rows
          (the zeros must stay in column order); logits with −0.0 and +0.0 interleaved, infinities, <code>±FLT_MAX</code>{' '}
          and subnormals; and rows with NaNs of both signs.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <code>sortedIdx</code> phải bằng đúng tham chiếu, đó là phép kiểm tra tính ổn định; <code>sortedProbs</code> phải bằng
            đúng các bit được gather; mọi hàng đầu ra phải là một hoán vị; 64 phần tử canh gác sau mỗi đầu ra phải không bị chạm, và{' '}
            <code>probs</code> không đổi. Không có dung sai: một phép sắp xếp không làm phép tính số học nào. Tham chiếu trong{' '}
            <code>main.cpp</code> là <code>std::stable_sort</code> trên các chỉ số cột theo khóa số nguyên, được đối chiếu chéo trên
            mọi hàng không có NaN với <code>std::stable_sort</code> dùng <code>&gt;</code> trên các float. Trình giả lập chạy 1 và 3
            hàng tới 4,099 cột, với các công thức dữ liệu luân phiên khi kích thước thay đổi; backend race chạy hai hàng tới 2,100
            cột, hai ô ngay cả ở 8 khóa mỗi luồng.
          </p>
        }
      >
        <p>
          <code>sortedIdx</code> must equal the reference exactly, which is the stability check; <code>sortedProbs</code> must
          equal the gathered bits exactly; every output row must be a permutation; 64 guard entries past every output must be
          untouched, and <code>probs</code> unchanged. No tolerance: a sort does no arithmetic. The reference in{' '}
          <code>main.cpp</code> is <code>std::stable_sort</code> of the column indices on the integer key, cross-checked on
          every NaN-free row against <code>std::stable_sort</code> with <code>&gt;</code> on the floats. The emulator runs rows
          1 and 3 up to 4,099 columns, with the data recipes rotating as the size changes; the race backend runs two rows up to
          2,100 columns, two tiles even at 8 keys per thread.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch14-facts.md A11, C2)', '2026-11-20 (notes/today/ch14-facts.md mục A11, C2)')}>
        <Translated
          vi={
            <p>
              <code>std::stable_sort</code> giữ thứ tự các phần tử tương đương. <code>vDSP_vsorti</code> của Accelerate được ghi trong
              tài liệu là không ổn định, nên nó không phải tham chiếu.
            </p>
          }
        >
          <p>
            <code>std::stable_sort</code> keeps equivalent elements in order. Accelerate's <code>vDSP_vsorti</code> is
            documented as not stable, so it is not the reference.
          </p>
        </Translated>
      </Today>
      <RaceStatus lab={l3} />

      <h3>
        <Translated vi="Con đường thư viện, theo từng backend">The library route, per backend</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Lời khuyên của sách là dùng phép sắp xếp của thư viện trừ khi bạn có lý do không dùng. Trên CUDA thao tác này là{' '}
            <code>cub::DeviceSegmentedRadixSort::SortPairsDescending</code> hoặc{' '}
            <code>cub::DeviceSegmentedSort::StableSortPairsDescending</code>, nhưng không phải{' '}
            <code>DeviceSegmentedSort::SortPairsDescending</code> thường, vốn được ghi là không ổn định. Trên HIP các tên tương tự tồn
            tại dưới <code>hipcub::</code>. Trên Metal không có phép sắp xếp thư viện cấp kernel; <code>argSort</code> của MPSGraph
            sắp xếp dọc một trục và Apple không nêu tính ổn định hay thứ tự khi bằng nhau cho nó, nên nếu các xác suất bằng nhau phải
            ra theo id token tăng dần, bạn tự viết kernel, như ở đây. <a href={href('ch14', '14.9')}>Mục 14.9</a> có toàn bộ bảng.
          </p>
        }
      >
        <p>
          The book's advice is to use a library sort unless you have a reason not to. On CUDA this operation is{' '}
          <code>cub::DeviceSegmentedRadixSort::SortPairsDescending</code> or{' '}
          <code>cub::DeviceSegmentedSort::StableSortPairsDescending</code>, but not the plain{' '}
          <code>DeviceSegmentedSort::SortPairsDescending</code>, which is documented as not stable. On HIP the same names exist
          under <code>hipcub::</code>. On Metal there is no kernel-level library sort; MPSGraph's <code>argSort</code> sorts
          along an axis and Apple states no stability or tie order for it, so if equal probabilities must come out by
          increasing token id, you write the kernel, as here. <a href={href('ch14', '14.9')}>Section 14.9</a> has the whole
          table.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch14-facts.md A5, B1, B2, C1)', '2026-11-20 (notes/today/ch14-facts.md mục A5, B1, B2, C1)')}>
        <Translated
          vi={
            <p>
              Trong hipCUB 4.7.0, <code>StableSortPairsDescending</code> gọi <code>SortPairsDescending</code>, hàm chạy segmented radix
              sort của rocPRIM; rocPRIM ghi trong tài liệu rằng radix sort của nó ổn định.
            </p>
          }
        >
          <p>
            In hipCUB 4.7.0, <code>StableSortPairsDescending</code> calls <code>SortPairsDescending</code>, which runs
            rocPRIM's segmented radix sort; rocPRIM documents its radix sort as stable.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Một lý do kernel này quan trọng trên HIP: <code>argsort</code> CUDA của llama.cpp sắp xếp các hàng dài bằng CUB, bản build
            HIP của nó không có nhánh CUB, và nhánh còn lại cần cả hàng, đệm lên lũy thừa của hai, trong bộ nhớ chia sẻ. Một hàng
            151,936 cột được đệm thành 262,144 số nguyên, 1 MiB, nên theo cách chúng tôi đọc, một <code>argsort</code> cỡ từ vựng bị
            báo là không hỗ trợ trên HIP và chạy trên backend khác. Một phép sắp xếp hàng mà bộ nhớ chia sẻ không tăng theo hàng, như
            phép này, chính là thứ nhánh đó thiếu.
          </p>
        }
      >
        <p>
          One reason this kernel matters on HIP: llama.cpp's CUDA <code>argsort</code> sorts long rows with CUB, its HIP build
          has no CUB path, and its other path needs the whole row, padded to a power of two, in shared memory. A 151,936-column
          row pads to 262,144 integers, 1 MiB, so by our reading a vocabulary-sized <code>argsort</code> is reported
          unsupported on HIP and runs on another backend. A row sort whose shared memory does not grow with the row, like this
          one, is what that path lacks.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Đây là cách chúng tôi đọc mã nguồn llama.cpp (notes/today/ch14-facts.md D1), không phải thứ chúng tôi đã chạy. Lab này
              không được biên dịch hay chạy trên phần cứng AMD ở đây.
            </p>
          }
        >
          <p>
            Our reading of llama.cpp's source (notes/today/ch14-facts.md D1), not something we ran. This lab is not compiled or
            run on AMD hardware here.
          </p>
        </Translated>
      </Beyond>

      <h3>
        <Translated vi="Trước khi đo hiệu năng">Before you benchmark</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Một khối mỗi hàng nghĩa là ở batch 1 một khối duy nhất sắp xếp cả hàng trong khi phần còn lại của thiết bị không có gì để
            làm: tình huống mục <a href={href('ch10', '10.10')}>10.10</a> đã cho thấy với softmax, và là tình huống phần cuối của{' '}
            <a href={href('ch14', '14.8')}>14.8</a> mô tả cho lab này. Hãy viết ra một con số, rồi bạn nghĩ tỉ số sẽ tăng hay giảm ở
            batch 64; phần kết quả cho thấy cả hai.
          </p>
        }
      >
        <p>
          One block per row means that at batch 1 a single block sorts the whole row while the rest of the device has nothing
          to do: the situation section <a href={href('ch10', '10.10')}>10.10</a> showed for a softmax, and the one the end of{' '}
          <a href={href('ch14', '14.8')}>14.8</a> describes for this lab. Write down a number, and then whether you expect the
          ratio to grow or shrink at batch 64; the reveal shows both.
        </p>
      </Translated>
      <LabPrediction
        id="ch14/labs/p-lab3-batch1"
        prompt={
          <Translated
            vi={
              <>
                Ở batch 1 và 151,936 cột, kernel của bạn sẽ nhanh hơn <code>std::stable_sort</code> trên một luồng CPU bao nhiêu lần?
                (Dưới 1 nghĩa là chậm hơn.)
              </>
            }
          >
            <>
              At batch 1 and 151,936 columns, how many times faster than one CPU thread's <code>std::stable_sort</code> will
              your kernel be? (Below 1 means slower.)
            </>
          </Translated>
        }
        unit={tr('× the speed of std::stable_sort', '× tốc độ của std::stable_sort')}
        measure={rowsOverCpu}
        missing={missingBench('14.3')}
        reading={
          <Translated
            vi={
              <>
                CPU sắp xếp lần lượt từng hàng, nên thời gian của nó tăng theo batch. GPU giao cho mỗi hàng một khối riêng, nên cho đến
                khi thiết bị đầy, thêm hàng tốn rất ít thời gian. Hãy đọc tỉ số batch 64 đối chiếu với câu trả lời thứ hai của bạn.
              </>
            }
          >
            <>
              The CPU sorts one row after another, so its time grows with the batch. The GPU gives each row its own block, so
              until the device is full more rows cost little extra time. Read the batch-64 ratio against your second answer.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l3}
        cols={[
          what(WHAT_3),
          rawCol(tr('Rows', 'Số hàng'), 'rows'),
          { head: tr('Cols', 'Số cột'), num: true, cell: (r) => big(r.cols) },
          msCol,
          numCol(tr('Rows/s', 'Hàng/s'), 'rows_per_s'),
          numCol('Mkeys/s', 'mkeys_per_s'),
        ]}
      />
      <Translated
        vi={
          <p>
            Benchmark hoạt động thế nào: xác suất softmax ở 1, 8 và 64 hàng nhân 32,000 và 151,936 cột. Mỗi lần chạy được đo lặp lại
            toàn bộ phép sắp xếp (khóa, mọi lượt, gather) cho đến khi kéo dài ít nhất 20 ms, và hàng kết quả là tốt nhất trong 5 lần
            chạy như vậy, tính mỗi lần gọi. Baseline là <code>std::stable_sort</code> với <code>&gt;</code> lần lượt trên từng hàng,
            cộng bước gather, trên một luồng CPU. Mọi kết quả được đo đều được kiểm tra trước khi báo cáo.
          </p>
        }
      >
        <p>
          How the benchmark works: softmax probabilities at 1, 8 and 64 rows by 32,000 and 151,936 columns. Each timed run
          repeats the whole sort (keys, every pass, gather) until it lasts at least 20 ms, and the row is the best of 5 such
          runs, per call. The baseline is <code>std::stable_sort</code> with <code>&gt;</code> on each row in turn, plus the
          gather, on one CPU thread. Every timed result is checked before it is reported.
        </p>
      </Translated>

      <h3>
        <Translated vi="Mốc dự án">The project milestone</Translated>
      </h3>
      <Translated
        vi={
          <p>
            <code>launchSortRowsDesc</code> là các ứng viên đã sắp xếp của mốc dự án M5. Nó theo các quy ước của dự án để các module
            sau có thể ghép nó: float32 thứ tự hàng, batch đứng đầu, con trỏ device thô và kích thước tường minh làm đối số, một hàm
            host <code>launch</code> bên cạnh các kernel, tham chiếu CPU trong <code>main.cpp</code>. <code>sortedProbs</code> của nó
            đúng là thứ mặt nạ giữ top-p của lab 11.6 nhận, một hàng xác suất được sắp lớn nhất trước, và <code>sortedIdx</code> cho
            biết mỗi ô được giữ chứa token nào. Module X1 ghép các mảnh và rút mẫu; top-k thì nó làm bằng một phép chọn, không phải
            phép sắp xếp này (<a href={href('ch14', '14.9')}>14.9</a> nói vì sao).
          </p>
        }
      >
        <p>
          <code>launchSortRowsDesc</code> is milestone M5's sorted candidates. It follows the project's conventions so that
          later modules can compose it: row-major float32, batch first, raw device pointers and explicit sizes as arguments, a{' '}
          <code>launch</code> host function beside the kernels, the CPU reference in <code>main.cpp</code>. Its{' '}
          <code>sortedProbs</code> is exactly what lab 11.6's top-p keep mask takes, a row of probabilities sorted largest
          first, and <code>sortedIdx</code> says which token each kept slot holds. Module X1 joins the pieces and draws the
          sample; top-k it does with a selection, not this sort (<a href={href('ch14', '14.9')}>14.9</a> says why).
        </p>
      </Translated>
      <ReferenceSolution lab={l3} tiers={TIERS_3} />
    </LabsPage>
  );
}
