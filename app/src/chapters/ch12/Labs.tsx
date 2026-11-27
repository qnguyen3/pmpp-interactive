import type { Question } from '../../components/Ask';
import { Predict } from '../../components/Ask';
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
  rawCol,
  useLabs,
  whose,
  type BenchSet,
  type Lab,
} from '../../components/LabKit';
import { fmt, labStatus, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 12's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch12', [
  { num: '12.1', dir: 'lab1_unstable', name: 'unstable filter, three ways', vi: { name: 'bộ lọc không ổn định, ba cách' } },
  { num: '12.2', dir: 'lab2_filterhunt', name: 'filter hunt', vi: { name: 'săn lỗi bộ lọc' } },
  { num: '12.3', dir: 'lab3_stable', name: 'stable filter by scan', vi: { name: 'bộ lọc ổn định bằng scan' } },
  { num: '12.4', dir: 'lab4_candidates', name: 'candidate compaction (M5)', vi: { name: 'nén gọn ứng viên (M5)' } },
  { num: '12.5', dir: 'lab5_inplace', name: 'in-place single pass', sources: ['kernel.cu'], vi: { name: 'tại chỗ, một lượt' } },
]);
const [l1, l2, l3, l4, l5] = LABS;

// Identifier colours for the reference solutions. Only names that mean one thing in every
// file of a lab (lab 12.3's `keep` is an array in one kernel and a bool in another: left out).
const tiers = (global: string[], shared: string[], reg: string[] = []): Record<string, Tier> => ({
  ...Object.fromEntries(global.map((n) => [n, 'global' as Tier])),
  ...Object.fromEntries(shared.map((n) => [n, 'shared' as Tier])),
  ...Object.fromEntries(reg.map((n) => [n, 'reg' as Tier])),
});
const UNSTABLE_TIERS = tiers(['input', 'output', 'outputSize'], ['output_s', 'outputSize_s', 'j_s'], ['val', 'vote', 'leader', 'offset']);
const STABLE_TIERS = tiers(['input', 'output', 'outputSize', 'blockCounts', 'blockOffsets'], ['staged_s', 'warpTotals_s', 'groupTotals']);
const ROW_TIERS = tiers(['logits', 'threshold', 'outIdx', 'outVal', 'counts', 'idxOut', 'valOut'], ['warpTotals_s', 'groupTotals']);
const SINGLE_TIERS = tiers(['input', 'output', 'outputSize', 'blockCounter', 'flags', 'inclusive'], ['tile_s', 'warpTotals_s', 'bid_s', 'runStart_s']);

// ---------------------------------------------------------------- reading the reports
const ms = (s: number) => `${fmt(s * 1000)} ms`; // i18n-ignore: unit
const count = (v: number) => v.toLocaleString('en-US');
const secs = (r: BenchRow | undefined) => num(r?.seconds);

/** The row called `name` at a given fraction kept (and other numeric fields). */
const at = (s: BenchSet | undefined, name: string, fraction: number, where: Record<string, number> = {}) =>
  benchRow(s, name, { fraction, ...where });

const missing = (lab: string, gpu = tr('a machine with a GPU', 'một máy có GPU')) => (
  <Translated vi={<>Khi lab {lab} đã vượt qua, hãy chạy <code>python3 labs/pmpp.py run {lab} --bench</code> trên {gpu}, rồi tải lại.</>}>
    <>
      Once lab {lab} passes, run <code>python3 labs/pmpp.py run {lab} --bench</code> on {gpu}, then reload.
    </>
  </Translated>
);

/** Lab 12.1: atomic time over warp-aggregated time at half the keys kept. */
const aggRatio = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const a = secs(at(s, 'atomic', 0.5));
    const w = secs(at(s, 'warp', 0.5));
    const r = at(s, 'atomic', 0.5);
    if (!a || !w || !r) return null;
    const p = secs(at(s, 'private', 0.5));
    return {
      value: a / w,
      detail:
        tr(
          `${ms(a)} plain atomic, ${ms(w)} warp-aggregated, ${count(num(r.n) ?? 0)} keys at half kept`,
          `${ms(a)} atomic thường, ${ms(w)} gộp theo warp, ${count(num(r.n) ?? 0)} khóa với một nửa được giữ`,
        ) + (p ? tr(`; privatised ${ms(p)}`, `; riêng tư hóa ${ms(p)}`) : ''),
    };
  });

/** Lab 12.1: per fraction, how many times as long the plain atomic took as the privatised kernel. */
function behind(s: BenchSet) {
  return s.rows
    .filter((r) => r.name === 'atomic' && num(r.fraction) != null)
    .flatMap((r) => {
      const f = num(r.fraction)!;
      const a = secs(r);
      const p = secs(at(s, 'private', f));
      return a && p ? [{ f, a, p, ratio: a / p }] : [];
    })
    .sort((x, y) => x.f - y.f);
}

function MissingLine({ lab, gpu }: { lab: string; gpu?: string }) {
  const { loading, reload } = useLabs();
  return loading ? (
    <p data-measured="loading"><Translated vi="Đang đọc báo cáo.">Reading reports.</Translated></p>
  ) : (
    <p data-measured="none">
      <Translated vi="Chưa có số đo nào trên máy này.">No measurement on this machine yet.</Translated> {missing(lab, gpu)}{' '}
      <button className="quiet" onClick={reload}>
        <Translated vi="Tải lại kết quả">Reload results</Translated>
      </button>
    </p>
  );
}

/** The fraction at which the plain atomic is furthest behind the privatised kernel, against the reader's pick. */
function WorstReveal({ committed }: { committed: number }) {
  const { reports } = useLabs();
  const found = preferredBench(reports, l1).flatMap((s) => {
    const rows = behind(s);
    return rows.length ? [{ s, rows }] : [];
  });
  return (
    <>
      {found.length ? (
        found.map(({ s, rows }) => {
          const worst = rows.reduce((a, b) => (b.ratio > a.ratio ? b : a));
          const mine = rows.reduce((a, b) => (Math.abs(b.f - committed) < Math.abs(a.f - committed) ? b : a));
          return (
            <p key={s.backend} data-measured={worst.f} data-backend={s.backend}>
              <Translated
                vi={
                  <>
                    Đo trên {s.device} ({s.backend}) với {whose(s)}, trong lần chạy ngày {s.date}: atomic thường tụt lại xa nhất ở
                    tỷ lệ <b>{worst.f}</b>, nơi nó tốn thời gian gấp {fmt(worst.ratio)} lần kernel riêng tư hóa ({ms(worst.a)} so
                    với {ms(worst.p)}).
                  </>
                }
              >
                Measured on {s.device} ({s.backend}) with {whose(s)}, in the run of {s.date}: the plain atomic was furthest
                behind at fraction <b>{worst.f}</b>, where it took {fmt(worst.ratio)} times as long as the privatised kernel (
                {ms(worst.a)} against {ms(worst.p)}).
              </Translated>{' '}
              {mine.f === worst.f
                ? tr('That is the fraction you picked.', 'Đó là tỷ lệ bạn đã chọn.')
                : tr(
                    `At ${mine.f !== committed ? `${mine.f}, the measured fraction nearest your pick,` : 'your pick'} the ratio was ${fmt(mine.ratio)}.`,
                    `Ở ${mine.f !== committed ? `${mine.f}, tỷ lệ đã đo gần lựa chọn của bạn nhất,` : 'lựa chọn của bạn'} tỷ số là ${fmt(mine.ratio)}.`,
                  )}
            </p>
          );
        })
      ) : (
        <MissingLine lab="12.1" />
      )}
      <Translated
        vi={
          <p className="muted">
            Mô hình của sách (§12.2, p. 291) cho mọi khóa được giữ xếp hàng ở một bộ đếm, nên tranh chấp tăng theo tỷ lệ được
            giữ, trong khi kernel riêng tư hóa thực hiện một phép atomic công khai mỗi khối bất kể tỷ lệ. Ngược lại, số byte ghi
            cũng tăng theo tỷ lệ, và khối lượng việc của các phép atomic riêng cũng vậy. Hãy đọc cả cột tỷ số trong bảng bên dưới,
            không chỉ giá trị lớn nhất: khi hai tỷ lệ chênh nhau vài phần trăm, cái nào đứng trước chỉ là nhiễu.
          </p>
        }
      >
        <p className="muted">
          The book's model (§12.2, p. 291) has every kept key queue at one counter, so contention grows with the fraction
          kept, while the privatised kernel makes one public atomic per block whatever the fraction. Against that, the
          bytes written grow with the fraction too, and so does the work of the private atomics. Read the whole column of
          ratios in the table below, not only its largest value: when two fractions are within a few percent, which one
          comes first is noise.
        </p>
      </Translated>
    </>
  );
}

/** Lab 12.3: part A over part B at coarse 4, half kept. */
const stagedRatio = (reports: Reports) =>
  measure(reports, l3, (s) => {
    const a = secs(at(s, 'flags', 0.5));
    const b = secs(at(s, 'staged', 0.5, { coarse: 4 }));
    if (!a || !b) return null;
    const r = at(s, 'flags', 0.5)!;
    return {
      value: a / b,
      detail: tr(
        `${ms(a)} for part A, ${ms(b)} for part B at coarse 4, ${count(num(r.n) ?? 0)} keys at half kept`,
        `${ms(a)} cho phần A, ${ms(b)} cho phần B ở coarse 4, ${count(num(r.n) ?? 0)} khóa với một nửa được giữ`,
      ),
    };
  });

/** Lab 12.3: part B's coarsening factors at half kept, fastest first. */
function CoarseReveal({ committed }: { committed: number }) {
  const { reports } = useLabs();
  const found = preferredBench(reports, l3).flatMap((s) => {
    const rows = s.rows
      .filter((r) => r.name === 'staged' && num(r.fraction) === 0.5 && num(r.coarse) != null && secs(r) != null)
      .map((r) => ({ c: num(r.coarse)!, t: secs(r)! }))
      .sort((a, b) => a.c - b.c);
    return rows.length ? [{ s, rows }] : [];
  });
  return (
    <>
      {found.length ? (
        found.map(({ s, rows }) => {
          const best = rows.reduce((a, b) => (b.t < a.t ? b : a));
          const worst = rows.reduce((a, b) => (b.t > a.t ? b : a));
          const mine = rows.find((r) => r.c === committed);
          return (
            <p key={s.backend} data-measured={best.c} data-backend={s.backend}>
              <Translated
                vi={
                  <>
                    Đo trên {s.device} ({s.backend}) với {whose(s)}, trong lần chạy ngày {s.date}, với một nửa được giữ: hệ số
                    nhanh nhất là <b>coarse {best.c}</b>, ở {ms(best.t)}, và chậm nhất là coarse {worst.c}, ở {ms(worst.t)}.
                  </>
                }
              >
                Measured on {s.device} ({s.backend}) with {whose(s)}, in the run of {s.date}, at half kept: the fastest
                factor was <b>coarse {best.c}</b>, at {ms(best.t)}, and the slowest coarse {worst.c}, at {ms(worst.t)}.
              </Translated>{' '}
              {best.c === committed
                ? tr('That is the factor you picked.', 'Đó là hệ số bạn đã chọn.')
                : mine
                  ? tr(`Your pick took ${ms(mine.t)}, ${fmt(mine.t / best.t)} times the fastest.`, `Lựa chọn của bạn tốn ${ms(mine.t)}, gấp ${fmt(mine.t / best.t)} lần mức nhanh nhất.`)
                  : tr(
                      `Coarse ${committed} was not measured: the factors timed are ${rows.map((r) => r.c).join(', ')}.`,
                      `Coarse ${committed} chưa được đo: các hệ số đã đo là ${rows.map((r) => r.c).join(', ')}.`,
                    )}
            </p>
          );
        })
      ) : (
        <MissingLine lab="12.3" />
      )}
      <Translated
        vi={
          <p className="muted">
            Gộp việc cho luồng làm đoạn được đưa vào bộ nhớ tạm của mỗi khối dài hơn và phép scan các tổng warp thưa hơn (§12.6,
            pp. 299–300); nó cũng tạo ra ít khối hơn và một buffer tạm lớn hơn mỗi khối. Hãy so độ chênh trước khi nhìn bên thắng,
            và xem các tỷ lệ khác trong bảng: thứ tự các hệ số không nhất thiết giống nhau ở mỗi tỷ lệ.
          </p>
        }
      >
        <p className="muted">
          Coarsening makes each block's staged run longer and its scan of warp totals rarer (§12.6, pp. 299–300); it also
          makes fewer blocks and a bigger staging buffer per block. Compare the spread before the winner, and look at the
          other fractions in the table: the order of the factors need not be the same at each.
        </p>
      </Translated>
    </>
  );
}

/** Lab 12.4: one thread per row over one block per row, batch 1, 128,256 columns, 1 percent kept. */
const VOCAB = 128256;
const compaction = (s: BenchSet, kernel: string, rows: number, cols: number, fraction: number) =>
  s.rows.find((r) => r.name === 'compaction' && r.kernel === kernel && num(r.rows) === rows && num(r.cols) === cols && num(r.fraction) === fraction);
const batch1Ratio = (reports: Reports) =>
  measure(reports, l4, (s) => {
    const a = secs(compaction(s, 'serial', 1, VOCAB, 0.01));
    const b = secs(compaction(s, 'block', 1, VOCAB, 0.01));
    if (!a || !b) return null;
    const a64 = secs(compaction(s, 'serial', 64, VOCAB, 0.01));
    const b64 = secs(compaction(s, 'block', 64, VOCAB, 0.01));
    return {
      value: a / b,
      detail:
        tr(
          `${ms(a)} with one thread per row, ${ms(b)} with one block per row, 1 row of ${count(VOCAB)} columns at 1% kept`,
          `${ms(a)} với một luồng mỗi hàng, ${ms(b)} với một khối mỗi hàng, 1 hàng ${count(VOCAB)} cột với 1% được giữ`,
        ) + (a64 && b64 ? tr(`; at 64 rows the ratio is ${fmt(a64 / b64)}`, `; ở 64 hàng tỷ số là ${fmt(a64 / b64)}`) : ''),
    };
  });
/** Lab 12.4: one block per row, 64 rows, time at half kept over time at 1 percent kept. */
const fractionRatio = (reports: Reports) =>
  measure(reports, l4, (s) => {
    const hi = secs(compaction(s, 'block', 64, VOCAB, 0.5));
    const lo = secs(compaction(s, 'block', 64, VOCAB, 0.01));
    if (!hi || !lo) return null;
    const hi32 = secs(compaction(s, 'block', 64, 32000, 0.5));
    const lo32 = secs(compaction(s, 'block', 64, 32000, 0.01));
    return {
      value: hi / lo,
      detail:
        tr(
          `${ms(hi)} at 50% kept, ${ms(lo)} at 1%, 64 rows of ${count(VOCAB)} columns`,
          `${ms(hi)} với 50% được giữ, ${ms(lo)} với 1%, 64 hàng ${count(VOCAB)} cột`,
        ) + (hi32 && lo32 ? tr(`; ${fmt(hi32 / lo32)} at 32,000 columns`, `; ${fmt(hi32 / lo32)} ở 32,000 cột`) : ''),
    };
  });

/** Lab 12.5: lab 12.3 part B over the single pass, coarse 4, half kept, out of place, same backend. */
const singlePassRatio = (reports: Reports) => {
  const partB = preferredBench(reports, l3);
  return measure(reports, l5, (s) => {
    const b = partB.find((x) => x.backend === s.backend);
    const one = s.rows.find((r) => r.name === 'singlepass' && num(r.fraction) === 0.5 && num(r.coarse) === 4 && (r.inplace as unknown) === false);
    const tB = secs(at(b, 'staged', 0.5, { coarse: 4 }));
    const t1 = secs(one);
    if (!b || !tB || !t1) return null;
    return {
      whose: tr(`lab 12.5 from ${whose(s)}, lab 12.3 from ${whose(b)}`, `lab 12.5 từ ${whose(s)}, lab 12.3 từ ${whose(b)}`),
      value: tB / t1,
      detail: tr(
        `${ms(tB)} for lab 12.3 part B, ${ms(t1)} for the single pass, coarse 4, half kept, out of place`,
        `${ms(tB)} cho lab 12.3 phần B, ${ms(t1)} cho bản một lượt, coarse 4, một nửa được giữ, không tại chỗ`,
      ),
    };
  });
};

// ---------------------------------------------------------------- the race flavour, which the status tables do not show
type FullResult = RunResult & { tsan?: unknown[]; log?: string };
const TSAN_EXIT = 66;

function RaceLine({ lab: l }: { lab: Lab }) {
  const { reports, loading } = useLabs();
  const run = reports.race;
  const rs = (run?.results.filter((x) => x.lab === l.id) ?? []) as FullResult[];
  const sol = rs.filter((x) => x.variant === 'solution');
  const work = rs.filter((x) => x.variant === 'work');
  const s = labStatus(run, l.id, 'solution');
  const n = sol.flatMap((x) => x.tests).length;
  const wTests = work.flatMap((x) => x.tests);
  const wRaces = work.reduce((k, x) => k + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0);
  const ws = labStatus(run, l.id, 'work');
  const races = wRaces ? tr(`, ${wRaces} race${wRaces === 1 ? '' : 's'} reported`, `, báo ${wRaces} race`) : '';
  return (
    <p data-race-status={loading ? 'loading' : s} data-race-lab={l.num}>
      {loading ? (
        tr('Reading reports.', 'Đang đọc báo cáo.')
      ) : !run || (s === 'none' && !work.length) ? (
        <Translated
          vi={
            <>
              Chưa có lần chạy nào của lab {l.num} trên backend race ở máy này:{' '}
              <code>python3 labs/pmpp.py run {l.num} --backend race</code>.
            </>
          }
        >
          <>
            No run of lab {l.num} on the race backend on this machine yet:{' '}
            <code>python3 labs/pmpp.py run {l.num} --backend race</code>.
          </>
        </Translated>
      ) : (
        <>
          <Translated vi="Trên backend race của máy này, lời giải tham chiếu">On the race backend of this machine, the reference solution</Translated>{' '}
          {s === 'pass'
            ? tr(
                `passed all ${n} tests with no race reported: emulator-checked under both flavours.`,
                `vượt qua cả ${n} bài kiểm thử, không báo race nào: emulator-checked dưới cả hai biến thể.`,
              )
            : s === 'fail'
              ? tr('failed: unverified there.', 'thất bại: unverified ở đó.')
              : tr('has no run yet.', 'chưa có lần chạy nào.')}{' '}
          <Translated vi="Bản làm của bạn ở đó:">Your work there:</Translated>{' '}
          {!work.length
            ? tr('no run yet.', 'chưa có lần chạy nào.')
            : ws === 'none'
              ? tr(
                  `not started (${wTests.filter((t) => t.pass).length} of ${wTests.length} tests pass as shipped${races}).`,
                  `chưa bắt đầu (${wTests.filter((t) => t.pass).length} / ${wTests.length} bài kiểm thử vượt qua ở trạng thái ban đầu${races}).`,
                )
              : ws === 'pass'
                ? tr(
                    `passes, ${wTests.length} of ${wTests.length} tests, no race reported.`,
                    `vượt qua, ${wTests.length} / ${wTests.length} bài kiểm thử, không báo race nào.`,
                  )
                : tr(
                    `not passing: ${wTests.filter((t) => t.pass).length} of ${wTests.length} tests${races}.`,
                    `chưa vượt qua: ${wTests.filter((t) => t.pass).length} / ${wTests.length} bài kiểm thử${races}.`,
                  )}
        </>
      )}
    </p>
  );
}

// ---------------------------------------------------------------- lab 12.2: what each backend showed of the shipped kernels
interface HuntRun {
  backend: string;
  device: string;
  date: string;
  untouched: boolean;
  passed: number;
  total: number;
  aborted: boolean;
  asan: boolean;
  races: number;
}
const HUNT_BACKENDS = ['emu', 'race', 'metal', 'cuda', 'hip'];
const BACKEND_NAME: Record<string, string> = {
  get emu() {
    return tr('the emulator (emu)', 'trình giả lập (emu)');
  },
  get race() {
    return tr('the race emulator', 'trình giả lập race');
  },
};

function huntRuns(reports: Reports): HuntRun[] {
  return HUNT_BACKENDS.flatMap((b) => {
    const run = reports[b];
    const rs = (run?.results.filter((x) => x.lab === l2.id && x.variant === 'work') ?? []) as FullResult[];
    if (!run || !rs.length) return [];
    const tests = rs.flatMap((x) => x.tests);
    return [
      {
        backend: b,
        device: String(run.device.name ?? b),
        date: run.generated.slice(0, 10),
        untouched: run.work_state?.[l2.id] === 'untouched',
        passed: tests.filter((t) => t.pass).length,
        total: tests.length,
        // ThreadSanitizer ends a run that reported a race with exit code 66; that is a report, not a stop.
        aborted: rs.some((x) => x.test_rc != null && x.test_rc !== 0 && x.test_rc !== 1 && !(b === 'race' && x.test_rc === TSAN_EXIT)),
        asan: rs.some((x) => (x.log ?? '').includes('AddressSanitizer')),
        races: rs.reduce((n, x) => n + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0),
      },
    ];
  });
}

function HuntReveal({ committed }: { committed: number }) {
  const { reports, loading, reload } = useLabs();
  const runs = huntRuns(reports);
  return (
    <>
      <Translated
        vi={
          <p data-answer="2">
            Khi lab này được xây, từng lỗi được đặt lại riêng lẻ vào lời giải tham chiếu và chạy trên mọi backend ở đây.{' '}
            <b>emu bắt được 2</b> trong ba lỗi, một cách tất định. Lỗi mà nó che đi là một lần cập nhật chỉ sai khi có luồng khác
            chen vào giữa lần đọc và lần ghi của nó, và emu không bao giờ ngắt một luồng ở chỗ đó.{' '}
            {committed === 2 ? 'Đó là con số của bạn.' : `Bạn đã dự đoán ${committed}.`}
          </p>
        }
      >
        <p data-answer="2">
          When this lab was built, each defect was put back alone into the reference solution and run on every backend
          here. <b>emu caught 2</b> of the three, deterministically. The one it hid is an update that goes wrong only if
          another thread gets in between its read and its write, and emu never interrupts a thread there.{' '}
          {committed === 2 ? 'That is your number.' : `You predicted ${committed}.`}
        </p>
      </Translated>
      {runs.map((r) => (
        <p key={r.backend} data-measured={r.passed} data-backend={r.backend}>
          <Translated
            vi={
              <>
                {BACKEND_NAME[r.backend] ? `Trên ${BACKEND_NAME[r.backend]}` : `Trên ${r.device} (${r.backend})`}, lần chạy cuối
                của <code>work/</code> của bạn ngày {r.date}
                {r.untouched ? ', vẫn ở trạng thái ban đầu, với cả ba lỗi' : ', sau khi bạn sửa'}:{' '}
                {r.aborted && r.total === 0
                  ? `dừng trước khi trường hợp đầu tiên hoàn tất${r.asan ? ', kèm một báo cáo AddressSanitizer' : ''}.`
                  : `${r.passed} / ${r.total} trường hợp đúng${r.aborted ? `, rồi lần chạy dừng${r.asan ? ' kèm một báo cáo AddressSanitizer' : ''}` : ''}.`}
                {r.races ? ` ThreadSanitizer báo ${r.races} race.` : ''}
              </>
            }
          >
            {BACKEND_NAME[r.backend] ? `On ${BACKEND_NAME[r.backend]}` : `On ${r.device} (${r.backend})`}, the last run of
            your <code>work/</code> on {r.date}
            {r.untouched ? ', still as shipped, with all three defects' : ', after your edits'}:{' '}
            {r.aborted && r.total === 0
              ? `stopped before its first case finished${r.asan ? ', with an AddressSanitizer report' : ''}.`
              : `${r.passed} of ${r.total} cases right${r.aborted ? `, then the run stopped${r.asan ? ' with an AddressSanitizer report' : ''}` : ''}.`}
            {r.races ? ` ThreadSanitizer reported ${r.races} race${r.races === 1 ? '' : 's'}.` : ''}
          </Translated>
        </p>
      ))}
      {!runs.length ? (
        loading ? (
          <p data-measured="loading"><Translated vi="Đang đọc báo cáo.">Reading reports.</Translated></p>
        ) : (
          <p data-measured="none">
            <Translated
              vi={
                <>
                  Chưa có lần chạy nào cho bản làm của bạn ở lab này. Hãy chạy{' '}
                  <code>python3 labs/pmpp.py run 12.2 --backend emu</code> trước khi thay đổi bất cứ gì, rồi tải lại.
                </>
              }
            >
              No run of your work for this lab yet. Run <code>python3 labs/pmpp.py run 12.2 --backend emu</code> before
              you change anything, then reload.
            </Translated>{' '}
            <button className="quiet" onClick={reload}>
              <Translated vi="Tải lại kết quả">Reload results</Translated>
            </button>
          </p>
        )
      ) : null}
      <Translated
        vi={
          <p className="muted">
            Khi cả ba lỗi đều có mặt, lỗi đầu tiên che các lỗi khác: một lần chạy trình giả lập dừng ở báo cáo AddressSanitizer
            không bao giờ tới các trường hợp sau đó. Sửa một lỗi, chạy lại, và bức tranh thay đổi. Một trường hợp vượt qua trên
            GPU của bạn chỉ cho bạn thấy vài lần khởi chạy trên một thiết bị, không chứng minh kernel đúng.
          </p>
        }
      >
        <p className="muted">
          With all three defects in, the first failure hides the others: an emulator run that stops at an AddressSanitizer
          report never reaches the cases after it. Fix one, run again, and the picture changes. A case that passes on your
          GPU has shown you a few launches on one device, not that the kernel is correct.
        </p>
      </Translated>
    </>
  );
}

// ---------------------------------------------------------------- milestone M5
function MilestoneStatus() {
  const { reports, loading } = useLabs();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const done = backends.filter((b) => labStatus(reports[b], l4.id, 'work') === 'pass');
  const name = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  return (
    <p data-milestone={loading ? 'loading' : done.join(',') || 'none'}>
      {loading
        ? tr('Reading reports.', 'Đang đọc báo cáo.')
        : !backends.length
          ? tr('No lab results are recorded on this machine yet.', 'Chưa có kết quả lab nào được ghi trên máy này.')
          : done.length
            ? tr(
                `Your own lab 12.4 passes on ${done.map(name).join(' and ')} on this machine: the compaction piece of M5 is in place there.`,
                `Lab 12.4 của chính bạn vượt qua trên ${done.map(name).join(' và ')} ở máy này: phần nén gọn của M5 đã sẵn sàng ở đó.`,
              )
            : tr('Your own lab 12.4 does not pass on any backend of this machine yet.', 'Lab 12.4 của chính bạn chưa vượt qua trên backend nào của máy này.')}
    </p>
  );
}

// ---------------------------------------------------------------- results-table columns
const UNSTABLE_NAME: Record<string, string> = {
  get atomic() {
    return tr('plain atomic (Fig. 12.2)', 'atomic thường (Hình 12.2)');
  },
  get warp() {
    return tr('warp-aggregated (Fig. 12.3)', 'gộp theo warp (Hình 12.3)');
  },
  get private() {
    return tr('privatised (Fig. 12.6)', 'riêng tư hóa (Hình 12.6)');
  },
  get copy() {
    return tr('copy (provided)', 'sao chép (cung cấp sẵn)');
  },
};
const keysCol = { get head() { return tr('Keys', 'Số khóa'); }, num: true, cell: (r: BenchRow) => (num(r.n) != null ? count(num(r.n)!) : '') };
const fractionCol = { get head() { return tr('Fraction kept', 'Tỷ lệ được giữ'); }, num: true, cell: (r: BenchRow) => (r.name === 'copy' ? '' : String(r.fraction ?? '')) };
const STABLE_NAME = (r: BenchRow) =>
  r.name === 'flags'
    ? tr('part A, three steps', 'phần A, ba bước')
    : r.name === 'staged'
      ? tr(`part B, coarse ${String(r.coarse)}`, `phần B, coarse ${String(r.coarse)}`)
      : r.name === 'copy'
        ? tr('copy (provided)', 'sao chép (cung cấp sẵn)')
        : r.name;

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS}>
      <Translated
        vi={
          <LabsIntro>
            Năm bài thực hành, mỗi bài trong thư mục riêng dưới <code>labs/ch12/</code>. Bài đầu viết ba bộ lọc không ổn định của
            chương và đo thời gian chúng trên một dải tỷ lệ được giữ. Bài thứ hai là cuộc săn ba lỗi trong cùng ba kernel đó. Bài
            thứ ba là bộ lọc ổn định của mục 12.5 và 12.6, bằng scan, theo hai thiết kế chạy được trên mọi backend. Bài thứ tư là
            một phần của mốc dự án M5: danh sách ứng viên của một bộ lấy mẫu, một hàng logits mỗi khối. Bài thứ năm, tùy chọn, là
            bộ lọc tại chỗ một lượt của mục 12.7, chỉ cho CUDA và HIP.
          </LabsIntro>
        }
      >
        <LabsIntro>
          Five labs, each in its own directory under <code>labs/ch12/</code>. The first writes the chapter's three
          unstable filters and times them across a sweep of the fraction kept. The second is a hunt for three defects in
          the same three kernels. The third is the stable filter of sections 12.5 and 12.6, by scan, in two designs that
          run on every backend. The fourth is a piece of milestone M5: the candidate list of a sampler, one row of logits
          per block. The fifth, optional, is the single-pass in-place filter of section 12.7, for CUDA and HIP only.
        </LabsIntro>
      </Translated>
      <Translated
        vi={
          <p>
            Lab 12.5 là ngoại lệ của cách bố trí đó: nó chỉ có <code>kernel.cu</code>, vì nó để một khối chờ khối khác, và Apple
            không ghi nhận bảo đảm nào rằng khối được chờ rồi sẽ chạy. <a href={href('ch12', '12.7')}>Mục 12.7</a> giải thích
            vì sao, và nên làm gì trên Metal thay vào đó.
          </p>
        }
      >
        <p>
          Lab 12.5 is the exception to that layout: it has <code>kernel.cu</code> only, because it lets one block wait for
          another, and Apple documents no guarantee that the block waited for will ever run.{' '}
          <a href={href('ch12', '12.7')}>Section 12.7</a> explains why, and what to do on Metal instead.
        </p>
      </Translated>
      <RunningLabs
        example="12.1"
        alsoShownIn={
          <Translated
            vi={
              <>
                bảng số byte ở <a href={href('ch12', '12.6')}>mục 12.6</a>
              </>
            }
          >
            <>
              the bytes table of <a href={href('ch12', '12.6')}>section 12.6</a>
            </>
          </Translated>
        }
      />
      <BackendsAndEmulator />
      <Translated
        vi={
          <p>
            Trình giả lập có biến thể thứ hai, <code>race</code>, và cả năm lab ở đây đều được dựng cho nó. Nó chạy các luồng giả
            lập đồng thời dưới ThreadSanitizer và báo hai luồng chạm vào cùng vùng nhớ mà không có gì sắp thứ tự cho chúng, kèm
            cả hai dòng mã nguồn. Đó là trình giả lập duy nhất thấy được một bộ đếm bị tăng mà không có atomic, và là trình duy
            nhất trong đó nhiều khối có thể chạy cùng lúc (khối tối đa 128 luồng), điều mà look-back của lab 12.5 cần. Yêu cầu
            nó bằng <code>--backend race</code>; <code>python3 labs/pmpp.py doctor</code> cho biết trình biên dịch của bạn có
            dựng được nó không. Các bảng trên trang này chỉ hiển thị biến thể đầu tiên; một dòng dưới mỗi lab cho kết quả race.
          </p>
        }
      >
        <p>
          The emulator has a second flavour, <code>race</code>, and all five labs here are built for it. It runs the
          emulated threads concurrently under ThreadSanitizer and reports two threads that touch the same memory with
          nothing ordering them, with both source lines. It is the only emulator that can see a counter bumped without an
          atomic, and the only one in which several blocks can run at once (blocks of at most 128 threads), which lab
          12.5's look-back needs. Ask for
          it with <code>--backend race</code>; <code>python3 labs/pmpp.py doctor</code> says whether your compiler can
          build it. The tables on this page show the first flavour only; a line under each lab gives the race results.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Không trình giả lập nào chạy warp rộng hơn 32 lane, và mặt nạ của chúng là 32 bit. Vì vậy cái bẫy 64 lane của mục
            12.3, một mặt nạ 32 bit hoặc <code>__popc</code> trên một phiếu bầu 64 bit, hoặc <code>1 &lt;&lt; lane</code> quá lane
            31, vượt qua ở mọi nơi một máy Mac có thể kiểm thử. Các lab giữ một mẫu kiểm thử cho nó, chỉ giữ các khóa 32 đến 63
            trong mỗi 64 khóa, mẫu này tìm ra lỗi trên wavefront AMD 64 lane. Trên Mac, hãy kiểm tra mặt nạ có kiểu bằng cách đọc
            mã của bạn, và thử dạng "naive HIP port" trong trình trực quan hóa của <a href={href('ch12', '12.3')}>mục 12.3</a>.
          </p>
        }
      >
        <p>
          Neither emulator runs a warp wider than 32 lanes, and their masks are 32-bit. The 64-lane trap of section 12.3,
          a 32-bit mask or <code>__popc</code> on a 64-bit vote, or <code>1 &lt;&lt; lane</code> past lane 31, therefore
          passes everywhere a Mac can test. The labs keep a test pattern for it, only keys 32 to 63 of every 64 kept, which
          finds it on a 64-lane AMD wavefront. On a Mac, check the typed mask by reading your code, and try the "naive HIP
          port" form in <a href={href('ch12', '12.3')}>section 12.3</a>'s visualiser.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch12-facts.md B1; ch05-facts B5; infra-facts B3)', '2026-11-20 (notes/today/ch12-facts.md, mục B1; ch05-facts B5; infra-facts B3)')}>
        <Translated
          vi={
            <p>
              Mọi hàm <code>_sync</code> của HIP đều đòi mặt nạ 64 bit, và <code>__ballot_sync</code> trả về{' '}
              <code>unsigned long long</code>. <code>warpSize</code> là 64 trên gfx9 và 32 trên gfx10 trở lên.
            </p>
          }
        >
          <p>
            Every HIP <code>_sync</code> function requires a 64-bit mask, and <code>__ballot_sync</code> returns{' '}
            <code>unsigned long long</code>. <code>warpSize</code> is 64 on gfx9 and 32 on gfx10 and above.
          </p>
        </Translated>
      </Today>
      <VerificationStatus />

      {/* ------------------------------------------------------------ 12.1 */}
      <h2><Translated vi="Lab 12.1: bộ lọc không ổn định theo ba cách">Lab 12.1: the unstable filter three ways</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn viết ba bộ lọc không ổn định của chương trên các khóa <code>unsigned int</code>, giữ một khóa khi nó nhỏ hơn một
            ngưỡng. (a) là Hình 12.2: một phép atomic mỗi khóa được giữ trên một bộ đếm. (b) là Hình 12.3 theo dạng mục 12.3 khuyên
            dùng: một lần bỏ phiếu mà mọi lane tham gia, một phép atomic mỗi warp, và ô của mỗi lane lấy từ các phiếu bên dưới nó.
            (c) là Hình 12.6: một danh sách và bộ đếm riêng mỗi khối trong bộ nhớ chia sẻ, ba rào chắn, một phép atomic công khai
            mỗi khối. Cả ba để các khóa được giữ trong <code>output[0, *outputSize)</code> theo thứ tự bất kỳ, và bên gọi đặt{' '}
            <code>*outputSize</code> về 0 trước mỗi lần gọi.
          </p>
        }
      >
        <p>
          You write the chapter's three unstable filters of <code>unsigned int</code> keys, keeping a key when it is
          below a threshold. (a) is Fig. 12.2: one atomic per kept key on one counter. (b) is Fig. 12.3 in the form
          section 12.3 recommends: a vote that every lane takes, one atomic per warp, and each lane's slot from the votes
          below it. (c) is Fig. 12.6: a private list and counter per block in shared memory, three barriers, one public
          atomic per block. All three leave the kept keys in <code>output[0, *outputSize)</code> in any order, and the
          caller sets <code>*outputSize</code> to 0 before every call.
        </p>
      </Translated>
      <Source section="12.2–12.4" pages="290–297" figs={tr('Figs. 12.2, 12.3, 12.6', 'Hình 12.2, 12.3, 12.6')} />
      <p><Translated vi="Chỗ mã khác trang sách, và vì sao:">Where the code departs from the page, and why:</Translated></p>
      <ul>
        <Translated
          vi={
            <li>
              <code>cond(val)</code> của sách không bao giờ được định nghĩa. Ở đây nó là <code>val &lt; threshold</code>, để các
              bài kiểm thử và benchmark chọn được bao nhiêu khóa còn lại.
            </li>
          }
        >
          <li>
            The book's <code>cond(val)</code> is never defined. Here it is <code>val &lt; threshold</code>, so that the
            tests and the benchmark choose how many keys survive.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Một ballot trong luồng điều khiển đồng nhất thay cho <code>__activemask()</code> bên trong nhánh: hướng dẫn của
              NVIDIA giờ nói mặt nạ hoạt động không thể cho biết lane nào đã vào nhánh. Vì vậy không luồng nào return trước lần
              bỏ phiếu, các luồng quá cuối mảng bỏ phiếu 0, và mặt nạ của phép shuffle là mặt nạ đầy đủ.
            </li>
          }
        >
          <li>
            A ballot in uniform control flow replaces <code>__activemask()</code> inside the branch: NVIDIA's guide now
            says the active mask cannot tell which lanes took a branch. So no thread returns before the vote, threads past
            the end vote 0, and the shuffle's mask is the full mask.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Mặt nạ có kiểu riêng, <code>lane_mask_t</code>, 32 bit trên CUDA và các trình giả lập, 64 bit dưới HIP. Các phép
              đếm bit của nó đi qua <code>maskPopc</code> và <code>maskFfs</code>, mà template bị xóa biến một kiểu đối số sai
              thành lỗi biên dịch, như llama.cpp làm. Lane lấy từ <code>warpSize</code>.
            </li>
          }
        >
          <li>
            The mask has its own type, <code>lane_mask_t</code>, 32 bits on CUDA and the emulators and 64 under HIP. Its
            bit counts go through <code>maskPopc</code> and <code>maskFfs</code>, whose deleted template turns a wrong
            argument type into a compile error, as llama.cpp does. The lane comes from <code>warpSize</code>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Dưới HIP các phép atomic là <code>atomicAdd</code> kiểu cũ, relaxed và hợp lệ trên bộ nhớ chia sẻ và toàn cục. Ở đó{' '}
              <code>&lt;cuda/atomic&gt;</code> đến từ libhipcxx. Bản dựng HIP của khóa học liên kết{' '}
              <code>libhipcxx::libhipcxx</code> khi CMake tìm thấy nó (nó đi kèm ROCm 10.1 trở lên, infra-facts B2). Không có nó,
              mã include <code>&lt;cuda/atomic&gt;</code> không dựng được. Có nó, chưa máy ROCm nào dựng mã của khóa học này
              (unverified). Vì vậy lab dùng lời gọi kiểu cũ, thứ HIP ghi nhận cho bộ nhớ chia sẻ và toàn cục (B1) và không cần thư
              viện bổ sung. Chưa dựng bằng ROCm ở đây.
            </li>
          }
        >
          <li>
            Under HIP the atomics are the legacy <code>atomicAdd</code>, relaxed and valid on shared and global memory.{' '}
            <code>&lt;cuda/atomic&gt;</code> comes from libhipcxx there. The course's HIP build links{' '}
            <code>libhipcxx::libhipcxx</code> when CMake finds it (it ships with ROCm 10.1 and later, infra-facts B2).
            Without it, code that includes <code>&lt;cuda/atomic&gt;</code> does not build. With it, no ROCm machine has
            built this course's code yet (unverified). So the lab uses the legacy call, which HIP documents for shared
            and global memory (B1) and which needs no extra library. Not built with ROCm here.
          </li>
        </Translated>
      </ul>
      <Translated
        vi={
          <p>
            Trên Metal các bộ đếm là một <code>device atomic_uint*</code> và một <code>threadgroup atomic_uint</code>, và các bước
            warp của (b) là bốn hàm SIMD group: <code>simd_sum</code>, <code>simd_prefix_exclusive_sum</code>,{' '}
            <code>simd_is_first</code> và <code>simd_broadcast_first</code> là phép đếm, hạng, leader và phát. Chúng được gọi từ
            mọi luồng, với phép atomic được bỏ qua khi một SIMD group không giữ gì.
          </p>
        }
      >
        <p>
          On Metal the counters are a <code>device atomic_uint*</code> and a <code>threadgroup atomic_uint</code>, and
          the warp steps of (b) are four SIMD-group functions: <code>simd_sum</code>,{' '}
          <code>simd_prefix_exclusive_sum</code>, <code>simd_is_first</code> and <code>simd_broadcast_first</code> are the
          count, the rank, the leader and the broadcast. They are called from every thread, with the atomic skipped when
          a SIMD-group keeps nothing.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch12-facts.md C1, C3; infra-facts C1)', '2026-11-20 (notes/today/ch12-facts.md, mục C1, C3; infra-facts C1)')}>
        <Translated
          vi={
            <p>
              Các hàm atomic của Metal chỉ nhận kiểu atomic. Apple không nói liệu các luồng hoạt động bên trong một nhánh có đúng
              là những luồng đã vào nhánh không.
            </p>
          }
        >
          <p>
            Metal's atomic functions take atomic types only. Apple does not say whether the active threads inside a
            branch are the ones that took it.
          </p>
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
                  <code>filterAtomicKernel</code>, <code>filterWarpAggregatedKernel</code>, <code>filterPrivatisedKernel</code>{' '}
                  và các hàm khởi chạy <code>launchFilterAtomic</code>, <code>launchFilterWarpAggregated</code>,{' '}
                  <code>launchFilterPrivatised</code>
                </>
              }
            >
              <>
                <code>filterAtomicKernel</code>, <code>filterWarpAggregatedKernel</code>, <code>filterPrivatisedKernel</code>{' '}
                and their launchers <code>launchFilterAtomic</code>, <code>launchFilterWarpAggregated</code>,{' '}
                <code>launchFilterPrivatised</code>
              </>
            </Translated>,
          ],
          ['kernel.metal', <><code>filter_atomic</code>, <code>filter_warp_aggregated</code>, <code>filter_privatised</code></>],
          ['launch.mm', <><code>encodeFilterAtomic</code>, <code>encodeFilterWarpAggregated</code>, <code>encodeFilterPrivatised</code></>],
        ]}
      />
      <Translated
        vi={
          <p>
            Được cung cấp sẵn và không sửa: một phép sao chép n khóa, mốc băng thông của benchmark, trong{' '}
            <code>labs/ch12/lab1_unstable/provided/copy_keys.cu</code>, <code>copy_keys.metal</code> và{' '}
            <code>copy_keys.mm</code>. Nó cũng được kiểm thử.
          </p>
        }
      >
        <p>
          Provided and not edited: a copy of n keys, the benchmark's bandwidth reference, in{' '}
          <code>labs/ch12/lab1_unstable/provided/copy_keys.cu</code>, <code>copy_keys.metal</code> and{' '}
          <code>copy_keys.mm</code>. It is tested too.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 12.1 --backend emu</code> trước, rồi <code>--backend race</code>, rồi GPU của
            bạn. Một bộ lọc không ổn định có thể đặt các khóa theo thứ tự bất kỳ, nên chương trình điều khiển (driver) không so
            sánh mảng. Nó kiểm tra rằng <code>*outputSize</code> bằng số khóa mà một bộ lọc tuần tự giữ, rằng{' '}
            <code>output[0, count)</code> sau khi sắp xếp bằng danh sách tuần tự đã sắp xếp (không mất gì, không ghi gì hai lần),
            và rằng mọi ô quá số đếm vẫn giữ giá trị canh (sentinel) được ghi trước lần gọi; rồi nó đặt bộ đếm về 0, gọi lại và
            kiểm tra thêm lần nữa. Kích thước chạy từ 1 đến 4,099, và trên thiết bị tới 3 × 2<sup>20</sup> + 17; tỷ lệ được giữ
            0, 0.001, 0.03, 0.5, 0.97 và 1; và các mẫu kết thúc ở biên warp và biên khối, kể cả một mẫu chỉ giữ các khóa 32 đến
            63 trong mỗi 64 khóa.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 12.1 --backend emu</code> first, then <code>--backend race</code>, then your
          GPU. An unstable filter may put its keys in any order, so the driver does not compare arrays. It checks that{' '}
          <code>*outputSize</code> equals the count a sequential filter keeps, that <code>output[0, count)</code> sorted
          equals the sequential list sorted (nothing lost, nothing written twice), and that every slot past the count still
          holds the sentinel written before the call; then it zeroes the counter, calls again and checks once more. Sizes
          run from 1 to 4,099, and on a device to 3 × 2<sup>20</sup> + 17; fractions kept 0, 0.001, 0.03, 0.5, 0.97 and
          1; and patterns that end on warp and block edges, including one that keeps only keys 32 to 63 of every 64.
        </p>
      </Translated>
      <RaceLine lab={l1} />
      <Translated
        vi={
          <p>
            Với <code>--bench</code>, driver tăng n cho đến khi phép sao chép được cung cấp tốn vài mili giây, rồi đo ba bộ lọc ở
            các tỷ lệ được giữ 0, 0.001, 0.01, 0.03, 0.1, 0.25, 0.5, 0.75 và 1, lấy lần tốt nhất trong 5 lần chạy mỗi mức, và
            kiểm tra mọi kết quả trước khi báo cáo. GB/s tính lưu lượng tối thiểu một bộ lọc cần: 4 byte đọc mỗi khóa và 4 byte
            ghi mỗi khóa được giữ. Hãy chốt hai dự đoán trước.
          </p>
        }
      >
        <p>
          With <code>--bench</code> the driver grows n until the provided copy takes a few milliseconds, then times the
          three filters at fractions kept 0, 0.001, 0.01, 0.03, 0.1, 0.25, 0.5, 0.75 and 1, best of 5 runs each, and
          checks every result before reporting it. GB/s counts the least traffic a filter needs: 4 bytes read per key and 4
          written per kept key. Commit two predictions first.
        </p>
      </Translated>
      <Predict
        id="ch12/labs/p-lab1-worst"
        prompt={
          <Translated
            vi={
              <>
                Trước khi chạy benchmark. Ở tỷ lệ được giữ nào, từ 0 đến 1, kernel atomic thường sẽ tụt lại xa nhất so với kernel
                riêng tư hóa, tức là tốn thời gian gấp nhiều lần nhất?
              </>
            }
          >
            <>
              Before you run the benchmark. At which fraction kept, from 0 to 1, will the plain-atomic kernel be furthest
              behind the privatised one, that is, take the most times as long?
            </>
          </Translated>
        }
        unit={tr('(fraction kept)', '(tỷ lệ được giữ)')}
      >
        {(v) => <WorstReveal committed={v} />}
      </Predict>
      <LabPrediction
        id="ch12/labs/p-lab1-agg"
        prompt={
          <Translated
            vi={
              <>
                Trước khi chạy benchmark. Với một nửa số khóa được giữ, thời gian của kernel atomic thường chia cho thời gian của
                kernel gộp theo warp sẽ là bao nhiêu?
              </>
            }
          >
            <>
              Before you run the benchmark. At half the keys kept, the plain-atomic kernel's time divided by the
              warp-aggregated kernel's time will be what?
            </>
          </Translated>
        }
        unit={tr('× the warp-aggregated time', '× thời gian gộp theo warp')}
        measure={aggRatio}
        missing={missing('12.1')}
        reading={
          <Translated
            vi={
              <>
                Trên 1, tự gộp bằng tay có lợi. Gần 1, atomic thường tốn ngang bản đã gộp: hoặc trình biên dịch đã tự gộp nó, như
                sách nói về nvcc (p. 294) và như backend LLVM của AMD làm ở <code>-O1</code> trở lên, hoặc bộ đếm không phải thứ
                giới hạn kernel nào. Apple không ghi nhận gì về cả hai hướng, và chỉ mã được sinh ra mới cho bạn biết là trường hợp
                nào (notes/today/ch12-facts.md A7, B4).
              </>
            }
          >
            <>
              Above 1, aggregating by hand paid. Near 1, the plain atomic cost what the aggregated one did: either the
              compiler already aggregates it, as the book says of nvcc (p. 294) and as AMD's LLVM backend does at{' '}
              <code>-O1</code> and above, or the counter was not what limited either kernel. Apple documents nothing either
              way, and only the generated code tells you which (notes/today/ch12-facts.md A7, B4).
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l1}
        cols={[
          { head: tr('Filter', 'Bộ lọc'), cell: (r) => UNSTABLE_NAME[r.name] ?? r.name },
          fractionCol,
          keysCol,
          msCol,
          numCol('GB/s', 'gbps'), // i18n-ignore: unit
          numCol('Gkeys/s', 'gkeys_per_s'), // i18n-ignore: unit
        ]}
      />
      <ReferenceSolution lab={l1} tiers={UNSTABLE_TIERS} />

      {/* ------------------------------------------------------------ 12.2 */}
      <h2><Translated vi="Lab 12.2: săn lỗi bộ lọc">Lab 12.2: filter hunt</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn nhận ba bộ lọc của lab 12.1, bằng CUDA và bằng Metal, được viết đúng trừ ba lỗi độc lập, cùng ba lỗi đó trong cả
            hai tệp. Chúng biên dịch được. Hãy tìm và sửa cả ba. Đó là những lỗi mà mục 12.2 đến 12.4 dễ dẫn tới: một lỗi về một
            lần cập nhật không được bị ngắt, một lỗi về chỗ ô của một lane bắt đầu, một lỗi về lúc luồng 0 được hành động thay
            khối của nó.
          </p>
        }
      >
        <p>
          You are handed lab 12.1's three filters, in CUDA and in Metal, written correctly except for three independent
          defects, the same three in both files. They compile. Find and fix all three. They are the mistakes sections 12.2
          to 12.4 invite: one about an update that must not be interrupted, one about where a lane's slot starts, one about
          when thread 0 may act for its block.
        </p>
      </Translated>
      <Source section="12.2–12.4" pages="290–297" figs={tr('Figs. 12.2, 12.3, 12.6; the defects are ours', 'Hình 12.2, 12.3, 12.6; các lỗi là của khóa học này')} />
      <LabFiles
        lab={l2}
        files={[
          ['kernel.cu', <Translated vi={<>ba kernel, có các lỗi; các hàm khởi chạy bên dưới chúng là đúng</>}><>the three kernels, with the defects; the launchers below them are correct</></Translated>],
          ['kernel.metal', <Translated vi={<>cùng ba kernel đó, với cùng ba lỗi</>}><>the same three kernels, with the same three defects</></Translated>],
          ['launch.mm', <Translated vi={<>đúng ngay từ đầu; không cần thay đổi gì</>}><>correct as shipped; nothing to change</></Translated>],
        ]}
      />
      <Translated
        vi={
          <p>
            Không phải lỗi: mặt nạ có kiểu và các hàm hỗ trợ của nó, lần bỏ phiếu mọi lane tham gia trước khi bất kỳ luồng nào
            return, chỉ số lane lấy từ <code>warpSize</code>, <code>atomicAdd</code> kiểu cũ dưới <code>__HIPCC__</code>, và trên
            Metal các hàm SIMD group được gọi từ mọi luồng. Lab 12.1 giải thích lý do cho từng thứ. Driver chạy các phép kiểm tra
            của lab 12.1 ở kích thước nhỏ hơn, và trên GPU nó lặp mỗi đầu vào năm lần. Trước khi chạy bất cứ gì, hãy đọc kernel
            và chốt một con số.
          </p>
        }
      >
        <p>
          Not defects: the typed mask and its helpers, the vote every lane takes before any thread returns, the lane index
          from <code>warpSize</code>, the legacy <code>atomicAdd</code> under <code>__HIPCC__</code>, and on Metal the
          SIMD-group functions called from every thread. Lab 12.1 says why for each. The driver runs lab 12.1's checks at
          smaller sizes, and on a GPU it repeats every input five times. Before you run anything, read the kernel and
          commit to a number.
        </p>
      </Translated>
      <Predict
        id="ch12/labs/p-lab2-emu"
        prompt={
          <Translated
            vi={
              <>
                Trước khi thay đổi bất cứ gì. Trong ba lỗi, <code>emu</code>, trình giả lập chạy từng luồng một theo ba thứ tự cố
                định, sẽ bắt được bao nhiêu?
              </>
            }
          >
            <>
              Before you change anything. Of the three defects, how many will <code>emu</code>, the emulator that runs
              one thread at a time under three fixed orders, catch?
            </>
          </Translated>
        }
        unit={tr('of 3 defects', 'trên 3 lỗi')}
      >
        {(v) => <HuntReveal committed={v} />}
      </Predict>
      <p><Translated vi="Rồi săn lỗi, theo thứ tự này:">Then hunt, in this order:</Translated></p>
      <ol>
        <Translated
          vi={
            <li>
              Trình giả lập trước: <code>python3 labs/pmpp.py run 12.2 --backend emu</code>. Một số đếm sai, một ô không bao giờ
              được ghi (nó vẫn giữ giá trị canh) hoặc một ô được ghi quá số đếm, mỗi thứ chỉ tới một chỗ khác nhau. Một lần ghi
              ngoài mảng đầu ra dừng lần chạy với một báo cáo AddressSanitizer: đọc <code>WRITE</code>, dòng{' '}
              <code>kernel.cu</code> của khung đầu tiên, và nó vượt quá vùng bao nhiêu byte. Một lỗi ở "threads in order" mà vượt
              qua ở "threads reversed" là về việc luồng 0 làm gì giữa hai rào chắn.
            </li>
          }
        >
          <li>
            Emulator first: <code>python3 labs/pmpp.py run 12.2 --backend emu</code>. A wrong count, a slot never written
            (it still holds the sentinel) or one written past the count each point somewhere different. A write outside
            the output array stops the run with an AddressSanitizer report: read <code>WRITE</code>, the first frame's{' '}
            <code>kernel.cu</code> line, and how many bytes past the region it was. A failure under "threads in order"
            that passes under "threads reversed" is about what thread 0 does between two barriers.
          </li>
        </Translated>
        <li><Translated vi="Sửa một thứ rồi chạy lại. Các lỗi thay đổi khi bạn tiến lên.">Fix one thing and run again. The failures change as you go.</Translated></li>
        <Translated
          vi={
            <li>
              <code>python3 labs/pmpp.py run 12.2 --backend race</code>. Hai luồng chạm vào cùng vùng nhớ mà không đồng bộ hóa sẽ
              được báo kèm cả hai dòng <code>kernel.cu</code>, dù đáp án có sai hay không. ThreadSanitizer báo mỗi cặp dòng một
              lần mỗi lần chạy, nên một trường hợp sau có thể vượt qua dù lỗi vẫn còn.
            </li>
          }
        >
          <li>
            <code>python3 labs/pmpp.py run 12.2 --backend race</code>. Two threads touching the same memory without
            synchronisation are reported with both <code>kernel.cu</code> lines, whether or not the answer came out wrong.
            ThreadSanitizer reports each pair of lines once per run, so a later case can pass with the defect still in.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Rồi GPU của bạn: <code>python3 labs/pmpp.py run 12.2</code>. Ghi lại lỗi nào nó cho bạn thấy và lỗi nào nó che, và
              kích thước nào thất bại: một lỗi giữa các warp vẫn ẩn khi toàn bộ đầu vào vừa trong một warp.
            </li>
          }
        >
          <li>
            Then your GPU: <code>python3 labs/pmpp.py run 12.2</code>. Note which defects it showed you and which it hid,
            and which sizes fail: a defect between warps stays hidden while the whole input fits in one warp.
          </li>
        </Translated>
      </ol>
      <Translated
        vi={
          <p>
            Dù GPU của bạn cho thấy gì, hãy đối chiếu với những gì nhà sản xuất ghi nhận, vì đó là những gì thiết bị tiếp theo
            được phép làm. Cũng không công cụ GPU nào gọi tên cả ba lỗi giúp bạn:{' '}
            <code>compute-sanitizer --tool racecheck</code> của NVIDIA chỉ kiểm tra bộ nhớ chia sẻ, nên không thấy được một bộ
            đếm toàn cục, còn AMD và Apple không ghi nhận trình phát hiện race nào.
          </p>
        }
      >
        <p>
          Whatever your GPU showed, read it against what the vendors document, because that is what the next device is
          entitled to do. No GPU tool would have named all three for you either: NVIDIA's{' '}
          <code>compute-sanitizer --tool racecheck</code> checks shared memory only, so it cannot see a global counter,
          and AMD and Apple document no race detector.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch12-facts.md, settled table: ch09 A3, A10, B6, C5)', '2026-11-20 (notes/today/ch12-facts.md, bảng đã chốt: ch09 A3, A10, B6, C5)')}>
        <Translated
          vi={
            <p>
              Các phép atomic trên một địa chỉ được tuần tự hóa không theo thứ tự cố định; một lần đọc, cộng và ghi thông thường
              không hứa hẹn gì.
            </p>
          }
        >
          <p>
            Atomics on one address are serialised in no fixed order; a plain read, add and write promises nothing.
          </p>
        </Translated>
      </Today>
      <RaceLine lab={l2} />
      <LabResults
        lab={l2}
        noBench={tr(
          'This lab has no benchmark. The result is the line above: every test passing on the emulators and on a device. As shipped, the kernels are meant to fail.',
          'Lab này không có benchmark. Kết quả là dòng ở trên: mọi bài kiểm thử vượt qua trên các trình giả lập và trên một thiết bị. Ở trạng thái ban đầu, các kernel được thiết kế để thất bại.',
        )}
      />
      <ReferenceSolution lab={l2} tiers={UNSTABLE_TIERS} />

      {/* ------------------------------------------------------------ 12.3 */}
      <h2><Translated vi="Lab 12.3: một bộ lọc ổn định bằng scan">Lab 12.3: a stable filter by scan</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn viết hai bộ lọc ổn định, không tại chỗ: giữ một khóa khi nó nhỏ hơn một ngưỡng, và giữ các khóa được giữ theo thứ
            tự đầu vào. Phần A là bài tập 1, ba bước: một kernel ghi các cờ keep, phép scan được cung cấp biến chúng thành các
            offset, và một kernel rải mỗi khóa được giữ tới offset của nó, luồng của khóa cuối ghi số đếm. Phần B là bài tập 3 và
            4 theo dạng reduce-then-scan. Mỗi khối đếm các khóa được giữ trong ô (tile) 256 × coarse khóa của nó bằng bỏ phiếu
            warp; phép scan được cung cấp biến các số đếm thành offset của khối; một kernel thứ hai đọc lại ô theo từng vòng, xếp
            hạng các khóa được giữ (một lần bỏ phiếu, popcount của các lane bên dưới, một scan các tổng warp) vào một buffer tạm
            trong bộ nhớ chia sẻ, rồi ghi đoạn đã đưa vào bộ nhớ tạm với các luồng liên tiếp trên các ô liên tiếp.
          </p>
        }
      >
        <p>
          You write two stable, out-of-place filters: keep a key when it is below a threshold, and keep the kept keys in
          input order. Part A is exercise 1, three steps: a kernel writes the keep flags, the provided scan turns them into
          offsets, and a kernel scatters each kept key to its offset, the thread of the last key writing the count. Part B
          is exercises 3 and 4 in reduce-then-scan form. Each block counts the kept keys of its tile of 256 × coarse keys
          with warp votes; the provided scan turns the counts into block offsets; a second kernel re-reads the tile round
          by round, ranks its kept keys (a vote, the popcount of the lanes below, a scan of warp totals) into a staging
          buffer in shared memory, and then writes the staged run with consecutive threads on consecutive slots.
        </p>
      </Translated>
      <Source section="12.5–12.6" pages="297–300" figs={tr('Figs. 12.7–12.10; exercises 1, 3 and 4, p. 302', 'Hình 12.7–12.10; bài tập 1, 3 và 4, p. 302')} />
      <Translated
        vi={
          <p>
            Không thiết kế nào để một khối chờ khối khác, nên cả hai chạy được trên CUDA, HIP và Metal.{' '}
            <code>gridExclusiveScan</code> của Hình 12.8 là một phép scan toàn lưới bên trong một kernel, nơi một khối chờ các
            khối trước nó, và chỉ CUDA ghi nhận khi nào điều đó an toàn (<a href={href('ch12', '12.7')}>mục 12.7</a>). Lab 12.5
            dựng phiên bản đó. Phần B là hình dạng mà <code>nonzero_static</code> của PyTorch có trên CUDA, và{' '}
            <code>nonzero</code> trên MPS của nó dùng ba dispatch theo cùng cách.
          </p>
        }
      >
        <p>
          Neither design lets one block wait for another, so both run on CUDA, HIP and Metal. Fig. 12.8's{' '}
          <code>gridExclusiveScan</code> is a grid-wide scan inside one kernel, where a block waits for the blocks before
          it, and only CUDA documents when that is safe (<a href={href('ch12', '12.7')}>section 12.7</a>). Lab 12.5 builds
          that version. Part B is the shape PyTorch's <code>nonzero_static</code> takes on CUDA, and its MPS{' '}
          <code>nonzero</code> takes three dispatches the same way.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Hai thiết kế này là các bài tập của sách dưới một dạng của khóa học này. Mã của PyTorch: notes/today/ch12-facts.md
              D4, D5.
            </p>
          }
        >
          <p>
            The two designs are the book's exercises in a form of ours. PyTorch's code: notes/today/ch12-facts.md D4, D5.
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
                  <code>keepFlagsKernel</code>, <code>scatterKernel</code>, <code>countKeptKernel</code>,{' '}
                  <code>stagedScatterKernel</code> và các hàm khởi chạy <code>launchStableFilterFlags</code> và{' '}
                  <code>launchStableFilterStaged</code>
                </>
              }
            >
              <>
                <code>keepFlagsKernel</code>, <code>scatterKernel</code>, <code>countKeptKernel</code>,{' '}
                <code>stagedScatterKernel</code> and the launchers <code>launchStableFilterFlags</code> and{' '}
                <code>launchStableFilterStaged</code>
              </>
            </Translated>,
          ],
          ['kernel.metal', <><code>keep_flags</code>, <code>scatter_flags</code>, <code>count_kept</code>, <code>staged_scatter</code></>],
          [
            'launch.mm',
            <Translated
              vi={
                <>
                  <code>encodeStableFilterFlags</code> và <code>encodeStableFilterStaged</code>, mỗi hàm là vài dispatch trong
                  compute encoder duy nhất nó nhận, phép scan được cung cấp nằm ở giữa
                </>
              }
            >
              <>
                <code>encodeStableFilterFlags</code> and <code>encodeStableFilterStaged</code>, each several dispatches in
                the one compute encoder it receives, the provided scan in the middle
              </>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Được cung cấp sẵn và không sửa: một phép scan loại trừ phân cấp trên <code>uint32</code> (mỗi khối scan một ô, các
            tổng của ô được scan theo cùng cách, một lượt cộng hoàn tất: scan-scan-add của{' '}
            <a href={href('ch11', '11.9')}>mục 11.9</a>, không chờ giữa các khối), và kernel sao chép mà benchmark dùng để so
            sánh, trong <code>labs/ch12/shared-g12/scan_u32.cu</code>, <code>scan_u32.metal</code> và{' '}
            <code>scan_u32_launch.mm</code>.
          </p>
        }
      >
        <p>
          Provided and not edited: a hierarchical exclusive scan of <code>uint32</code> (every block scans a tile, the tile
          totals are scanned the same way, an add pass finishes: <a href={href('ch11', '11.9')}>section 11.9</a>'s
          scan-scan-add, with no waiting between blocks), and the copy kernel the benchmark compares against, in{' '}
          <code>labs/ch12/shared-g12/scan_u32.cu</code>, <code>scan_u32.metal</code> and{' '}
          <code>scan_u32_launch.mm</code>.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md C9; ch05-facts C8)', '2026-11-20 (notes/today/ch02-facts.md, mục C9; ch05-facts C8)')}>
        <Translated
          vi={
            <p>
              Các dispatch trong một compute encoder thông thường chạy theo thứ tự, mỗi dispatch thấy các lần ghi của dispatch
              trước: tương ứng của Metal với các lần khởi chạy trên một stream CUDA. Bộ nhớ threadgroup đặt bằng{' '}
              <code>setThreadgroupMemoryLength:atIndex:</code> là bội số của 16 byte.
            </p>
          }
        >
          <p>
            Dispatches in one plain compute encoder run in order, each seeing the previous one's writes: Metal's
            counterpart of launches on one CUDA stream. Threadgroup memory set with{' '}
            <code>setThreadgroupMemoryLength:atIndex:</code> is a multiple of 16 bytes.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 12.3 --backend emu</code> trước; nó mất vài phút, vì mọi luồng GPU chạy trên CPU.
            Phép scan được cung cấp được kiểm thử riêng so với một phép scan CPU. Rồi phần A, và phần B ở coarse 1, 2, 4, 8 và 16
            (16 chỉ khi buffer tạm của nó vừa thiết bị), ở các kích thước từ 1 đến 4,099 và trên thiết bị tới 3 × 2<sup>20</sup>{' '}
            + 17, với tỷ lệ được giữ 0, 0.001, 0.5, 0.97 và 1 và các đoạn vượt qua biên ô. Các khóa được giữ đều khác nhau và
            không theo thứ tự tăng, nên đầu ra phải bằng bộ lọc ổn định CPU từng phần tử một; mọi ô quá số đếm vẫn phải giữ giá trị
            canh; và <code>keep</code>, <code>offset</code>, các số đếm của khối và vùng nhớ nháp của phép scan bắt đầu đầy rác.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 12.3 --backend emu</code> first; it takes a few minutes, because every GPU
          thread runs on the CPU. The provided scan is tested alone against a CPU scan. Then part A, and part B at coarse 1,
          2, 4, 8 and 16 (16 only where its staging buffer fits the device), at sizes from 1 to 4,099 and on a device to 3 ×
          2<sup>20</sup> + 17, with fractions kept 0, 0.001, 0.5, 0.97 and 1 and runs that cross tile boundaries. The kept
          keys are all different and not in increasing order, so the output must equal the CPU stable filter element by
          element; every slot past the count must still hold the sentinel; and <code>keep</code>, <code>offset</code>, the
          block counts and the scan's scratch start full of garbage.
        </p>
      </Translated>
      <RaceLine lab={l3} />
      <Translated
        vi={
          <p>
            Với <code>--bench</code>, driver tăng n cho đến khi một phép sao chép thường tốn vài mili giây và đo phần A ("flags")
            và phần B ("staged") ở mọi hệ số gộp việc vừa thiết bị, cho các tỷ lệ được giữ từ 0.001 đến 1. Cách đếm byte của mục
            12.5 là mô hình: các mảng có tên của phần A di chuyển 28 + 4f byte mỗi khóa ở tỷ lệ được giữ f (đầu vào đọc hai lần,{' '}
            <code>keep</code> ghi một lần và đọc hai lần, <code>offset</code> ghi một lần và đọc một lần, các khóa được giữ được
            ghi), 30 ở mức giữ một nửa; phần B di chuyển 8 + 4f, khoảng 10. Bên cạnh đó driver in "at copy speed", số byte mỗi
            khóa mà một phép sao chép thường lẽ ra đã di chuyển trong cùng thời gian: con số của mô hình so với cái giá thiết bị
            của bạn tính. Phép scan được cung cấp di chuyển nhiều hơn mức mô hình tính cho nó, và bộ nhớ đệm quyết định phần còn
            lại.
          </p>
        }
      >
        <p>
          With <code>--bench</code> the driver grows n until a plain copy takes a few milliseconds and times part A
          ("flags") and part B ("staged") at every coarsening factor that fits, for fractions kept 0.001 to 1. Section
          12.5's byte count is the model: part A's named arrays move 28 + 4f bytes per key at kept fraction f (the input
          read twice, <code>keep</code> written once and read twice, <code>offset</code> written once and read once, the
          kept keys written), 30 at half kept; part B moves 8 + 4f, about 10. Beside it the driver prints "at copy speed",
          the bytes per key a plain copy would have moved in the same time: the model's count against what your device
          charged. The provided scan moves more than the model charges it, and caches decide the rest.
        </p>
      </Translated>
      <LabPrediction
        id="ch12/labs/p-lab3-staged"
        prompt={
          <Translated
            vi={
              <>
                Trước khi chạy benchmark. Ở coarse 4 và một nửa số khóa được giữ, thời gian của phần A chia cho thời gian của
                phần B sẽ là bao nhiêu?
              </>
            }
          >
            <>
              Before you run the benchmark. At coarse 4 and half the keys kept, part A's time divided by part B's time will
              be what?
            </>
          </Translated>
        }
        unit={tr("× part B's time", '× thời gian của phần B')}
        measure={stagedRatio}
        missing={missing('12.3')}
        reading={
          <Translated
            vi={
              <>
                Tỷ số byte của mô hình là 30 trên 10. Tỷ số thời gian là những gì thiết bị của bạn tạo ra từ đó: các lượt thêm
                của phép scan tính vào phần A, và một ô được phần B đọc hai lần có thể đến từ bộ nhớ đệm ở lần thứ hai. Hãy so cột
                "at copy speed" với cột của mô hình trong bảng.
              </>
            }
          >
            <>
              The model's ratio of bytes is 30 to 10. The time ratio is what your device made of it: the scan's extra
              passes count against part A, and a tile read twice by part B may come from a cache the second time. Compare
              the "at copy speed" column with the model's column in the table.
            </>
          </Translated>
        }
      />
      <Predict
        id="ch12/labs/p-lab3-coarse"
        prompt={
          <Translated vi={<>Và ở mức giữ một nửa, hệ số gộp việc nào của phần B, 1, 2, 4, 8 hay 16, sẽ nhanh nhất?</>}>
            <>And at half kept, which coarsening factor of part B, 1, 2, 4, 8 or 16, will be fastest?</>
          </Translated>
        }
        unit={tr('(coarsening factor)', '(hệ số gộp việc)')}
      >
        {(v) => <CoarseReveal committed={v} />}
      </Predict>
      <LabResults
        lab={l3}
        cols={[
          { head: tr('Design', 'Thiết kế'), cell: STABLE_NAME },
          fractionCol,
          keysCol,
          msCol,
          numCol('GB/s', 'gbps'), // i18n-ignore: unit
          numCol(tr('Model, bytes per key', 'Mô hình, byte mỗi khóa'), 'model_bytes_per_key'),
          numCol(tr('At copy speed, bytes per key', 'Ở tốc độ sao chép, byte mỗi khóa'), 'copy_equivalent_bytes_per_key'),
        ]}
      />
      <ReferenceSolution lab={l3} tiers={STABLE_TIERS} />

      {/* ------------------------------------------------------------ 12.4 */}
      <h2><Translated vi="Lab 12.4: nén gọn ứng viên (M5)">Lab 12.4: candidate compaction (M5)</Translated></h2>
      <Translated
        vi={
          <p>
            Một bộ lấy mẫu chỉ giữ một số token, bằng top-k, top-p hay min-p, trước hết cần, cho mỗi hàng logits, danh sách các
            token vượt qua một ngưỡng. Bạn viết danh sách đó. Với một tensor logits float32 <code>[rows × cols]</code>, thứ tự hàng
            (row-major) và batch ở chiều đầu, cùng một ngưỡng mỗi hàng, kernel của bạn ghi cho mỗi hàng các cặp (cột, logit) có{' '}
            <code>logit &gt;= threshold[r]</code>, theo thứ tự cột tăng dần, và số lượng của chúng. Một khối 256 luồng mỗi hàng
            đi qua hàng theo từng vòng <code>blockDim.x</code> cột, nên các lần load của mỗi vòng được hợp nhất. Vị trí của một
            cặp được giữ là số cặp đã giữ ở các vòng trước, cộng số cặp được giữ bởi các warp trước trong vòng này (một scan các
            tổng warp trong bộ nhớ chia sẻ), cộng số cặp được giữ bởi các lane thấp hơn trong warp của nó (lần bỏ phiếu và tổng
            tiền tố nhị phân của Hình 12.3). Không có atomic toàn cục, và không khối nào chờ khối khác.
          </p>
        }
      >
        <p>
          A sampler that keeps only some tokens, by top-k, top-p or min-p, first needs, for each row of logits, the list of
          tokens that pass a threshold. You write that list. For a <code>[rows × cols]</code> float32 logits tensor,
          row-major and batch first, and one threshold per row, your kernel writes for every row the pairs (column, logit)
          with <code>logit &gt;= threshold[r]</code>, in ascending column order, and their count. One block of 256 threads
          per row walks the row in rounds of <code>blockDim.x</code> columns, so each round's loads are coalesced. A kept
          pair's position is the pairs kept in earlier rounds, plus those kept by earlier warps this round (a scan of warp
          totals in shared memory), plus those kept by lower lanes of its warp (the vote and the binary prefix sum of Fig.
          12.3). No global atomics, and no block waits for another.
        </p>
      </Translated>
      <Source section="12.3, 12.5" pages="292–293, 297–298" figs={tr('Figs. 12.3, 12.7, 12.8', 'Hình 12.3, 12.7, 12.8')} />
      <Translated
        vi={
          <p>
            Đó là bộ lọc ổn định vì một bộ lấy mẫu duyệt các ứng viên theo thứ tự, hoặc phá thế hòa theo cột, cần cùng một đáp án
            ở mọi lần chạy; <code>torch.topk</code> của PyTorch từ chối atomic vì lý do đó. Và đó là một khối mỗi hàng vì khi đó
            số đếm đang chạy của hàng không bao giờ rời khỏi khối, đây là thiết kế mà top-k Metal của llama.cpp dùng và là thiết
            kế an toàn theo tài liệu trên cả ba backend.
          </p>
        }
      >
        <p>
          It is the stable filter because a sampler that walks its candidates in order, or breaks ties by column, needs
          the same answer every run; PyTorch's <code>torch.topk</code> refuses atomics for that reason. And it is one
          block per row because the row's running count then never leaves the block, which is the design llama.cpp's
          Metal top-k uses and the one that is documented-safe on all three backends.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Các cặp, một khối mỗi hàng và bộ lấy mẫu là của khóa học này, không phải của sách. Sự kiện về nhà sản xuất và thư
              viện: notes/today/ch12-facts.md C4, D2, D4.
            </p>
          }
        >
          <p>
            Pairs, one block per row and the sampler are ours, not the book's. Vendor and library facts:
            notes/today/ch12-facts.md C4, D2, D4.
          </p>
        </Translated>
      </Beyond>
      <LabFiles
        lab={l4}
        files={[
          ['kernel.cu', <Translated vi={<><code>compactCandidatesKernel</code> và <code>launchCompactCandidates</code></>}><><code>compactCandidatesKernel</code> and <code>launchCompactCandidates</code></></Translated>],
          ['kernel.metal', <><code>compact_candidates</code></>],
          ['launch.mm', <Translated vi={<><code>encodeCompactCandidates</code>, với độ rộng SIMD group đọc từ pipeline</>}><><code>encodeCompactCandidates</code>, with the SIMD-group width read from the pipeline</></Translated>],
        ]}
      />
      <Translated
        vi={
          <p>
            Được cung cấp sẵn và không sửa: một mức cơ sở cho mỗi hàng một luồng chạy vòng lặp min-p phía host của llama.cpp,
            trong <code>labs/ch12/lab4_candidates/provided/rowwise_serial.cu</code>, <code>rowwise_serial.metal</code> và{' '}
            <code>rowwise_serial.mm</code>. Nó đúng, nó cũng được kiểm thử, và benchmark đo kernel của bạn so với nó.
          </p>
        }
      >
        <p>
          Provided and not edited: a baseline that gives each row one thread running llama.cpp's host min-p loop, in{' '}
          <code>labs/ch12/lab4_candidates/provided/rowwise_serial.cu</code>, <code>rowwise_serial.metal</code> and{' '}
          <code>rowwise_serial.mm</code>. It is correct, it is tested too, and it is what the benchmark measures your
          kernel against.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 12.4</code>, bắt đầu với <code>--backend emu</code>. Số hàng 1, 4 và 7 nhân số
            cột từ 1 đến 128,256, ba mức cuối bằng cỡ bộ từ vựng (ít hơn trên các trình giả lập). Mỗi hàng của một batch có ngưỡng
            riêng: trên giá trị lớn nhất của hàng, bằng nó, dưới giá trị nhỏ nhất, các phân vị giữ khoảng 1, 10 và 50 phần trăm,
            và min-p 0.1 tính như llama.cpp. Các giá trị cài đúng bằng ngưỡng nằm ở các lane cạnh nhau, vắt qua biên warp 32 lane
            và 64 lane, vắt qua biên vòng và ở cả hai đầu, và một hàng chỉ giữ các cặp ở vòng cuối của nó. Số đếm phải chính xác,
            các cột tăng ngặt và bằng danh sách CPU, các giá trị bằng nhau từng bit, và mọi ô quá số đếm của một hàng không bị
            chạm tới. Không phép tính nào chạm vào giá trị, nên so sánh bằng chính xác là phép kiểm tra đúng, kể cả dưới chế độ
            toán mặc định của Metal. NaN và vô cực nằm ngoài hợp đồng, như ở lab 10.3.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 12.4</code>, starting with <code>--backend emu</code>. Rows 1, 4 and 7 by
          columns from 1 to 128,256, the last three vocabulary-sized (fewer on the emulators). Each row of a batch has its
          own threshold: above the row's max, equal to it, below its min, the quantiles that keep about 1, 10 and 50
          percent, and min-p 0.1 computed as llama.cpp does. Values planted exactly at the threshold sit in neighbouring
          lanes, across a 32-lane and a 64-lane warp boundary, across a round boundary and at both ends, and one row keeps
          only pairs in its last round. Counts must be exact, columns strictly ascending and equal to the CPU list, values
          equal bit for bit, and every slot past a row's count untouched. No arithmetic touches a value, so exact
          equality is the right test even under Metal's default math. NaN and infinity are outside the contract, as in
          lab 10.3.
        </p>
      </Translated>
      <RaceLine lab={l4} />
      <Translated
        vi={
          <p>
            Một trường hợp mà backend race có mặt để bắt: các tổng warp của một vòng nằm trong cùng mảng dùng chung với các tổng
            của vòng sau, nên một warp xong sớm có thể ghi đè một tổng mà warp chậm hơn vẫn đang cộng. Khi lab được xây, lời giải
            bị bỏ rào chắn đó vượt qua mọi bài kiểm thử <code>emu</code> và <code>race</code> đã báo nó.
          </p>
        }
      >
        <p>
          One case the race backend is there for: the warp totals of one round live in the same shared array as the next
          round's, so a warp that finishes early can overwrite a total a slower warp is still adding. When the lab was
          built, the solution with that barrier removed passed every <code>emu</code> test and <code>race</code> reported
          it.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Với <code>--bench</code>, driver đo các mức 1, 8 và 64 hàng nhân 32,000 và 128,256 cột ở các tỷ lệ được giữ 0.001,
            0.01, 0.1 và 0.5, kernel của bạn ("block") so với mức cơ sở ("serial"), mỗi mức lấy lần tốt nhất trong 5 lần chạy, mỗi
            lần lặp lời gọi trong vài mili giây. GB/s tính 4 byte mỗi logit đọc, 8 byte mỗi cặp được giữ ghi ra và 4 byte mỗi hàng
            cho số đếm. Hãy chốt hai dự đoán trước.
          </p>
        }
      >
        <p>
          With <code>--bench</code> the driver times rows 1, 8 and 64 by 32,000 and 128,256 columns at fractions kept
          0.001, 0.01, 0.1 and 0.5, your kernel ("block") against the baseline ("serial"), each the best of 5 runs that
          repeat the call for a couple of milliseconds. GB/s counts 4 bytes per logit read, 8 per kept pair written and 4
          per row for the count. Commit two predictions first.
        </p>
      </Translated>
      <LabPrediction
        id="ch12/labs/p-lab4-batch1"
        prompt={
          <Translated
            vi={
              <>
                Trước khi chạy benchmark. Với 1 hàng 128,256 cột và 1 phần trăm được giữ, thời gian một-luồng-mỗi-hàng được cung
                cấp chia cho thời gian một-khối-mỗi-hàng của bạn sẽ là bao nhiêu?
              </>
            }
          >
            <>
              Before you run the benchmark. At 1 row of 128,256 columns with 1 percent kept, the provided thread-per-row
              time divided by your block-per-row time will be what?
            </>
          </Translated>
        }
        unit={tr('× the block-per-row time', '× thời gian một-khối-mỗi-hàng')}
        measure={batch1Ratio}
        missing={missing('12.4')}
        reading={
          <Translated
            vi={
              <>
                Ở batch 1 cả hai phiên bản chỉ dùng một phần rất nhỏ của GPU: một luồng, hoặc một khối. Ý của{' '}
                <a href={href('ch10', '10.10')}>mục 10.10</a> cũng áp dụng ở đây. Tỷ số ở 64 hàng nằm trong chi tiết ở trên; hãy so
                thời gian của kernel khối tăng thế nào từ 1 hàng lên 64 trong bảng.
              </>
            }
          >
            <>
              At batch 1 both versions use a tiny part of the GPU: one thread, or one block.{' '}
              <a href={href('ch10', '10.10')}>Section 10.10</a>'s point applies here too. The ratio at 64 rows is in the
              detail above;
              compare how the block kernel's time grows from 1 row to 64 in the table.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch12/labs/p-lab4-fraction"
        prompt={
          <Translated
            vi={
              <>
                Và ở 64 hàng 128,256 cột: thời gian một-khối-mỗi-hàng của bạn ở 50 phần trăm được giữ chia cho thời gian của nó ở
                1 phần trăm được giữ sẽ là bao nhiêu?
              </>
            }
          >
            <>
              And at 64 rows of 128,256 columns: your block-per-row time at 50 percent kept divided by its time at 1
              percent kept will be what?
            </>
          </Translated>
        }
        unit={tr('× the time at 1% kept', '× thời gian ở 1% được giữ')}
        measure={fractionRatio}
        missing={missing('12.4')}
        reading={
          <Translated
            vi={
              <>
                Hãy hỏi điều gì thay đổi giữa 1 và 50 phần trăm: số byte ghi tăng năm mươi lần, số logits đọc và số vòng thì không.
                Một tỷ số gần 1 cho biết các lần đọc và các vòng quyết định nhịp độ.
              </>
            }
          >
            <>
              Ask what changes between 1 and 50 percent: the bytes written grow fifty-fold, the logits read and the number
              of rounds do not. A ratio near 1 says the reads and the rounds set the pace.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l4}
        filter={(r) => r.name === 'compaction'}
        cols={[
          { head: tr('Version', 'Phiên bản'), cell: (r) => (r.kernel === 'serial' ? tr('one thread per row (provided)', 'một luồng mỗi hàng (cung cấp sẵn)') : tr('one block per row', 'một khối mỗi hàng')) },
          rawCol(tr('Rows', 'Số hàng'), 'rows'),
          { head: tr('Columns', 'Số cột'), num: true, cell: (r) => (num(r.cols) != null ? count(num(r.cols)!) : '') },
          rawCol(tr('Fraction kept', 'Tỷ lệ được giữ'), 'fraction'),
          { head: tr('Kept pairs', 'Số cặp được giữ'), num: true, cell: (r) => (num(r.kept) != null ? count(num(r.kept)!) : '') },
          msCol,
          numCol('GB/s', 'gb_per_s'), // i18n-ignore: unit
          numCol(tr('Rows per second', 'Số hàng mỗi giây'), 'rows_per_s'),
        ]}
      />
      <h3><Translated vi="Mốc dự án M5, phần nén gọn">Project milestone M5, the compaction piece</Translated></h3>
      <Translated
        vi={
          <p>
            Mốc dự án M5 của dự án xuyên suốt là lấy mẫu trên GPU: top-k và top-p, có phạt lặp lại. Lab 9.3 cho nó số đếm token;
            lab 12.4 cho nó danh sách ứng viên, các cặp (cột, logit) ổn định và một số đếm mỗi hàng. Module X1 ghép nó với giá
            trị lớn nhất theo hàng của lab 10.3 (mốc dự án M4) ở <a href={href('x1', 'X1.6')}>mục X1.6</a>: ngưỡng min-p là giá
            trị lớn nhất cộng ln p, không cần softmax. Ngưỡng top-k đến từ một phép chọn (<a href={href('x1', 'X1.4')}>mục
            X1.4</a>, một radix select được đo so với phép sắp xếp của chương 14); các phần tử hòa tại ngưỡng đều được giữ, theo
            thứ tự cột, nên để giữ đúng k phần tử hãy bỏ các cặp thừa có giá trị bằng ngưỡng từ cuối, cách này giữ các cột thấp
            nhất, đúng quy tắc phá thế hòa của llama.cpp. README của lab đi qua cả ba.
          </p>
        }
      >
        <p>
          Milestone M5 of the through-line project is sampling on the GPU: top-k and top-p, with repetition penalties.
          Lab 9.3 gave it token counts; lab 12.4 gives it the candidate list, stable (column, logit) pairs and a count per
          row. Module X1 composes it with lab 10.3's row max (milestone M4) in{' '}
          <a href={href('x1', 'X1.6')}>section X1.6</a>: the min-p threshold is the max plus ln p, which needs no
          softmax. A top-k threshold comes from a selection (<a href={href('x1', 'X1.4')}>section X1.4</a>, a radix
          select measured against chapter 14's sort); ties at it are all kept, in column order, so to keep exactly k drop
          the surplus threshold-valued pairs from the end, which keeps the lowest columns, llama.cpp's tie rule. The lab's
          README works through all three.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Dự án xuyên suốt, <code>nanoinfer</code>, là của khóa học này: một đường suy luận GPU nhỏ cho một transformer chỉ
              có decoder, lớn dần theo từng mốc dự án. Quy tắc phá thế hòa và min-p: notes/today/ch12-facts.md D1, D3.
            </p>
          }
        >
          <p>
            The through-line project, <code>nanoinfer</code>, is ours: one small GPU inference path for a decoder-only
            transformer, grown a milestone at a time. Tie rule and min-p: notes/today/ch12-facts.md D1, D3.
          </p>
        </Translated>
      </Beyond>
      <MilestoneStatus />
      <ReferenceSolution lab={l4} tiers={ROW_TIERS} />

      {/* ------------------------------------------------------------ 12.5 */}
      <h2><Translated vi="Lab 12.5: bộ lọc ổn định tại chỗ trong một lượt (tùy chọn)">Lab 12.5: in-place stable filter in a single pass (optional)</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn viết bộ lọc ổn định của lab 12.3 thành một kernel đọc mỗi khóa một lần, và làm nó chạy được tại chỗ: driver gọi
            nó với <code>output == input</code> cũng như với một đầu ra riêng. Luồng 0 lấy chỉ số của khối từ một bộ đếm, không
            phải từ <code>blockIdx.x</code>; khối load ô của nó vào bộ nhớ chia sẻ và qua một rào chắn, nên mọi lần đọc đầu vào
            của nó đã xong; nó nén gọn các khóa được giữ của ô theo thứ tự; luồng 0 chờ cờ của khối trước bằng một acquire, cộng
            tổng của nó, công bố tổng của mình bằng một release; một rào chắn thứ hai; rồi khối ghi các khóa được giữ tại offset
            đó, các luồng liên tiếp trên các ô liên tiếp.
          </p>
        }
      >
        <p>
          You write lab 12.3's stable filter as one kernel that reads every key once, and make it work in place: the
          driver calls it with <code>output == input</code> as well as with a separate output. Thread 0 takes the block's
          index from a counter, not from <code>blockIdx.x</code>; the block loads its tile into shared memory and passes a
          barrier, so every read of its input is done; it compacts the tile's kept keys in order; thread 0 waits for the
          previous block's flag with an acquire, adds its total, publishes its own with a release; a second barrier; then
          the block writes its kept keys at that offset, consecutive threads on consecutive slots.
        </p>
      </Translated>
      <Source section="12.7" pages="300, 302" figs={tr('Fig. 12.11; exercises 2 to 4; Figs. 11.17, 11.18', 'Hình 12.11; bài tập 2 đến 4; Hình 11.17, 11.18')} />
      <Translated
        vi={
          <p>
            Chỉ cho họ CUDA. Hai điều kiện làm nó đúng khi chạy tại chỗ, release và acquire trên các cờ với mọi lần đọc diễn ra
            trước release, và chỉ chờ các khối đã bắt đầu, là của <a href={href('ch12', '12.7')}>mục 12.7</a>. CUDA ghi nhận điều
            kiện thứ hai; AMD và Apple thì không, nên trên HIP lab vẫn là unverified cho đến khi một báo cáo ROCm cho thấy nó vượt
            qua, còn trên Metal thì hoàn toàn không có phiên bản nào.
          </p>
        }
      >
        <p>
          CUDA family only. The two conditions that make it correct in place, the release and acquire on the flags with
          every read before the release, and waiting only on blocks that have started, are{' '}
          <a href={href('ch12', '12.7')}>section 12.7</a>'s. CUDA documents the second; AMD and Apple do not, so on HIP the
          lab stays unverified until a ROCm report shows it passing, and on Metal there is no version at all.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch12-facts.md A9, A10, B6, C4; ch11-facts B3, C5)', '2026-11-20 (notes/today/ch12-facts.md, mục A9, A10, B6, C4; ch11-facts B3, C5)')}>
        <Translated
          vi={
            <p>
              CUB chuyển look-back sang acquire và release khi một phép select có thể làm tại chỗ. Một khi một luồng của khối
              CUDA đã tiến triển, mọi luồng trong khối của nó rồi cũng sẽ tiến triển. AMD không ghi nhận đảm bảo tiến triển nào
              cho workgroup, Apple không ghi nhận gì giữa các threadgroup.
            </p>
          }
        >
          <p>
            CUB switches its look-back to acquire and release when a selection may be in place. Once a thread of a CUDA
            block has made progress, every thread of its block eventually does. AMD documents no forward progress for
            workgroups, Apple none between threadgroups.
          </p>
        </Translated>
      </Today>
      <LabFiles
        lab={l5}
        files={[
          [
            'kernel.cu',
            <Translated
              vi={
                <>
                  <code>stableFilterSinglePassKernel</code> và <code>launchStableFilterSinglePass</code>; mặt nạ lane,{' '}
                  <code>maskPopc</code> và <code>spinPause()</code> được cung cấp sẵn ở đầu tệp
                </>
              }
            >
              <>
                <code>stableFilterSinglePassKernel</code> and <code>launchStableFilterSinglePass</code>; the lane mask,{' '}
                <code>maskPopc</code> and <code>spinPause()</code> are provided at the top
              </>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 12.5 --backend emu</code> cho phần logic và <code>--backend race</code> cho phần
            thứ tự. <code>emu</code> chạy các khối lần lượt theo thứ tự chỉ số, nên look-back không bao giờ phải chờ ở đó và mối
            nguy tại chỗ không thể xuất hiện. <code>race</code> chạy các khối 64 luồng, nhiều khối cùng lúc, với acquire và
            release thật, và báo một truy cập không có thứ tự vào <code>inclusive[]</code> hoặc vào các khóa kèm cả hai dòng. Mọi
            trường hợp chạy cả không tại chỗ lẫn tại chỗ, ở coarse 1, 2 và 4, với hai mẫu khiến các khối ghi đè lên đầu vào của
            các khối trước. Khi tại chỗ, phần mảng quá số đếm không được xác định, như <code>remove_if</code> của Thrust để lại,
            và không được kiểm tra. README có một thí nghiệm tùy chọn: chuyển các cờ sang relaxed và xem race báo gì.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 12.5 --backend emu</code> for the logic and <code>--backend race</code> for
          the ordering. <code>emu</code> runs blocks one after another in index order, so the look-back never waits there
          and the in-place hazard cannot appear. <code>race</code> runs blocks of 64 threads, several at once, with real
          acquire and release, and reports an unordered access to <code>inclusive[]</code> or to the keys with both lines.
          Every case runs out of place and in place, at coarse 1, 2 and 4, with two patterns that make blocks write over
          earlier blocks' input. In place, the array past the count is unspecified, as Thrust's <code>remove_if</code>{' '}
          leaves it, and is not checked. The README has an optional experiment: make the flags relaxed and see what race
          reports.
        </p>
      </Translated>
      <RaceLine lab={l5} />
      <LabPrediction
        id="ch12/labs/p-lab5-singlepass"
        prompt={
          <Translated
            vi={
              <>
                Trước khi đo hiệu năng trên một thiết bị CUDA hoặc HIP. Ở coarse 4, một nửa được giữ, không tại chỗ: thời gian của
                lab 12.3 phần B chia cho thời gian của kernel này sẽ là bao nhiêu?
              </>
            }
          >
            <>
              Before you benchmark on a CUDA or HIP device. At coarse 4 and half kept, out of place: lab 12.3 part B's time
              divided by this kernel's time will be what?
            </>
          </Translated>
        }
        unit={tr('× the single-pass time', '× thời gian của bản một lượt')}
        measure={singlePassRatio}
        missing={
          <Translated
            vi={
              <>
                Phép so sánh này cần các hàng benchmark của lab 12.5 và lab 12.3 từ cùng một thiết bị CUDA hoặc HIP. Hãy chạy{' '}
                <code>python3 labs/pmpp.py run 12.5 --bench</code> và <code>python3 labs/pmpp.py run 12.3 --bench</code> ở đó,
                rồi tải lại. Không có phiên bản Metal nào để đo.
              </>
            }
          >
            <>
              This comparison needs benchmark rows for lab 12.5 and lab 12.3 from the same CUDA or HIP device. Run{' '}
              <code>python3 labs/pmpp.py run 12.5 --bench</code> and <code>python3 labs/pmpp.py run 12.3 --bench</code>{' '}
              there, then reload. There is no Metal version to measure.
            </>
          </Translated>
        }
        reading={
          <Translated
            vi={
              <>
                Mô hình: bản một lượt di chuyển 4 + 4f byte mỗi khóa, 6 ở mức giữ một nửa, trong khi phần B đọc mỗi khóa hai lần, 8
                + 4f, khoảng 10. Ngược lại, mỗi lần chuyển giao giữa các khối phải chờ khối trước nó, và chuỗi dài bằng số khối.
              </>
            }
          >
            <>
              The model: the single pass moves 4 + 4f bytes per key, 6 at half kept, where part B reads every key twice, 8
              + 4f, about 10. Against that, every block's hand-off waits for the block before it, and the chain is as long
              as there are blocks.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l5}
        filter={(r) => r.name === 'singlepass'}
        cols={[
          rawCol(tr('Fraction kept', 'Tỷ lệ được giữ'), 'fraction'),
          rawCol('Coarse', 'coarse'), // i18n-ignore: coarsening factor name used in the lab output
          { head: tr('In place', 'Tại chỗ'), cell: (r) => ((r.inplace as unknown) === true ? tr('yes', 'có') : tr('no', 'không')) },
          keysCol,
          msCol,
          numCol('GB/s', 'gbps'), // i18n-ignore: unit
        ]}
      />
      <ReferenceSolution lab={l5} tiers={SINGLE_TIERS} />
    </LabsPage>
  );
}
