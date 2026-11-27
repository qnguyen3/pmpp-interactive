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
  defineLabs,
  measure,
  msCol,
  numCol,
  preferredBench,
  rawCol,
  useLabs,
  whose,
  type Lab,
} from '../../components/LabKit';
import { Translated, tr } from '../../lib/i18n';
import { fmt, labStatus, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';

import { NoteGroup } from './NoteGroup';
// The page is built from components/LabKit.tsx; what is here is chapter 10's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch10', [
  { num: '10.1', dir: 'lab1_sum', name: 'one sum, three kernels', vi: { name: 'một tổng, ba kernel' } },
  { num: '10.2', dir: 'lab2_reducehunt', name: 'reduction hunt', vi: { name: 'truy lỗi phép rút gọn' } },
  { num: '10.3', dir: 'lab3_softmax_argmax', name: 'softmax and argmax (M4)', vi: { name: 'softmax và argmax (M4)' } },
]);
const [l1, l2, l3] = LABS;

// Identifier colours for the reference solutions. Only names that mean one thing in all
// three files of a lab.
const tiers = (global: string[], shared: string[], reg: string[] = []): Record<string, Tier> => ({
  ...Object.fromEntries(global.map((n) => [n, 'global' as Tier])),
  ...Object.fromEntries(shared.map((n) => [n, 'shared' as Tier])),
  ...Object.fromEntries(reg.map((n) => [n, 'reg' as Tier])),
});
const SUM_TIERS = tiers(['input', 'output'], ['input_s', 'partialSums_s'], ['partialSum']);
const HUNT_TIERS = tiers(['in', 'out'], ['partials_s']);
const ROW_TIERS = tiers(['x', 'y', 'out'], ['scratch', 'vals_s', 'idxs_s', 'vals', 'idxs']);

// ---------------------------------------------------------------- reading the reports
const ms = (s: number) => `${fmt(s * 1000)} ms`; // i18n-ignore: number and unit only
const count = (v: number) => v.toLocaleString('en-US');
const GPUS = ['metal', 'cuda', 'hip'];

const KERNEL_NAME: Record<string, string> = {
  get simple() {
    return tr('simple (Fig. 10.5, provided)', 'đơn giản (Fig. 10.5, có sẵn)');
  },
  get convergent() {
    return tr('convergent (Fig. 10.8, provided)', 'hội tụ (Fig. 10.8, có sẵn)');
  },
  get shared() {
    return tr('shared memory, kernel (a)', 'bộ nhớ chia sẻ, kernel (a)');
  },
  get warp() {
    return tr('two-stage, kernel (b)', 'hai giai đoạn, kernel (b)');
  },
};
const rowsKernel = (rows: BenchRow[], kernel: string) => rows.find((r) => r.name === 'rows' && r.kernel === kernel);

/** Lab 10.1: per row, how many times as fast the two-stage kernel is as the simple one. */
const warpOverSimple = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const a = num(rowsKernel(s.rows, 'simple')?.seconds);
    const b = num(rowsKernel(s.rows, 'warp')?.seconds);
    const r = rowsKernel(s.rows, 'warp');
    if (!a || !b || !r) return null;
    const conv = num(rowsKernel(s.rows, 'convergent')?.seconds);
    const shared = num(rowsKernel(s.rows, 'shared')?.seconds);
    return {
      value: a / b,
      detail: tr(
        `${ms(a)} for the simple kernel and ${ms(b)} for the two-stage kernel, ${count(num(r.rows) ?? 0)} rows of ${2 * (num(r.block) ?? 0)} floats` +
          (conv && shared ? `; convergent ${ms(conv)}, shared memory ${ms(shared)}` : ''),
        `${ms(a)} cho kernel đơn giản và ${ms(b)} cho kernel hai giai đoạn, ${count(num(r.rows) ?? 0)} hàng, mỗi hàng ${2 * (num(r.block) ?? 0)} float` +
          (conv && shared ? `; hội tụ ${ms(conv)}, bộ nhớ chia sẻ ${ms(shared)}` : ''),
      ),
    };
  });

/** Lab 10.3: at batch 1 and 128,256 columns, one thread per row over one block per row; batch 64 alongside. */
const COLS = 128256;
const softmaxRow = (rows: BenchRow[], kernel: string, batch: number) =>
  rows.find((r) => r.name === 'softmax' && r.kernel === kernel && num(r.rows) === batch && num(r.cols) === COLS);
const naiveOverBlock = (reports: Reports) =>
  measure(reports, l3, (s) => {
    const a = num(softmaxRow(s.rows, 'naive', 1)?.seconds);
    const b = num(softmaxRow(s.rows, 'block', 1)?.seconds);
    if (!a || !b) return null;
    const a64 = num(softmaxRow(s.rows, 'naive', 64)?.seconds);
    const b64 = num(softmaxRow(s.rows, 'block', 64)?.seconds);
    return {
      value: a / b,
      detail: tr(
        `${ms(a)} with one thread per row, ${ms(b)} with one block per row, batch 1, ${count(COLS)} columns` +
          (a64 && b64 ? `; at batch 64 the ratio is ${fmt(a64 / b64)} (${ms(a64)} against ${ms(b64)})` : ''),
        `${ms(a)} với một luồng cho mỗi hàng, ${ms(b)} với một khối cho mỗi hàng, batch 1, ${count(COLS)} cột` +
          (a64 && b64 ? `; ở batch 64 tỉ lệ là ${fmt(a64 / b64)} (${ms(a64)} so với ${ms(b64)})` : ''),
      ),
    };
  });

// ---------------------------------------------------------------- lab 10.1: the two predictions shared with 10.9 and 10.10
const BENCH_MISSING = (
  <Translated
    vi={
      <>
        Khi lab 10.1 đã đạt, hãy chạy <code>python3 labs/pmpp.py run 10.1 --bench</code> trên một máy có GPU, rồi tải lại.
      </>
    }
  >
    <>
      Once lab 10.1 passes, run <code>python3 labs/pmpp.py run 10.1 --bench</code> on a machine with a GPU, then reload.
    </>
  </Translated>
);

function MissingLine() {
  const { loading, reload } = useLabs();
  return loading ? (
    <p data-measured="loading">
      <Translated vi="Đang đọc báo cáo.">Reading reports.</Translated>
    </p>
  ) : (
    <p data-measured="none">
      <Translated vi="Chưa có phép đo nào trên máy này.">No measurement on this machine yet.</Translated> {BENCH_MISSING}{' '}
      <button className="quiet" onClick={reload}>
        <Translated vi="Tải lại kết quả">Reload results</Translated>
      </button>
    </p>
  );
}

/** The coarsening sweep: the fastest factor, and how the reader's pick did against it. */
function CoarseReveal({ committed }: { committed: number }) {
  const { reports } = useLabs();
  const found = preferredBench(reports, l1).flatMap((s) => {
    const rows = s.rows
      .filter((r) => r.name === 'any' && num(r.coarse) != null && num(r.gb_per_s) != null)
      .map((r) => ({ c: num(r.coarse)!, g: num(r.gb_per_s)!, n: num(r.n), block: num(r.block) }))
      .sort((a, b) => a.c - b.c);
    return rows.length ? [{ s, rows }] : [];
  });
  return (
    <>
      {found.length ? (
        found.map(({ s, rows }) => {
          const best = rows.reduce((a, b) => (b.g > a.g ? b : a));
          const worst = rows.reduce((a, b) => (b.g < a.g ? b : a));
          const far = (c: number) => Math.abs(Math.log2(Math.max(c, 1e-9)) - Math.log2(Math.max(committed, 1e-9)));
          const mine = rows.reduce((a, b) => (far(b.c) < far(a.c) ? b : a));
          return (
            <Translated
              key={s.backend}
              vi={
                <p data-measured={best.c} data-backend={s.backend}>
                  Đo trên {s.device} ({s.backend}) với {whose(s)}, trong lần chạy ngày {s.date}
                  {best.n ? <>, {count(best.n)} float trong các khối {best.block}</> : null}: hệ số nhanh nhất là{' '}
                  <b>C = {best.c}</b>, ở {fmt(best.g)} GB/s, và chậm nhất là C = {worst.c}, ở {fmt(worst.g)} GB/s.{' '}
                  {mine.c === best.c
                    ? 'Đó chính là hệ số bạn đã chọn.'
                    : `Lựa chọn của bạn${mine.c !== committed ? ` (giá trị đo gần nhất: C = ${mine.c})` : ''} chạy ở ${fmt(mine.g)} GB/s, bằng ${Math.round((mine.g / best.g) * 100)}% hệ số nhanh nhất.`}
                </p>
              }
            >
              <p key={s.backend} data-measured={best.c} data-backend={s.backend}>
                Measured on {s.device} ({s.backend}) with {whose(s)}, in the run of {s.date}
                {best.n ? <>, {count(best.n)} floats in blocks of {best.block}</> : null}: the fastest factor was{' '}
                <b>C = {best.c}</b>, at {fmt(best.g)} GB/s, and the slowest C = {worst.c}, at {fmt(worst.g)} GB/s.{' '}
                {mine.c === best.c
                  ? 'That is the factor you picked.'
                  : `Your pick${mine.c !== committed ? ` (nearest measured: C = ${mine.c})` : ''} ran at ${fmt(mine.g)} GB/s, ${Math.round((mine.g / best.g) * 100)}% of the fastest.`}
              </p>
            </Translated>
          );
        })
      ) : (
        <MissingLine />
      )}
      <Translated
        vi={
          <p className="muted">
            Hãy đọc độ chênh trước khi đọc người thắng. Khi hệ số nhanh nhất và chậm nhất chỉ cách nhau vài phần trăm, mọi
            hệ số đều đọc mảng ở tốc độ xấp xỉ nhau và việc hệ số nào về nhất chỉ là nhiễu. Số bước của mục 10.10 không tính
            thời gian bộ nhớ, nên một đường cong phẳng không mâu thuẫn với nó.
          </p>
        }
      >
        <p className="muted">
          Read the spread before the winner. When the fastest and slowest factors are a few percent apart, every factor
          reads the array at about the same rate and which one comes first is noise. Section 10.10's step count has no
          memory time in it, so a flat curve does not contradict it.
        </p>
      </Translated>
    </>
  );
}

/** The determinism count: distinct bit patterns among 20 runs of the atomic kernel. */
function DistinctReveal({ committed }: { committed: number }) {
  const { reports } = useLabs();
  const found = preferredBench(reports, l1).flatMap((s) => {
    const r = s.rows.find((x) => x.name === 'determinism');
    const d = num(r?.distinct);
    return r && d != null ? [{ s, d, runs: num(r.runs), n: num(r.n) }] : [];
  });
  return (
    <>
      {found.length ? (
        found.map(({ s, d, runs, n }) => (
          <Translated
            key={s.backend}
            vi={
              <p data-measured={d} data-backend={s.backend}>
                Đo trên {s.device} ({s.backend}) với {whose(s)}, trong lần chạy ngày {s.date}: <b>{d}</b> kết quả khác nhau
                {runs ? <> trong {runs} lần chạy</> : null}
                {n ? <> trên cùng {count(n)} float</> : null}.{' '}
                {d === committed ? 'Đúng như bạn đã dự đoán.' : `Bạn đã dự đoán ${committed}.`}
              </p>
            }
          >
            <p key={s.backend} data-measured={d} data-backend={s.backend}>
              Measured on {s.device} ({s.backend}) with {whose(s)}, in the run of {s.date}: <b>{d}</b> distinct result
              {d === 1 ? '' : 's'}
              {runs ? <> in {runs} runs</> : null}
              {n ? <> on the same {count(n)} floats</> : null}.{' '}
              {d === committed ? 'That is what you predicted.' : `You predicted ${committed}.`}
            </p>
          </Translated>
        ))
      ) : (
        <MissingLine />
      )}
      <Translated
        vi={
          <p className="muted">
            Một kết quả duy nhất chỉ có nghĩa là các khối tình cờ commit theo những thứ tự làm tròn giống nhau. Đó không phải
            một lời hứa: một atomic không sắp thứ tự gì cả (§10.9, p. 245). Cách sửa thực sự là một lời hứa là lượt thứ hai,
            cũng là điều <code>cub::DeviceReduce</code> làm theo mặc định (notes/today/ch10-facts.md A13).
          </p>
        }
      >
        <p className="muted">
          One distinct result would mean the blocks happened to commit in orders that round alike. That is not a promise:
          an atomic orders nothing (§10.9, p. 245). The fix that is a promise is the second pass, which is also what{' '}
          <code>cub::DeviceReduce</code> does by default (notes/today/ch10-facts.md A13).
        </p>
      </Translated>
    </>
  );
}

// ---------------------------------------------------------------- lab 10.2: what each backend showed of the shipped kernel
type FullResult = RunResult & { tsan?: unknown[]; log?: string };
interface HuntRun {
  backend: string;
  device: string;
  date: string;
  untouched: boolean;
  passed: number;
  total: number;
  /** Stopped by a sanitizer or a crash before the cases finished. */
  aborted: boolean;
  asan: boolean;
  races: number;
  failing: { name: string; detail: string }[];
}
const HUNT_BACKENDS = ['metal', 'cuda', 'hip', 'emu', 'race'];
const TSAN_EXIT = 66;
const BACKEND_NAME: Record<string, string> = {
  get emu() {
    return tr('the emulator', 'trình giả lập');
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
    return [{
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
      failing: tests.filter((t) => !t.pass).map((t) => ({ name: t.case, detail: /wrong in \d+ of \d+ launches/.exec(t.detail ?? '')?.[0] ?? 'wrong' })),
    }];
  });
}

function HuntReveal({ committed }: { committed: number }) {
  const { reports, loading, reload } = useLabs();
  const runs = huntRuns(reports);
  const gpu = runs.filter((r) => GPUS.includes(r.backend));
  const emu = runs.filter((r) => !GPUS.includes(r.backend));
  return (
    <>
      {gpu.map((r) => (
        <div key={r.backend} data-measured={r.passed} data-backend={r.backend}>
          <Translated
            vi={
              <p>
                Trên {r.device} ({r.backend}), lần chạy gần nhất <code>work/</code> của bạn vào {r.date}
                {r.untouched ? ', vẫn là kernel như khi phát hành' : ', sau khi bạn sửa'}: <b>{r.passed}</b> trên {r.total}{' '}
                trường hợp đúng trong mọi lần khởi chạy.{' '}
                {r.untouched
                  ? r.passed === committed
                    ? 'Đó là dự đoán của bạn.'
                    : `Bạn đã dự đoán ${committed}.`
                  : 'Bản làm của bạn đã thay đổi so với mã khởi đầu, nên lần chạy này không còn là thí nghiệm nữa.'}
              </p>
            }
          >
            <p>
              On {r.device} ({r.backend}), the last run of your <code>work/</code> on {r.date}
              {r.untouched ? ', still the kernel as shipped' : ', after your edits'}: <b>{r.passed}</b> of {r.total} cases
              right in every launch.{' '}
              {r.untouched
                ? r.passed === committed
                  ? 'That is your prediction.'
                  : `You predicted ${committed}.`
                : 'Your work has changed since it was the starter, so this run is not the experiment any more.'}
            </p>
          </Translated>
          {r.failing.length ? (
            <ul className="muted">
              {r.failing.map((f) => (
                <li key={f.name}>
                  {f.name}: {f.detail}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      ))}
      {emu.map((r) => (
        <Translated
          key={r.backend}
          vi={
            <p data-measured={r.passed} data-backend={r.backend}>
              Trên {BACKEND_NAME[r.backend]}, lần chạy gần nhất <code>work/</code> của bạn
              {r.untouched ? ' như khi phát hành' : ''}:{' '}
              {r.aborted && r.total === 0
                ? `dừng trước khi trường hợp đầu tiên kết thúc${r.asan ? ', với một báo cáo AddressSanitizer' : ''}.`
                : `${r.passed} trên ${r.total} trường hợp đúng${r.aborted ? ', rồi lần chạy dừng lại' : ''}.`}
              {r.races ? ` ThreadSanitizer báo cáo ${r.races} race.` : ''}
            </p>
          }
        >
          <p key={r.backend} data-measured={r.passed} data-backend={r.backend}>
            On {BACKEND_NAME[r.backend]}, the last run of your <code>work/</code>
            {r.untouched ? ' as shipped' : ''}:{' '}
            {r.aborted && r.total === 0
              ? `stopped before its first case finished${r.asan ? ', with an AddressSanitizer report' : ''}.`
              : `${r.passed} of ${r.total} cases right${r.aborted ? ', then the run stopped' : ''}.`}
            {r.races ? ` ThreadSanitizer reported ${r.races} race${r.races === 1 ? '' : 's'}.` : ''}
          </p>
        </Translated>
      ))}
      {!runs.length ? (
        loading ? (
          <p data-measured="loading">
            <Translated vi="Đang đọc báo cáo.">Reading reports.</Translated>
          </p>
        ) : (
          <p data-measured="none">
            <Translated
              vi={
                <>
                  Chưa có lần chạy nào bản làm của bạn cho lab này. Hãy chạy <code>python3 labs/pmpp.py run 10.2</code> trước
                  khi bạn thay đổi bất cứ điều gì, rồi tải lại.
                </>
              }
            >
              <>
                No run of your work for this lab yet. Run <code>python3 labs/pmpp.py run 10.2</code> before you change
                anything, then reload.
              </>
            </Translated>{' '}
            <button className="quiet" onClick={reload}>
              <Translated vi="Tải lại kết quả">Reload results</Translated>
            </button>
          </p>
        )
      ) : null}
      {runs.length && !gpu.length ? (
        <p data-measured="none" data-backend="gpu">
          <Translated vi="Không có lần chạy GPU nào bản làm của bạn cho lab này trên máy này, nên chưa có gì để so sánh với dự đoán của bạn.">
            No GPU run of your work for this lab on this machine, so there is nothing yet to compare your prediction with.
          </Translated>
        </p>
      ) : null}
      <Translated
        vi={
          <p className="muted">
            Một trường hợp đúng trong cả 20 lần khởi chạy chưa chứng tỏ kernel đúng ở đó. Đọc một lane đã rời khỏi lệnh
            shuffle và đọc bộ nhớ chia sẻ mà không ai ghi đều là hành vi không xác định, và không xác định có thể nghĩa là
            đúng giá trị, mọi lần, trên một thiết bị (notes/today/ch10-facts.md A3; infra-facts C2).
          </p>
        }
      >
        <p className="muted">
          A case right in all 20 launches has not shown that the kernel is correct there. Reading a lane that has left the
          shuffle and reading shared memory nobody wrote are undefined, and undefined can mean the right value, every
          time, on one device (notes/today/ch10-facts.md A3; infra-facts C2).
        </p>
      </Translated>
    </>
  );
}

// ---------------------------------------------------------------- the race flavour, which the status tables do not show
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
  return (
    <p data-race-status={loading ? 'loading' : s} data-race-lab={l.num}>
      {loading ? (
        tr('Reading reports.', 'Đang đọc báo cáo.')
      ) : !run || (s === 'none' && !work.length) ? (
        <Translated
          vi={
            <>
              Chưa có lần chạy nào của lab {l.num} trên backend race trên máy này:{' '}
              <code>python3 labs/pmpp.py run {l.num} --backend race</code>.
            </>
          }
        >
          <>
            No run of lab {l.num} on the race backend on this machine yet: <code>python3 labs/pmpp.py run {l.num} --backend race</code>.
          </>
        </Translated>
      ) : (
        <Translated
          vi={
            <>
              Trên backend race của máy này, lời giải tham chiếu{' '}
              {s === 'pass'
                ? `đạt cả ${n} bài kiểm thử, không race nào được báo cáo: emulator-checked dưới cả hai biến thể.`
                : s === 'fail'
                  ? 'không đạt: unverified ở đó.'
                  : 'chưa có lần chạy nào.'}{' '}
              Bản làm của bạn ở đó:{' '}
              {!work.length
                ? 'chưa có lần chạy nào.'
                : ws === 'none'
                  ? `chưa bắt đầu (${wTests.filter((t) => t.pass).length} trên ${wTests.length} bài kiểm thử đạt như khi phát hành${wRaces ? `, ${wRaces} race được báo cáo` : ''}).`
                  : ws === 'pass'
                    ? `đạt, ${wTests.length} trên ${wTests.length} bài kiểm thử, không race nào được báo cáo.`
                    : `chưa đạt: ${wTests.filter((t) => t.pass).length} trên ${wTests.length} bài kiểm thử${wRaces ? `, ${wRaces} race được báo cáo` : ''}.`}
            </>
          }
        >
          <>
            On the race backend of this machine, the reference solution{' '}
            {s === 'pass'
              ? `passed all ${n} tests with no race reported: emulator-checked under both flavours.`
              : s === 'fail'
                ? 'failed: unverified there.'
                : 'has no run yet.'}{' '}
            Your work there:{' '}
            {!work.length
              ? 'no run yet.'
              : ws === 'none'
                ? `not started (${wTests.filter((t) => t.pass).length} of ${wTests.length} tests pass as shipped${wRaces ? `, ${wRaces} race${wRaces === 1 ? '' : 's'} reported` : ''}).`
                : ws === 'pass'
                  ? `passes, ${wTests.length} of ${wTests.length} tests, no race reported.`
                  : `not passing: ${wTests.filter((t) => t.pass).length} of ${wTests.length} tests${wRaces ? `, ${wRaces} race${wRaces === 1 ? '' : 's'} reported` : ''}.`}
          </>
        </Translated>
      )}
    </p>
  );
}

// ---------------------------------------------------------------- milestone M4
function MilestoneStatus() {
  const { reports, loading } = useLabs();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const done = backends.filter((b) => labStatus(reports[b], l3.id, 'work') === 'pass');
  const name = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  return (
    <p data-milestone={loading ? 'loading' : done.join(',') || 'none'}>
      {loading
        ? tr('Reading reports.', 'Đang đọc báo cáo.')
        : !backends.length
          ? tr('No lab results are recorded on this machine yet.', 'Chưa có kết quả lab nào được ghi lại trên máy này.')
          : done.length
            ? tr(
                `Your own lab 10.3 passes on ${done.map(name).join(' and ')} on this machine, greedy decoding included: M4 is done there.`,
                `Lab 10.3 của bạn đạt trên ${done.map(name).join(' và ')} trên máy này, kể cả decode greedy: M4 đã hoàn thành ở đó.`,
              )
            : tr(
                'Your own lab 10.3 does not pass on any backend of this machine yet.',
                'Lab 10.3 của bạn chưa đạt trên backend nào của máy này.',
              )}
    </p>
  );
}

// ---------------------------------------------------------------- results-table columns for lab 10.1
const what = (r: BenchRow) =>
  r.name === 'rows'
    ? tr(`rows: ${KERNEL_NAME[String(r.kernel)] ?? String(r.kernel)}`, `theo hàng: ${KERNEL_NAME[String(r.kernel)] ?? String(r.kernel)}`)
    : r.name === 'any'
      ? tr(`any length (c), C = ${String(r.coarse)}`, `độ dài bất kỳ (c), C = ${String(r.coarse)}`)
      : r.name === 'determinism'
        ? tr('determinism (c), C = 1', 'tính tất định (c), C = 1')
        : r.name;
const floats = (r: BenchRow) => {
  const v = r.name === 'rows' ? (num(r.rows) ?? NaN) * 2 * (num(r.block) ?? NaN) : num(r.n);
  return v == null || !isFinite(v) ? '' : count(v);
};

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS}>
      <LabsIntro>
        <Translated
          vi={
            <>
              Ba lab, mỗi lab trong thư mục riêng dưới <code>labs/ch10/</code>. Lab thứ nhất viết ba kernel tốt nhất của
              chương và đo thời gian chúng so với hai kernel đầu tiên của sách. Lab thứ hai là cuộc truy tìm bốn lỗi trong một
              phép rút gọn biên dịch được. Lab thứ ba là mốc dự án M4 của dự án khóa học: một softmax và một argmax trên các
              hàng logits, và một bộ decode greedy chạy trên chúng.
            </>
          }
        >
          <>
            Three labs, each in its own directory under <code>labs/ch10/</code>. The first writes the chapter's three best
            kernels and times them against the book's first two. The second is a hunt for four defects in a reduction that
            compiles. The third is milestone M4 of the course project: a softmax and an argmax over rows of logits, and a
            greedy decoder that runs on them.
          </>
        </Translated>
      </LabsIntro>
      <RunningLabs
        example="10.1"
        alsoShownIn={
          <Translated
            vi={
              <>
                các mục <a href={href('ch10', '10.8')}>10.8</a>, <a href={href('ch10', '10.9')}>10.9</a> và{' '}
                <a href={href('ch10', '10.10')}>10.10</a>
              </>
            }
          >
            <>
              sections <a href={href('ch10', '10.8')}>10.8</a>, <a href={href('ch10', '10.9')}>10.9</a> and{' '}
              <a href={href('ch10', '10.10')}>10.10</a>
            </>
          </Translated>
        }
      />
      <BackendsAndEmulator />
      <Translated
        vi={
          <p>
            Trình giả lập có một biến thể thứ hai, <code>race</code>, và cả ba lab ở đây đều được dựng cho nó. Nó chạy các
            luồng được giả lập đồng thời dưới ThreadSanitizer và báo cáo hai luồng chạm vào cùng một vùng nhớ mà không có gì
            sắp thứ tự giữa chúng, kèm cả hai dòng mã nguồn. Đây là trình giả lập duy nhất có thể thấy một lần commit vào{' '}
            <code>*out</code> được thực hiện mà không có atomic; GPU chỉ cho thấy nó dưới dạng một tổng sai, khi các khối
            tình cờ va nhau. Yêu cầu nó bằng <code>--backend race</code>;{' '}
            <code>python3 labs/pmpp.py doctor</code> cho biết trình biên dịch của bạn có dựng được nó không. Các bảng trên
            trang này chỉ hiện biến thể thứ nhất; một dòng dưới mỗi lab cho kết quả race.
          </p>
        }
      >
        <p>
          The emulator has a second flavour, <code>race</code>, and all three labs here are built for it. It runs the
          emulated threads concurrently under ThreadSanitizer and reports two threads that touch the same memory with
          nothing ordering them, with both source lines. It is the only emulator that can see a commit to{' '}
          <code>*out</code> made without an atomic; a GPU shows one only as a wrong sum, when blocks happen to collide. Ask for it with <code>--backend race</code>;{' '}
          <code>python3 labs/pmpp.py doctor</code> says whether your compiler can build it. The tables on this page show
          the first flavour only; a line under each lab gives the race results.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Không trình giả lập nào chạy warp rộng hơn 32 lane. Vì vậy hai lỗi HIP mà chương này cảnh báo, một mask{' '}
            <code>0xffffffff</code> và một vòng lặp shuffle bắt đầu từ 16, đều đạt ở mọi nơi một máy Mac có thể kiểm thử. HIP
            từ chối lỗi thứ nhất lúc biên dịch; lỗi thứ hai cộng riêng từng nửa của một warp 64 lane. Hãy kiểm tra cả hai bằng
            cách đọc mã của bạn.
          </p>
        }
      >
        <p>
          Neither emulator runs a warp wider than 32 lanes. The two HIP mistakes this chapter warns about, a{' '}
          <code>0xffffffff</code> mask and a shuffle loop that starts at 16, therefore pass everywhere a Mac can test.
          HIP rejects the first at compile time; the second sums each half of a 64-lane warp separately. Check both by
          reading your code.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch10-facts.md B1, B2)">{/* i18n-ignore: date and citation only */}
        <Translated
          vi={
            <p>
              Mọi hàm <code>_sync</code> của HIP đều đòi một mask 64-bit. <code>warpSize</code> là 64 trên gfx9 và 32 trên các
              dòng AMD mới hơn, và không phải là hằng số lúc biên dịch ở phía host.
            </p>
          }
        >
          <p>
            Every HIP <code>_sync</code> function requires a 64-bit mask. <code>warpSize</code> is 64 on gfx9 and 32 on
            later AMD parts, and is not a compile-time constant on the host.
          </p>
        </Translated>
      </Today>
      <VerificationStatus />

      {/* ------------------------------------------------------------ 10.1 */}
      <h2>
        <Translated vi="Lab 10.1: một tổng, ba kernel">Lab 10.1: one sum, three kernels</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn viết ba phép rút gọn mà chương kết thúc bằng và đo thời gian chúng so với hai phép mà chương bắt đầu. (a) là
            kernel dùng bộ nhớ chia sẻ của Fig. 10.10, với tham số <code>output</code> mà trang sách bỏ sót. (b) là phép rút
            gọn warp hai giai đoạn của Figs. 10.13 và 10.16, được viết để biên dịch cho CUDA và HIP và đúng ở mọi kích thước
            khối. (c) là kernel nhiều khối của Figs. 10.18 và 10.20 được hoàn thiện: độ dài bất kỳ, hệ số gộp việc chọn lúc
            chạy, và một lần commit atomic cho mỗi khối. Trong (a) và (b), khối b rút gọn hàng b gồm{' '}
            <code>2 × blockDim.x</code> float vào <code>out[b]</code>, đúng dạng của một softmax trên các hàng ngắn. (b) cũng
            là phép rút gọn khối mà lab 10.3 dùng lại.
          </p>
        }
      >
        <p>
          You write the three reductions the chapter ends with and time them against the two it starts with. (a) is Fig.
          10.10's shared-memory kernel, with the <code>output</code> parameter the page leaves out. (b) is the two-stage
          warp reduction of Figs. 10.13 and 10.16, written to compile for CUDA and HIP and to be right at any block size. (c)
          is the multi-block kernel of Figs. 10.18 and 10.20 completed: any length, a coarsening factor chosen at run
          time, and one atomic commit per block. In (a) and (b) block b reduces row b of <code>2 × blockDim.x</code>{' '}
          floats into <code>out[b]</code>, the shape of a softmax over short rows. (b) is also the block reduction lab
          10.3 reuses.
        </p>
      </Translated>
      <Source
        section="10.6–10.10"
        pages="236–248"
        figs={tr('Figs. 10.10, 10.13, 10.16, 10.18, 10.20; exercise 5, p. 250', 'Hình 10.10, 10.13, 10.16, 10.18, 10.20; bài tập 5, p. 250')}
      />
      <p>
        <Translated vi="Những chỗ (b) và (c) khác với trang sách, và lý do:">Where (b) and (c) depart from the page, and why:</Translated>
      </p>
      <Translated
        vi={
          <ul>
            <li>
              Lane mask có kiểu riêng, 32 bit trên CUDA và 64 trên HIP, không bao giờ là literal <code>0xffffffff</code>. Vòng
              lặp shuffle bắt đầu từ <code>warpSize / 2</code>, và lane cùng warp được tính từ <code>warpSize</code>.
            </li>
            <li>
              Mảng theo warp được cấp kích thước cho warp hẹp nhất mà các backend ở đây có, <code>1024 / 32</code> phần tử,
              và giai đoạn 2 chỉ đọc nó ở những chỗ một warp đã ghi, đệm bằng 0. Fig. 10.16 đọc 32 phần tử bất kể khối, điều
              chỉ đúng với 1,024 luồng trong warp rộng 32.
            </li>
            <li>
              Mọi lần nạp trong (c) đều được kiểm tra so với <code>n</code> và đóng góp 0 khi vượt quá cuối mảng; không luồng
              nào return trước các lệnh shuffle. Launcher đặt <code>*out</code> về 0, điều mà các hình của sách cần nhưng
              không bao giờ cho thấy.
            </li>
            <li>
              Dưới HIP, lần commit trong (c) là <code>atomicAdd</code> legacy, cũng relaxed và có phạm vi toàn thiết bị, còn{' '}
              <code>&lt;cuda/atomic&gt;</code> chỉ được include cho CUDA và các trình giả lập: trên HIP, header đó đến từ
              libhipcxx, thứ mà bản dựng HIP của khóa học chỉ liên kết ở nơi ROCm cung cấp nó, nên lab này không dựa vào nó
              (infra-facts A1, B1, B2). Chưa được dựng với ROCm ở đây.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            The lane mask has its own type, 32 bits on CUDA and 64 on HIP, never the literal <code>0xffffffff</code>.
            The shuffle loop starts at <code>warpSize / 2</code>, and lane and warp come from <code>warpSize</code>.
          </li>
          <li>
            The per-warp array is sized for the narrowest warp any backend here has, <code>1024 / 32</code> entries, and
            stage 2 reads it only where a warp wrote, padding with 0. Fig. 10.16 reads 32 entries whatever the block,
            which is right only for 1,024 threads in 32-wide warps.
          </li>
          <li>
            Every load in (c) is checked against <code>n</code> and contributes 0 past the end; no thread returns before
            the shuffles. The launcher sets <code>*out</code> to 0, which the book's figures need and never show.
          </li>
          <li>
            Under HIP the commit in (c) is the legacy <code>atomicAdd</code>, also relaxed and device-wide, and{' '}
            <code>&lt;cuda/atomic&gt;</code> is included only for CUDA and the emulators: on HIP that header comes from
            libhipcxx, which the course's HIP build links only where ROCm provides it, so this lab does not rely on it
            (infra-facts A1, B1, B2). Not built with ROCm here.
          </li>
        </ul>
      </Translated>
      <p>
        <Translated vi="Bạn sửa ba tệp:">You edit three files:</Translated>
      </p>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <Translated
              vi={
                <>
                  <code>SharedMemorySumReductionKernel</code>, <code>warp_reduce</code>,{' '}
                  <code>TwoStageWarpLevelSumReductionKernel</code>, <code>CoarsenedSumReductionKernel</code> và các launcher
                  của chúng
                </>
              }
            >
              <>
                <code>SharedMemorySumReductionKernel</code>, <code>warp_reduce</code>,{' '}
                <code>TwoStageWarpLevelSumReductionKernel</code>, <code>CoarsenedSumReductionKernel</code> and their
                launchers
              </>
            </Translated>,
          ],
          ['kernel.metal', <><code>shared_memory_sum</code>, <code>warp_reduce</code>, <code>two_stage_warp_sum</code>, <code>coarsened_sum</code></>],
          ['launch.mm', <><code>encodeSharedMemorySum</code>, <code>encodeTwoStageWarpSum</code>, <code>encodeCoarsenedSum</code></>],
        ]}
      />
      <Translated
        vi={
          <p>
            Có sẵn và không cần sửa: các kernel đơn giản và hội tụ của sách (Figs. 10.5 và 10.8), một hàng cho mỗi khối, trong{' '}
            <code>labs/ch10/shared/inplace_sum.cu</code>, <code>inplace_sum.metal</code> và{' '}
            <code>inplace_launch.mm</code>. Các mục <a href={href('ch10', '10.3')}>10.3</a> và{' '}
            <a href={href('ch10', '10.4')}>10.4</a> chạy từng bước qua chúng. Phiên bản Metal của chúng giữ các tổng bộ phận
            trong bộ nhớ device, nên rào chắn của chúng chỉ định bộ nhớ device, không phải bộ nhớ threadgroup.
          </p>
        }
      >
        <p>
          Provided and not edited: the book's simple and convergent kernels (Figs. 10.5 and 10.8), one row per block, in{' '}
          <code>labs/ch10/shared/inplace_sum.cu</code>, <code>inplace_sum.metal</code> and{' '}
          <code>inplace_launch.mm</code>. Sections <a href={href('ch10', '10.3')}>10.3</a> and{' '}
          <a href={href('ch10', '10.4')}>10.4</a> step through them. Their Metal versions keep partial sums in device
          memory, so their barrier names device memory, not threadgroup memory.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch10-facts C1, C2; ch04-facts C2)">{/* i18n-ignore: date and citation only */}
        <Translated
          vi={
            <p>
              <code>simd_shuffle_down</code> của Metal không nhận mask; độ rộng đến từ{' '}
              <code>[[threads_per_simdgroup]]</code>. Lần commit float là <code>atomic_fetch_add_explicit</code> trên một{' '}
              <code>device atomic_float*</code>.
            </p>
          }
        >
          <p>
            Metal's <code>simd_shuffle_down</code> takes no mask; the width comes from{' '}
            <code>[[threads_per_simdgroup]]</code>. The float commit is <code>atomic_fetch_add_explicit</code> on a{' '}
            <code>device atomic_float*</code>.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 10.1 --backend emu</code> trước, rồi <code>--backend race</code>, rồi GPU của
            bạn. (a), (b) và cả hai kernel có sẵn được kiểm thử ở 1, 3 và 37 hàng và các khối 64, 128, 256 và 1,024 luồng (ít
            hơn trên các trình giả lập; trên Metal, các kích thước mà pipeline cho phép). (c) chạy các độ dài từ 1 đến 4,097,
            cùng 100,003 và 2<sup>22</sup> trên thiết bị, ở các hệ số gộp việc 1, 2, 3 và 8 và các khối 64 và 256. Mọi đầu ra
            bắt đầu là NaN, và mọi trường hợp của (c) được khởi chạy hai lần trên cùng đầu ra, nên một launcher không đặt lại{' '}
            <code>*out</code> sẽ không đạt. Dữ liệu có tổng chính xác phải khớp chính xác; dữ liệu ngẫu nhiên phải nằm trong
            10<sup>−6</sup> của tổng các độ lớn, cộng 10<sup>−6</sup>, vì các atomic float trên bộ nhớ toàn cục đẩy các số
            denormal về không (ch10-facts A6).
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 10.1 --backend emu</code> first, then <code>--backend race</code>, then your
          GPU. (a), (b) and both provided kernels are tested at 1, 3 and 37 rows and blocks of 64, 128, 256 and 1,024
          threads (fewer on the emulators; on Metal, the sizes the pipeline allows). (c) runs lengths from 1 to 4,097, and
          100,003 and 2<sup>22</sup> on a device, at coarsening factors 1, 2, 3 and 8 and blocks of 64 and 256. Every
          output starts as NaN, and every case of (c) is launched twice on the same output, so a launcher that does not
          reset <code>*out</code> fails. Data with exact sums must match exactly; random data must be within 10
          <sup>−6</sup> of the sum of magnitudes, plus 10<sup>−6</sup>, because float atomics on global memory flush
          denormals to zero (ch10-facts A6).
        </p>
      </Translated>
      <RaceLine lab={l1} />
      <Translated
        vi={
          <p>
            Với <code>--bench</code>, chương trình điều khiển (driver) đo ba thứ trên thiết bị: bốn kernel theo hàng trên
            2<sup>24</sup> float trong các hàng 512 với khối 256; (c) trên một mảng 2<sup>26</sup> float trở lên ở các hệ số
            gộp việc 1 đến 128; và 20 lần chạy (c) trên cùng 2<sup>24</sup> float, đếm số mẫu bit khác nhau của kết quả. GB/s
            là số byte mà một phép tổng phải đọc chia cho thời gian, không phải lượng hệ thống bộ nhớ đã di chuyển. Hãy chốt
            ba dự đoán trước.
          </p>
        }
      >
        <p>
          With <code>--bench</code> the driver times three things on a device: the four row kernels on 2<sup>24</sup>{' '}
          floats in rows of 512 with blocks of 256; (c) on one array of 2<sup>26</sup> floats or more at coarsening factors
          1 to 128; and 20 runs of (c) on the same 2<sup>24</sup> floats, counting the distinct bit patterns of the result.
          GB/s is the bytes a sum must read divided by the time, not what the memory system moved. Commit three
          predictions first.
        </p>
      </Translated>
      <LabPrediction
        id="ch10/labs/p-lab1-warp"
        prompt={
          <Translated
            vi={
              <>
                Trước khi bạn chạy benchmark. Trên các hàng 512 float, một khối 256 luồng cho mỗi hàng, kernel hai giai đoạn
                của bạn sẽ chạy nhanh gấp bao nhiêu lần kernel đơn giản (Fig. 10.5)?
              </>
            }
          >
            <>
              Before you run the benchmark. On rows of 512 floats, one block of 256 threads per row, how many times as
              fast as the simple kernel (Fig. 10.5) will your two-stage kernel run?
            </>
          </Translated>
        }
        unit={tr('× the simple kernel', '× kernel đơn giản')}
        measure={warpOverSimple}
        missing={BENCH_MISSING}
        reading={
          <Translated
            vi={
              <>
                Phân tích của sách cho 256 phần tử cho hiệu suất tài nguyên 30% so với 66% ở hai kernel đầu, và 141, 36 và 9
                yêu cầu bộ nhớ ở ba kernel đầu (pp. 231, 234, 235, 237). Đó là các con số đếm. Hãy nhìn cả bốn hàng của bảng
                trước khi bạn quyết định thiết bị của mình thưởng cho bước nào của chương.
              </>
            }
          >
            <>
              The book's analysis for 256 elements gives 30% against 66% resource efficiency for the first two kernels and
              141, 36 and 9 memory requests for the first three (pp. 231, 234, 235, 237). Those are counts. Look at all
              four rows of the table before you decide which of the chapter's steps your device rewarded.
            </>
          </Translated>
        }
      />
      <Predict
        id="ch10/10.10/p-coarse"
        prompt={
          <Translated
            vi={
              <>
                Cùng dự đoán như ở mục 10.10: chốt nó ở trang nào cũng được. (c) được đo ở các hệ số gộp việc 1, 2, 4, và cứ
                thế tới 128, với các khối 256 luồng. Hệ số nào sẽ nhanh nhất trên thiết bị của bạn?
              </>
            }
          >
            <>
              The same prediction as in section 10.10: commit it on either page. (c) is timed at coarsening factors 1, 2,
              4, and so on up to 128, with blocks of 256 threads. Which factor will be fastest on your device?
            </>
          </Translated>
        }
        unit={tr('(coarsening factor)', '(hệ số gộp việc)')}
      >
        {(v) => <CoarseReveal committed={v} />}
      </Predict>
      <Predict
        id="ch10/10.9/p-distinct"
        prompt={
          <Translated
            vi={
              <>
                Cùng dự đoán như ở mục 10.9. 20 lần chạy (c) ở C = 1 trên cùng 2<sup>24</sup> float: bao nhiêu kết quả khác
                nhau, tính theo mẫu bit?
              </>
            }
          >
            <>
              The same prediction as in section 10.9. 20 runs of (c) at C = 1 on the same 2<sup>24</sup> floats: how
              many distinct results, as bit patterns?
            </>
          </Translated>
        }
        unit={tr('distinct results', 'kết quả khác nhau')}
      >
        {(v) => <DistinctReveal committed={v} />}
      </Predict>
      <LabResults
        lab={l1}
        filter={(r) => r.name === 'rows' || r.name === 'any' || r.name === 'determinism'}
        cols={[
          { head: tr('Measurement', 'Phép đo'), cell: what },
          { head: tr('Floats', 'Số float'), num: true, cell: floats },
          rawCol(tr('Block', 'Khối'), 'block'),
          {
            head: tr('Blocks', 'Số khối'),
            num: true,
            cell: (r) => {
              const v = num(r.name === 'rows' ? r.rows : r.name === 'any' ? r.blocks : undefined);
              return v == null ? '' : count(v);
            },
          },
          msCol,
          numCol('GB/s', 'gb_per_s'),
          {
            head: tr('Distinct of runs', 'Khác nhau trên số lần chạy'),
            num: true,
            cell: (r) =>
              r.name === 'determinism'
                ? tr(`${String(r.distinct)} of ${String(r.runs)}`, `${String(r.distinct)} trên ${String(r.runs)}`)
                : '',
          },
        ]}
      />
      <ReferenceSolution lab={l1} tiers={SUM_TIERS} />

      {/* ------------------------------------------------------------ 10.2 */}
      <h2>
        <Translated vi="Lab 10.2: truy lỗi phép rút gọn">Lab 10.2: reduction hunt</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn được giao <code>reduceAnyKernel</code>, một phép tổng trên độ dài bất kỳ: các khối 128 luồng, hai phần tử mỗi
            luồng, một phép rút gọn warp hai giai đoạn, và một lần commit cho mỗi khối vào <code>*out</code>, được launcher
            đặt về 0. Nó biên dịch được, và có bốn lỗi độc lập, cùng bốn lỗi đó trong bản CUDA và bản Metal. Hãy tìm và sửa cả
            bốn. Chúng là những lỗi mà các mục 10.7 đến 10.9 dễ dẫn tới: một lỗi về các lane rời đi sớm, một lỗi về một mảng bị
            đọc quá phần đã được ghi, một lỗi về thứ tự giữa các warp, một lỗi về các khối commit cùng lúc.
          </p>
        }
      >
        <p>
          You are handed <code>reduceAnyKernel</code>, a sum over any length: blocks of 128 threads, two elements per
          thread, a two-stage warp reduction, and one commit per block into <code>*out</code>, which the launcher sets to
          0. It compiles, and it has four independent defects, the same four in the CUDA and the Metal version. Find and
          fix all four. They are the mistakes sections 10.7 to 10.9 invite: one about lanes that leave early, one about
          an array read past what was written, one about ordering between warps, one about blocks that commit at the same
          time.
        </p>
      </Translated>
      <Source
        section="10.7–10.9"
        pages="237–246"
        figs={tr(
          'Figs. 10.13, 10.16, 10.18; exercise 5, p. 250; the defects are ours, except one that Fig. 10.16 has as printed',
          'Hình 10.13, 10.16, 10.18; bài tập 5, p. 250; các lỗi là của khóa học này, trừ một lỗi mà Fig. 10.16 có sẵn như khi in',
        )}
      />
      <LabFiles
        lab={l2}
        files={[
          [
            'kernel.cu',
            <Translated
              vi={
                <>
                  <code>reduceAnyKernel</code>, với bốn lỗi; <code>launchReduceAny</code> bên dưới nó là đúng
                </>
              }
            >
              <>
                <code>reduceAnyKernel</code>, with the four defects; <code>launchReduceAny</code> below it is correct
              </>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated
              vi={
                <>
                  <code>reduce_any</code>, với cùng bốn lỗi đó
                </>
              }
            >
              <>
                <code>reduce_any</code>, with the same four
              </>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated vi={<>đúng như khi phát hành; không có gì phải sửa</>}>
              <>correct as shipped; nothing to change</>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Không phải lỗi: mask có kiểu riêng và vòng lặp shuffle bắt đầu từ <code>warpSize / 2</code>, như trong lab 10.1.
            Chương trình điều khiển chạy các độ dài là bội của 256 phần tử của một khối và các độ dài không phải, mỗi độ dài
            hai lần trên cùng đầu ra, với các số nguyên từ 1 đến 8 để mọi thứ tự cộng đều cho cùng một float. Một tổng sai là
            một lỗi, không bao giờ là do làm tròn. Trước khi chạy bất cứ thứ gì, hãy đọc kernel và chốt một con số.
          </p>
        }
      >
        <p>
          Not defects: the mask has its own type and the shuffle loop starts at <code>warpSize / 2</code>, as in lab
          10.1. The driver runs lengths that are multiples of a block's 256 elements and lengths that are not, each twice on
          the same output, with whole numbers from 1 to 8 so that every order of addition gives the same float. A wrong
          sum is a defect, never rounding. Before you run anything, read the kernel and commit to a number.
        </p>
      </Translated>
      <Predict
        id="ch10/labs/p-lab2-gpu"
        prompt={
          <Translated
            vi={
              <>
                Trước khi bạn thay đổi bất cứ điều gì. Trên GPU của bạn, chương trình điều khiển chạy chín độ dài (256, 512,
                1,024, 65,536, 1, 33, 1,000, 4,097 và 100,003), mỗi độ dài 20 lần. Với cả bốn lỗi còn nguyên, bao nhiêu trong
                chín trường hợp sẽ ra đúng trong cả 20 lần khởi chạy?
              </>
            }
          >
            <>
              Before you change anything. On your GPU the driver runs nine lengths (256, 512, 1,024, 65,536, 1, 33, 1,000,
              4,097 and 100,003), each 20 times. With all four defects in, how many of the nine will come out right in all
              20 launches?
            </>
          </Translated>
        }
        unit={tr('of 9 cases', 'trên 9 trường hợp')}
      >
        {(v) => <HuntReveal committed={v} />}
      </Predict>
      <p>
        <Translated vi="Rồi truy tìm, theo thứ tự này:">Then hunt, in this order:</Translated>
      </p>
      <Translated
        vi={
          <ol>
            <li>
              Trình giả lập trước: <code>python3 labs/pmpp.py run 10.2 --backend emu</code>. Một lệnh shuffle đọc một lane đã
              rời đi sẽ trả về các byte <code>0xCD</code>, một số float âm lớn, kèm một ghi chú nêu dòng mã. Một lần đọc vượt
              quá một mảng sẽ dừng lần chạy với một báo cáo AddressSanitizer: hãy đọc <code>READ</code>, dòng{' '}
              <code>kernel.cu</code> của khung đầu tiên, và nó vượt quá vùng nhớ bao nhiêu byte.
            </li>
            <li>
              Sửa một chỗ và chạy lại. Những độ dài nào không đạt, và dưới thứ tự luồng nào trong ba thứ tự, là bằng chứng của
              bạn cho lỗi tiếp theo.
            </li>
            <li>
              <code>python3 labs/pmpp.py run 10.2 --backend race</code>. Một lỗi vô hình với <code>emu</code>, vốn không bao
              giờ ngắt một luồng giữa chừng một lần cập nhật, và chỉ lộ ra ở đây. ThreadSanitizer báo cáo mỗi cặp dòng một lần
              cho mỗi lần chạy, nên một trường hợp về sau có thể đạt dù lỗi vẫn còn.
            </li>
            <li>
              Rồi tới GPU của bạn: <code>python3 labs/pmpp.py run 10.2</code>. Ghi lại những lỗi nào nó cho bạn thấy và những
              lỗi nào nó che đi.
            </li>
          </ol>
        }
      >
        <ol>
          <li>
            Emulator first: <code>python3 labs/pmpp.py run 10.2 --backend emu</code>. A shuffle that reads a lane that
            has left returns <code>0xCD</code> bytes, a large negative float, with a note naming the line. A read past an
            array stops the run with an AddressSanitizer report: read <code>READ</code>, the first frame's{' '}
            <code>kernel.cu</code> line, and how many bytes past the region it was.
          </li>
          <li>
            Fix one thing and run again. Which lengths fail, and under which of the three thread orders, is your evidence
            for the next defect.
          </li>
          <li>
            <code>python3 labs/pmpp.py run 10.2 --backend race</code>. One defect is invisible to <code>emu</code>,
            which never interrupts a thread in the middle of an update, and shows only here. ThreadSanitizer reports each
            pair of lines once per run, so a later case can pass with the defect still in.
          </li>
          <li>
            Then your GPU: <code>python3 labs/pmpp.py run 10.2</code>. Note which defects it showed you and which it hid.
          </li>
        </ol>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch10-facts.md A1, A3; infra-facts C1, C2)">{/* i18n-ignore: date and citation only */}
        <Translated
          vi={
            <p>
              Đọc một lane không hoạt động là không xác định trên CUDA và trên Metal. Hai luồng ghi vào một vị trí mà không
              đồng bộ hóa là không xác định trên CUDA. Các hàm atomic của Metal chỉ nhận kiểu atomic, nên cách sửa ở đó thay
              đổi kiểu của con trỏ.
            </p>
          }
        >
          <p>
            Reading an inactive lane is undefined on CUDA and on Metal. Two threads writing one location without
            synchronisation is undefined on CUDA. Metal's atomic functions take only atomic types, so the fix there changes
            the pointer's type.
          </p>
        </Translated>
      </Today>
      <RaceLine lab={l2} />
      <LabResults
        lab={l2}
        noBench={tr(
          'This lab has no benchmark. The result is the line above: every test passing on the emulators and on a device. As shipped, the kernel is meant to fail.',
          'Lab này không có benchmark. Kết quả là dòng ở trên: mọi bài kiểm thử đạt trên các trình giả lập và trên một thiết bị. Như khi phát hành, kernel được cố ý để không đạt.',
        )}
      />
      <ReferenceSolution lab={l2} tiers={HUNT_TIERS} />

      {/* ------------------------------------------------------------ 10.3 */}
      <h2>
        <Translated vi="Lab 10.3: softmax và argmax trên logits (M4)">Lab 10.3: softmax and argmax over logits (M4)</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mỗi bước decode kết thúc bằng một hàng logits, mỗi logit cho một mục từ vựng. Bạn viết ba kernel theo hàng trên các
            logits lưu dưới dạng <code>[rows × cols]</code> float32, thứ tự hàng (row-major), batch trước: giá trị lớn nhất của
            mỗi hàng (toán tử của bài tập 4), <code>y = softmax(scale × x)</code> được tính ổn định (một phép rút gọn max, rồi
            một tổng của <code>exp(scale × x − max)</code>, rồi phép chia), và chỉ số của logit lớn nhất với hòa thì lấy chỉ số
            nhỏ nhất, chính là bước decode greedy. Một khối 256 luồng cho mỗi hàng; mỗi luồng đi qua hàng của mình bằng một vòng
            lặp có bước nhảy bằng khối (gộp việc và được hợp nhất, các mục 10.10 và 10.5), và khối kết thúc bằng phép rút gọn
            hai giai đoạn của lab 10.1.
          </p>
        }
      >
        <p>
          Every decode step ends in a row of logits, one per vocabulary entry. You write three row-wise kernels over
          logits stored as <code>[rows × cols]</code> float32, row-major, batch first: the largest value of each row
          (exercise 4's operator), <code>y = softmax(scale × x)</code> computed stably (a max reduction, then a sum of{' '}
          <code>exp(scale × x − max)</code>, then the division), and the index of the largest logit with ties to the lowest
          index, which is the greedy-decoding step. One block of 256 threads per row; each thread walks its row with a
          loop strided by the block (coarsened and coalesced, sections 10.10 and 10.5), and the block finishes with lab
          10.1's two-stage reduction.
        </p>
      </Translated>
      <NoteGroup>
        <Source
          section="10.8–10.10"
          pages="241–248, 250"
          figs={tr(
            'Fig. 10.16, p. 243; Fig. 10.20, p. 247; exercise 4, p. 250',
            'Hình 10.16, p. 243; Hình 10.20, p. 247; bài tập 4, p. 250',
          )}
        />
        <Beyond>
          <Translated
            vi={
              <p>
                Softmax, argmax và bộ decode là của khóa học này, dựng từ các thành phần của chương. Sách không nhắc tới mô
                hình ngôn ngữ trong chương này.
              </p>
            }
          >
            <p>
              The softmax, the argmax and the decoder are ours, built from the chapter's parts. The book does not mention
              language models in this chapter.
            </p>
          </Translated>
        </Beyond>
      </NoteGroup>
      <LabFiles
        lab={l3}
        files={[
          [
            'kernel.cu',
            <Translated
              vi={
                <>
                  <code>rowMaxKernel</code>, <code>softmaxRowsKernel</code>, <code>argmaxRowsKernel</code> và{' '}
                  <code>launchRowMax</code>, <code>launchSoftmaxRows</code>, <code>launchArgmaxRows</code>
                </>
              }
            >
              <>
                <code>rowMaxKernel</code>, <code>softmaxRowsKernel</code>, <code>argmaxRowsKernel</code> and <code>launchRowMax</code>, <code>launchSoftmaxRows</code>, <code>launchArgmaxRows</code>
              </>
            </Translated>,
          ],
          ['kernel.metal', <><code>row_max</code>, <code>softmax_rows</code>, <code>argmax_rows</code></>],
          ['launch.mm', <><code>encodeRowMax</code>, <code>encodeSoftmaxRows</code>, <code>encodeArgmaxRows</code></>],
        ]}
      />
      <Translated
        vi={
          <p>
            Có sẵn và không cần sửa: một baseline giao cho mỗi hàng một luồng, trong{' '}
            <code>labs/ch10/shared/rowwise_naive.cu</code>, <code>rowwise_naive.metal</code> và{' '}
            <code>rowwise_naive_launch.mm</code>. Nó đúng, và là thứ mà benchmark dùng để so sánh các kernel của bạn.
          </p>
        }
      >
        <p>
          Provided and not edited: a baseline that gives each row one thread, in <code>labs/ch10/shared/rowwise_naive.cu</code>,{' '}
          <code>rowwise_naive.metal</code> and <code>rowwise_naive_launch.mm</code>. It is correct and it is what the
          benchmark measures your kernels against.
        </p>
      </Translated>
      <p>
        <Translated vi="Hợp đồng, được README của lab giải thích đầy đủ:">The contract, which the lab's README explains in full:</Translated>
      </p>
      <Translated
        vi={
          <ul>
            <li>
              <code>rows</code> và <code>cols</code> bất kỳ từ 1 trở lên; mọi logit đều hữu hạn. NaN và vô cực nằm ngoài hợp
              đồng: dưới chế độ fast math của Metal, hành vi của chúng là không xác định, và các thư viện không thống nhất về
              việc argmax nên làm gì với một NaN.
            </li>
            <li>
              Phần tử đơn vị của max là <code>-FLT_MAX</code>, không phải −∞ của sách (p. 222), vì cùng lý do đó. Argmax kết
              hợp các cặp (giá trị, chỉ số): giá trị lớn hơn thắng, và giữa hai giá trị bằng nhau thì chỉ số nhỏ hơn thắng, điều
              này làm cho phép kết hợp có tính giao hoán và kết hợp trên các giá trị hữu hạn như các mục{' '}
              <a href={href('ch10', '10.2')}>10.2</a> và <a href={href('ch10', '10.4')}>10.4</a> đòi hỏi.
            </li>
            <li>
              Các phép rút gọn max và tổng dùng chung bộ nhớ nháp, nên một rào chắn ngăn cách chúng: đó là race mà PyTorch và
              llama.cpp đề phòng, và là race mà backend <code>race</code> báo cáo nếu bạn bỏ nó đi.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            Any <code>rows</code> and <code>cols</code> from 1 up; every logit finite. NaN and infinity are outside it:
            under Metal's fast math their behaviour is undefined, and libraries disagree on what argmax should do with a
            NaN.
          </li>
          <li>
            The identity of max is <code>-FLT_MAX</code>, not the book's −∞ (p. 222), for the same reason. The argmax
            combines (value, index) pairs: the larger value wins, and of two equal values the lower index, which makes the
            combine commutative and associative on finite values as sections <a href={href('ch10', '10.2')}>10.2</a> and{' '}
            <a href={href('ch10', '10.4')}>10.4</a> require.
          </li>
          <li>
            The max and the sum reductions share scratch memory, so a barrier separates them: the race PyTorch and
            llama.cpp guard against, and the one the <code>race</code> backend reports if you leave it out.
          </li>
        </ul>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 (notes/today/ch10-facts.md A10, A11, C3, C4, D1 to D4)',
          '2026-11-20 (notes/today/ch10-facts.md A10, A11, C3, C4, D1 đến D4)',
        )}
      >
        <Translated
          vi={
            <p>
              Dùng <code>expf</code>, không dùng <code>__expf</code>, vốn có sai số tăng theo |x|. PyTorch ghi rõ quy tắc chỉ
              số nhỏ nhất cho <code>torch.argmax</code>; các bước greedy trên CUDA, Metal và host của llama.cpp phá hòa chính
              xác theo ba cách khác nhau.
            </p>
          }
        >
          <p>
            Use <code>expf</code>, not <code>__expf</code>, whose error grows with |x|. PyTorch documents the
            lowest-index rule for <code>torch.argmax</code>; llama.cpp's CUDA, Metal and host greedy steps break exact ties
            three different ways.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 10.3</code>, bắt đầu với <code>--backend emu</code>. Các hàng 1, 4 và 7 nhân
            với số cột từ 1 đến 128,256, ba giá trị cuối có kích thước bằng bộ từ vựng (ít hơn trên các trình giả lập). Các hàng
            lệch ±90 bắt được một softmax bỏ qua bước trừ max; các hàng toàn số âm bắt được phần tử đơn vị max bằng 0; các điểm
            hòa chính xác được cài sẵn ở các lane kề nhau, ở hai warp, trong vòng lặp của một luồng và ở cuối hàng bắt được từng
            quy tắc phá hòa sai. Max và argmax phải khớp chính xác; softmax phải nằm trong sai số tương đối 10<sup>−4</sup> so
            với một tham chiếu độ chính xác kép, một giới hạn mà README suy ra từ sai số <code>exp</code> của từng backend, và
            mọi hàng phải có tổng bằng 1. Rồi tới bài kiểm thử mốc dự án: một bộ decode greedy đồ chơi có "mô hình" là một bảng
            tra logits với một điểm hòa cài sẵn trong mỗi hàng. Mỗi bước sao chép các hàng của token hiện tại vào một buffer
            logits <code>[4 × V]</code>, gọi <code>launchArgmaxRows</code> của bạn và đưa các chỉ số trở lại. Sau 16 bước, bốn
            chuỗi phải khớp với chuỗi của CPU.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 10.3</code>, starting with <code>--backend emu</code>. Rows 1, 4 and 7 by
          columns from 1 to 128,256, the last three of them vocabulary-sized (fewer on the emulators). Rows offset by ±90
          catch a softmax that skips the max subtraction; all-negative rows catch a max identity of 0; planted exact ties
          in neighbouring lanes, in two warps, within one thread's loop and at the end of the row catch each wrong tie
          rule. Max and argmax must match exactly; softmax must be within 10<sup>−4</sup> relative of a double-precision
          reference, a bound the README derives from each backend's <code>exp</code> error, and every row must sum to 1.
          Then the milestone test: a toy greedy decoder whose "model" is a lookup table of logits with a planted tie in
          every row. Each step copies the current tokens' rows into a <code>[4 × V]</code> logits buffer, calls your{' '}
          <code>launchArgmaxRows</code> and feeds the indices back. After 16 steps the four sequences must match the CPU's.
        </p>
      </Translated>
      <RaceLine lab={l3} />
      <LabPrediction
        id="ch10/labs/p-lab3-naive"
        prompt={
          <Translated
            vi={
              <>
                Trước khi bạn chạy benchmark. Ở batch 1 và 128,256 cột, softmax một khối cho mỗi hàng của bạn sẽ nhanh gấp bao
                nhiêu lần một luồng cho mỗi hàng? Và khoảng cách sẽ tăng hay giảm ở batch 64?
              </>
            }
          >
            <>
              Before you run the benchmark. At batch 1 and 128,256 columns, how many times as fast as one thread per row
              will your block-per-row softmax be? And will the gap grow or shrink at batch 64?
            </>
          </Translated>
        }
        unit={tr('× one thread per row', '× một luồng cho mỗi hàng')}
        measure={naiveOverBlock}
        missing={
          <Translated
            vi={
              <>
                Khi lab 10.3 đã đạt, hãy chạy <code>python3 labs/pmpp.py run 10.3 --bench</code> trên một máy có GPU, rồi tải
                lại.
              </>
            }
          >
            <>
              Once lab 10.3 passes, run <code>python3 labs/pmpp.py run 10.3 --bench</code> on a machine with a GPU, then
              reload.
            </>
          </Translated>
        }
        reading={
          <Translated
            vi={
              <>
                Ở batch 1, baseline chạy một luồng trên toàn bộ thiết bị, còn kernel theo khối chạy một khối: không cái nào lấp
                đầy một GPU, và <a href={href('ch10', '10.10')}>mục 10.10</a> cho thấy lý do. Ở batch 64, cả hai có lượng công
                việc song song gấp 64 lần. Tỉ lệ ở batch 64 nằm trong phần chi tiết ở trên; nó trả lời câu hỏi thứ hai.
              </>
            }
          >
            <>
              At batch 1 the baseline runs one thread on the whole device, and the block kernel one block: neither fills a
              GPU, and <a href={href('ch10', '10.10')}>section 10.10</a> shows why. At batch 64 both have 64 times the
              parallel work. The ratio at batch 64 is in the detail above; it answers the second question.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l3}
        filter={(r) => r.name === 'softmax' || r.name === 'argmax'}
        cols={[
          { head: tr('Operation', 'Phép toán'), cell: (r) => r.name },
          {
            head: tr('Version', 'Phiên bản'),
            cell: (r) =>
              r.kernel === 'naive'
                ? tr('one thread per row (provided)', 'một luồng cho mỗi hàng (có sẵn)')
                : tr('one block per row', 'một khối cho mỗi hàng'),
          },
          rawCol(tr('Rows', 'Số hàng'), 'rows'),
          { head: tr('Columns', 'Số cột'), num: true, cell: (r) => (num(r.cols) != null ? count(num(r.cols)!) : '') },
          msCol,
          numCol('GB/s', 'gb_per_s'),
        ]}
      />
      <Translated
        vi={
          <p>
            GB/s tính lưu lượng ít nhất mà một kernel có thể tạo ra: 8 byte mỗi phần tử cho softmax (một lần đọc, một lần
            ghi), 4 cho argmax. Softmax của bạn đọc mỗi hàng ba lần, nên nó di chuyển nhiều hơn mức được ghi nhận.
          </p>
        }
      >
        <p>
          GB/s counts the least traffic a kernel could make: 8 bytes per element for softmax (one read, one write), 4 for
          argmax. Your softmax reads each row three times, so it moves more than it is credited with.
        </p>
      </Translated>
      <h3>
        <Translated vi="Mốc dự án M4">Project milestone M4</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Lab 10.3 là mốc dự án M4 của dự án xuyên suốt: max, softmax và argmax theo hàng trên logits{' '}
            <code>[batch × vocab]</code>, và decode greedy từ đầu đến cuối trên một mô hình đồ chơi. Đó là decode greedy cho một
            bảng tra, không phải cho một mô hình ngôn ngữ. Mốc dự án M10 (<a href={href('x3', 'labs')}>lab X3.2</a>) chạy một mô
            hình nhỏ kiểu Llama, được một script trong repository sinh ra từ một seed cố định, và bước greedy của nó gọi lời
            giải tham chiếu của <code>launchArgmaxRows</code> trong lab này. <code>scale</code> của softmax trở thành nhiệt độ
            lấy mẫu ở module X1 và 1/√d trong attention của chương 20.
          </p>
        }
      >
        <p>
          Lab 10.3 is milestone M4 of the through-line project: row-wise max, softmax and argmax over{' '}
          <code>[batch × vocab]</code> logits, and greedy decoding end to end on a toy model. It is greedy decoding for a
          lookup table, not for a language model. Milestone M10 (<a href={href('x3', 'labs')}>lab X3.2</a>) runs a small
          Llama-style model, written by a script in the repository from a fixed seed, and its greedy step calls the reference solution of this lab's{' '}
          <code>launchArgmaxRows</code>. The softmax's <code>scale</code> becomes the temperature of sampling in module X1 and 1/√d in the
          attention of chapter 20.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Dự án xuyên suốt, <code>nanoinfer</code>, là của khóa học này: một đường suy luận GPU nhỏ cho một transformer
              chỉ có decoder, lớn dần qua từng mốc dự án.
            </p>
          }
        >
          <p>
            The through-line project, <code>nanoinfer</code>, is ours: one small GPU inference path for a decoder-only
            transformer, grown a milestone at a time.
          </p>
        </Translated>
      </Beyond>
      <MilestoneStatus />
      <ReferenceSolution lab={l3} tiers={ROW_TIERS} />
    </LabsPage>
  );
}
