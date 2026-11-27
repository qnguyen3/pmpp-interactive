import type { ReactNode } from 'react';
import type { Question } from '../../components/Ask';
import { Predict } from '../../components/Ask';
import { Aside, Beyond, NoteGroup, Source, Today, type Tier } from '../../components/notes';
import {
  BackendsAndEmulator,
  LabFiles,
  LabPrediction,
  LabResults,
  LabsPage,
  ReferenceSolution,
  Reveal,
  RunningLabs,
  VerificationStatus,
  compare,
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
import { Translated, tr } from '../../lib/i18n';
import { fmt, labStatus, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';
import { rssOverSsa, type Row } from '../../sim/scantraffic';

// The page is built from components/LabKit.tsx; what is here is chapter 11's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

export const LABS = defineLabs('ch11', [
  { num: '11.1', dir: 'lab1_blockscan', name: 'one block scan, three ways', vi: { name: 'một phép scan khối, ba cách' } },
  { num: '11.2', dir: 'lab2_coarsened', name: 'coarsening and register tiling', vi: { name: 'gộp việc cho luồng và phân ô thanh ghi' } },
  { num: '11.3', dir: 'lab3_scanhunt', name: 'scan race hunt', vi: { name: 'săn race condition trong scan' } },
  { num: '11.4', dir: 'lab4_global', name: 'any length, hierarchically', vi: { name: 'độ dài bất kỳ, theo phân cấp' } },
  { num: '11.5', dir: 'lab5_singlepass', name: 'single pass with lookback', sources: ['kernel.cu'], vi: { name: 'single pass với lookback' } },
  { num: '11.6', dir: 'lab6_topp', name: 'top-p keep mask (M5)', vi: { name: 'mask giữ lại top-p (M5)' } },
]);
const [l1, l2, l3, l4, l5, l6] = LABS;

// Identifier colours for the reference solutions. Only names that mean one thing in every
// file of a lab.
const tiers = (global: string[], shared: string[], reg: string[] = []): Record<string, Tier> => ({
  ...Object.fromEntries(global.map((n) => [n, 'global' as Tier])),
  ...Object.fromEntries(shared.map((n) => [n, 'shared' as Tier])),
  ...Object.fromEntries(reg.map((n) => [n, 'reg' as Tier])),
});
const BLOCK_TIERS = tiers(['input', 'output'], ['buffer1_s', 'buffer2_s', 'inBuffer_s', 'outBuffer_s', 'warpSums_s'], ['leftVal']);
const COARSE_TIERS = tiers(['input', 'output'], ['buffer_s', 'threadSums', 'warpSums_s'], ['buffer_r', 'threadSum']);
const HUNT_TIERS = tiers(['in', 'out', 'rowIn', 'rowOut'], ['stage_s', 'warpSums_s', 'simdSums_s', 'myStage'], ['carry', 'threadSum']);
const GLOBAL_TIERS = tiers(['input', 'output', 'scratch', 'blockSums', 'scannedBlockSums'], ['buffer_s', 'threadSums', 'warpSums_s', 'partialSums_s'], ['buffer_r', 'carry']);
const SINGLE_TIERS = tiers(
  ['input', 'output', 'partialSums', 'flags', 'blockCounter'],
  ['buffer_s', 'threadSums', 'warpSums_s', 'bid_s', 'prevBlockPartialSum_s'],
  ['buffer_r', 'threadSum', 'prevPartialSum'],
);
const TOPP_TIERS = tiers(
  ['probs', 'massBefore', 'keep', 'kept', 'rowProbs', 'rowMass', 'rowKeep'],
  ['stage_s', 'warpSums_s', 'warpCounts_s', 'simdSums_s', 'simdCounts_s', 'myStage'],
  ['carry', 'threadSum'],
);

// ---------------------------------------------------------------- reading the reports
const ms = (s: number) => `${fmt(s * 1000)} ms`; // i18n-ignore: a number and its unit
const count = (v: number) => v.toLocaleString('en-US');
const pct = (v: number) => `${Math.round(v * 100)}%`;
const N26 = 2 ** 26;
const VOCAB = 128256;

/** The first row of a set with this kernel and these numeric fields. */
const kernelRow = (set: BenchSet | undefined, kernel: string, where: Record<string, number> = {}) =>
  set?.rows.find((r) => r.kernel === kernel && Object.entries(where).every(([k, v]) => num(r[k]) === v));
const secondsOf = (r: BenchRow | undefined) => num(r?.seconds);

/** Lab 11.1: at blocks of 1,024, how many times as fast the warp-shuffle scan runs as Fig. 11.3. */
const warpOverTwo = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const two = kernelRow(s, 'two_barrier', { block: 1024 });
    const warp = kernelRow(s, 'warp_shuffle', { block: 1024 });
    const a = secondsOf(two);
    const b = secondsOf(warp);
    if (!a || !b) return null;
    const dbl = secondsOf(kernelRow(s, 'double_buffer', { block: 1024 }));
    const prefix = secondsOf(kernelRow(s, 'warp_prefix', { block: 1024 }));
    const share = num(warp?.share_of_copy);
    return {
      value: a / b,
      detail: tr(
        `${ms(a)} for the two-barrier kernel and ${ms(b)} for the warp-shuffle scan, ${count(num(warp?.n) ?? 0)} floats in blocks of 1,024` +
        (dbl ? `; double buffering ${ms(dbl)}` : '') +
        (prefix ? `; the simd_prefix_inclusive_sum version ${ms(prefix)}` : '') +
        (share != null ? `; the warp-shuffle scan reached ${pct(share)} of the copy kernel's rate` : ''),
        `${ms(a)} cho kernel hai rào chắn và ${ms(b)} cho phép scan warp-shuffle, ${count(num(warp?.n) ?? 0)} số float trong các khối 1,024` +
        (dbl ? `; buffer kép ${ms(dbl)}` : '') +
        (prefix ? `; phiên bản simd_prefix_inclusive_sum ${ms(prefix)}` : '') +
        (share != null ? `; phép scan warp-shuffle đạt ${pct(share)} tốc độ của kernel copy` : ''),
      ),
    };
  });

/** Lab 11.4 (and the prediction of section 11.9): reduce-scan-scan's rate over scan-scan-add's. */
const rssOverSsaMeasured = (reports: Reports) =>
  measure(reports, l4, (s) => {
    const r = rssOverSsa(s.rows as Row[], N26);
    if (!r) return null;
    return {
      value: r.ratio,
      detail: tr(
        `${fmt(r.rss / 1e9)} against ${fmt(r.ssa / 1e9)} × 10⁹ elements per second at n = ${count(r.n)}, the faster block size of each`,
        `${fmt(r.rss / 1e9)} so với ${fmt(r.ssa / 1e9)} × 10⁹ phần tử mỗi giây ở n = ${count(r.n)}, với kích thước khối nhanh hơn của mỗi bên`,
      ),
    };
  });

/** Lab 11.6: the kernel's share of the copy kernel's bandwidth at 64 rows of 128,256 tokens, in percent. */
const toppShare = (reports: Reports) =>
  measure(reports, l6, (s) => {
    const mine = kernelRow(s, 'yours', { rows: 64, cols: VOCAB });
    const copy = kernelRow(s, 'copy', { rows: 64, cols: VOCAB });
    const share = num(mine?.share_of_copy);
    if (share == null || !mine) return null;
    const one = num(kernelRow(s, 'yours', { rows: 1, cols: VOCAB })?.share_of_copy);
    return {
      value: share * 100,
      detail: tr(
        `${fmt(num(mine.gb_per_s) ?? NaN)} GB/s at 9 bytes per token, against the copy's ${fmt(num(copy?.gb_per_s) ?? NaN)} GB/s at the same shape` +
        (one != null ? `; at 1 row of ${count(VOCAB)} the share is ${pct(one)}` : ''),
        `${fmt(num(mine.gb_per_s) ?? NaN)} GB/s với 9 byte mỗi token, so với ${fmt(num(copy?.gb_per_s) ?? NaN)} GB/s của kernel copy ở cùng kích thước` +
        (one != null ? `; ở 1 hàng ${count(VOCAB)} token, tỷ lệ là ${pct(one)}` : ''),
      ),
    };
  });

// ---------------------------------------------------------------- lab 11.2: the fastest factor at block 256
const BENCH_MISSING = (lab: string) => (
  <Translated
    vi={
      <>
        Khi lab {lab} đã qua, hãy chạy <code>python3 labs/pmpp.py run {lab} --bench</code> trên một máy có GPU, rồi tải lại.
      </>
    }
  >
    <>
      Once lab {lab} passes, run <code>python3 labs/pmpp.py run {lab} --bench</code> on a machine with a GPU, then reload.
    </>
  </Translated>
);

function MissingLine({ lab, children }: { lab: string; children?: ReactNode }) {
  const { loading, reload } = useLabs();
  return loading ? (
    <p data-measured="loading"><Translated vi="Đang đọc báo cáo.">Reading reports.</Translated></p>
  ) : (
    <p data-measured="none">
      <Translated vi="Máy này chưa có số đo nào.">No measurement on this machine yet.</Translated> {children ?? BENCH_MISSING(lab)}{' '}
      <button className="quiet" onClick={reload}>
        <Translated vi="Tải lại kết quả">Reload results</Translated>
      </button>
    </p>
  );
}

/** Fastest coarsening factor of the register-tiled kernel at block 256, and the reader's pick against it. */
function FactorReveal({ committed }: { committed: number }) {
  const { reports } = useLabs();
  const found = preferredBench(reports, l2).flatMap((s) => {
    const rows = s.rows
      .filter((r) => r.kernel === 'register_tiled' && num(r.block) === 256 && num(r.coarse) != null && num(r.seconds) != null)
      .map((r) => ({ c: num(r.coarse)!, t: num(r.seconds)!, share: num(r.share_of_copy) }))
      .sort((a, b) => a.c - b.c);
    return rows.length ? [{ s, rows }] : [];
  });
  return (
    <>
      {found.length ? (
        found.map(({ s, rows }) => {
          const best = rows.reduce((a, b) => (b.t < a.t ? b : a));
          const worst = rows.reduce((a, b) => (b.t > a.t ? b : a));
          const far = (c: number) => Math.abs(Math.log2(Math.max(c, 1e-9)) - Math.log2(Math.max(committed, 1e-9)));
          const mine = rows.reduce((a, b) => (far(b.c) < far(a.c) ? b : a));
          return (
            <Translated
              key={s.backend}
              vi={
                <p data-measured={best.c} data-backend={s.backend}>
                  Đo trên {s.device} ({s.backend}) với {whose(s)}, trong lần chạy ngày {s.date}, kernel phân ô thanh ghi với
                  khối 256: hệ số nhanh nhất là <b>{best.c}</b>, mất {ms(best.t)}
                  {best.share != null ? ` (${pct(best.share)} tốc độ của kernel copy)` : ''}, và chậm nhất là {worst.c}, mất{' '}
                  {ms(worst.t)}.{' '}
                  {mine.c === best.c
                    ? 'Đó chính là hệ số bạn đã chọn.'
                    : `Lựa chọn của bạn${mine.c !== committed ? ` (gần nhất trong số đã đo: ${mine.c})` : ''} mất ${ms(mine.t)}, bằng ${Math.round((mine.t / best.t) * 100)}% thời gian nhanh nhất.`}
                </p>
              }
            >
              <p key={s.backend} data-measured={best.c} data-backend={s.backend}>
                Measured on {s.device} ({s.backend}) with {whose(s)}, in the run of {s.date}, register-tiled kernel at blocks
                of 256: the fastest factor was <b>{best.c}</b>, at {ms(best.t)}
                {best.share != null ? ` (${pct(best.share)} of the copy kernel's rate)` : ''}, and the slowest {worst.c}, at{' '}
                {ms(worst.t)}.{' '}
                {mine.c === best.c
                  ? 'That is the factor you picked.'
                  : `Your pick${mine.c !== committed ? ` (nearest measured: ${mine.c})` : ''} took ${ms(mine.t)}, ${Math.round((mine.t / best.t) * 100)}% of the fastest time.`}
              </p>
            </Translated>
          );
        })
      ) : (
        <MissingLine lab="11.2" />
      )}
      <Translated
        vi={
          <p className="muted">
            Hãy đọc độ chênh lệch trước khi xem bên thắng. Khi hệ số nhanh nhất và chậm nhất chỉ cách nhau vài phần trăm, mọi
            hệ số đều di chuyển dữ liệu với tốc độ gần như nhau và việc hệ số nào về nhất chỉ là nhiễu. Phép đếm bước của phần
            11.6 không có thời gian bộ nhớ trong đó, nên một đường cong phẳng không mâu thuẫn với nó.
          </p>
        }
      >
        <p className="muted">
          Read the spread before the winner. When the fastest and slowest factors are a few percent apart, every factor moves
          the bytes at about the same rate and which one comes first is noise. The step count of section 11.6 has no memory
          time in it, so a flat curve does not contradict it.
        </p>
      </Translated>
    </>
  );
}

// ---------------------------------------------------------------- lab 11.3: what each backend showed of the shipped kernel
type FullResult = RunResult & { tsan?: unknown[]; log?: string };
interface HuntRun {
  backend: string;
  device: string;
  date: string;
  untouched: boolean;
  passed: number;
  total: number;
  aborted: boolean;
  races: number;
  failing: { name: string; detail: string }[];
}
const GPUS = ['metal', 'cuda', 'hip'];
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
    const rs = (run?.results.filter((x) => x.lab === l3.id && x.variant === 'work') ?? []) as FullResult[];
    if (!run || !rs.length) return [];
    const tests = rs.flatMap((x) => x.tests);
    return [
      {
        backend: b,
        device: String(run.device.name ?? b),
        date: run.generated.slice(0, 10),
        untouched: run.work_state?.[l3.id] === 'untouched',
        passed: tests.filter((t) => t.pass).length,
        total: tests.length,
        aborted: rs.some((x) => x.test_rc != null && x.test_rc !== 0 && x.test_rc !== 1 && !(b === 'race' && x.test_rc === TSAN_EXIT)),
        races: rs.reduce((n, x) => n + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0),
        failing: tests
          .filter((t) => !t.pass)
          .map((t) => ({ name: t.case, detail: /wrong in \d+ of \d+ launches/.exec(t.detail ?? '')?.[0] ?? tr('wrong', 'sai') })),
      },
    ];
  });
}

function HuntReveal({ committed }: { committed: number }) {
  const { reports, loading, reload } = useLabs();
  const runs = huntRuns(reports);
  const gpu = runs.filter((r) => GPUS.includes(r.backend));
  const emu = runs.filter((r) => !GPUS.includes(r.backend));
  return (
    <>
      <Translated vi={<p>Bạn đã dự đoán rằng {committed} trong ba lỗi sẽ làm một bài kiểm tra thất bại trên GPU của bạn.</p>}>
        <p>You predicted that {committed} of the three defects will make a test fail on your GPU.</p>
      </Translated>
      {gpu.map((r) => (
        <div key={r.backend} data-measured={r.passed} data-backend={r.backend}>
          <Translated
            vi={
              <p>
                Trên {r.device} ({r.backend}), lần chạy gần nhất <code>work/</code> của bạn ngày {r.date}
                {r.untouched ? ', vẫn là kernel như được phát hành' : ', sau khi bạn sửa'}: <b>{r.passed}</b> trên {r.total}{' '}
                trường hợp đúng ở mọi lần khởi chạy.
              </p>
            }
          >
            <p>
              On {r.device} ({r.backend}), the last run of your <code>work/</code> on {r.date}
              {r.untouched ? ', still the kernel as shipped' : ', after your edits'}: <b>{r.passed}</b> of {r.total} cases
              right in every launch.
            </p>
          </Translated>
          {r.failing.length ? (
            <ul className="muted">
              {r.failing.slice(0, 6).map((f) => (
                <li key={f.name}>
                  {f.name}: {f.detail}
                </li>
              ))}
              {r.failing.length > 6 ? <li><Translated vi={<>và {r.failing.length - 6} trường hợp nữa</>}>and {r.failing.length - 6} more</Translated></li> : null}
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
              {r.untouched ? ' như được phát hành' : ''}: {r.passed} trên {r.total} trường hợp đúng{r.aborted ? ', rồi lần chạy dừng lại' : ''}.
              {r.races ? ` ThreadSanitizer báo cáo ${r.races} race condition.` : ''}
            </p>
          }
        >
          <p key={r.backend} data-measured={r.passed} data-backend={r.backend}>
            On {BACKEND_NAME[r.backend]}, the last run of your <code>work/</code>
            {r.untouched ? ' as shipped' : ''}: {r.passed} of {r.total} cases right{r.aborted ? ', then the run stopped' : ''}.
            {r.races ? ` ThreadSanitizer reported ${r.races} race${r.races === 1 ? '' : 's'}.` : ''}
          </p>
        </Translated>
      ))}
      {!runs.length ? (
        loading ? (
          <p data-measured="loading"><Translated vi="Đang đọc báo cáo.">Reading reports.</Translated></p>
        ) : (
          <p data-measured="none">
            <Translated
              vi={
                <>
                  Chưa có lần chạy nào của bản làm của bạn cho lab này. Hãy chạy <code>python3 labs/pmpp.py run 11.3</code>{' '}
                  trước khi bạn thay đổi bất cứ thứ gì, rồi tải lại.
                </>
              }
            >
              <>
                No run of your work for this lab yet. Run <code>python3 labs/pmpp.py run 11.3</code> before you change
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
          <Translated vi="Máy này chưa có lần chạy GPU nào của bản làm của bạn cho lab này, nên chưa có gì để so với dự đoán của bạn.">
            No GPU run of your work for this lab on this machine, so there is nothing yet to compare your prediction with.
          </Translated>
        </p>
      ) : null}
      <Translated
        vi={
          <p className="muted">
            Các báo cáo đếm trường hợp, không đếm lỗi: một trường hợp thất bại có thể đến từ bất kỳ lỗi nào trong ba. Để trả lời
            câu hỏi của chính bạn, hãy sửa từng lỗi một và chạy lại trên GPU sau mỗi lần, điền vào bảng trong README của lab.
            Một trường hợp đúng ở mọi lần khởi chạy chưa chứng tỏ kernel đúng ở đó: một race condition mà thiết bị hôm nay không
            để lộ vẫn là race condition (notes/today/ch10-facts.md A1; ch04-facts A10).
          </p>
        }
      >
        <p className="muted">
          The reports count cases, not defects: a failing case could come from any of the three. To answer your own
          question, fix one defect at a time and rerun on the GPU after each, filling in the table in the lab's README. A case
          that is right in every launch has not shown that the kernel is correct there: a race that a device does not show
          today is still a race (notes/today/ch10-facts.md A1; ch04-facts A10).
        </p>
      </Translated>
    </>
  );
}

// ---------------------------------------------------------------- lab 11.5 against lab 11.4, on the same device
interface SinglePass {
  backend: string;
  device: string;
  date: string;
  whose: string;
  value: number;
  detail: string;
}
function singleOverSsa(reports: Reports): SinglePass[] {
  const four = preferredBench(reports, l4);
  return preferredBench(reports, l5).flatMap((s) => {
    const other = four.find((f) => f.backend === s.backend);
    const pairs = s.rows
      .filter((r) => r.kernel === 'single_pass' && num(r.n) === N26 && num(r.seconds) != null)
      .flatMap((r) => {
        const block = num(r.block)!;
        const ssa = secondsOf(kernelRow(other, 'scan_scan_add', { block, n: N26 }));
        return ssa ? [{ block, single: num(r.seconds)!, ssa }] : [];
      });
    if (!pairs.length || !other) return [];
    const main = pairs.find((p) => p.block === 256) ?? pairs[0];
    return [
      {
        backend: s.backend,
        device: s.device,
        date: s.date,
        whose: tr(`${whose(s)} for lab 11.5, and ${whose(other)} for lab 11.4`, `${whose(s)} cho lab 11.5, và ${whose(other)} cho lab 11.4`),
        value: main.ssa / main.single,
        detail: tr(
          `${ms(main.ssa)} for scan-scan-add and ${ms(main.single)} for the single pass, ${count(N26)} floats in blocks of ${main.block}` +
          pairs
            .filter((p) => p !== main)
            .map((p) => `; at blocks of ${p.block} the ratio is ${fmt(p.ssa / p.single)}`)
            .join(''),
          `${ms(main.ssa)} cho scan-scan-add và ${ms(main.single)} cho single pass, ${count(N26)} số float trong các khối ${main.block}` +
          pairs
            .filter((p) => p !== main)
            .map((p) => `; với khối ${p.block}, tỷ lệ là ${fmt(p.ssa / p.single)}`)
            .join(''),
        ),
      },
    ];
  });
}

function SinglePassReveal({ committed }: { committed: number }) {
  const { reports } = useLabs();
  const found = singleOverSsa(reports);
  return (
    <>
      {found.length ? (
        found.map((m) => (
          <Translated
            key={m.backend}
            vi={
              <p data-measured={fmt(m.value)} data-backend={m.backend}>
                Đo trên {m.device} ({m.backend}) với {m.whose}, trong lần chạy ngày {m.date}: <b>{fmt(m.value)}</b> × scan-scan-add
                của lab 11.4 ({m.detail}). {compare(committed, m.value)}
              </p>
            }
          >
            <p key={m.backend} data-measured={fmt(m.value)} data-backend={m.backend}>
              Measured on {m.device} ({m.backend}) with {m.whose}, in the run of {m.date}: <b>{fmt(m.value)}</b> × lab 11.4's
              scan-scan-add ({m.detail}). {compare(committed, m.value)}
            </p>
          </Translated>
        ))
      ) : (
        <MissingLine lab="11.5">
          <Translated
            vi={
              <>
                Lab 11.5 chỉ chạy trên thiết bị CUDA và HIP. Trên một thiết bị như vậy, hãy chạy{' '}
                <code>python3 labs/pmpp.py run 11.5 --bench</code> và <code>python3 labs/pmpp.py run 11.4 --bench</code>, rồi
                tải lại: các dòng của hai lab được so sánh trên cùng một thiết bị.
              </>
            }
          >
            <>
              Lab 11.5 runs only on CUDA and HIP devices. On one, run <code>python3 labs/pmpp.py run 11.5 --bench</code> and{' '}
              <code>python3 labs/pmpp.py run 11.4 --bench</code>, then reload: the two labs' rows are compared on the same
              device.
            </>
          </Translated>
        </MissingLine>
      )}
      <Translated
        vi={
          <p className="muted">
            Số byte nói là 2: 16 mỗi phần tử so với 8. Chuỗi thì nói ít hơn: mỗi khối chờ khối đứng trước nó, mỗi khối một mắt
            xích (p. 279). Số đo của bạn thấp hơn 2 bao nhiêu chính là cái giá mà chuỗi đó, và việc chờ, gây ra trên thiết bị
            của bạn.
          </p>
        }
      >
        <p className="muted">
          The bytes say 2: 16 per element against 8. The chain says less: every block waits for the one before it, one link
          per block (p. 279). How far your measurement falls below 2 is what that chain, and the waiting, cost on your
          device.
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
              Máy này chưa có lần chạy nào của lab {l.num} trên backend race:{' '}
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
        <Translated
          vi={
            <>
              Trên backend race của máy này, lời giải tham chiếu{' '}
              {s === 'pass'
                ? `đã qua cả ${n} bài kiểm tra mà không có race nào được báo cáo: emulator-checked với cả hai biến thể.`
                : s === 'fail'
                  ? 'thất bại: unverified ở đó.'
                  : 'chưa có lần chạy nào.'}{' '}
              Bản làm của bạn ở đó:{' '}
              {!work.length
                ? 'chưa có lần chạy nào.'
                : ws === 'none'
                  ? `chưa bắt đầu (${wTests.filter((t) => t.pass).length} trên ${wTests.length} bài kiểm tra qua khi giữ nguyên${wRaces ? `, ${wRaces} race được báo cáo` : ''}).`
                  : ws === 'pass'
                    ? `đã qua, ${wTests.length} trên ${wTests.length} bài kiểm tra, không có race nào được báo cáo.`
                    : `chưa qua: ${wTests.filter((t) => t.pass).length} trên ${wTests.length} bài kiểm tra${wRaces ? `, ${wRaces} race được báo cáo` : ''}.`}
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

// ---------------------------------------------------------------- milestone M5, part 1
function MilestoneStatus() {
  const { reports, loading } = useLabs();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const done = backends.filter((b) => labStatus(reports[b], l6.id, 'work') === 'pass');
  const name = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  return (
    <p data-milestone={loading ? 'loading' : done.join(',') || 'none'}>
      {loading
        ? tr('Reading reports.', 'Đang đọc báo cáo.')
        : !backends.length
          ? tr('No lab results are recorded on this machine yet.', 'Máy này chưa ghi nhận kết quả lab nào.')
          : done.length
            ? tr(`Your own lab 11.6 passes on ${done.map(name).join(' and ')} on this machine: the top-p piece of M5 is in place there.`, `Lab 11.6 của bạn đã qua trên ${done.map(name).join(' và ')} ở máy này: mảnh top-p của M5 đã có ở đó.`)
            : tr('Your own lab 11.6 does not pass on any backend of this machine yet.', 'Lab 11.6 của bạn chưa qua trên backend nào của máy này.')}
    </p>
  );
}

// ---------------------------------------------------------------- results-table columns
const KERNEL_1: Record<string, string> = {
  get copy() {
    return tr('copy (provided)', 'copy (có sẵn)');
  },
  get two_barrier() {
    return tr('two barriers (Fig. 11.3, provided)', 'hai rào chắn (Fig. 11.3, có sẵn)');
  },
  get double_buffer() {
    return tr('double buffer, (a)', 'buffer kép, (a)');
  },
  get warp_shuffle() {
    return tr('warp shuffle, (b)', 'warp shuffle, (b)');
  },
  warp_prefix: 'simd_prefix_inclusive_sum, (b2)', // i18n-ignore: a function name and a part label
};
const KERNEL_2: Record<string, string> = {
  get copy() {
    return tr('copy (provided)', 'copy (có sẵn)');
  },
  get coarsened() {
    return tr('coarsened, (a)', 'gộp việc, (a)');
  },
  get register_tiled() {
    return tr('register-tiled, (b)', 'phân ô thanh ghi, (b)');
  },
  get thread_major() {
    return tr('thread-major (provided)', 'thread-major (có sẵn)');
  },
};
const KERNEL_4: Record<string, string> = {
  get copy() {
    return tr('copy (provided)', 'copy (có sẵn)');
  },
  scan_scan_add: 'scan-scan-add',
  reduce_scan_scan: 'reduce-scan-scan',
  get single_pass() {
    return tr('single pass', 'single pass');
  },
};
const KERNEL_6: Record<string, string> = {
  get copy() {
    return tr('copy (provided)', 'copy (có sẵn)');
  },
  get naive() {
    return tr('one thread per row (provided)', 'một luồng mỗi hàng (có sẵn)');
  },
  get yours() {
    return tr('top-p mask kernel', 'kernel mask top-p');
  },
};
const named = (map: Record<string, string>) => (r: BenchRow) => map[String(r.kernel)] ?? String(r.kernel ?? r.name);
const shareCol = {
  get head() {
    return tr('Of copy', 'So với copy');
  },
  num: true,
  cell: (r: BenchRow) => (num(r.share_of_copy) != null ? pct(num(r.share_of_copy)!) : ''),
};
const floatsCol = {
  get head() {
    return tr('Floats', 'Số float');
  },
  num: true,
  cell: (r: BenchRow) => (num(r.n) != null ? count(num(r.n)!) : ''),
};
// A function, so the column names are read in the language of the moment they are shown.
const globalCols = () => [
  {
    get head() {
      return tr('Design', 'Thiết kế');
    },
    cell: named(KERNEL_4),
  },
  rawCol(tr('Block', 'Khối'), 'block'),
  rawCol('S', 'segment'),
  floatsCol,
  msCol,
  {
    get head() {
      return tr('10⁹ elements/s', '10⁹ phần tử/s');
    },
    num: true,
    cell: (r: BenchRow) => (num(r.elements_per_s) != null ? fmt(num(r.elements_per_s)! / 1e9) : ''),
  },
  numCol(tr('Book B/element', 'B/phần tử theo sách'), 'book_bytes_per_element', 4),
  shareCol,
];

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS}>
      <Translated
        vi={
          <p>
            Sáu lab, mỗi lab trong thư mục riêng dưới <code>labs/ch11/</code>. Hai lab đầu xây dựng và đo các phép scan khối:
            ba cách scan một khối, rồi gộp việc cho luồng (thread coarsening) và phân ô thanh ghi. Lab thứ ba là cuộc săn ba
            lỗi trong phép scan theo hàng mà các thư viện thật đang dùng. Lab thứ tư và thứ năm scan một mảng có độ dài bất kỳ,
            bằng nhiều kernel trên mọi backend và bằng một kernel chỉ trên CUDA và HIP. Lab thứ sáu là mảnh top-p của mốc dự án
            M5 của dự án khóa học: mask giữ lại top-p trên các xác suất đã sắp xếp.
          </p>
        }
      >
        <p>
          Six labs, each in its own directory under <code>labs/ch11/</code>. The first two build and measure block scans:
          three ways to scan one block, then coarsening and register tiling. The third is a hunt for three defects in the
          row-wise scan that real libraries ship. The fourth and fifth scan an array of any length, with several kernels on
          every backend and with one kernel on CUDA and HIP only. The sixth is the top-p piece of milestone M5 of the course
          project: the top-p keep mask over sorted probabilities.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Mỗi lab giữ các tệp của nó ba lần: <code>starter/</code> là bài tập như được phát hành, <code>work/</code>{' '}
            là của bạn và bắt đầu là một bản sao của starter, và <code>solution/</code> là lời giải tham chiếu. Một chương
            trình điều khiển (driver), <code>main.cpp</code>, kiểm tra những gì có trong <code>work/</code> so với một tham
            chiếu CPU (vòng lặp của Fig. 11.1, chạy với double) và, trên một thiết bị thật, đo thời gian. Bạn không sửa driver.
          </p>
        }
      >
        <p>
          Every lab holds its files three times over: <code>starter/</code> is the exercise as shipped, <code>work/</code>{' '}
          is yours and starts as a copy of the starter, and <code>solution/</code> is the reference. A driver,{' '}
          <code>main.cpp</code>, tests what is in <code>work/</code> against a CPU reference (Fig. 11.1's loop, run in
          double) and, on a real device, times it. You do not edit the driver.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Năm lab có ba tệp: <code>kernel.cu</code>, CUDA thuần, được build theo ba cách từ một mã nguồn (cho CUDA, cho HIP
            qua một header ánh xạ tên, và cho một trình giả lập CPU), cùng <code>kernel.metal</code> và{' '}
            <code>launch.mm</code>, cùng các kernel đó và phần khởi chạy chúng bằng ngôn ngữ shading và API host của Metal.
            Lab 11.5 chỉ có <code>kernel.cu</code>. Các khối của nó chờ nhau bên trong một kernel, và Metal không ghi điều gì
            trong tài liệu làm cho việc đó an toàn; trên Metal, phép scan toàn cục là của lab 11.4.
          </p>
        }
      >
        <p>
          Five labs have three files: <code>kernel.cu</code>, plain CUDA, built three ways from the one source (for CUDA,
          for HIP through a header of name mappings, and for a CPU emulator), and <code>kernel.metal</code> and{' '}
          <code>launch.mm</code>, the same kernels and their launch in Metal's shading language and host API. Lab 11.5 has{' '}
          <code>kernel.cu</code> only. Its blocks wait for each other inside one kernel, and Metal documents nothing that
          makes that safe; on Metal the global scan is lab 11.4's.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch11-facts.md C5)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Đặc tả Metal không đưa ra bảo đảm forward progress nào giữa các threadgroup. Khi được hỏi về một phép scan dạng
              chuỗi, một kỹ sư Apple đã gọi các giả định của nó là "undefined behavior".
            </p>
          }
        >
          <p>
            The Metal spec gives no forward-progress guarantee between threadgroups. Asked about a chained scan, an Apple
            engineer called its assumptions "undefined behavior".
          </p>
        </Translated>
      </Today>
      <RunningLabs
        example="11.1"
        alsoShownIn={
          <Translated
            vi={
              <>
                các phần <a href={href('ch11', '11.3')}>11.3</a>, <a href={href('ch11', '11.6')}>11.6</a>,{' '}
                <a href={href('ch11', '11.8')}>11.8</a> và <a href={href('ch11', '11.9')}>11.9</a>
              </>
            }
          >
            <>
              sections <a href={href('ch11', '11.3')}>11.3</a>, <a href={href('ch11', '11.6')}>11.6</a>,{' '}
              <a href={href('ch11', '11.8')}>11.8</a> and <a href={href('ch11', '11.9')}>11.9</a>
            </>
          </Translated>
        }
      />
      <BackendsAndEmulator />
      <Translated
        vi={
          <p>
            Trình giả lập có một biến thể thứ hai, <code>race</code>, và cả sáu lab ở đây đều được build cho nó. Nó chạy các
            luồng giả lập, và nhiều khối, cùng lúc dưới ThreadSanitizer, và báo cáo hai luồng chạm vào cùng một vùng nhớ mà
            không có gì sắp thứ tự chúng, kèm cả hai dòng mã nguồn. Một phép scan đọc các giá trị mà luồng khác đã ghi, nên một
            rào chắn bị thiếu là lỗi mà chương này liên tục cảnh báo (pp. 257–258), và có những lỗi chỉ <code>race</code> mới
            thấy được. Hãy yêu cầu nó bằng <code>--backend race</code>; <code>python3 labs/pmpp.py doctor</code> cho biết
            trình biên dịch của bạn có build được nó không. Các bảng trên trang này chỉ hiện biến thể thứ nhất; một dòng dưới
            mỗi lab cho kết quả race.
          </p>
        }
      >
        <p>
          The emulator has a second flavour, <code>race</code>, and all six labs here are built for it. It runs the emulated
          threads, and several blocks, at the same time under ThreadSanitizer, and reports two threads that touch the same
          memory with nothing ordering them, with both source lines. A scan reads values other threads wrote, so a missing
          barrier is the mistake this chapter keeps warning about (pp. 257–258), and some only <code>race</code> can see. Ask for it with{' '}
          <code>--backend race</code>; <code>python3 labs/pmpp.py doctor</code> says whether your compiler can build it. The
          tables on this page show the first flavour only; a line under each lab gives the race results.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Không trình giả lập nào chạy warp rộng hơn 32 lane, và Metal không có lane mask. Vì vậy hai lỗi HIP mà chương này
            cảnh báo, một mask <code>0xffffffff</code> và một vòng lặp shuffle dừng ở hằng số 32, đều qua ở mọi nơi mà một máy
            Mac kiểm tra được. HIP từ chối lỗi thứ nhất lúc biên dịch. Với lỗi thứ hai, trên một wavefront 64 lane, các lane 32
            đến 63 ra kết quả sai, vì <code>laneIdx()</code> đếm theo nhóm 64 trong khi các bước nhảy chỉ với ngược lại được 31
            lane; nó chỉ đúng nếu mọi phép đếm cũng dùng 32. Hãy kiểm tra cả hai bằng cách đọc mã của bạn.
          </p>
        }
      >
        <p>
          Neither emulator runs a warp wider than 32 lanes, and Metal has no lane masks. The two HIP mistakes this chapter
          warns about, a <code>0xffffffff</code> mask and a shuffle loop that stops at a literal 32, therefore pass everywhere
          a Mac can test. HIP rejects the first at compile time. With the second, on a 64-lane wavefront lanes 32 to 63 come
          out wrong, because <code>laneIdx()</code> counts in 64s while the strides only reach 31 lanes back; it would be
          right only if every count also used 32. Check both by reading your code.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch11-facts.md B1; infra-facts B3)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Mọi hàm <code>_sync</code> của HIP đều đòi mask 64-bit. <code>warpSize</code> là 64 trên gfx9 và 32 trên các chip
              AMD đời sau, và không phải hằng số lúc biên dịch.
            </p>
          }
        >
          <p>
            Every HIP <code>_sync</code> function requires a 64-bit mask. <code>warpSize</code> is 64 on gfx9 and 32 on
            later AMD parts, and is not a compile-time constant.
          </p>
        </Translated>
      </Today>
      <VerificationStatus />
      <Translated
        vi={
          <p>
            <code>&lt;cuda/atomic&gt;</code> của lab 11.5 đến từ libhipcxx trên ROCm, thứ mà bản build HIP của khóa học liên
            kết vào ở nơi ROCm cung cấp nó; chưa có máy ROCm nào build nó. Dù sao nó vẫn là unverified trên AMD cho đến khi một
            báo cáo từ máy AMD cho thấy nó qua, vì AMD không ghi trong tài liệu forward progress nào giữa các workgroup.
          </p>
        }
      >
        <p>
          Lab 11.5's <code>&lt;cuda/atomic&gt;</code> comes from libhipcxx on ROCm, which the course's HIP build links
          where ROCm provides it; no ROCm machine has built it yet. It stays unverified on AMD in any case until a report from an AMD machine shows it passing, because AMD documents no forward progress between
          workgroups.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch11-facts.md B3, B4; infra-facts B2)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              libhipcxx (ROCm 10.1 trở về sau) giữ cách viết của CUDA cho <code>cuda::atomic_ref</code>. rocPRIM của chính AMD
              chỉ nhận chỉ số khối từ một bộ đếm atomic trên một số kiến trúc.
            </p>
          }
        >
          <p>
            libhipcxx (ROCm 10.1 and later) keeps CUDA's spelling of <code>cuda::atomic_ref</code>. AMD's own rocPRIM claims
            block indices from an atomic counter only on some architectures.
          </p>
        </Translated>
      </Today>

      {/* ------------------------------------------------------------ 11.1 */}
      <h2><Translated vi="Lab 11.1: một phép scan khối, ba cách">Lab 11.1: one block scan, three ways</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn viết hai phép scan bao gồm mức khối và đo thời gian chúng so với kernel hai rào chắn của sách, Fig. 11.3, được
            cung cấp sẵn. (a) là Fig. 11.5: hai buffer chia sẻ đổi vai cho nhau, một rào chắn mỗi lần lặp, và bước sao chép cho
            những luồng không cộng. (b) là phép scan warp-shuffle của Fig. 11.8 đến 11.10, được làm cho biên dịch được trên
            CUDA và HIP và đúng ở mọi kích thước khối. Trên Metal còn có (b2), cùng phép scan khối đó với phép scan của mỗi
            SIMD group được làm bằng một lời gọi <code>simd_prefix_inclusive_sum</code>. Phép scan khối của (b) là thứ mà các
            lab 11.2, 11.3 và 11.6 dùng lại.
          </p>
        }
      >
        <p>
          You write two block-level inclusive scans and time them against the book's two-barrier kernel, Fig. 11.3, which
          is provided. (a) is Fig. 11.5: two shared buffers that swap roles, one barrier per iteration, and the copy for
          threads that do not add. (b) is the warp-shuffle scan of Figs. 11.8 to 11.10, made to compile for CUDA and HIP
          and to be right at any block size. On Metal there is also (b2), the same block scan with each SIMD-group's scan
          done by one call to <code>simd_prefix_inclusive_sum</code>. The block scan of (b) is the one labs 11.2, 11.3 and
          11.6 reuse.
        </p>
      </Translated>
      <Source section="11.2–11.4" pages="254–265" figs={tr('Figs. 11.3, 11.5, 11.8 to 11.10', 'Figs. 11.3, 11.5, 11.8 đến 11.10')} />
      <Translated vi={<p>Những chỗ (b) khác với trang sách, và lý do:</p>}>
        <p>Where (b) departs from the page, and why:</p>
      </Translated>
      <ul>
        <Translated
          vi={
            <li>
              Lane mask có kiểu riêng, 32 bit trên CUDA và 64 trên HIP, không bao giờ là hằng số <code>0xffffffff</code>. Vòng
              lặp chạy khi <code>stride &lt; warpSize</code>, và lane cùng warp được tính từ <code>warpSize</code>.
            </li>
          }
        >
          <li>
            The lane mask has its own type, 32 bits on CUDA and 64 on HIP, never the literal <code>0xffffffff</code>. The
            loop runs while <code>stride &lt; warpSize</code>, and lane and warp come from <code>warpSize</code>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Fig. 11.9 khai báo các tổng của warp bên trong <code>blockScan</code>. Metal từ chối điều đó và HIP không ghi nó
              trong tài liệu, nên kernel khai báo mảng, với kích thước cho warp hẹp nhất mà bất kỳ backend nào ở đây có, và
              truyền một con trỏ.
            </li>
          }
        >
          <li>
            Fig. 11.9 declares the warp totals inside <code>blockScan</code>. Metal refuses that and HIP does not document
            it, so the kernel declares the array, sized for the narrowest warp any backend here has, and passes a pointer.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Lần đọc ở giai đoạn 2 được bảo vệ: các lane không có warp tương ứng nạp 0. Mọi luồng đều gọi{' '}
              <code>blockScan</code>, dù trong phạm vi hay không, vì nó chứa rào chắn và shuffle với mask đầy đủ.
            </li>
          }
        >
          <li>
            The stage-2 read is guarded: lanes with no warp behind them feed 0. Every thread calls{' '}
            <code>blockScan</code>, in range or not, because it holds barriers and full-mask shuffles.
          </li>
        </Translated>
      </ul>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <Translated
              vi={
                <>
                  <code>KoggeStoneDoubleBufferKernel</code>, <code>warpScan</code>, <code>blockScan</code>,{' '}
                  <code>WarpScanKernel</code> và các hàm khởi chạy <code>launchKoggeStoneDB</code> và <code>launchWarpScan</code>
                </>
              }
            >
              <>
                <code>KoggeStoneDoubleBufferKernel</code>, <code>warpScan</code>, <code>blockScan</code>,{' '}
                <code>WarpScanKernel</code> and the launchers <code>launchKoggeStoneDB</code> and <code>launchWarpScan</code>
              </>
            </Translated>,
          ],
          ['kernel.metal', <Translated vi={<><code>ks_double_buffer</code>, <code>warp_scan_shuffle</code>, <code>warp_scan_prefix</code> và các hàm phụ trợ của chúng</>}><><code>ks_double_buffer</code>, <code>warp_scan_shuffle</code>, <code>warp_scan_prefix</code> and their helpers</></Translated>],
          ['launch.mm', <><code>encodeKoggeStoneDB</code>, <code>encodeWarpScanShuffle</code>, <code>encodeWarpScanPrefix</code></>],
        ]}
      />
      <Translated
        vi={
          <p>
            Có sẵn và không sửa: Fig. 11.3 và một kernel copy, trong <code>labs/ch11/shared-g11/bs_two_barrier.cu</code>,{' '}
            <code>bs_copy.cu</code>, các bản sinh đôi <code>.metal</code> của chúng và <code>bs_launch.mm</code>. Kernel copy di
            chuyển cùng lượng byte như một phép scan và không làm gì khác, nên nó là thước đo: tốc độ của mọi kernel cũng được
            cho dưới dạng tỷ lệ so với tốc độ của copy.
          </p>
        }
      >
        <p>
          Provided and not edited: Fig. 11.3 and a copy kernel, in <code>labs/ch11/shared-g11/bs_two_barrier.cu</code>,{' '}
          <code>bs_copy.cu</code>, their <code>.metal</code> twins and <code>bs_launch.mm</code>. The copy moves the same
          bytes as a scan and nothing else, so it is the yardstick: every kernel's rate is also given as a share of the
          copy's.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Hãy chạy <code>python3 labs/pmpp.py run 11.1 --backend emu</code> trước. Các bài kiểm tra chạy độ dài từ 1 đến 2
            <sup>22</sup> với khối từ 64 đến 1,024 luồng (ít hơn trên trình giả lập), với toàn số 1, số nguyên từ −3 đến 3 (cả
            hai đều chính xác theo mọi thứ tự) và giá trị phân bố đều trong [−1, 1], so với vòng lặp của Fig. 11.1 chạy với
            double, trong phạm vi 10<sup>−6</sup> của tổng các trị tuyệt đối. Đầu ra bắt đầu là NaN, và các từ canh gác phía
            sau chúng bắt được một luồng vượt quá cuối mà vẫn ghi.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 11.1 --backend emu</code> first. The tests run lengths from 1 to 2
          <sup>22</sup> at blocks of 64 to 1,024 threads (fewer on the emulators), with all ones, integers from −3 to 3
          (both exact in any order) and values uniform in [−1, 1], compared with Fig. 11.1's loop in double within 10
          <sup>−6</sup> of the sum of magnitudes. Outputs start as NaN, and guard words after them catch a thread past the
          end that stores.
        </p>
      </Translated>
      <RaceLine lab={l1} />
      <LabPrediction
        id="ch11/labs/p-lab1-warp"
        prompt={
          <Translated
            vi={
              <>
                Trước khi chạy benchmark. Ở khối 1,024 luồng, phép scan warp-shuffle của bạn sẽ chạy nhanh gấp bao nhiêu lần
                kernel hai rào chắn (Fig. 11.3)?
              </>
            }
          >
            <>
              Before you run the benchmark. At blocks of 1,024 threads, how many times as fast as the two-barrier kernel
              (Fig. 11.3) will your warp-shuffle scan run?
            </>
          </Translated>
        }
        unit={tr('× the two-barrier kernel', '× kernel hai rào chắn')}
        measure={warpOverTwo}
        missing={BENCH_MISSING('11.1')}
        reading={
          <Translated
            vi={
              <>
                Đếm từ các vòng lặp: 20 rào chắn cho Fig. 11.3 ở 1,024 luồng, 10 cho Fig. 11.5, 2 cho Fig. 11.9 (pp.
                257–265). Một rào chắn tốn bao nhiêu phụ thuộc vào việc các luồng chờ ở đó bao lâu, và một phép scan khối cũng
                phải di chuyển dữ liệu của nó. Khi cả hai kernel đều gần với tốc độ của kernel copy, số byte quyết định và các
                rào chắn hầu như không lộ ra. Hãy so sánh cả các dòng ở 256 luồng.
              </>
            }
          >
            <>
              Counted from the loops: 20 barriers for Fig. 11.3 at 1,024 threads, 10 for Fig. 11.5, 2 for Fig. 11.9 (pp.
              257–265). What a barrier costs depends on how long threads wait at it, and a block scan also has to move its
              bytes. When both kernels come close to the copy kernel's rate, the bytes decide and the barriers hardly show.
              Compare the rows at 256 threads too.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Phần <a href={href('ch11', '11.3')}>11.3</a> hỏi một dự đoán thứ hai về các dòng này, riêng buffer kép sẽ nhanh
            gấp bao nhiêu lần. Hãy chốt nó ở đó trước khi bạn chạy benchmark.
          </p>
        }
      >
        <p>
          Section <a href={href('ch11', '11.3')}>11.3</a> asks a second prediction about these rows, how many times as fast
          double buffering alone will be. Commit it there before you run the benchmark.
        </p>
      </Translated>
      <LabResults
        lab={l1}
        filter={(r) => r.name === 'scan'}
        cols={[{ head: tr('Measurement', 'Phép đo'), cell: named(KERNEL_1) }, rawCol(tr('Block', 'Khối'), 'block'), floatsCol, msCol, numCol('GB/s', 'gb_per_s'), shareCol]}
      />
      <ReferenceSolution lab={l1} tiers={BLOCK_TIERS} />

      {/* ------------------------------------------------------------ 11.2 */}
      <h2><Translated vi="Lab 11.2: gộp việc cho luồng và phân ô thanh ghi">Lab 11.2: coarsening and register tiling</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn viết phép scan khối có gộp việc của Fig. 11.12 và dạng phân ô thanh ghi của nó, Fig. 11.13. Mỗi khối scan một
            đoạn gồm <code>COARSE_FACTOR × blockDim.x</code> phần tử: nó nạp đoạn vào bộ nhớ chia sẻ trong{' '}
            <code>COARSE_FACTOR</code> lượt được hợp nhất, mỗi luồng scan tuần tự đoạn của riêng nó, <code>blockScan</code>{' '}
            của bạn từ lab 11.1 scan các tổng của đoạn, và khối cộng rồi ghi ra theo từng lượt như khi nạp. Trong (b), đoạn
            nằm trong một mảng cục bộ <code>buffer_r</code> và bộ nhớ chia sẻ chỉ bị chạm một lần lúc vào và một lần lúc ra.
          </p>
        }
      >
        <p>
          You write the coarsened block scan of Fig. 11.12 and its register-tiled form, Fig. 11.13. Each block scans a
          segment of <code>COARSE_FACTOR × blockDim.x</code> elements: it loads the segment into shared memory in{' '}
          <code>COARSE_FACTOR</code> coalesced passes, each thread scans its own run sequentially, your{' '}
          <code>blockScan</code> from lab 11.1 scans the run totals, and the block adds and stores in passes as it loaded.
          In (b) the run lives in a local array <code>buffer_r</code> and shared memory is touched once on the way in and
          once on the way out.
        </p>
      </Translated>
      <Source section="11.6–11.8" pages="267–273" figs={tr('Figs. 11.11 to 11.13; exercise 4, p. 286', 'Figs. 11.11 đến 11.13; bài tập 4, p. 286')} />
      <ul>
        <Translated
          vi={
            <li>
              Các kernel của sách không bao giờ đọc <code>N</code>, nên chúng chỉ đúng với các đoạn trọn vẹn. Kernel của bạn
              kiểm tra mọi lần nạp và ghi toàn cục, và một phần tử ngoài phạm vi đóng góp 0.
            </li>
          }
        >
          <li>
            The book's kernels never read <code>N</code>, so they are right only for whole segments. Yours check every
            global load and store, and an element out of range contributes 0.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <code>COARSE_FACTOR</code> là một tham số template, nên một lần build kiểm tra và đo các hệ số 1, 2, 4, 8 và 16;
              các hàm khởi chạy nhận hệ số làm đối số cuối. Trên Metal, nó là một macro được đặt khi driver build pipeline.
            </li>
          }
        >
          <li>
            <code>COARSE_FACTOR</code> is a template parameter, so one build tests and times factors 1, 2, 4, 8 and 16; the
            launchers take the factor as their last argument. On Metal it is a macro set when the driver builds the
            pipeline.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Buffer của đoạn là bộ nhớ chia sẻ động có kích thước định lúc khởi chạy, nên benchmark ở 256 luồng không giữ chỗ
              cho lượng mà 1,024 luồng sẽ cần. Một cấu hình có bộ nhớ không vừa sẽ bị bỏ qua, và driver nói lý do.
            </li>
          }
        >
          <li>
            The segment buffer is dynamic shared memory sized at launch, so the benchmark at 256 threads does not reserve
            what 1,024 threads would need. A configuration whose memory does not fit is skipped, and the driver says why.
          </li>
        </Translated>
      </ul>
      <LabFiles
        lab={l2}
        files={[
          [
            'kernel.cu',
            <Translated
              vi={
                <>
                  <code>CoarsenedScanKernel</code>, <code>RegisterTiledScanKernel</code>, <code>blockScan</code> của bạn từ lab
                  11.1, <code>launchCoarsenedScan</code> và <code>launchRegisterTiledScan</code>
                </>
              }
            >
              <>
                <code>CoarsenedScanKernel</code>, <code>RegisterTiledScanKernel</code>, your <code>blockScan</code> from lab
                11.1, <code>launchCoarsenedScan</code> and <code>launchRegisterTiledScan</code>
              </>
            </Translated>,
          ],
          ['kernel.metal', <Translated vi={<><code>coarsened_scan</code>, <code>register_tiled_scan</code> và <code>block_scan</code> của bạn</>}><><code>coarsened_scan</code>, <code>register_tiled_scan</code> and your <code>block_scan</code></></Translated>],
          ['launch.mm', <><code>encodeCoarsenedScan</code>, <code>encodeRegisterTiledScan</code></>],
        ]}
      />
      <Translated
        vi={
          <p>
            Có sẵn: kernel copy, và một biến thể thread-major trong <code>labs/ch11/shared-g11/bs_threadmajor.cu</code> và{' '}
            <code>.metal</code>: Fig. 11.13 với mỗi luồng nạp đoạn của riêng nó thẳng từ bộ nhớ toàn cục, kiểu mẫu mà p. 269
            cảnh báo và cũng là kiểu mà cumsum của llama.cpp dùng. Benchmark cho thấy cái giá của nó trên thiết bị của bạn.
          </p>
        }
      >
        <p>
          Provided: the copy kernel, and a thread-major variant in <code>labs/ch11/shared-g11/bs_threadmajor.cu</code> and{' '}
          <code>.metal</code>: Fig. 11.13 with each thread loading its own run straight from global memory, the pattern p.
          269 warns against and the one llama.cpp's cumsum uses. The benchmark shows what that costs on your device.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch11-facts.md D1, D3)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Cumsum CUDA của llama.cpp lấy <code>idx = i + tid * num_unroll</code>, rồi đọc <code>src_row[idx + j]</code>,
              với <code>num_unroll = 4</code>; phép scan Metal của MLX cũng cho mỗi luồng đọc các giá trị liên tiếp của riêng
              nó.
            </p>
          }
        >
          <p>
            llama.cpp's CUDA cumsum takes <code>idx = i + tid * num_unroll</code>, then reads <code>src_row[idx + j]</code>,
            with <code>num_unroll = 4</code>; MLX's Metal scan also has each thread read its own contiguous values.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Các bài kiểm tra chạy độ dài từ 1 đến 2<sup>22</sup> + 3 ở mọi hệ số và với khối 64, 256 và 1,024, với dữ liệu và
            dung sai của lab 11.1 được kiểm tra theo từng đoạn, và phần tử đầu và cuối của mọi đoạn được kiểm tra trước phần
            còn lại.
          </p>
        }
      >
        <p>
          The tests run lengths from 1 to 2<sup>22</sup> + 3 at every factor and blocks of 64, 256 and 1,024, with the data
          and tolerance of lab 11.1 checked segment by segment, and the first and last element of every segment checked
          before the rest.
        </p>
      </Translated>
      <RaceLine lab={l2} />
      <Predict
        id="ch11/labs/p-lab2-best"
        prompt={
          <Translated
            vi={
              <>
                Trước khi chạy benchmark. Kernel phân ô thanh ghi được đo trên 2<sup>26</sup> số float ở các hệ số 1, 2, 4, 8
                và 16 với khối 256 luồng. Hệ số nào sẽ nhanh nhất trên thiết bị của bạn?
              </>
            }
          >
            <>
              Before you run the benchmark. The register-tiled kernel is timed on 2<sup>26</sup> floats at factors 1, 2, 4, 8
              and 16 with blocks of 256 threads. Which factor will be fastest on your device?
            </>
          </Translated>
        }
        unit={tr('(coarsening factor)', '(hệ số gộp việc)')}
      >
        {(v) => <FactorReveal committed={v} />}
      </Predict>
      <Translated
        vi={
          <p>
            Hai dự đoán nữa đọc các dòng này, và được hỏi ở nơi các phần của chúng giải thích: phần{' '}
            <a href={href('ch11', '11.6')}>11.6</a> hỏi lần nạp theo giai đoạn nhanh hơn lần nạp thread-major bao nhiêu, và
            phần <a href={href('ch11', '11.8')}>11.8</a> hỏi kernel phân ô thanh ghi đạt bao nhiêu phần băng thông copy của bạn.
          </p>
        }
      >
        <p>
          Two more predictions read these rows, and are asked where their sections explain them: section{' '}
          <a href={href('ch11', '11.6')}>11.6</a> asks how much faster the staged load is than the thread-major one, and
          section <a href={href('ch11', '11.8')}>11.8</a> what share of your copy bandwidth the register-tiled kernel reaches.
        </p>
      </Translated>
      <LabResults
        lab={l2}
        filter={(r) => r.name === 'scan'}
        cols={[{ head: tr('Measurement', 'Phép đo'), cell: named(KERNEL_2) }, rawCol(tr('Block', 'Khối'), 'block'), rawCol(tr('Factor', 'Hệ số'), 'coarse'), msCol, numCol('GB/s', 'gb_per_s'), shareCol]}
      />
      <Translated
        vi={
          <p>
            Đi xa hơn, không có bài kiểm tra: bài tập 4. Nạp vector cho các lượt giữa bộ nhớ toàn cục và bộ nhớ chia sẻ (trên
            Metal, khai báo buffer là <code>device const float4*</code>; phép ép kiểu CUDA từ <code>float*</code> sang{' '}
            <code>float4*</code> là kỹ thuật của sách và không nhà cung cấp nào ghi nó trong tài liệu), và đệm thêm cho các
            truy cập theo bước nhảy <code>buffer_s[t*F + c]</code>, thứ mà bản đồ bank của phần{' '}
            <a href={href('ch11', '11.7')}>11.7</a> vẽ ra. README có chi tiết.
          </p>
        }
      >
        <p>
          Go further, untested: exercise 4. Vector loads for the passes between global and shared memory (on Metal, declare
          the buffer <code>device const float4*</code>; the CUDA cast from <code>float*</code> to <code>float4*</code> is the
          book's technique and no vendor documents it), and padding for the strided accesses <code>buffer_s[t*F + c]</code>,
          which section <a href={href('ch11', '11.7')}>11.7</a>'s bank map draws. The README has the details.
        </p>
      </Translated>
      <ReferenceSolution lab={l2} tiers={COARSE_TIERS} />

      {/* ------------------------------------------------------------ 11.3 */}
      <h2><Translated vi="Lab 11.3: săn race condition trong scan">Lab 11.3: scan race hunt</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn được giao một phép scan loại trừ theo hàng với một khối mỗi hàng, hàng được duyệt theo từng chunk với một tiền
            tố được mang theo, như llama.cpp, MLX và PyTorch dùng cho các hàng theo batch. Bên trong mỗi chunk, đoạn 4 phần tử
            của mỗi luồng được scan trong thanh ghi, và phép scan khối warp-shuffle của lab 11.1 kết hợp các đoạn, mảng tổng
            của warp được dùng lại từ chunk này sang chunk khác. Phiên bản của lab này còn đưa phần của mỗi warp qua bộ nhớ chia
            sẻ để các lần nạp được hợp nhất, như sách đưa cả một đoạn qua đó; việc đưa qua theo từng warp là của khóa học này,
            còn llama.cpp và MLX hoàn toàn không làm vậy, mỗi luồng tự nạp đoạn của mình trực tiếp. Nó biên dịch được, và nó có
            ba lỗi độc lập, cùng ba lỗi đó trong phiên bản CUDA và Metal. Hãy tìm và sửa cả ba. Kernel đã sửa là phép scan mà
            lab 11.6 xây dựng tiếp.
          </p>
        }
      >
        <p>
          You are handed a row-wise exclusive scan with one block per row, the row walked in chunks with a carried prefix,
          as llama.cpp, MLX and PyTorch ship for batched rows. Inside each chunk, each thread's run of 4 is scanned in
          registers, and the warp-shuffle block scan of lab 11.1 combines the runs, its array of warp totals reused from
          chunk to chunk. This lab's version also stages each warp's slice through shared memory so the loads are
          coalesced, as the book stages a whole segment; the per-warp staging is ours, and llama.cpp and MLX do not stage
          at all, each thread loading its own run directly. It compiles, and it has
          three independent defects, the same three in the CUDA and the Metal version. Find and fix all three. The fixed
          kernel is the scan lab 11.6 builds on.
        </p>
      </Translated>
      <Source section="11.2–11.7" pages="257–259, 264, 268–271" figs={tr('Figs. 11.3, 11.8, 11.9, 11.12, 11.13; the defects are ours', 'Figs. 11.3, 11.8, 11.9, 11.12, 11.13; các lỗi là của khóa học này')} />
      <LabFiles
        lab={l3}
        files={[
          ['kernel.cu', <Translated vi={<><code>ExclusiveRowScanKernel</code>, với ba lỗi; <code>launchExclusiveRowScan</code> bên dưới nó là đúng</>}><><code>ExclusiveRowScanKernel</code>, with the three defects; <code>launchExclusiveRowScan</code> below it is correct</></Translated>],
          ['kernel.metal', <Translated vi={<><code>exclusive_row_scan</code>, với cùng ba lỗi đó</>}><><code>exclusive_row_scan</code>, with the same three</></Translated>],
          ['launch.mm', <Translated vi={<>đúng như được phát hành; không cần thay đổi gì</>}><>correct as shipped; nothing to change</></Translated>],
        ]}
      />
      <Translated
        vi={
          <p>
            Không phải lỗi: lane mask có kiểu và <code>warpSize</code>, các kiểm tra biên mà Fig. 11.12 và 11.13 của sách
            thiếu, mỗi warp đưa phần chunk của riêng nó qua bộ nhớ chia sẻ, và trên Metal là chỉ số lane được truyền vào hàm
            phụ trợ. Driver trộn các hàng toàn số 1, số nguyên từ −3 đến 3 và giá trị phân bố đều trong một batch, và so sánh
            chính xác hai loại đầu: mọi tổng riêng phần đều là số nguyên mà float giữ chính xác, nên một sai khác là lỗi, không
            bao giờ là làm tròn. Trước khi chạy bất cứ thứ gì, hãy đọc kernel và chốt một con số.
          </p>
        }
      >
        <p>
          Not defects: the typed lane mask and <code>warpSize</code>, the bounds checks the book's Figs. 11.12 and 11.13
          lack, each warp staging its own slice of the chunk, and on Metal the lane index passed into the helper. The driver
          mixes rows of all ones, integers from −3 to 3 and uniform values in one batch, and compares the first two
          exactly: every partial sum is a whole number a float holds exactly, so a difference is a defect, never rounding.
          Before you run anything, read the kernel and commit to a number.
        </p>
      </Translated>
      <Predict
        id="ch11/labs/p-lab3-gpu"
        prompt={
          <Translated
            vi={
              <>
                Trước khi bạn thay đổi bất cứ thứ gì. Chỉ chạy trên GPU của bạn, bao nhiêu trong ba lỗi sẽ làm một bài kiểm tra
                thất bại?
              </>
            }
          >
            <>
              Before you change anything. Running only on your GPU, how many of the three defects will make a test fail?
            </>
          </Translated>
        }
        unit={tr('of 3 defects', 'trên 3 lỗi')}
      >
        {(v) => <HuntReveal committed={v} />}
      </Predict>
      <Translated vi={<p>Sau đó hãy săn, theo thứ tự này:</p>}>
        <p>Then hunt, in this order:</p>
      </Translated>
      <ol>
        <Translated
          vi={
            <li>
              Trình giả lập trước: <code>python3 labs/pmpp.py run 11.3 --backend emu</code>. Bộ nhớ chia sẻ chưa ai ghi chứa
              các byte <code>0xCD</code>, khoảng −4.3 × 10<sup>8</sup> khi đọc như float. Một giá trị gấp đúng đôi giá trị lẽ
              ra phải có, hoặc lệch đúng tổng của một đoạn, chỉ ra lỗi số học, không phải lỗi thời điểm.
            </li>
          }
        >
          <li>
            Emulator first: <code>python3 labs/pmpp.py run 11.3 --backend emu</code>. Shared memory nobody has written holds{' '}
            <code>0xCD</code> bytes, about −4.3 × 10<sup>8</sup> as a float. A value exactly double what it should be, or
            off by a whole run's total, points at arithmetic, not timing.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Sửa một thứ và chạy lại. Kích thước khối nào, bao nhiêu chunk, và cột nào sai đầu tiên là bằng chứng cho lỗi tiếp
              theo.
            </li>
          }
        >
          <li>
            Fix one thing and run again. Which block size, how many chunks, and which column goes wrong first are your
            evidence for the next defect.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <code>python3 labs/pmpp.py run 11.3 --backend race</code>. Trình giả lập chỉ giải quyết một rào chắn hay một
              shuffle khi không luồng nào đi xa hơn được, nên một lỗi cần một warp vượt hẳn một quãng so với warp khác sẽ ẩn ở
              đó và chỉ lộ ra ở đây.
            </li>
          }
        >
          <li>
            <code>python3 labs/pmpp.py run 11.3 --backend race</code>. The emulator resolves a barrier or a shuffle only when
            no thread can go further, so a defect that needs one warp to be a whole stretch ahead of another stays hidden
            there and shows only here.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Rồi đến GPU của bạn: <code>python3 labs/pmpp.py run 11.3</code>. Ghi lại những lỗi nó cho bạn thấy và những lỗi
              nó giấu.
            </li>
          }
        >
          <li>
            Then your GPU: <code>python3 labs/pmpp.py run 11.3</code>. Note which defects it showed you and which it hid.
          </li>
        </Translated>
      </ol>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A10; infra-facts C4; ch11-facts D4)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              NVIDIA yêu cầu dùng <code>__syncwarp()</code> thay vì dựa vào lockstep trong một warp;{' '}
              <code>simdgroup_barrier</code> của Metal là "an execution and memory barrier". Phép scan MPS của PyTorch có một
              rào chắn mà chú thích nói rằng không có nó thì "long (multi-iteration) scans race".
            </p>
          }
        >
          <p>
            NVIDIA asks for <code>__syncwarp()</code> rather than lockstep within a warp; Metal's{' '}
            <code>simdgroup_barrier</code> is "an execution and memory barrier". PyTorch's MPS scan has a barrier whose
            comment says that without it "long (multi-iteration) scans race".
          </p>
        </Translated>
      </Today>
      <RaceLine lab={l3} />
      <LabResults
        lab={l3}
        noBench={tr(
          'This lab has no benchmark. The result is the line above: every test passing on the emulators and on a device. As shipped, the kernel is meant to fail.',
          'Lab này không có benchmark. Kết quả là dòng ở trên: mọi bài kiểm tra đều qua trên các trình giả lập và trên một thiết bị. Khi được phát hành, kernel được cố ý để thất bại.',
        )}
      />
      <ReferenceSolution lab={l3} tiers={HUNT_TIERS} />

      {/* ------------------------------------------------------------ 11.4 */}
      <h2><Translated vi="Lab 11.4: phép scan độ dài bất kỳ, theo phân cấp">Lab 11.4: a scan of any length, hierarchically</Translated></h2>
      <Translated
        vi={
          <p>
            Lab 11.1 và 11.2 scan đoạn của mỗi khối riêng rẽ. Ở đây bạn nối các đoạn thành một phép scan bao gồm trên cả n
            phần tử, với hai thiết kế ba kernel của phần 11.9. Scan-scan-add (Fig. 11.14) scan mọi đoạn và ghi tổng của nó,
            scan các tổng, rồi cộng các tổng đã scan trở lại. Reduce-scan-scan (Fig. 11.15) chỉ rút gọn mọi đoạn thành tổng của
            nó, scan các tổng, rồi scan mỗi đoạn với tổng trước đó được cộng vào phần tử đầu tiên. Sách đếm 16 + 8/S và 12 +
            12/S byte mỗi phần tử, với S phần tử mỗi đoạn.
          </p>
        }
      >
        <p>
          Labs 11.1 and 11.2 scan each block's segment on its own. Here you join the segments into one inclusive scan of
          all n elements, with the two three-kernel designs of section 11.9. Scan-scan-add (Fig. 11.14) scans every
          segment and writes its total, scans the totals, and adds the scanned totals back. Reduce-scan-scan (Fig. 11.15)
          only reduces every segment to its total, scans the totals, then scans each segment with the earlier total added
          to its first element. The book counts 16 + 8/S and 12 + 12/S bytes per element, with S elements per segment.
        </p>
      </Translated>
      <Source section="11.9" pages="273–276" figs="Figs. 11.14, 11.15" />
      <Translated
        vi={
          <p>
            Sách không in mã cho cả hai, và kernel thứ hai của nó scan các tổng "in a single block" (p. 273), giới hạn n ở
            S × S. Ở đây bước thứ hai lại là toàn bộ thiết kế, áp dụng lên các tổng cho đến khi chúng vừa một đoạn, nên độ dài
            nào cũng được. Trên Metal, ba kernel là nhiều lần dispatch trong một compute encoder, vốn chạy chúng lần lượt và
            cho mỗi lần thấy những gì lần trước đã ghi; các threadgroup không bao giờ trao đổi với nhau bên trong một lần
            dispatch.
          </p>
        }
      >
        <p>
          The book prints no code for either, and its second kernel scans the totals "in a single block" (p. 273), which
          caps n at S × S. Here the second step is the whole design again, applied to the totals until they fit one
          segment, so any length works. On Metal the three kernels are several dispatches in one compute encoder, which
          runs them one after another and makes each see what the last wrote; threadgroups never talk to each other inside
          a dispatch.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md C9; ch11-facts D1, D4)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Một compute encoder thường "dispatches its compute commands serially", và các buffer device mặc định được theo dõi
              hazard. llama.cpp đưa scan-scan-add ba lần dispatch vào sản phẩm trên Metal; backend MPS của PyTorch dùng ba lượt
              reduce, scan và carry.
            </p>
          }
        >
          <p>
            A plain compute encoder "dispatches its compute commands serially", and device buffers are hazard-tracked by
            default. llama.cpp ships the three-dispatch scan-scan-add on Metal; PyTorch's MPS backend a three-pass reduce,
            scan and carry.
          </p>
        </Translated>
      </Today>
      <LabFiles
        lab={l4}
        files={[
          [
            'kernel.cu',
            <Translated
              vi={
                <>
                  <code>scanSegment</code>, <code>ScanSegmentsKernel</code>, <code>AddBlockSumsKernel</code>,{' '}
                  <code>ReduceSegmentsKernel</code>, <code>ScanWithCarryKernel</code>, <code>scratchFloats</code>,{' '}
                  <code>launchScanScanAdd</code> và <code>launchReduceScanScan</code>
                </>
              }
            >
              <>
                <code>scanSegment</code>, <code>ScanSegmentsKernel</code>, <code>AddBlockSumsKernel</code>,{' '}
                <code>ReduceSegmentsKernel</code>, <code>ScanWithCarryKernel</code>, <code>scratchFloats</code>,{' '}
                <code>launchScanScanAdd</code> and <code>launchReduceScanScan</code>
              </>
            </Translated>,
          ],
          ['kernel.metal', <><code>scan_segment</code>, <code>scan_segments</code>, <code>add_block_sums</code>, <code>reduce_segments</code>, <code>scan_with_carry</code></>],
          ['launch.mm', <><code>scratchFloats</code>, <code>encodeScanScanAdd</code>, <code>encodeReduceScanScan</code></>],
        ]}
      />
      <Translated
        vi={
          <p>
            Có sẵn trong starter: các phép scan warp và khối của lab 11.1 và phép rút gọn warp của lab 10.1; cùng một kernel
            copy trong <code>labs/ch11/shared-g13/global_copy.cu</code> và <code>.metal</code>. Các bài kiểm tra chạy độ dài
            đến 2<sup>24</sup> + 7, bốn mức đệ quy với khối 64, và trên trình giả lập n = 70,001, tức ba mức. Mỗi thiết kế chạy
            hai lần trên cùng các buffer, lần đầu với đầu ra và vùng nháp đầy NaN, nên một tổng bị đọc trước khi được ghi, hoặc
            bị cộng dồn thay vì ghi đè, sẽ thất bại. Lần chạy trên trình giả lập chậm; phần lớn thời gian là trường hợp ba mức
            đó.
          </p>
        }
      >
        <p>
          Provided in the starter: the warp and block scans of lab 11.1 and the warp reduction of lab 10.1; and a copy
          kernel in <code>labs/ch11/shared-g13/global_copy.cu</code> and <code>.metal</code>. The tests run lengths up to 2
          <sup>24</sup> + 7, four levels of recursion at blocks of 64, and on the emulator n = 70,001, which is three. Each
          design runs twice on the same buffers, the first time with output and scratch full of NaN, so a total read before
          it is written, or added to instead of overwritten, fails. The emulator run is slow; most of it is that
          three-level case.
        </p>
      </Translated>
      <RaceLine lab={l4} />
      <Predict
        id="ch11/11.9/p-rss"
        prompt={
          <Translated
            vi={
              <>
                Cùng dự đoán với phần 11.9: chốt nó ở trang nào cũng được. Trên thiết bị của bạn, ở n = 2<sup>26</sup>,
                reduce-scan-scan sẽ chạy nhanh gấp bao nhiêu lần scan-scan-add? Phép đếm byte nói 16/12.
              </>
            }
          >
            <>
              The same prediction as in section 11.9: commit it on either page. How many times faster than scan-scan-add
              will reduce-scan-scan run at n = 2<sup>26</sup> on your device? The byte count says 16/12.
            </>
          </Translated>
        }
        unit="×"
      >
        {(v) => (
          <Reveal
            committed={v}
            measure={rssOverSsaMeasured}
            unit={tr("× scan-scan-add's rate", '× tốc độ của scan-scan-add')}
            reading={
              <Translated
                vi={
                  <>
                    Phép đếm byte là mô hình của lưu lượng, không phải của thời gian: nó bỏ qua các lần khởi chạy giữa các kernel,
                    vài khối ít ỏi của kernel thứ hai và các cache. Benchmark ở đây nhiễu, nên hãy chạy nó hơn một lần trước khi
                    đọc tỷ lệ; nếu các lần chạy không khớp nhau, độ chênh đó là một phần của những gì bạn đã đo.
                  </>
                }
              >
                <>
                  The byte count is a model of traffic, not of time: it leaves out the launches between kernels, the second
                  kernel's few blocks and the caches. The benchmark is noisy here, so run it more than once before you
                  read the ratio; if the runs disagree, that spread is part of what you measured.
                </>
              </Translated>
            }
            missing={BENCH_MISSING('11.4')}
          />
        )}
      </Predict>
      <LabResults lab={l4} filter={(r) => r.name === 'global_scan'} cols={globalCols()} />
      <Translated
        vi={
          <p>
            <code>Book B/element</code> là số byte bộ nhớ toàn cục mỗi phần tử theo cách đếm của sách: 8 cho copy, 16 + 8/S và
            12 + 12/S cho các thiết kế. Đó là mô hình của lưu lượng. Tỷ lệ so với copy so sánh thời gian: thời gian của copy
            chia cho thời gian của thiết kế.
          </p>
        }
      >
        <p>
          <code>Book B/element</code> is the book's count of global-memory bytes per element: 8 for the copy, 16 + 8/S and 12
          + 12/S for the designs. It is a model of the traffic. The share of copy compares times: the copy's time over the
          design's.
        </p>
      </Translated>
      <ReferenceSolution lab={l4} tiers={GLOBAL_TIERS} />

      {/* ------------------------------------------------------------ 11.5 */}
      <h2><Translated vi="Lab 11.5: phép scan single-pass với lookback (họ CUDA, không bắt buộc)">Lab 11.5: single-pass scan with lookback (CUDA family, optional)</Translated></h2>
      <Translated
        vi={
          <p>
            Một kernel làm toàn bộ phép scan. Mỗi khối scan đoạn của nó; luồng cuối của nó, luồng dẫn đầu, chờ cho đến khi khối
            đứng trước đã công bố tổng của mọi thứ tính đến cuối nó, công bố điều tương tự cho khối của mình, và khối cộng tổng
            trước đó vào các phần tử của mình. Đầu vào được nạp một lần và đầu ra được ghi một lần: 8 byte mỗi phần tử thay vì
            16 hay 12 của lab 11.4. Bạn viết chỉ số khối động của Fig. 11.18, giao thức cờ của Fig. 11.17 và hàm khởi chạy đặt
            bộ đếm và các cờ về 0 trước mỗi lần khởi chạy. Phép scan mức khối bên dưới được cung cấp sẵn.
          </p>
        }
      >
        <p>
          One kernel does the whole scan. Each block scans its segment; its last thread, the leader, waits until the block
          before it has published the sum of everything up to its end, publishes the same for its own block, and the block
          adds the earlier sum to its elements. The input is loaded once and the output stored once: 8 bytes per element
          instead of lab 11.4's 16 or 12. You write the dynamic block index of Fig. 11.18, the flag protocol of Fig. 11.17
          and the launcher that zeroes the counter and the flags before every launch. The block-level scan underneath is
          provided.
        </p>
      </Translated>
      <Source section="11.9" pages="276–282" figs="Figs. 11.16(b), 11.17, 11.18" />
      <Translated vi={<p>Những chỗ lab khác với trang sách:</p>}>
        <p>Where the lab departs from the page:</p>
      </Translated>
      <ul>
        <Translated
          vi={
            <li>
              Danh sách tham số của Fig. 11.18 thiếu dấu phẩy trước <code>unsigned int N</code> (p. 281), và kernel không bao
              giờ dùng <code>N</code>. Ở đây mọi lần nạp và ghi đều được kiểm tra với nó.
            </li>
          }
        >
          <li>
            Fig. 11.18's parameter list is missing the comma before <code>unsigned int N</code> (p. 281), and the kernel
            never uses <code>N</code>. Every load and store here is checked against it.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Cờ được đọc bằng <code>load(cuda::memory_order_acquire)</code> và được đặt bằng{' '}
              <code>store(1, cuda::memory_order_release)</code>, dạng của chính ví dụ message-passing của NVIDIA;{' '}
              <code>fetch_add(0, …)</code> của sách là một thao tác read-modify-write ở chỗ chỉ cần một lần đọc.
            </li>
          }
        >
          <li>
            The flag is read with <code>load(cuda::memory_order_acquire)</code> and set with{' '}
            <code>store(1, cuda::memory_order_release)</code>, the form of NVIDIA's own message-passing example; the book's{' '}
            <code>fetch_add(0, …)</code> is a read-modify-write where a read suffices.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Sách ngụ ý các cờ bắt đầu ở 0 và nói bộ đếm được "zero-initialized" (pp. 278, 280). Hàm khởi chạy làm cả hai,
              bằng <code>cudaMemset</code>, trước mỗi lần khởi chạy.
            </li>
          }
        >
          <li>
            The book implies the flags start at zero and says the counter is "zero-initialized" (pp. 278, 280). The launcher
            does both, with <code>cudaMemset</code>, before every launch.
          </li>
        </Translated>
      </ul>
      <Today checked="2026-11-20 (notes/today/ch11-facts.md A5, A6)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Mô hình bộ nhớ của NVIDIA in giao thức này dưới tên "message passing". Mô hình thực thi của nó bảo đảm rằng một
              khối đã thực thi một bước sẽ tiếp tục tiến triển, nên việc chờ một khối đã nhận chỉ số từ bộ đếm sẽ kết thúc;
              chờ <code>blockIdx.x − 1</code> thì không có chỗ dựa như vậy.
            </p>
          }
        >
          <p>
            NVIDIA's memory model prints this protocol as "message passing". Its execution model guarantees that a block
            which has executed a step keeps making progress, so waiting on a block that claimed its index from the counter
            ends; waiting on <code>blockIdx.x − 1</code> has no such backing.
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
                  chỉ số khối động trong <code>SinglePassScanKernel</code>, <code>interBlockScan</code> và{' '}
                  <code>launchSinglePassScan</code>. Không có phiên bản Metal.
                </>
              }
            >
              <>
                the dynamic block index in <code>SinglePassScanKernel</code>, <code>interBlockScan</code> and{' '}
                <code>launchSinglePassScan</code>. There is no Metal version.
              </>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Hãy chạy <code>python3 labs/pmpp.py run 11.5 --backend emu</code>, rồi <code>--backend race</code>. Một khối chờ
            một cờ không ai đặt sẽ chờ mãi, và lần chạy khi đó treo cho đến khi <code>pmpp.py</code> bỏ cuộc sau 15 phút; thay
            vào đó hãy nhấn Ctrl-C. Những gì các trình giả lập thấy và không thấy được ở đây quan trọng hơn ở bất kỳ lab nào
            khác:
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 11.5 --backend emu</code>, then <code>--backend race</code>. A block that waits
          for a flag nobody sets never stops waiting, and the run then hangs until <code>pmpp.py</code> gives up after 15
          minutes; press Ctrl-C instead. What the emulators can and cannot see here matters more than in any other lab:
        </p>
      </Translated>
      <ul>
        <Translated
          vi={
            <li>
              <code>race</code> báo cáo một cờ relaxed: khi acquire hoặc release bị đổi thành relaxed, ThreadSanitizer đã báo
              cáo một race giữa lần đọc <code>partialSums[bid − 1]</code> và lần ghi <code>partialSums[bid]</code> vào lúc lab
              này được viết, còn <code>emu</code> thì qua. Nó cũng báo cáo các cờ mà hàm khởi chạy không đặt về 0.
            </li>
          }
        >
          <li>
            <code>race</code> reports a relaxed flag: with the acquire or the release made relaxed, ThreadSanitizer reported
            a race between the read of <code>partialSums[bid − 1]</code> and the write of <code>partialSums[bid]</code> when
            this lab was written, and <code>emu</code> passed. It also reports flags the launcher did not zero.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <code>emu</code> chạy các khối lần lượt theo thứ tự chỉ số, nên lookback không bao giờ phải chờ ở đó. Cả hai đều
              bỏ qua phạm vi luồng, nên một cờ phạm vi block vẫn qua, dù trên GPU nó là một race.
            </li>
          }
        >
          <li>
            <code>emu</code> runs blocks one after another in index order, so the lookback never waits there. Thread scopes
            are ignored by both, so a block-scope flag passes, though on a GPU it is a race.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Không trình giả lập nào tạo ra được deadlock của việc chờ <code>blockIdx.x − 1</code>. Trình mô phỏng ở phần{' '}
              <a href={href('ch11', '11.9')}>11.9</a> cho thấy nó. Một lần chạy sạch ở đây không chứng minh kernel chạy được
              trên GPU.
            </li>
          }
        >
          <li>
            Neither can produce the deadlock of waiting on <code>blockIdx.x − 1</code>. The simulator in section{' '}
            <a href={href('ch11', '11.9')}>11.9</a> shows it. A clean run here does not prove the kernel runs on a GPU.
          </li>
        </Translated>
      </ul>
      <RaceLine lab={l5} />
      <Predict
        id="ch11/labs/p-lab5-single"
        prompt={
          <Translated
            vi={
              <>
                Trước khi bạn chạy benchmark trên một thiết bị CUDA hoặc HIP. Ở n = 2<sup>26</sup>, kernel single-pass sẽ chạy
                nhanh gấp bao nhiêu lần scan-scan-add của lab 11.4? Phép đếm byte nói 2.
              </>
            }
          >
            <>
              Before you run the benchmark on a CUDA or HIP device. How many times faster than lab 11.4's scan-scan-add will
              the single-pass kernel run at n = 2<sup>26</sup>? The byte count says 2.
            </>
          </Translated>
        }
        unit={tr('× scan-scan-add', '× scan-scan-add')}
      >
        {(v) => <SinglePassReveal committed={v} />}
      </Predict>
      <LabResults lab={l5} filter={(r) => r.name === 'global_scan'} cols={globalCols()} />
      <Translated
        vi={
          <p>
            Đi xa hơn trên một thiết bị CUDA, không có bài kiểm tra ở đây: một cơ chế lùi lại bằng <code>__nanosleep</code>{' '}
            trong vòng chờ (chỉ CUDA, nên đặt sau <code>#if defined(PMPP_BACKEND_CUDA)</code>), và bài tập 5, decoupled
            lookback, thứ mà trình mô phỏng của phần 11.9 mô hình hóa.
          </p>
        }
      >
        <p>
          Go further on a CUDA device, untested here: a back-off with <code>__nanosleep</code> in the wait loop (CUDA only,
          so behind <code>#if defined(PMPP_BACKEND_CUDA)</code>), and exercise 5, decoupled lookback, which the section 11.9
          simulator models.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch11-facts.md A7, A8, B4)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Look-back của CUB gọi <code>__nanosleep</code>; HIP không ghi hàm tương đương nào trong tài liệu.{' '}
              <code>DeviceScan</code> của CUB là thiết kế của bài tập 5, và nói rằng nó "typically proceeds at "memcpy" speeds".
            </p>
          }
        >
          <p>
            CUB's look-back calls <code>__nanosleep</code>; HIP documents no counterpart. CUB's <code>DeviceScan</code> is
            exercise 5's design, and says it "typically proceeds at "memcpy" speeds".
          </p>
        </Translated>
      </Today>
      <ReferenceSolution lab={l5} tiers={SINGLE_TIERS} />

      {/* ------------------------------------------------------------ 11.6 */}
      <h2><Translated vi="Lab 11.6: mask giữ lại top-p (mốc dự án M5)">Lab 11.6: top-p keep mask (milestone M5)</Translated></h2>
      <Translated
        vi={
          <p>
            Lấy mẫu top-p giữ lại tập nhỏ nhất các token khả dĩ nhất có tổng xác suất đạt p. Sau khi các xác suất được sắp
            xếp, tập đó là một tiền tố của mỗi hàng, và việc tìm nó là một phép scan. Bạn viết một kernel theo hàng trên{' '}
            <code>[rows × cols]</code> xác suất float32, thứ tự hàng (row-major), batch đứng đầu, mỗi hàng được sắp giảm dần:
            nó ghi <code>massBefore</code>, phép scan loại trừ của hàng; <code>keep</code>, một byte mỗi token; và{' '}
            <code>kept</code>, số lượng số 1 trong mỗi hàng.
          </p>
        }
      >
        <p>
          Top-p sampling keeps the smallest set of most likely tokens whose probability reaches p. After the probabilities
          are sorted, that set is a prefix of each row, and finding it is a scan. You write one row-wise kernel over{' '}
          <code>[rows × cols]</code> float32 probabilities, row-major, batch first, each row sorted in descending order: it
          writes <code>massBefore</code>, the exclusive scan of the row; <code>keep</code>, one byte per token; and{' '}
          <code>kept</code>, the number of ones in each row.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="11.1, 11.6, 11.7" pages="253, 268–271" figs={tr('the exclusive scan, Figs. 11.12 and 11.13', 'phép scan loại trừ, Figs. 11.12 và 11.13')} />
        <Beyond>
          <Translated vi={<p>Lấy mẫu top-p không có trong sách. Quy tắc, mask và phép đếm là của khóa học này, lấy từ các dự án bên dưới.</p>}>
            <p>Top-p sampling is not in the book. The rule, the mask and the count are ours, from the projects below.</p>
          </Translated>
        </Beyond>
      </NoteGroup>
      <Translated vi={<p>Hợp đồng, mà README của lab giải thích từng điều khoản:</p>}>
        <p>The contract, which the lab's README explains clause by clause:</p>
      </Translated>
      <ul>
        <Translated
          vi={
            <li>
              <code>keep[i] = 1</code> đúng khi <code>massBefore[i] &lt; p</code>: token vượt qua p được giữ lại. Đó là phép
              scan loại trừ của phần <a href={href('ch11', '11.1')}>11.1</a>; so sánh bằng phép scan bao gồm sẽ bỏ mất nó.
            </li>
          }
        >
          <li>
            <code>keep[i] = 1</code> exactly when <code>massBefore[i] &lt; p</code>: the token that crosses p is kept. That
            is section <a href={href('ch11', '11.1')}>11.1</a>'s exclusive scan; comparing the inclusive one drops it.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Token 0 luôn được giữ, và <code>p ≥ 1</code> giữ mọi token, vì khối lượng float trước các token cuối có thể làm
              tròn thành 1 hoặc nhỉnh hơn một chút.
            </li>
          }
        >
          <li>
            Token 0 is always kept, and <code>p ≥ 1</code> keeps every token, because the float mass before the last tokens
            can round to 1 or just above.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Khi bằng nhau thì phân định theo vị trí, tức là theo phép sắp xếp. mlx-lm thì giữ mọi token bằng với ngưỡng biên;
              đó là một phương án thay thế được nêu rõ, không phải quy tắc của lab này.
            </li>
          }
        >
          <li>
            Ties are broken by position, that is by the sort. mlx-lm keeps every token tied with the boundary instead; that
            is a stated alternative, not this lab's rule.
          </li>
        </Translated>
      </ul>
      <Today checked="2026-11-20 (notes/today/ch11-facts.md D2, D3, D4)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              llama.cpp giữ token có tổng chạy đạt p và trả về sớm khi <code>p &gt;= 1</code>; transformers và vLLM bỏ một
              token khi tổng tăng dần tính đến nó tối đa là 1 − p; mlx-lm giữ mọi token bằng với ngưỡng.
            </p>
          }
        >
          <p>
            llama.cpp keeps the token whose running sum reaches p and returns early for <code>p &gt;= 1</code>; transformers
            and vLLM drop a token when the ascending sum up to it is at most 1 − p; mlx-lm keeps all tokens tied with the
            threshold.
          </p>
        </Translated>
      </Today>
      <LabFiles
        lab={l6}
        files={[
          ['kernel.cu', <Translated vi={<><code>TopPMaskKernel</code> và <code>launchTopPMask</code></>}><><code>TopPMaskKernel</code> and <code>launchTopPMask</code></></Translated>],
          ['kernel.metal', <><code>top_p_mask</code></>],
          ['launch.mm', <><code>encodeTopPMask</code></>],
        ]}
      />
      <Translated
        vi={
          <p>
            Thiết kế được gợi ý là phép scan theo hàng đã sửa của lab 11.3 với ngưỡng và phép đếm được gộp vào, để các xác suất
            chỉ được đọc một lần. Có sẵn: một baseline giao mỗi hàng cho một luồng, và một kernel copy di chuyển cùng 9 byte mỗi
            token, trong <code>labs/ch11/shared-g12/</code>.
          </p>
        }
      >
        <p>
          The suggested design is lab 11.3's fixed row scan with the threshold and a count fused in, so the probabilities are
          read once. Provided: a baseline that gives each row one thread, and a copy that moves the same 9 bytes per token,
          in <code>labs/ch11/shared-g12/</code>.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Các bài kiểm tra chạy các hàng 1, 4 và 7 với tới 128,256 cột; ba trong số các độ rộng (32,000, 50,257 và 128,256)
            có kích thước cỡ vocabulary. Các hàng xác suất dyadic, có mọi tổng riêng phần chính xác trong float theo mọi thứ
            tự, đặt khối lượng bao gồm của một token đúng bằng p và được so sánh chính xác: chúng bắt được cả "inclusive &lt;
            p" lẫn "massBefore ≤ p". Các hàng softmax của logits ngẫu nhiên được so sánh trong một dải 10<sup>−4</sup> của
            khối lượng, mà README suy ra từ số lần làm tròn; bên trong dải đó, quyết định nào cũng được chấp nhận.
          </p>
        }
      >
        <p>
          The tests run rows of 1, 4 and 7 by up to 128,256 columns; three of the widths (32,000, 50,257 and 128,256) are
          vocabulary-sized. Rows of dyadic probabilities,
          whose every partial sum is exact in float in any order, put a token's inclusive mass exactly on p and are compared
          exactly: they catch both "inclusive &lt; p" and "massBefore ≤ p". Softmax rows of random logits are compared within
          a band of 10<sup>−4</sup> of the mass, which the README derives from the number of roundings; inside the band
          either decision is accepted.
        </p>
      </Translated>
      <RaceLine lab={l6} />
      <LabPrediction
        id="ch11/labs/p-lab6-copy"
        prompt={
          <Translated
            vi={
              <>
                Trước khi chạy benchmark. Ở 64 hàng gồm 128,256 token, kernel của bạn sẽ đạt bao nhiêu phần băng thông copy của
                bạn, tính theo phần trăm?
              </>
            }
          >
            <>
              Before you run the benchmark. At 64 rows of 128,256 tokens, what share of your copy bandwidth will your kernel
              reach, in percent?
            </>
          </Translated>
        }
        unit={tr('% of copy', '% của copy')}
        measure={toppShare}
        missing={BENCH_MISSING('11.6')}
        reading={
          <Translated
            vi={
              <>
                Kernel copy chạy một luồng mỗi token; thiết kế được gợi ý chạy một khối mỗi hàng, 64 khối mỗi khối duyệt lần
                lượt 126 chunk. Phần <a href={href('ch11', '11.8')}>11.8</a> gọi một kernel bị giới hạn bởi bộ nhớ là được tối
                ưu tốt khi trên 80% mà không có truy cập thừa (p. 273). Tỷ lệ 1 hàng trong phần chi tiết là những gì một khối mỗi
                hàng làm được ở batch 1.
              </>
            }
          >
            <>
              The copy runs one thread per token; the suggested design runs one block per row, 64 blocks each walking 126
              chunks one after another. Section <a href={href('ch11', '11.8')}>11.8</a> calls a memory-bound kernel well
              optimised above 80% with no redundant accesses (p. 273). The 1-row share in the detail is what one block per
              row does at batch 1.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l6}
        filter={(r) => r.name === 'top_p_mask'}
        cols={[
          { head: tr('Measurement', 'Phép đo'), cell: named(KERNEL_6) },
          { head: tr('Rows × cols', 'Hàng × cột'), cell: (r) => `${String(r.rows)} × ${num(r.cols) != null ? count(num(r.cols)!) : ''}` },
          msCol,
          numCol('GB/s', 'gb_per_s'),
          shareCol,
          { head: tr('Over baseline', 'So với baseline'), num: true, cell: (r) => (num(r.speedup_vs_naive) != null ? `${fmt(num(r.speedup_vs_naive)!)}×` : '') },
        ]}
      />
      <Translated
        vi={
          <p>
            GB/s đếm 9 byte mỗi token: đọc 4, ghi 4 + 1. Ở 1 hoặc 8 hàng, các mảng chỉ vài megabyte và một lần chạy có đo thời
            gian lặp lại lời gọi trên cùng các buffer, nên nếu tốc độ của copy ở đó cao hơn ở 64 hàng, các lần lặp được phục vụ
            từ thứ gì đó nhanh hơn bộ nhớ chính. Hãy đọc câu hỏi băng thông ở 64 × 128,256 và 256 × 32,000.
          </p>
        }
      >
        <p>
          GB/s counts 9 bytes per token: 4 read, 4 + 1 written. At 1 or 8 rows the arrays are a few megabytes and a timed run
          repeats the call on the same buffers, so if the copy's rate there is higher than at 64 rows, the repeats are served
          from something faster than main memory. Read the bandwidth question at 64 × 128,256 and 256 × 32,000.
        </p>
      </Translated>
      <h3><Translated vi="Mốc dự án M5, mảnh top-p">Project milestone M5, the top-p piece</Translated></h3>
      <Translated
        vi={
          <p>
            Lab 11.6 là mảnh top-p của mốc dự án M5 của dự án xuyên suốt, lấy mẫu top-k và top-p trên GPU, kèm phạt lặp lại
            (repetition penalty); số đếm token của lab 9.3 cho phần phạt là mảnh đầu tiên. Chương 12 nén gọn các token được giữ
            lại thành một danh sách ngắn, chương 14 sắp xếp các logits mà lab này nhận được ở dạng đã sắp xếp, và module X1 rút
            mẫu: khúc xúc xích của phần 11.1, với một số ngẫu nhiên phân bố đều thay cho con dao.
          </p>
        }
      >
        <p>
          Lab 11.6 is the top-p piece of milestone M5 of the through-line project, top-k and top-p sampling on the GPU, with repetition
          penalties; lab 9.3's token counts for the penalties were the first piece. Chapter
          12 compacts the kept tokens into a short list, chapter 14 sorts the logits that this lab receives already sorted,
          and module X1 draws the sample: the sausage of section 11.1, with a uniform random number in place of the knife.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Dự án xuyên suốt, <code>nanoinfer</code>, là của khóa học này: một đường suy luận GPU nhỏ cho một transformer chỉ
              có decoder, lớn dần qua từng mốc.
            </p>
          }
        >
          <p>
            The through-line project, <code>nanoinfer</code>, is ours: one small GPU inference path for a decoder-only
            transformer, grown a milestone at a time.
          </p>
        </Translated>
      </Aside>
      <MilestoneStatus />
      <ReferenceSolution lab={l6} tiers={TOPP_TIERS} />
    </LabsPage>
  );
}
