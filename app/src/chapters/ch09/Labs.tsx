import type { ReactNode } from 'react';
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
  Reveal,
  RunningLabs,
  VerificationStatus,
  defineLabs,
  measure,
  msCol,
  numCol,
  useLabs,
  type Lab,
} from '../../components/LabKit';
import { fmt, labStatus, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 9's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch09', [
  { num: '9.1', dir: 'lab1_racehunt', name: 'race hunt (optional)', vi: { name: 'săn race (tùy chọn)' } },
  { num: '9.2', dir: 'lab2_privatised', name: 'privatised, coarsened, run-merged', vi: { name: 'riêng tư hóa, gộp việc, gộp chuỗi' } },
  { num: '9.3', dir: 'lab3_tokencounts', name: 'token counts (M5 piece)', vi: { name: 'đếm token (một phần của M5)' } },
]);
const [l1, l2, l3] = LABS;

// Identifier colours for the reference solutions. Only names that mean one thing in all
// three files of a lab.
const tiers = (global: string[], shared: string[] = [], reg: string[] = []): Record<string, Tier> => ({
  ...Object.fromEntries(global.map((n) => [n, 'global' as Tier])),
  ...Object.fromEntries(shared.map((n) => [n, 'shared' as Tier])),
  ...Object.fromEntries(reg.map((n) => [n, 'reg' as Tier])),
});
const HUNT_TIERS = tiers(['image', 'bins', 'binsPool', 'binsPriv']);
const SHARED_TIERS = tiers(['image', 'bins'], ['bins_s'], ['bin_r', 'bNext', 'segment']);
const TOKEN_TIERS = tiers(['tokens', 'counts', 'logits', 'newTokens', 'rowTokens', 'rowCounts']);

// ---------------------------------------------------------------- reading the reports
const gps = (v: number) => `${fmt(v)} Gpixel/s`; // i18n-ignore: number and unit
const ms = (s: number) => `${fmt(s * 1000)} ms`; // i18n-ignore: number and unit
const count = (v: number) => v.toLocaleString('en-US');
const g = (r: BenchRow | undefined) => num(r?.gpixels_per_s);

/** Lab 9.1: the basic kernel, uniform bytes over a constant image (section 9.3's prediction). */
const skewSlowdown = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const uni = g(s.rows.find((r) => r.name === 'basic' && r.data === 'uniform'));
    const con = g(s.rows.find((r) => r.name === 'basic' && r.data === 'constant'));
    if (!uni || !con) return null;
    return {
      value: uni / con,
      detail: tr(`${gps(uni)} on uniform bytes, ${gps(con)} on the constant image`, `${gps(uni)} trên các byte phân bố đều, ${gps(con)} trên ảnh hằng`),
    };
  });

/** Lab 9.1: on the constant image, the pool kernel over the basic one. */
const poolGap = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const basic = g(s.rows.find((r) => r.name === 'basic' && r.data === 'constant'));
    const pool = g(s.rows.find((r) => r.name === 'pool' && r.data === 'constant'));
    if (!basic || !pool) return null;
    const pu = g(s.rows.find((r) => r.name === 'pool' && r.data === 'uniform'));
    return {
      value: pool / basic,
      detail:
        tr(
          `${gps(pool)} for the pool kernel and ${gps(basic)} for the basic kernel on the constant image`,
          `${gps(pool)} cho kernel vùng chung và ${gps(basic)} cho kernel cơ bản trên ảnh hằng`,
        ) +
        (pu ? tr(`; the pool kernel ran at ${gps(pu)} on uniform bytes`, `; kernel vùng chung chạy ở ${gps(pu)} trên các byte phân bố đều`) : ''),
    };
  });

const lab2 = (rows: BenchRow[], name: string, data: string, coarse: number) =>
  rows.find((r) => r.name === name && r.data === data && num(r.coarse) === coarse);

/** Lab 9.2: shared (factor 16) over basic, constant image (section 9.4's prediction). */
const sharedSpeedup = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const basic = g(lab2(s.rows, 'basic', 'constant', 1));
    const shared = g(lab2(s.rows, 'shared', 'constant', 16));
    if (!basic || !shared) return null;
    return {
      value: shared / basic,
      detail: tr(
        `${gps(shared)} for the shared-memory kernel at factor 16, ${gps(basic)} for the basic kernel, constant image`,
        `${gps(shared)} cho kernel bộ nhớ chia sẻ ở hệ số 16, ${gps(basic)} cho kernel cơ bản, ảnh hằng`,
      ),
    };
  });

/** Lab 9.2: interleaved over contiguous at factor 16, uniform bytes (section 9.5's prediction). */
const interleavedGain = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const inter = num(lab2(s.rows, 'shared', 'uniform', 16)?.seconds);
    const contig = num(lab2(s.rows, 'contiguous', 'uniform', 16)?.seconds);
    if (!inter || !contig) return null;
    return {
      value: contig / inter,
      detail: tr(
        `${ms(inter)} interleaved, ${ms(contig)} contiguous, factor 16, uniform bytes`,
        `${ms(inter)} xen kẽ, ${ms(contig)} liền kề, hệ số 16, các byte phân bố đều`,
      ),
    };
  });

/** Lab 9.2: runs over shared at factor 16, on the sky; uniform bytes alongside. */
const runsOnSky = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const runs = g(lab2(s.rows, 'runs', 'sky', 16));
    const shared = g(lab2(s.rows, 'shared', 'sky', 16));
    if (!runs || !shared) return null;
    const ru = g(lab2(s.rows, 'runs', 'uniform', 16));
    const su = g(lab2(s.rows, 'shared', 'uniform', 16));
    return {
      value: runs / shared,
      detail:
        tr(`${gps(runs)} with run merging and ${gps(shared)} without, factor 16, sky`, `${gps(runs)} khi gộp chuỗi và ${gps(shared)} khi không, hệ số 16, bầu trời`) +
        (ru && su
          ? tr(
              `; on uniform bytes the ratio is ${fmt(ru / su)} (${gps(ru)} against ${gps(su)})`,
              `; trên các byte phân bố đều tỉ lệ là ${fmt(ru / su)} (${gps(ru)} so với ${gps(su)})`,
            )
          : ''),
    };
  });

const tokenRow = (rows: BenchRow[], name: string, data: string) => rows.find((r) => r.name === name && r.data === data);
const shape = (r: BenchRow) =>
  tr(
    `batch ${String(r.batch)}, ${count(num(r.len) ?? 0)} tokens per row, vocabulary ${count(num(r.vocab) ?? 0)}`,
    `batch ${String(r.batch)}, ${count(num(r.len) ?? 0)} token mỗi hàng, bộ từ vựng ${count(num(r.vocab) ?? 0)}`,
  );

/** Lab 9.3: the provided pool over direct atomics, in time. */
const poolOverCount = (reports: Reports) =>
  measure(reports, l3, (s) => {
    const c = tokenRow(s.rows, 'count', 'uniform');
    const p = tokenRow(s.rows, 'count_pool', 'uniform');
    const cs = num(c?.seconds);
    const ps = num(p?.seconds);
    if (!c || !cs || !ps) return null;
    const cz = num(tokenRow(s.rows, 'count', 'zipf')?.seconds);
    const pz = num(tokenRow(s.rows, 'count_pool', 'zipf')?.seconds);
    return {
      value: ps / cs,
      detail:
        tr(
          `${ms(ps)} for count_pool, ${ms(cs)} for direct atomics, uniform ids, ${shape(c)}`,
          `${ms(ps)} cho count_pool, ${ms(cs)} cho phép toán nguyên tử trực tiếp, id phân bố đều, ${shape(c)}`,
        ) +
        (num(p?.blocks_per_row) ? tr(`, ${String(p!.blocks_per_row)} pool blocks per row`, `, ${String(p!.blocks_per_row)} khối vùng chung mỗi hàng`) : '') +
        (cz && pz ? tr(`; Zipf-like ids: ${fmt(pz / cz)}`, `; id kiểu Zipf: ${fmt(pz / cz)}`) : ''),
    };
  });

/** Lab 9.3: recounting the context over one decode step. */
const recountOverAppend = (reports: Reports) =>
  measure(reports, l3, (s) => {
    const c = tokenRow(s.rows, 'count', 'uniform');
    const cs = num(c?.seconds);
    const as = num(tokenRow(s.rows, 'append', 'uniform')?.seconds);
    if (!c || !cs || !as) return null;
    return {
      value: cs / as,
      detail: tr(
        `${ms(cs)} to count the context, ${ms(as)} for one append step, uniform ids, ${shape(c)}`,
        `${ms(cs)} để đếm ngữ cảnh, ${ms(as)} cho một bước append, id phân bố đều, ${shape(c)}`,
      ),
    };
  });

// ---------------------------------------------------------------- the race flavour, which the status tables do not show
type FullResult = RunResult & { tsan?: unknown[] };

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
  const races = wRaces
    ? tr(`, ThreadSanitizer reported ${wRaces} race${wRaces === 1 ? '' : 's'}`, `, ThreadSanitizer báo ${wRaces} race`)
    : '';
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
        <Translated
          vi={
            <>
              Trên backend race của máy này, lời giải tham chiếu{' '}
              {s === 'pass'
                ? `qua cả ${n} bài kiểm tra mà không có race nào được báo: emulator-checked trên backend race.`
                : s === 'fail'
                  ? 'thất bại: unverified ở đó.'
                  : 'chưa có lần chạy nào.'}{' '}
              Bản làm của bạn ở đó:{' '}
              {!work.length
                ? 'chưa có lần chạy nào.'
                : ws === 'none'
                  ? `chưa bắt đầu (${wTests.filter((t) => t.pass).length} trên ${wTests.length} bài kiểm tra qua khi mới phát hành${races}).`
                  : ws === 'pass'
                    ? `qua, ${wTests.length} trên ${wTests.length} bài kiểm tra, không có race nào được báo.`
                    : `chưa qua: ${wTests.filter((t) => t.pass).length} trên ${wTests.length} bài kiểm tra${races}.`}
            </>
          }
        >
          <>
            On the race backend of this machine, the reference solution{' '}
            {s === 'pass'
              ? `passed all ${n} tests with no race reported: emulator-checked on the race backend.`
              : s === 'fail'
                ? 'failed: unverified there.'
                : 'has no run yet.'}{' '}
            Your work there:{' '}
            {!work.length
              ? 'no run yet.'
              : ws === 'none'
                ? `not started (${wTests.filter((t) => t.pass).length} of ${wTests.length} tests pass as shipped${races}).`
                : ws === 'pass'
                  ? `passes, ${wTests.length} of ${wTests.length} tests, no race reported.`
                  : `not passing: ${wTests.filter((t) => t.pass).length} of ${wTests.length} tests${races}.`}
          </>
        </Translated>
      )}
    </p>
  );
}

// ---------------------------------------------------------------- milestone M5
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
                `Your own lab 9.3 passes on ${done.map(name).join(' and ')} on this machine: the counting piece of M5 is in place there.`,
                `Lab 9.3 của chính bạn qua trên ${done.map(name).join(' và ')} ở máy này: phần đếm của M5 đã sẵn sàng ở đó.`,
              )
            : tr('Your own lab 9.3 does not pass on any backend of this machine yet.', 'Lab 9.3 của chính bạn chưa qua trên backend nào của máy này.')}
    </p>
  );
}

// ---------------------------------------------------------------- results-table columns
const IMAGE: Record<string, string> = {
  get uniform() {
    return tr('uniform bytes', 'byte phân bố đều');
  },
  get constant() {
    return tr('constant', 'hằng');
  },
  get sky() {
    return tr('sky', 'bầu trời');
  },
};
const image = (r: BenchRow) => IMAGE[String(r.data)] ?? String(r.data ?? '');
const pixels = {
  get head() {
    return tr('Pixels', 'Số điểm ảnh');
  },
  num: true,
  cell: (r: BenchRow) => (num(r.pixels) != null ? count(num(r.pixels)!) : ''),
};
const LAB2_NAME: Record<string, string> = {
  get basic() {
    return tr('basic (Fig. 9.6, provided)', 'basic (Hình 9.6, được cung cấp)');
  },
  get contiguous() {
    return tr('contiguous (Fig. 9.12, provided)', 'contiguous (Hình 9.12, được cung cấp)');
  },
  get shared() {
    return tr('shared (Fig. 9.14)', 'shared (Hình 9.14)');
  },
  get runs() {
    return tr('runs (Fig. 9.15)', 'runs (Hình 9.15)');
  },
};
const LAB3_NAME: Record<string, string> = {
  count: 'count',
  get count_pool() {
    return tr('count_pool (provided)', 'count_pool (được cung cấp)');
  },
  get append() {
    return tr('append (one step)', 'append (một bước)');
  },
  penalties: 'penalties',
};
/** "5.62 G tokens/s" for a rate in items per second. */
const perSecond = (v: number, what: string) =>
  v >= 1e9 ? `${fmt(v / 1e9)} G ${what}/s` : v >= 1e6 ? `${fmt(v / 1e6)} M ${what}/s` : v >= 1e3 ? `${fmt(v / 1e3)} k ${what}/s` : `${fmt(v)} ${what}/s`;
const rate = (r: BenchRow) => {
  const t = num(r.tokens_per_s);
  const e = num(r.elements_per_s);
  const text = t != null ? perSecond(t, tr('tokens', 'token')) : e != null ? perSecond(e, 'logits') : '';
  return <span style={{ whiteSpace: 'nowrap' }}>{text}</span>;
};
const nowrap = (t: string) => <span style={{ whiteSpace: 'nowrap' }}>{t}</span>;

const BENCH_MISSING = (lab: string) => (
  <Translated
    vi={
      <>
        Khi lab {lab} qua, chạy <code>python3 labs/pmpp.py run {lab} --bench</code> trên một máy có GPU, rồi tải lại trang.
        Để đo lời giải tham chiếu trước, thêm <code>--solution</code>.
      </>
    }
  >
    <>
      Once lab {lab} passes, run <code>python3 labs/pmpp.py run {lab} --bench</code> on a machine with a GPU, then reload.
      To measure the reference solution first, add <code>--solution</code>.
    </>
  </Translated>
);

/** A prediction shared with a section: the same progress id, committed on either page. */
function SharedPrediction({
  id,
  prompt,
  unit,
  read,
  lab,
  reading,
}: {
  id: string;
  prompt: ReactNode;
  unit: string;
  read: (r: Reports) => ReturnType<typeof measure>;
  lab: string;
  reading: ReactNode;
}) {
  return (
    <Predict id={id} prompt={prompt} unit={unit}>
      {(v) => <Reveal committed={v} measure={read} unit={unit} missing={BENCH_MISSING(lab)} reading={reading} />}
    </Predict>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS}>
      <LabsIntro>
        <Translated
          vi={
            <>
              Hai lab và một lab tùy chọn thứ ba, mỗi lab trong một thư mục riêng dưới <code>labs/ch09/</code>. Lab 9.2 viết các
              kernel nhanh của chương, riêng tư hóa trong bộ nhớ chia sẻ, gộp việc và gộp chuỗi, và đo thời gian chúng so với kernel cơ
              bản trên các ảnh lệch. Lab 9.3 đếm token id trên cả một bộ từ vựng cho phạt lặp (repetition penalty), phần
              đếm của mốc dự án M5. Lab 9.1 là tùy chọn: một cuộc săn ba race trong hai kernel histogram biên dịch được và
              thường cho số đếm đúng. Cuộc săn lỗi của mục 9.4 bao quát cùng nội dung trong trình duyệt; lab bổ sung các công cụ
              thật và đo dự đoán về độ lệch của mục 9.3.
            </>
          }
        >
          <>
            Two labs and an optional third, each in its own directory under <code>labs/ch09/</code>. Lab 9.2 writes the
            chapter's fast kernels, privatised in shared memory, coarsened and run-merged, and times them against the basic
            one on skewed images. Lab 9.3 counts token ids over a whole vocabulary for repetition penalties, the counting
            piece of milestone M5. Lab 9.1 is optional: a hunt for three races in two histogram kernels that compile and
            often give the right counts. Section 9.4's bug hunt covers the same ground in the browser; the lab adds the real
            tools and times the skew prediction of section 9.3.
          </>
        </Translated>
      </LabsIntro>
      <RunningLabs
        example="9.1"
        alsoShownIn={
          <Translated
            vi={
              <>
                các mục <a href={href('ch09', '9.3')}>9.3</a>, <a href={href('ch09', '9.4')}>9.4</a> và{' '}
                <a href={href('ch09', '9.5')}>9.5</a>
              </>
            }
          >
            <>
              sections <a href={href('ch09', '9.3')}>9.3</a>, <a href={href('ch09', '9.4')}>9.4</a> and{' '}
              <a href={href('ch09', '9.5')}>9.5</a>
            </>
          </Translated>
        }
      />
      <BackendsAndEmulator />
      <Translated
        vi={
          <p>
            Trình giả lập có một biến thể thứ hai, <code>race</code>, và cả ba lab ở đây đều được build cho nó. Nó chạy các
            luồng được giả lập đồng thời dưới ThreadSanitizer và báo cáo hai luồng chạm vào cùng vùng nhớ mà
            không có gì sắp thứ tự chúng, kèm cả hai dòng mã nguồn, kể cả trong một lần chạy mà số đếm ra đúng. Nó là backend duy nhất
            ở đây thấy được một phép toán nguyên tử bị thiếu: <code>emu</code> không bao giờ ngắt một luồng giữa lần đọc và lần ghi của nó,
            nên một lệnh <code>++bins[b]</code> thường ở đó luôn đúng. Gọi nó bằng <code>--backend race</code>;{' '}
            <code>python3 labs/pmpp.py doctor</code> cho biết trình biên dịch của bạn có build được nó không. Các bảng trên trang này chỉ hiển thị
            biến thể thứ nhất; một dòng dưới mỗi lab cho kết quả race.
          </p>
        }
      >
        <p>
          The emulator has a second flavour, <code>race</code>, and all three labs here are built for it. It runs the
          emulated threads concurrently under ThreadSanitizer and reports two threads that touch the same memory with
          nothing ordering them, with both source lines, even in a run whose counts came out right. It is the only backend
          here that can see a missing atomic: <code>emu</code> never interrupts a thread between its read and its write,
          so a plain <code>++bins[b]</code> is always right there. Ask for it with <code>--backend race</code>;{' '}
          <code>python3 labs/pmpp.py doctor</code> says whether your compiler can build it. The tables on this page show the
          first flavour only; a line under each lab gives the race results.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch09-facts.md A10, B6, C5, E1)', '2026-11-20 (notes/today/ch09-facts.md, mục A10, B6, C5, E1)')}>
        <Translated
          vi={
            <p>
              <code>compute-sanitizer --tool racecheck</code> của NVIDIA chỉ kiểm tra bộ nhớ chia sẻ, nên không thể thấy một race trên
              các bin toàn cục. AMD không ghi trong tài liệu trình phát hiện race nào, và Shader Validation của Metal không liệt kê kiểm tra race nào. ThreadSanitizer
              chỉ xem xét những lịch chạy đã xảy ra.
            </p>
          }
        >
          <p>
            NVIDIA's <code>compute-sanitizer --tool racecheck</code> checks shared memory only, so it cannot see a race on
            global bins. AMD documents no race detector, and Metal's Shader Validation lists no race check. ThreadSanitizer
            examines only the schedules that happened.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Không trình giả lập nào cho phạm vi hay thứ tự bộ nhớ tác dụng gì (<code>labs/README.md</code>), và Metal hoàn toàn không có phạm vi trên
            thao tác đọc-sửa-ghi. Một bước ghi vào các bin công khai viết với <code>cuda::thread_scope_block</code> qua
            mọi bài kiểm tra trên mọi backend mà một máy Mac chạy được, và vẫn là một data race theo mô hình bộ nhớ của CUDA, khiến nó
            là hành vi không xác định (notes/today/ch09-facts.md A2, C1). Hãy kiểm tra mọi phạm vi bằng cách đọc mã.
          </p>
        }
      >
        <p>
          Neither emulator gives scopes or memory orders any effect (<code>labs/README.md</code>), and Metal has no scope on
          a read-modify-write at all. A commit to the public bins written with <code>cuda::thread_scope_block</code> passes
          every test on every backend a Mac can run, and is still a data race under CUDA's memory model, which makes it
          undefined behaviour (notes/today/ch09-facts.md A2, C1). Check every scope by reading it.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Trên HIP, <code>&lt;cuda/atomic&gt;</code> đến từ libhipcxx, một phần của ROCm từ 10.1. Bản build HIP của khóa học
            liên kết <code>libhipcxx::libhipcxx</code> khi CMake tìm thấy nó (<code>labs/cmake/Backends.cmake</code>); khi không
            tìm thấy, <code>doctor</code> sẽ báo, và cả ba lab sẽ không build được ở đó. Chưa ai thử: không có máy ROCm
            nào đằng sau khóa học này, nên điều này là unverified, và cột HIP bên dưới chỉ có thể được điền bởi máy của bạn.
          </p>
        }
      >
        <p>
          On HIP, <code>&lt;cuda/atomic&gt;</code> comes from libhipcxx, part of ROCm from 10.1. The course's HIP build
          links <code>libhipcxx::libhipcxx</code> when CMake finds it (<code>labs/cmake/Backends.cmake</code>); when it
          does not, <code>doctor</code> says so and all three labs will not build there. Nobody has tried: there is no ROCm
          machine behind this course, so this is unverified, and the HIP column below can only be filled by yours.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/infra-facts.md B2; ch09-facts.md B1)', '2026-11-20 (notes/today/infra-facts.md, mục B2; ch09-facts.md B1)')}>
        <Translated
          vi={
            <p>
              libhipcxx giữ cách viết của CUDA và đối số phạm vi, nhưng không có phạm vi cluster.
            </p>
          }
        >
          <p>
            libhipcxx keeps the CUDA spelling and the scope argument, without a cluster scope.
          </p>
        </Translated>
      </Today>
      <VerificationStatus />

      {/* ------------------------------------------------------------ 9.1 */}
      <h2><Translated vi="Lab 9.1 (tùy chọn): săn race, hai kernel histogram">Lab 9.1 (optional): race hunt, two histogram kernels</Translated></h2>
      <Translated
        vi={
          <p>
            Một đồng nghiệp đưa bạn hai histogram 256 bin của một ảnh 8-bit. <code>histogramBasic</code> là Hình 9.6, một
            luồng mỗi điểm ảnh. <code>histogramPool</code> là Hình 9.9: một histogram riêng mỗi khối trong một vùng chung ở bộ nhớ
            toàn cục, cập nhật bằng phép toán nguyên tử phạm vi khối, rồi ghi các bin riêng khác không vào các bin công khai. Giữa
            hai kernel có ba lỗi độc lập, cùng ba lỗi đó trong bản CUDA và bản Metal. Mọi thứ đều biên dịch được.
            Hãy tìm và sửa cả ba.
          </p>
        }
      >
        <p>
          A colleague hands you two 256-bin histograms of an 8-bit image. <code>histogramBasic</code> is Fig. 9.6, one
          thread per pixel. <code>histogramPool</code> is Fig. 9.9: one private histogram per block in a pool in global
          memory, updated with block-scope atomics, then a commit of the non-zero private bins to the public bins. Between
          them they have three independent defects, the same three in the CUDA and the Metal version. Everything compiles.
          Find and fix all three.
        </p>
      </Translated>
      <Source section="9.2–9.4" pages="203–213" figs={tr('Figs. 9.6, 9.9; the defects are ours', 'Hình 9.6, 9.9; các lỗi là của khóa học này')} />
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <Translated vi={<><code>histogramBasic</code> và <code>histogramPool</code>; hai hàm khởi chạy bên dưới chúng là đúng</>}>
              <><code>histogramBasic</code> and <code>histogramPool</code>; the two launchers below them are correct</>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated vi={<><code>histogram_basic</code> và <code>histogram_pool</code>, với cùng ba lỗi</>}>
              <><code>histogram_basic</code> and <code>histogram_pool</code>, with the same three defects</>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated vi={<>đúng khi phát hành; có mặt cho đầy đủ</>}>
              <>correct as shipped; there for completeness</>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Bên gọi xóa các bin và vùng chung về không, vùng chung giữ một histogram 256 bin mỗi khối. Sách không in mã host nào
            và không bao giờ nói ai xóa chúng; các hàm khởi chạy và quy tắc đó là của khóa học này. Mỗi trường hợp so sánh cả 256 số đếm với
            vòng lặp tuần tự của Hình 9.2, chính xác. Các ảnh từ 1 × 1 đến 64 × 64 chạy ở mọi nơi, còn 1024 × 1024 và 3000 ×
            1001 trên GPU; mỗi ảnh ở bốn loại (byte ngẫu nhiên phân bố đều, mọi điểm ảnh bằng 200, 0 và 255 thành các chuỗi dài, một
            dải chuyển mượt), với các khối 64 và 128 luồng, ít hơn số bin, nên vòng lặp ghi phải nhảy bước. Trường hợp đầu tiên của mỗi kernel
            là ảnh hằng trong 64 khối 64 luồng, đầu vào có tranh chấp nặng nhất. Trên GPU, mỗi
            cấu hình được khởi chạy 20 lần trên các buffer vừa được xóa về không.
          </p>
        }
      >
        <p>
          The caller zeroes the bins and the pool, which holds one 256-bin histogram per block. The book prints no host code
          and never says who zeroes them; the launchers and that rule are ours. Every case compares all 256 counts with the
          sequential loop of Fig. 9.2, exactly. Images from 1 × 1 to 64 × 64 run everywhere, and 1024 × 1024 and 3000 ×
          1001 on a GPU; each in four kinds (uniform random bytes, every pixel 200, 0 and 255 in long runs, a smooth
          gradient), with blocks of 64 and 128 threads, fewer than the bins, so the commit loop must stride. Each kernel's
          first case is the constant image in 64 blocks of 64 threads, the most contended input there is. On a GPU every
          configuration is launched 20 times on freshly zeroed buffers.
        </p>
      </Translated>
      <Translated vi={<p>Trước khi chạy bất cứ thứ gì, hãy đọc hai kernel và ghi lại mỗi lỗi bạn thấy, và backend nào bạn nghĩ sẽ cho thấy nó. Rồi săn, theo thứ tự này:</p>}>
        <p>Before you run anything, read the two kernels and write down each defect you can see, and which backend you expect to show it. Then hunt, in this order:</p>
      </Translated>
      <ol>
        <Translated
          vi={
            <li>
              Trình giả lập trước: <code>python3 labs/pmpp.py run 9.1 --backend emu</code>. Một rào chắn bị thiếu sẽ sai ở đó
              dưới ít nhất một trong ba thứ tự luồng. Một phép toán nguyên tử bị thiếu thì không bao giờ.
            </li>
          }
        >
          <li>
            Emulator first: <code>python3 labs/pmpp.py run 9.1 --backend emu</code>. A missing barrier is wrong there
            under at least one of the three thread orders. A missing atomic never is.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <code>python3 labs/pmpp.py run 9.1 --backend race</code>. Đọc số dòng <code>kernel.cu:</code> trong mỗi
              báo cáo. Một truy cập trong một cặp có thể nằm bên trong phép toán nguyên tử của chính trình giả lập (<code>pmpp_emu::atomic_rmw</code>):
              đó là một lời gọi <code>atomic_ref</code> của bạn đang race với một truy cập thường ở dòng kia. Mỗi cặp
              dòng chỉ được báo một lần mỗi lần chạy, nên hãy đọc trường hợp đầu tiên của mỗi kernel.
            </li>
          }
        >
          <li>
            <code>python3 labs/pmpp.py run 9.1 --backend race</code>. Read the <code>kernel.cu:</code> line numbers in each
            report. One access of a pair can sit inside the emulator's own atomic (<code>pmpp_emu::atomic_rmw</code>):
            that is one of your <code>atomic_ref</code> calls racing with a plain access on the other line. Each pair of
            lines is reported once per run, so read each kernel's first case.
          </li>
        </Translated>
        <Translated vi={<li>Sửa một thứ và chạy lại. Các lỗi thay đổi khi bạn tiến lên.</li>}>
          <li>Fix one thing and run again. The failures change as you go.</li>
        </Translated>
        <Translated
          vi={
            <li>
              Rồi đến GPU của bạn: <code>python3 labs/pmpp.py run 9.1</code>. Ghi lại những lỗi nào nó đã cho bạn thấy và những lỗi nào nó che đi.
            </li>
          }
        >
          <li>
            Then your GPU: <code>python3 labs/pmpp.py run 9.1</code>. Note which defects it showed you and which it hid.
          </li>
        </Translated>
      </ol>
      <Translated
        vi={
          <p>
            Viết các bản sửa nguyên tử như sách làm, <code>cuda::atomic_ref&lt;unsigned int, scope&gt;</code> và{' '}
            <code>fetch_add(…, cuda::memory_order_relaxed)</code>, với phạm vi được chọn theo các luồng dùng chung
            bộ đếm. Trên Metal, bản sửa đổi kiểu của tham số thành <code>device atomic_uint*</code>. Và một rào chắn Metal
            nêu tên vùng nhớ mà nó sắp thứ tự: hãy quyết định rào chắn của kernel vùng chung cần cờ nào trước khi bạn tra cứu.
          </p>
        }
      >
        <p>
          Write the atomic fixes as the book does, <code>cuda::atomic_ref&lt;unsigned int, scope&gt;</code> and{' '}
          <code>fetch_add(…, cuda::memory_order_relaxed)</code>, with the scope chosen from the threads that share the
          counter. On Metal the fix changes the parameter's type to <code>device atomic_uint*</code>. And a Metal barrier
          names the memory it orders: decide which flag the pool kernel's barrier needs before you look it up.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch09-facts.md A1, A3, C1; ch05-facts.md C2)', '2026-11-20 (notes/today/ch09-facts.md, mục A1, A3, C1; ch05-facts.md C2)')}>
        <Translated
          vi={
            <p>
              Nếu bỏ đi, phạm vi mặc định là system và thứ tự là <code>seq_cst</code>. Một lệnh ghi thường từ nhiều lane
              của một warp để cho "which thread performs the final write" (luồng nào thực hiện lần ghi cuối) là không xác định. Các hàm atomic của Metal chỉ nhận kiểu atomic.
            </p>
          }
        >
          <p>
            Left out, the scope defaults to system and the order to <code>seq_cst</code>. A plain write from several lanes
            of a warp leaves "which thread performs the final write" undefined. Metal's atomic functions take atomic types
            only.
          </p>
        </Translated>
      </Today>
      <RaceLine lab={l1} />
      <Translated
        vi={
          <p>
            Với <code>--bench</code>, chương trình điều khiển đo thời gian cả hai kernel trên 2<sup>26</sup> điểm ảnh (ít hơn nếu thiết bị báo quá
            ít bộ nhớ cho ảnh và vùng chung), 256 luồng mỗi khối, trên các byte phân bố đều và trên ảnh hằng, lấy kết quả tốt nhất
            trong 5 lần chạy. Hãy chốt hai dự đoán trước. Mô hình của sách cho phép tới 256 lần cho dự đoán đầu tiên: các phép toán nguyên tử trên một
            vị trí không thể chồng lấn, và các byte phân bố đều rải chúng ra 256 vị trí.
          </p>
        }
      >
        <p>
          With <code>--bench</code> the driver times both kernels on 2<sup>26</sup> pixels (fewer if the device reports too
          little memory for the image and the pool), 256 threads per block, on uniform bytes and on the constant image, best
          of 5 runs. Commit two predictions first. The book's model allows up to 256 times for the first: atomics on one
          location cannot overlap, and uniform bytes spread them over 256 locations.
        </p>
      </Translated>
      <SharedPrediction
        id="ch09/9.3/p-skew-slowdown"
        lab="9.1"
        prompt={
          <Translated
            vi={
              <>
                Cùng dự đoán như ở mục 9.3: chốt nó ở trang nào cũng được. Trên GPU <em>của bạn</em>, kernel cơ bản sẽ chạy chậm hơn
                bao nhiêu lần trên một ảnh hằng so với trên các byte ngẫu nhiên phân bố đều?
              </>
            }
          >
            <>
              The same prediction as in section 9.3: commit it on either page. On <em>your</em> GPU, how many times slower
              will the basic kernel run on a constant image than on uniform random bytes?
            </>
          </Translated>
        }
        unit={tr('× slower', '× chậm hơn')}
        read={skewSlowdown}
        reading={
          <Translated
            vi={
              <>
                Hai điều mà mô hình bỏ qua có thể làm con số thật nhỏ hơn 256 rất nhiều. Phép toán nguyên tử phạm vi device chạy trong
                bộ nhớ đệm L2 trên NVIDIA và AMD, không phải trong DRAM; Apple không ghi gì trong tài liệu (ch09-facts A4, B2, C4). Và một trình biên dịch có thể
                gộp các phép toán nguyên tử cùng địa chỉ của một warp: NVIDIA chỉ nói điều đó trong một bài blog năm 2017, "in many cases" (trong nhiều trường hợp); trình biên dịch của AMD
                chỉ làm vậy với một địa chỉ giống nhau trên toàn wavefront, mà một bin đọc từ ảnh thì không như vậy (A9,
                B5).
              </>
            }
          >
            <>
              Two things the model leaves out can make the real figure far smaller than 256. Device-scope atomics run in the
              L2 cache on NVIDIA and AMD, not in DRAM; Apple documents nothing (ch09-facts A4, B2, C4). And a compiler may
              merge one warp's same-address atomics: NVIDIA says so only in a 2017 blog post, "in many cases"; AMD's compiler
              does it only for an address that is the same across the wavefront, which a bin read from the image is not (A9,
              B5).
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch09/labs/p-lab1-pool-gap"
        prompt={
          <Translated vi={<>Trên ảnh hằng, kernel vùng chung sẽ chạy nhanh gấp bao nhiêu lần kernel cơ bản trên GPU của bạn?</>}>
            <>
              On the constant image, how many times as fast as the basic kernel will the pool kernel run on your GPU?
            </>
          </Translated>
        }
        unit={tr('× the basic kernel', '× kernel cơ bản')}
        measure={poolGap}
        missing={BENCH_MISSING('9.1')}
        reading={
          <Translated
            vi={
              <>
                Trên ảnh hằng, mọi phép toán nguyên tử của kernel cơ bản đều nhập vào một hàng đợi. Vùng chung chia hàng đợi đó thành một hàng cho mỗi
                khối đang chạy, và thêm một bước ghi tối đa 256 bin mỗi khối (<a href={href('ch09', '9.4')}>mục 9.4</a>
                ). Tỉ lệ trên các byte phân bố đều, trong bảng kết quả, cho biết việc chia tốn bao nhiêu khi không có hàng đợi nào để chia.
              </>
            }
          >
            <>
              On the constant image every atomic of the basic kernel joins one queue. The pool splits that queue into one per
              block in flight, and adds a commit of at most 256 bins per block (<a href={href('ch09', '9.4')}>section 9.4</a>
              ). The ratio on uniform bytes, in the results table, says what the split costs when there is no queue to split.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l1}
        cols={[
          { head: tr('Histogram', 'Histogram'), cell: (r) => (r.name === 'basic' ? tr('basic (Fig. 9.6)', 'basic (Hình 9.6)') : r.name === 'pool' ? tr('pool (Fig. 9.9)', 'pool (Hình 9.9)') : r.name) },
          { head: tr('Image', 'Ảnh'), cell: image },
          pixels,
          msCol,
          numCol('Gpixel/s', 'gpixels_per_s'),
        ]}
      />
      <Translated
        vi={
          <p>
            Dù GPU của bạn cho thấy gì, một cấu hình đã qua 20 lần cũng chỉ cho bạn thấy 20 lịch chạy trên một thiết bị. Một phép
            tăng đồng thời thường là một data race, nên một số đếm bị mất là kết quả điển hình, không phải kết quả được hứa hẹn, và một lần chạy sạch
            không chứng minh điều gì về thiết bị tiếp theo.
          </p>
        }
      >
        <p>
          Whatever your GPU showed, a configuration that passed 20 times showed you 20 schedules on one device. A plain
          concurrent increment is a data race, so a lost count is the typical outcome, not a promised one, and a clean run
          proves nothing about the next device.
        </p>
      </Translated>
      <ReferenceSolution lab={l1} tiers={HUNT_TIERS} />

      {/* ------------------------------------------------------------ 9.2 */}
      <h2><Translated vi="Lab 9.2: riêng tư hóa trong bộ nhớ chia sẻ, gộp việc và gộp chuỗi">Lab 9.2: privatised in shared memory, coarsened and run-merged</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn viết hai kernel nhanh của chương, bằng CUDA và bằng Metal. <code>histogramShared</code> là Hình 9.14: xóa
            một bản sao riêng của các bin trong bộ nhớ chia sẻ về không bằng một vòng lặp nhảy bước, rào chắn, đếm <code>coarseFactor</code> điểm ảnh
            mỗi luồng với chỉ số xen kẽ và phép toán nguyên tử phạm vi khối, rào chắn, rồi cộng mỗi bin riêng khác không vào các
            bin công khai bằng một phép toán nguyên tử phạm vi device. <code>histogramRuns</code> là Hình 9.15: như trên, cộng thêm một bộ đếm chuỗi
            mỗi luồng, để các điểm ảnh bằng nhau liên tiếp của một luồng chỉ tốn một phép toán nguyên tử. Hệ số gộp việc của sách là một
            macro lúc biên dịch; ở đây nó là một đối số lúc chạy, để một bản build có thể được đo ở nhiều hệ số.
          </p>
        }
      >
        <p>
          You write the chapter's two fast kernels, in CUDA and in Metal. <code>histogramShared</code> is Fig. 9.14: zero
          a private copy of the bins in shared memory with a strided loop, barrier, count <code>coarseFactor</code> pixels
          per thread with interleaved indexing and block-scope atomics, barrier, then add each non-zero private bin to the
          public bins with a device-scope atomic. <code>histogramRuns</code> is Fig. 9.15: the same, plus a run counter
          per thread, so that consecutive equal pixels of one thread cost one atomic. The book's coarsening factor is a
          compile-time macro; here it is a run-time argument, so that one build can be measured at several factors.
        </p>
      </Translated>
      <Source section="9.4–9.6" pages="211–219" figs={tr('Figs. 9.10, 9.14, 9.15', 'Hình 9.10, 9.14, 9.15')} />
      <LabFiles
        lab={l2}
        files={[
          [
            'kernel.cu',
            <Translated vi={<><code>histogramShared</code> và <code>histogramRuns</code>; các hàm khởi chạy bên dưới chúng đã được viết sẵn cho bạn</>}>
              <><code>histogramShared</code> and <code>histogramRuns</code>; the launchers below them are written for you</>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated vi={<><code>histogram_shared</code> và <code>histogram_runs</code></>}>
              <><code>histogram_shared</code> and <code>histogram_runs</code></>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated vi={<>đã viết sẵn cho bạn: 256 luồng mỗi threadgroup, hoặc ít hơn nếu pipeline không chạy được 256</>}>
              <>written for you: 256 threads per threadgroup, or fewer if the pipeline cannot run 256</>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Được cung cấp và không chỉnh sửa, trong <code>labs/ch09/lab2_privatised/provided/</code>: hai bản đối chứng của benchmark,{' '}
            <code>refHistogramBasic</code> (Hình 9.6) và <code>refHistogramContiguous</code> (Hình 9.12, riêng tư hóa trong bộ nhớ chia sẻ
            với chỉ số liền kề), bằng CUDA và Metal. Các bài kiểm tra cũng chạy chúng. Mục{' '}
            <a href={href('ch09', '9.5')}>9.5</a> hiển thị bản liền kề bên cạnh bản của bạn.
          </p>
        }
      >
        <p>
          Provided and not edited, in <code>labs/ch09/lab2_privatised/provided/</code>: the benchmark's two comparators,{' '}
          <code>refHistogramBasic</code> (Fig. 9.6) and <code>refHistogramContiguous</code> (Fig. 9.12, shared-memory
          privatisation with contiguous indexing), in CUDA and Metal. The tests run them too. Section{' '}
          <a href={href('ch09', '9.5')}>9.5</a> shows the contiguous one beside yours.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Trên Metal, các bin riêng là một mảng <code>threadgroup atomic_uint</code> khai báo trong thân kernel, được xóa về không
            bằng <code>atomic_store_explicit</code> và được đọc trong bước ghi bằng <code>atomic_load_explicit</code>. Khai báo đó
            là dạng mà mã đang phát hành dùng; đặc tả không đưa ví dụ nào về nó. Nó đã biên dịch và qua trên
            máy Mac mà lab này được xây dựng.
          </p>
        }
      >
        <p>
          On Metal the private bins are a <code>threadgroup atomic_uint</code> array declared in the kernel body, zeroed
          with <code>atomic_store_explicit</code> and read in the commit with <code>atomic_load_explicit</code>. That
          declaration is the form shipping code uses; the specification shows no example of it. It compiled and passed on
          the Mac this lab was built on.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch09-facts.md C1, D2)', '2026-11-20 (notes/today/ch09-facts.md, mục C1, D2)')}>
        <Translated
          vi={
            <p>
              Nếu trình biên dịch Metal của bạn từ chối mảng đó, hãy dùng một tham số <code>threadgroup atomic_uint* [[threadgroup(0)]]</code>{' '}
              có kích thước đặt ở host bằng <code>setThreadgroupMemoryLength</code>, như histogram MPS của PyTorch làm.
            </p>
          }
        >
          <p>
            If your Metal compiler rejects the array, use a <code>threadgroup atomic_uint* [[threadgroup(0)]]</code>{' '}
            parameter sized on the host with <code>setThreadgroupMemoryLength</code>, as PyTorch's MPS histogram does.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 9.2</code>, bắt đầu với <code>--backend emu</code> rồi đến{' '}
            <code>--backend race</code>. Cả hai kernel phải khớp chính xác số đếm của CPU trên các ảnh từ 1 × 1 đến 100 × 100
            (trên GPU còn có 1024 × 1024 và 4096 × 1025), phần lớn để khối cuối thiếu điểm ảnh, ở các hệ số gộp việc
            1, 3, 4 và 16, trên các byte phân bố đều, một ảnh hằng, một bầu trời (mỗi hàng một giá trị, thay đổi chậm từ hàng này sang hàng khác)
            và các sọc gồm 8 byte bằng nhau. Mỗi backend có thể cho bạn thấy gì:
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 9.2</code>, starting with <code>--backend emu</code> and then{' '}
          <code>--backend race</code>. Both kernels must match the CPU's counts exactly on images from 1 × 1 to 100 × 100
          (on a GPU also 1024 × 1024 and 4096 × 1025), most leaving the last block short of pixels, at coarsening factors
          1, 3, 4 and 16, on uniform bytes, a constant image, a sky (each row one value, changing slowly from row to row)
          and stripes of 8 equal bytes. What each backend can show you:
        </p>
      </Translated>
      <ul>
        <Translated
          vi={
            <li>
              Bộ nhớ chia sẻ không ai xóa thì chứa các byte <code>0xCD</code> trên <code>emu</code> và <code>race</code>, nên một
              vòng lặp xóa bị thiếu sẽ sai ở đó mọi lần. Một GPU có thể đưa bạn toàn số không, và khi đó các bài kiểm tra qua trên nó.
            </li>
          }
        >
          <li>
            Shared memory nobody zeroed holds <code>0xCD</code> bytes on <code>emu</code> and <code>race</code>, so a
            missing zeroing loop is wrong there every time. A GPU may hand you zeros, and then the tests pass on it.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Một rào chắn bị thiếu sẽ sai trên <code>emu</code> dưới ít nhất một thứ tự luồng, và <code>race</code> nêu tên
              các dòng liên quan.
            </li>
          }
        >
          <li>
            A missing barrier is wrong on <code>emu</code> under at least one thread order, and <code>race</code> names
            the lines involved.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <code>race</code> chạy một khối 256 luồng riêng lẻ, nên nó không thể thấy hai khối race nhau trong bước ghi. Một
              lệnh <code>+=</code> thường ở đó có thể hiện ra trên GPU của bạn dưới dạng số đếm bị mất.
            </li>
          }
        >
          <li>
            <code>race</code> runs a block of 256 threads on its own, so it cannot see two blocks racing in the commit. A
            plain <code>+=</code> there may show on your GPU as lost counts.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Có hai lỗi qua được mọi bài kiểm tra. Một vòng lặp xóa hoặc ghi có cận là <code>blockDim.x</code> cho ra số đếm
              đúng, vì các hàm khởi chạy dùng 256 luồng cho 256 bin; nó sai với mọi kích thước khối khác, điều mà lab 9.1
              có kiểm tra. Và chỉ số liền kề cũng cho số đếm đúng: một luồng lấy những điểm ảnh nào thay đổi tốc độ,
              không thay đổi kết quả.
            </li>
          }
        >
          <li>
            Two mistakes pass every test. A zeroing or commit loop bounded by <code>blockDim.x</code> gives the right
            counts, because the launchers use 256 threads for 256 bins; it is wrong for any other block size, which lab 9.1
            does test. And contiguous indexing gives the right counts too: which pixels a thread takes changes the speed,
            not the result.
          </li>
        </Translated>
      </ul>
      <RaceLine lab={l2} />
      <Translated
        vi={
          <p>
            Với <code>--bench</code>, chương trình điều khiển đo thời gian, trên 2<sup>26</sup> điểm ảnh (8192 × 8192, để một hàng bầu trời đủ dài),
            các byte phân bố đều, ảnh hằng và bầu trời: kernel cơ bản được cung cấp, kernel liền kề được cung cấp ở hệ số
            16, <code>shared</code> của bạn ở các hệ số 1, 4, 16 và 64, và <code>runs</code> của bạn ở 16, mỗi cái lấy kết quả tốt nhất trong 5 lần chạy.
            Hãy chốt ba dự đoán trước.
          </p>
        }
      >
        <p>
          With <code>--bench</code> the driver times, on 2<sup>26</sup> pixels (8192 × 8192, so that a sky row is long),
          uniform bytes, the constant image and the sky: the provided basic kernel, the provided contiguous kernel at factor
          16, your <code>shared</code> at factors 1, 4, 16 and 64, and your <code>runs</code> at 16, best of 5 runs each.
          Commit three predictions first.
        </p>
      </Translated>
      <SharedPrediction
        id="ch09/9.4/p-shared-speedup"
        lab="9.2"
        prompt={
          <Translated
            vi={
              <>
                Cùng dự đoán như ở mục 9.4: chốt nó ở trang nào cũng được. Với mọi điểm ảnh như nhau, kernel bộ nhớ chia sẻ sẽ chạy nhanh hơn
                kernel cơ bản bao nhiêu lần trên GPU của bạn, ở hệ số gộp việc 16?
              </>
            }
          >
            <>
              The same prediction as in section 9.4: commit it on either page. With every pixel the same, how many times
              faster than the basic kernel will the shared-memory kernel run on your GPU, at a coarsening factor of 16?
            </>
          </Translated>
        }
        unit={tr('× faster', '× nhanh hơn')}
        read={sharedSpeedup}
        reading={
          <Translated
            vi={
              <>
                Các phép toán nguyên tử của kernel cơ bản đều xếp hàng trên một bin trong bộ nhớ đệm cấp cuối; của kernel chia sẻ thì xếp hàng trên một bin
                mỗi khối trong bộ nhớ chia sẻ, rồi ghi một lần mỗi khối. Các dòng từ hệ số 1 đến hệ số 64 của bảng cho thấy bao nhiêu
                phần lợi là từ riêng tư hóa và bao nhiêu là từ gộp việc.
              </>
            }
          >
            <>
              The basic kernel's atomics all queue on one bin in the last-level cache; the shared kernel's queue on one bin
              per block in shared memory, then commit once per block. The factor-1 to factor-64 rows of the table show how
              much of the gain is privatisation and how much is coarsening.
            </>
          </Translated>
        }
      />
      <SharedPrediction
        id="ch09/9.5/p-interleaved-gain"
        lab="9.2"
        prompt={
          <Translated
            vi={
              <>
                Cùng dự đoán như ở mục 9.5: chốt nó ở trang nào cũng được. Ở hệ số gộp việc 16 trên các byte
                ngẫu nhiên phân bố đều, kernel xen kẽ sẽ chạy nhanh hơn kernel liền kề bao nhiêu lần trên GPU của bạn?
              </>
            }
          >
            <>
              The same prediction as in section 9.5: commit it on either page. At a coarsening factor of 16 on uniform
              random bytes, how many times faster than the contiguous kernel will the interleaved one run on your GPU?
            </>
          </Translated>
        }
        unit={tr('× faster', '× nhanh hơn')}
        read={interleavedGain}
        reading={
          <Translated
            vi={
              <>
                Lập luận của sách là hợp nhất truy cập: trong một bước, các luồng của một warp nên đọc các byte kế nhau (p. 216).
                NVIDIA ghi quy tắc đó trong tài liệu dưới dạng các transaction 32 byte mỗi warp, AMD dưới dạng các yêu cầu dòng cache; Apple không ghi trong tài liệu quy tắc
                hợp nhất truy cập nào, nên trên Mac số đo này là bằng chứng duy nhất (ch09-facts A7, B3, C4). Một con số
                nhỏ hơn 1 nghĩa là kernel liền kề thắng trên thiết bị này.
              </>
            }
          >
            <>
              The book's argument is coalescing: in one step the threads of a warp should read neighbouring bytes (p. 216).
              NVIDIA documents that rule as 32-byte transactions per warp, AMD as cache-line requests; Apple documents no
              coalescing rule, so on a Mac this measurement is the only evidence there is (ch09-facts A7, B3, C4). A number
              below 1 means contiguous won on this device.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch09/labs/p-lab2-runs-sky"
        prompt={
          <Translated
            vi={
              <>
                Trên ảnh bầu trời ở hệ số 16, <code>runs</code> sẽ nhanh gấp bao nhiêu lần <code>shared</code>? Một
                con số nhỏ hơn 1 nghĩa là bạn nghĩ gộp chuỗi tốn nhiều hơn phần nó tiết kiệm. Cũng hãy quyết định bạn kỳ vọng gì trên
                các byte phân bố đều.
              </>
            }
          >
            <>
              On the sky image at factor 16, how many times as fast as <code>shared</code> will <code>runs</code> be? A
              number below 1 means you expect run merging to cost more than it saves. Decide too what you expect on
              uniform bytes.
            </>
          </Translated>
        }
        unit={tr('× shared', '× shared')}
        measure={runsOnSky}
        missing={BENCH_MISSING('9.2')}
        reading={
          <Translated
            vi={
              <>
                Một hàng bầu trời gồm 8,192 byte bằng nhau, dài hơn nhiều so với 256 điểm ảnh giữa các lần đọc liên tiếp của một luồng, nên
                gần như mọi lần lặp đều nối dài một chuỗi. Các byte phân bố đều gần như không bao giờ lặp lại ở khoảng cách đó, nên ở đó{' '}
                <code>runs</code> trả giá cho phép so sánh mà chẳng tiết kiệm được gì (<a href={href('ch09', '9.6')}>mục 9.6</a>).
                Tỉ lệ trên các byte phân bố đều nằm trong phần chi tiết ở trên.
              </>
            }
          >
            <>
              A sky row is 8,192 equal bytes, far longer than the 256 pixels between one thread's consecutive reads, so
              nearly every iteration extends a run. Uniform bytes almost never repeat at that spacing, so there{' '}
              <code>runs</code> pays for its comparison and saves nothing (<a href={href('ch09', '9.6')}>section 9.6</a>).
              The uniform ratio is in the detail above.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l2}
        cols={[
          { head: tr('Histogram', 'Histogram'), cell: (r) => LAB2_NAME[r.name] ?? r.name },
          { head: tr('Image', 'Ảnh'), cell: image },
          { head: tr('Coarse', 'Hệ số gộp'), num: true, cell: (r) => String(r.coarse ?? '') },
          msCol,
          numCol('Gpixel/s', 'gpixels_per_s'),
        ]}
      />
      <ReferenceSolution lab={l2} tiers={SHARED_TIERS} />

      {/* ------------------------------------------------------------ 9.3 */}
      <h2><Translated vi="Lab 9.3: đếm token cho phạt lặp (M5)">Lab 9.3: token counts for repetition penalties (M5)</Translated></h2>
      <Translated
        vi={
          <p>
            Một sampler có hình phạt tần suất và hiện diện cần biết, với mỗi request trong batch, mỗi id của
            bộ từ vựng đã xuất hiện bao nhiêu lần trong ngữ cảnh của request đó. Đó là một histogram mà các bin là cả một bộ từ vựng:
            50,257 id cho GPT-2, 151,936 cho Qwen3-8B, 201,088 cho gpt-oss-20b. Bạn viết ba kernel trên các token id int32
            thứ tự hàng (row-major) và các số đếm uint32, batch đứng đầu: <code>countTokens</code> đếm token của mỗi hàng vào số đếm
            của hàng đó, <code>appendTokens</code> cộng một token mới cho mỗi hàng (một bước decode), và <code>applyPenalties</code>{' '}
            trừ <code>count × frequencyPenalty + (count &gt; 0) × presencePenalty</code> khỏi mọi logit.
          </p>
        }
      >
        <p>
          A sampler with frequency and presence penalties needs to know, for every request in the batch, how often each
          vocabulary id already occurs in that request's context. That is a histogram whose bins are a whole vocabulary:
          50,257 ids for GPT-2, 151,936 for Qwen3-8B, 201,088 for gpt-oss-20b. You write three kernels over row-major
          int32 token ids and uint32 counts, batch first: <code>countTokens</code> counts each row's tokens into that row's
          counts, <code>appendTokens</code> adds one new token per row (a decode step), and <code>applyPenalties</code>{' '}
          subtracts <code>count × frequencyPenalty + (count &gt; 0) × presencePenalty</code> from every logit.
        </p>
      </Translated>
      <Source section="9.2–9.5" pages="203–217" figs={tr('Figs. 9.6, 9.9, 9.14', 'Hình 9.6, 9.9, 9.14')} />
      <Translated
        vi={
          <p>
            Trước khi mở <code>work/</code>, hãy trả lời sáu câu hỏi thiết kế mà README của lab mở đầu: kiểu riêng tư hóa nào của
            chương vẫn áp dụng được khi một bản sao riêng chiếm hàng trăm KiB? Bảng "vừa hay không" của mục{' '}
            <a href={href('ch09', '9.4')}>9.4</a> đặt ba bộ từ vựng cạnh ngân sách trên chip mà các
            thiết bị của bạn báo. README đưa đáp án của chúng tôi ở cuối.
          </p>
        }
      >
        <p>
          Before you open <code>work/</code>, answer the six design questions the lab's README opens with: which of the
          chapter's privatisations still applies when one private copy is hundreds of KiB? Section{' '}
          <a href={href('ch09', '9.4')}>9.4</a>'s fit table puts the three vocabularies against the on-chip budgets your
          devices report. The README gives our answers at its end.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Đếm token id, bước decode và các hình phạt không có trong sách. Công thức là của llama.cpp, vốn
              đếm trên CPU; vLLM giữ một tensor số đếm <code>[requests × vocabulary]</code> trên GPU (ch09-facts D4, D6,
              D7).
            </p>
          }
        >
          <p>
            Counting token ids, the decode step and the penalties are not in the book. The formula is llama.cpp's, which
            counts on the CPU; vLLM keeps a <code>[requests × vocabulary]</code> count tensor on the GPU (ch09-facts D4, D6,
            D7).
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
                  <code>countTokens</code>, <code>appendTokens</code>, <code>applyPenalties</code> và{' '}
                  <code>launchCountTokens</code>, <code>launchAppendTokens</code>, <code>launchApplyPenalties</code>
                </>
              }
            >
              <>
                <code>countTokens</code>, <code>appendTokens</code>, <code>applyPenalties</code> and{' '}
                <code>launchCountTokens</code>, <code>launchAppendTokens</code>, <code>launchApplyPenalties</code>
              </>
            </Translated>,
          ],
          ['kernel.metal', <><code>count_tokens</code>, <code>append_tokens</code>, <code>apply_penalties</code></>],
          [
            'launch.mm',
            <Translated vi={<><code>encodeCountTokens</code>, <code>encodeAppendTokens</code>, <code>encodeApplyPenalties</code>; kích thước lưới là do bạn chọn</>}>
              <><code>encodeCountTokens</code>, <code>encodeAppendTokens</code>, <code>encodeApplyPenalties</code>; the grid sizes are yours</>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Được cung cấp và không chỉnh sửa: <code>provided/count_pool.cu</code>, <code>.metal</code> và <code>.mm</code>, thiết kế của
            Hình 9.9 áp dụng cho số đếm token. Mỗi khối có một bản sao riêng của cả bộ từ vựng trong một vùng chung ở bộ nhớ toàn cục,
            đếm ở đó bằng phép toán nguyên tử phạm vi khối, rồi ghi mọi bin khác không vào hàng của request tương ứng. Hàm khởi chạy
            xóa vùng chung về không, và việc đó được tính giờ cùng. Đó là bản đối chứng của benchmark, và các bài kiểm tra cũng chạy nó.
          </p>
        }
      >
        <p>
          Provided and not edited: <code>provided/count_pool.cu</code>, <code>.metal</code> and <code>.mm</code>, Fig.
          9.9's design applied to token counts. Every block gets a private copy of the whole vocabulary in a global-memory
          pool, counts there with block-scope atomics, then commits every non-zero bin to its request's row. The launcher
          zeroes the pool, and that is timed with it. It is the benchmark's comparator, and the tests run it too.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Trên Metal không có góc nhìn nguyên tử lên một mảng thường: <code>count_tokens</code> khai báo các số đếm là{' '}
            <code>device atomic_uint*</code> và cập nhật chúng bằng <code>atomic_fetch_add_explicit</code>, relaxed và không có
            phạm vi. Buffer ở host vẫn giữ các bộ đếm 32-bit thường.
          </p>
        }
      >
        <p>
          On Metal there is no atomic view of a plain array: <code>count_tokens</code> declares the counts{' '}
          <code>device atomic_uint*</code> and updates them with <code>atomic_fetch_add_explicit</code>, relaxed and with no
          scope. The host buffer still holds plain 32-bit counters.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch09-facts.md C1)', '2026-11-20 (notes/today/ch09-facts.md, mục C1)')}>
        <Translated
          vi={
            <p>
              PyTorch gắn một tensor <code>uint32</code> thường thành <code>device atomic_uint*</code> theo cùng cách.
            </p>
          }
        >
          <p>
            PyTorch binds a plain <code>uint32</code> tensor as <code>device atomic_uint*</code> in the same way.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 9.3</code>, bắt đầu với <code>--backend emu</code>. Các dạng dữ liệu đi từ một
            token trên 1,000 id tới 8 hàng 2,048 token và bộ từ vựng tới 201,088 id (các dạng lớn chỉ trên GPU),
            mỗi dạng
            với bốn phân bố có seed: id phân bố đều, id kiểu Zipf nơi vài id chiếm phần lớn hàng, một id lặp lại
            suốt cả hàng, và các hàng chứa id 0 và <code>vocab − 1</code>, để bắt lỗi sai vị trí hàng. Số đếm
            phải khớp chính xác; đếm rồi append một token mỗi hàng phải bằng đếm tất cả; hình phạt phải
            khớp CPU trong một dung sai, vì Metal không mặc định dùng số học IEEE chặt, và khi cả hai
            hình phạt bằng 0 thì mọi logit phải trở về y nguyên từng bit. Chương trình điều khiển xóa số đếm về không trước: runtime của khóa học
            lấp các buffer device mới bằng byte <code>0xA5</code> để một bước xóa bị thiếu lộ ra. Một phép tăng thường trong{' '}
            <code>appendTokens</code> không được bị <code>race</code> báo: mỗi hàng do một luồng sở hữu.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 9.3</code>, starting with <code>--backend emu</code>. Shapes run from one
          token over 1,000 ids up to 8 rows of 2,048 tokens and vocabularies of up to 201,088 ids (the large ones on a GPU
          only), each
          with four seeded distributions: uniform ids, Zipf-like ids where a few take most of the row, one id repeated
          through the whole row, and rows holding ids 0 and <code>vocab − 1</code>, which catch a wrong row offset. Counts
          must match exactly; counting and then appending one token per row must equal counting all of them; penalties must
          match the CPU within a tolerance, because Metal does not default to strict IEEE arithmetic, and with both
          penalties 0 every logit must come back bit for bit. The driver zeroes the counts first: the course's runtime
          fills new device buffers with <code>0xA5</code> bytes so that a missing zeroing shows. A plain increment in{' '}
          <code>appendTokens</code> must not be reported by <code>race</code>: one thread owns each row.
        </p>
      </Translated>
      <RaceLine lab={l3} />
      <Translated
        vi={
          <p>
            Với <code>--bench</code>, chương trình điều khiển chạy batch 8, 4,096 token mỗi hàng, bộ từ vựng 151,936, trên các id phân bố đều và
            kiểu Zipf: kernel count của bạn, vùng chung được cung cấp, một bước append và các hình phạt, mỗi cái lấy kết quả tốt nhất trong ít nhất 10 lần chạy.
            Vùng chung được cung cấp dùng một token mỗi luồng, nên 16 khối mỗi hàng, mỗi khối xóa và ghi bản sao riêng gồm
            151,936 bin của nó. Hãy chốt hai dự đoán trước.
          </p>
        }
      >
        <p>
          With <code>--bench</code> the driver runs batch 8, 4,096 tokens per row, vocabulary 151,936, on uniform and
          Zipf-like ids: your count, the provided pool, one append step and the penalties, each the best of at least 10 runs.
          The provided pool uses one token per thread, so 16 blocks per row, each zeroing and committing its own copy of
          151,936 bins. Commit two predictions first.
        </p>
      </Translated>
      <LabPrediction
        id="ch09/labs/p-lab3-pool"
        prompt={
          <Translated
            vi={
              <>
                Id phân bố đều: <code>count_pool</code> được cung cấp sẽ mất thời gian gấp bao nhiêu lần các phép toán nguyên tử trực tiếp vào mỗi hàng
                của bạn? Một con số nhỏ hơn 1 nghĩa là bạn nghĩ vùng chung nhanh hơn. Hãy lập luận từ những gì một
                khối vùng chung làm: bao nhiêu cập nhật, so với bao nhiêu bin nó phải xóa và đọc lại.
              </>
            }
          >
            <>
              Uniform ids: how many times as long as your direct atomics into each row will the provided{' '}
              <code>count_pool</code> take? A number below 1 means you expect the pool to be faster. Reason from what one
              pool block does: how many updates, against how many bins it must zero and read back.
            </>
          </Translated>
        }
        unit={tr('× the time of direct atomics', '× thời gian của phép toán nguyên tử trực tiếp')}
        measure={poolOverCount}
        missing={BENCH_MISSING('9.3')}
        reading={
          <Translated
            vi={
              <>
                Riêng tư hóa có lợi khi số cập nhật của một khối nhiều hơn số bin nó phải khởi tạo và ghi. Ở đây mỗi khối vùng chung
                đếm 256 token và xử lý 151,936 bin, nên chi phí phụ của nó không co lại theo hàng đợi mà nó loại bỏ.
                Cách chữa của mục <a href={href('ch09', '9.5')}>9.5</a> là ít khối hơn nhưng bận hơn; phần "Go further" của README tính
                xem điều đó đi xa được tới đâu.
              </>
            }
          >
            <>
              Privatisation pays when a block's updates outnumber the bins it must initialise and commit. Here each pool
              block counts 256 tokens and handles 151,936 bins, so its overhead does not shrink with the queue it removes.
              Section <a href={href('ch09', '9.5')}>9.5</a>'s cure is fewer, busier blocks; the README's "Go further" works
              out how far that goes.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch09/labs/p-lab3-append"
        prompt={
          <Translated
            vi={
              <>
                Đếm lại cả ngữ cảnh bằng <code>countTokens</code> sẽ mất thời gian gấp bao nhiêu lần một bước{' '}
                <code>appendTokens</code>? Hãy viết tỉ lệ công việc trước, rồi tỉ lệ thời gian bạn kỳ vọng.
              </>
            }
          >
            <>
              How many times as long as one <code>appendTokens</code> step will recounting the whole context with{' '}
              <code>countTokens</code> take? Write the ratio of the work first, then the ratio of time you expect.
            </>
          </Translated>
        }
        unit={tr('× one append step', '× một bước append')}
        measure={recountOverAppend}
        missing={BENCH_MISSING('9.3')}
        reading={
          <Translated
            vi={
              <>
                Công việc khác nhau theo độ dài ngữ cảnh, 4,096 token so với một token mỗi hàng. Cả hai lần khởi chạy đều ngắn, nên một chi phí
                cố định cho mỗi lần khởi chạy có thể chiếm phần lớn cả hai thời gian; trên CUDA và HIP, chương trình điều khiển đo bằng đồng hồ host, kể cả lần khởi chạy,
                còn trên Metal thì bằng thời điểm bắt đầu và kết thúc trên GPU của một command buffer.
              </>
            }
          >
            <>
              The work differs by the context length, 4,096 tokens against one per row. Both launches are short, so a fixed
              cost per launch can dominate both times; on CUDA and HIP the driver times with the host clock, launch included,
              and on Metal with the GPU's start and end times of one command buffer.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l3}
        cols={[
          { head: tr('Operation', 'Thao tác'), cell: (r) => LAB3_NAME[r.name] ?? r.name },
          { head: tr('Ids', 'Id'), cell: (r) => (r.data === 'zipf' ? 'Zipf' : String(r.data ?? '')) },
          {
            head: tr('Batch × tokens, vocabulary', 'Batch × token, bộ từ vựng'),
            num: true,
            cell: (r) => nowrap(`${String(r.batch ?? '')} × ${num(r.len) != null ? count(num(r.len)!) : ''}, ${num(r.vocab) != null ? count(num(r.vocab)!) : ''}`),
          },
          msCol,
          { head: tr('Rate', 'Tốc độ'), num: true, cell: rate },
        ]}
      />
      <Translated
        vi={
          <p>
            Tốc độ của <code>append</code> đếm một token mỗi hàng, nên nó nhỏ theo cách dựng; hãy so sánh thời gian của nó, không phải
            tốc độ. Tốc độ của các hình phạt đếm theo logit.
          </p>
        }
      >
        <p>
          The rate for <code>append</code> counts one token per row, so it is small by construction; compare its time, not
          its rate. The penalties' rate counts logits.
        </p>
      </Translated>
      <h3><Translated vi="Mốc dự án M5, phần đầu tiên">Project milestone M5, first piece</Translated></h3>
      <Translated
        vi={
          <p>
            Mốc M5 của dự án xuyên suốt là lấy mẫu trên GPU: top-k và top-p, với phạt lặp.
            Lab 9.3 là phần đếm của nó: số đếm token theo từng request trên <code>[batch × vocab]</code>, được cập nhật mỗi
            bước decode một lần, và các hình phạt tần suất và hiện diện được áp lên logits. Chương 11, 12 và 14
            bổ sung tổng tích lũy, nén gọn (compaction) và sắp xếp mà top-p và top-k cần, và module X1 ghép chúng lại
            với lab này trong lab X1.3 của nó. Phạt lặp có xét dấu (sign-aware) được bỏ qua ở đây; mục{' '}
            <a href={href('x1', 'X1.3')}>X1.3, Penalties from token counts</a> bổ sung nó.
          </p>
        }
      >
        <p>
          Milestone M5 of the through-line project is sampling on the GPU: top-k and top-p, with repetition penalties.
          Lab 9.3 is its counting piece: per-request token counts over <code>[batch × vocab]</code>, kept up to date one
          decode step at a time, and the frequency and presence penalties applied to the logits. Chapters 11, 12 and 14
          add the cumulative sum, the compaction and the sort that top-p and top-k need, and module X1 puts them together
          with this lab in its lab X1.3. The sign-aware repetition penalty is left out here; section{' '}
          <a href={href('x1', 'X1.3')}>X1.3, Penalties from token counts</a>, adds it.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Dự án xuyên suốt, <code>nanoinfer</code>, là của khóa học này: một đường suy luận GPU nhỏ cho một transformer
              chỉ có decoder, lớn dần qua từng mốc.
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
      <ReferenceSolution lab={l3} tiers={TOKEN_TIERS} />
    </LabsPage>
  );
}
