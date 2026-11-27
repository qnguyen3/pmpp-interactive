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
  rawCol,
  useLabs,
  type BenchSet,
  type Col,
  type Lab,
} from '../../components/LabKit';
import { Translated, localized, tr } from '../../lib/i18n';
import { fmt, labStatus, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 15's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

export const LABS = defineLabs('ch15', [
  { num: '15.1', dir: 'lab1_coarse', name: 'coarsened', vi: { name: 'gộp việc cho luồng' } },
  { num: '15.2', dir: 'lab2_regtile', name: 'register tiling', vi: { name: 'phân ô thanh ghi' } },
  { num: '15.3', dir: 'lab3_remap', name: 'coalesced stores', vi: { name: 'ghi được hợp nhất' } },
  { num: '15.4', dir: 'lab4_padded', name: 'padding, vector loads', vi: { name: 'đệm, nạp vector' } },
  { num: '15.5', dir: 'lab5_pipelined', name: 'pipelined', vi: { name: 'pipeline hóa' } },
  { num: '15.6', dir: 'lab6_linear', name: 'linear layer (M3)', vi: { name: 'lớp tuyến tính (M3)' } },
  { num: '15.7', dir: 'lab7_bughunt', name: 'bug hunt', vi: { name: 'săn lỗi' } },
]);
const [l1, l2, l3, l4, l5, l6, l7] = LABS;
const STAGES = [l1, l2, l3, l4, l5];

// Identifier colours. Labs 15.3 to 15.5 call their warp-width template parameter W, and
// labs 15.6 and 15.7 call the weight W, so each family has its own map.
const TIERS_GEMM: Record<string, Tier> = {
  A: 'global', B: 'global', C: 'global', A_s: 'shared', B_s: 'shared', C_r: 'reg', a_r: 'reg', b_r: 'reg',
};
const TIERS_LINEAR: Record<string, Tier> = {
  X: 'global', W: 'global', Y: 'global', A_s: 'shared', B_s: 'shared', C_r: 'reg', a_r: 'reg', b_r: 'reg',
};

// ---------------------------------------------------------------- reading the benchmark rows
// Labs 15.1 to 15.5 (labs/ch15/shared/stage_driver.hpp) record, per shape, the rows
// tiled_5_3 (lab 5.3's reference kernel), previous (the stage before, from 15.2 on), stage
// and vendor, each with m, k, n, seconds and gflops; plus one pipeline row (Metal) or
// occupancy row (CUDA, HIP). Lab 15.6 records m2, m3 and vendor with batch, in, out.
const CUBE = { m: 2048, k: 2048, n: 2048 };
const SHORT_K = { m: 4096, k: 128, n: 4096 };
const LAYER_1 = { m: 1, k: 1024, n: 4096 };
const shapeText = (w: { m: number; k: number; n: number }) => `${w.m} × ${w.k} × ${w.n}`;
const gflops = (r?: BenchRow) => num(r?.gflops);

/** One stage row over another of the same run and shape, as GFLOP/s over GFLOP/s. */
const stageRatio =
  (lab: Lab, top: string, bottom: string, where: { m: number; k: number; n: number }, topName: string, bottomName: string) =>
  (reports: Reports) =>
    measure(reports, lab, (s: BenchSet) => {
      const a = gflops(benchRow(s, top, where));
      const b = gflops(benchRow(s, bottom, where));
      if (!a || !b) return null;
      return {
        value: a / b,
        detail: tr(
          `${topName} ${fmt(a)} against ${bottomName} ${fmt(b)} GFLOP/s at ${shapeText(where)}`,
          `${topName} ${fmt(a)} so với ${bottomName} ${fmt(b)} GFLOP/s ở ${shapeText(where)}`,
        ),
      };
    });

/** Lab 15.6: M3 over M2 at batch 1 on the 1024 → 4096 layer. */
const m3OverM2Batch1 = (reports: Reports) =>
  measure(reports, l6, (s: BenchSet) => {
    const where = { batch: 1, in: 1024, out: 4096 };
    const a = gflops(benchRow(s, 'm3', where));
    const b = gflops(benchRow(s, 'm2', where));
    if (!a || !b) return null;
    return {
      value: a / b,
      detail: tr(
        `${fmt(a)} against ${fmt(b)} GFLOP/s, batch 1, 1024 → 4096 layer`,
        `${fmt(a)} so với ${fmt(b)} GFLOP/s, batch 1, lớp tuyến tính 1024 → 4096`,
      ),
    };
  });

// ---------------------------------------------------------------- results columns
const SHAPE_NAME: Record<string, string> = {
  get square() {
    return tr('square', 'vuông');
  },
  get layer() {
    return tr('1024 → 4096 layer', 'lớp 1024 → 4096');
  },
  get short_k() {
    return tr('short K', 'K ngắn');
  },
};
const stageKernel: Col = {
  get head() {
    return tr('Kernel measured', 'Kernel được đo');
  },
  cell: (r) =>
    r.name === 'tiled_5_3'
      ? 'lab 5.3 (T = 16)'
      : r.name === 'previous'
        ? tr(`lab ${String(r.kernel)} (the stage before)`, `lab ${String(r.kernel)} (bước trước)`)
        : r.name === 'stage'
          ? tr(`this stage (${String(r.kernel)})`, `bước này (${String(r.kernel)})`)
          : r.name === 'vendor'
            ? tr('vendor library', 'thư viện của nhà cung cấp')
            : r.name,
};
const shapeCol: Col = {
  get head() {
    return tr('Shape', 'Hình dạng');
  },
  cell: (r) => SHAPE_NAME[String(r.shape)] ?? String(r.shape ?? ''),
};
const mknCol: Col = { head: 'M × K × N', num: true, cell: (r) => `${String(r.m)} × ${String(r.k)} × ${String(r.n)}` };
const STAGE_ROWS = new Set(['tiled_5_3', 'previous', 'stage', 'vendor']);
const stageFilter = (r: BenchRow) => STAGE_ROWS.has(r.name);
const stageCols = (width: boolean): Col[] => [
  stageKernel,
  shapeCol,
  mknCol,
  ...(width ? [rawCol(tr('Width', 'Độ rộng'), 'width')] : []),
  msCol,
  numCol('GFLOP/s', 'gflops'),
];
const linearKernel: Col = {
  get head() {
    return tr('Kernel measured', 'Kernel được đo');
  },
  cell: (r) =>
    r.name === 'm2'
      ? 'M2 (lab 6.4, T = 16)'
      : r.name === 'm3'
        ? tr('M3 (this lab)', 'M3 (lab này)')
        : r.name === 'vendor'
          ? String(r.library ?? tr('vendor library', 'thư viện của nhà cung cấp'))
          : r.name,
};

// ---------------------------------------------------------------- what the device said about each kernel
/** Lab 15.6 spells the Metal pipeline's fields as Metal does; labs 15.1 to 15.5 in snake case. */
const pipelineOf = (r: BenchRow) => ({
  cap: num(r.max_threads_per_threadgroup) ?? num(r.maxTotalThreadsPerThreadgroup),
  simd: num(r.simd_width) ?? num(r.threadExecutionWidth),
  bytes: num(r.static_threadgroup_bytes) ?? num(r.staticThreadgroupMemoryLength),
});

/** The pipeline row (Metal) or occupancy row (CUDA, HIP) of a lab's benchmark, per backend. */
function KernelFacts({ lab }: { lab: Lab }) {
  const { reports, loading } = useLabs();
  const sets = preferredBench(reports, lab);
  const lines = sets.flatMap((s) =>
    s.rows
      .filter((r) => r.name === 'pipeline' || r.name === 'occupancy')
      .map((r) => {
        if (r.name === 'pipeline') {
          const p = pipelineOf(r);
          return {
            key: s.backend,
            text: tr(
              `${s.backend} (${s.device}): the compiled pipeline accepts at most ${p.cap ?? '?'} threads per threadgroup, runs SIMD-groups of ${p.simd ?? '?'}, and uses ${p.bytes != null ? fmt(p.bytes) : '?'} bytes of static threadgroup memory.`,
              `${s.backend} (${s.device}): pipeline đã biên dịch chấp nhận tối đa ${p.cap ?? '?'} luồng mỗi threadgroup, chạy SIMD group rộng ${p.simd ?? '?'}, và dùng ${p.bytes != null ? fmt(p.bytes) : '?'} byte bộ nhớ threadgroup tĩnh.`,
            ),
          };
        }
        const blocks = num(r.blocks_per_sm);
        const regs = num(r.registers_per_thread);
        return {
          key: s.backend,
          text:
            blocks == null || blocks < 0
              ? tr(
                  `${s.backend} (${s.device}): the occupancy query was not available in this build.`,
                  `${s.backend} (${s.device}): truy vấn occupancy không có trong bản build này.`,
                )
              : tr(
                  `${s.backend} (${s.device}): the runtime fits ${blocks} block${blocks === 1 ? '' : 's'} of ${String(r.threads_per_block ?? 256)} threads per SM${regs != null ? `, at ${regs} registers per thread` : ''}.`,
                  `${s.backend} (${s.device}): runtime xếp vừa ${blocks} khối ${String(r.threads_per_block ?? 256)} luồng mỗi SM${regs != null ? `, với ${regs} thanh ghi mỗi luồng` : ''}.`,
                ),
        };
      }),
  );
  return (
    <p className="status-line" data-kernel-facts={lab.num}>
      {loading
        ? tr('Reading reports.', 'Đang đọc báo cáo.')
        : lines.length
          ? lines.map((l) => <span key={l.key}>{l.text} </span>)
          : tr(
              'The benchmark also records what the device says about the compiled kernel (blocks per SM on CUDA and HIP, the pipeline’s limits on Metal); there is no such row on this machine yet.',
              'Benchmark cũng ghi lại những gì thiết bị nói về kernel đã biên dịch (số khối mỗi SM trên CUDA và HIP, các giới hạn của pipeline trên Metal); máy này chưa có dòng nào như vậy.',
            )}
    </p>
  );
}

// ---------------------------------------------------------------- the ladder
/** Every rung at 2048 cubed: each stage in its own run, and against the rung below in the same run. */
function Ladder() {
  const { reports, loading } = useLabs();
  const backends = [...new Set(STAGES.flatMap((l) => preferredBench(reports, l).map((s) => s.backend)))];
  const rungs = backends.map((b) => {
    const setOf = (l: Lab) => preferredBench(reports, l).find((s) => s.backend === b);
    const rows = STAGES.map((l, i) => {
      const s = setOf(l);
      const mine = gflops(benchRow(s, 'stage', CUBE));
      const below = gflops(benchRow(s, i === 0 ? 'tiled_5_3' : 'previous', CUBE));
      return { lab: l, s, mine, ratio: mine && below ? mine / below : undefined };
    });
    const first = setOf(l1);
    const last = STAGES.map(setOf).filter(Boolean).pop();
    return {
      b,
      device: rows.find((r) => r.s)?.s?.device ?? b,
      base: gflops(benchRow(first, 'tiled_5_3', CUBE)),
      vendor: gflops(benchRow(last, 'vendor', CUBE)),
      rows,
    };
  });
  const any = rungs.some((r) => r.rows.some((x) => x.mine));
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
      {loading ? (
        <p className="muted">
          <Translated vi="Đang đọc báo cáo.">Reading reports.</Translated>
        </p>
      ) : !any ? (
        <Translated
          vi={
            <p className="muted" data-ladder="none">
              Chưa bước nào có benchmark ở {shapeText(CUBE)} trên máy này. Hãy chạy{' '}
              <code>python3 labs/pmpp.py run 15.1 --solution --bench</code> (và 15.2 đến 15.5) trên một máy có GPU, rồi tải
              lại.
            </p>
          }
        >
          <p className="muted" data-ladder="none">
            No stage has a benchmark at {shapeText(CUBE)} on this machine yet. Run{' '}
            <code>python3 labs/pmpp.py run 15.1 --solution --bench</code> (and 15.2 to 15.5) on a machine with a GPU,
            then reload.
          </p>
        </Translated>
      ) : (
        rungs.map((g) => (
          <table className="data" key={g.b} data-ladder={g.b}>
            <thead>
              <tr>
                <th>
                  {g.b} <span className="muted">({g.device})</span>, {shapeText(CUBE)}
                </th>
                <th className="num">GFLOP/s</th>
                <th className="num">
                  <Translated vi="× bậc bên dưới, cùng lần chạy">× the rung below, same run</Translated>
                </th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>
                  <Translated vi="lab 5.3 (T = 16), từ lần chạy của lab 15.1">lab 5.3 (T = 16), from lab 15.1&rsquo;s run</Translated>
                </td>
                <td className="num">{g.base ? fmt(g.base) : ''}</td>
                <td className="num" />
              </tr>
              {g.rows.map(({ lab, s, mine, ratio }) => (
                <tr key={lab.num} data-rung={lab.num}>
                  <td>
                    <Translated vi={<>lab {lab.num}, {localized(lab.name, lab.vi?.name)}</>}>
                      lab {lab.num}, {lab.name}
                    </Translated>
                    {s ? (
                      <span className="muted">
                        {' '}
                        <Translated vi={<>({s.variant === 'work' ? 'của bạn' : 'lời giải tham chiếu'}, {s.date})</>}>
                          ({s.variant === 'work' ? 'yours' : 'reference solution'}, {s.date})
                        </Translated>
                      </span>
                    ) : null}
                  </td>
                  <td className="num">{mine ? fmt(mine) : ''}</td>
                  <td className="num">{ratio ? fmt(ratio) : ''}</td>
                </tr>
              ))}
              <tr>
                <td>
                  <Translated vi="thư viện của nhà cung cấp, từ lần chạy của lab 15.5">vendor library, from lab 15.5&rsquo;s run</Translated>
                </td>
                <td className="num">{g.vendor ? fmt(g.vendor) : ''}</td>
                <td className="num" />
              </tr>
            </tbody>
          </table>
        ))
      )}
    </div>
  );
}

// ---------------------------------------------------------------- the race backend
type RaceResult = RunResult & { tsan?: unknown[] };
/** Every lab of the page on the race backend, which the status table does not show. */
function RaceTable() {
  const { reports, loading } = useLabs();
  const run = reports.race;
  const line = (l: Lab, variant: 'solution' | 'work') => {
    const rs = (run?.results.filter((x) => x.lab === l.id && x.variant === variant) ?? []) as RaceResult[];
    const tests = rs.flatMap((x) => x.tests);
    const races = rs.reduce((n, x) => n + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0);
    const s = labStatus(run, l.id, variant);
    const text =
      s === 'none'
        ? variant === 'work'
          ? tr('not started', 'chưa bắt đầu')
          : tr('no run on this machine', 'chưa chạy trên máy này')
        : tr(
            `${s === 'pass' ? 'passes' : 'not passing'}: ${tests.filter((t) => t.pass).length} of ${tests.length} tests, ${races ? `${races} ThreadSanitizer report${races === 1 ? '' : 's'}` : 'no race reported'}`,
            `${s === 'pass' ? 'đạt' : 'chưa đạt'}: ${tests.filter((t) => t.pass).length} trên ${tests.length} bài test, ${races ? `${races} báo cáo ThreadSanitizer` : 'không có race nào được báo cáo'}`,
          );
    return { s, text };
  };
  if (loading)
    return (
      <p className="muted">
        <Translated vi="Đang đọc báo cáo.">Reading reports.</Translated>
      </p>
    );
  if (!run)
    return (
      <Translated vi={<p data-race="none">Máy này chưa có lần chạy nào trên backend race.</p>}>
        <p data-race="none">No run on the race backend on this machine yet.</p>
      </Translated>
    );
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
      <table className="data" data-table="race">
        <thead>
          <tr>
            <th>
              <Translated vi="Trên backend race">On the race backend</Translated>
            </th>
            <th>
              <Translated vi="Lời giải tham chiếu">Reference solution</Translated>
            </th>
            <th>
              <Translated vi="Bản làm của bạn">Your work</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          {LABS.map((l) => {
            const sol = line(l, 'solution');
            const mine = line(l, 'work');
            return (
              <tr key={l.id} data-race-lab={l.num} data-race-solution={sol.s} data-race-work={mine.s}>
                <td>
                  <Translated vi={<>Lab {l.num}, {localized(l.name, l.vi?.name)}</>}>
                    Lab {l.num}, {l.name}
                  </Translated>
                </td>
                <td>{sol.s === 'pass' ? `emulator-checked; ${sol.text}` : sol.text}</td>
                <td>{mine.text}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------- milestone M3
function MilestoneStatus() {
  const { reports, loading } = useLabs();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const pass = backends.filter((b) => labStatus(reports[b], l6.id, 'work') === 'pass');
  const name = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  return (
    <p data-milestone={loading ? 'loading' : pass.join(',') || 'none'}>
      {loading ? (
        tr('Reading reports.', 'Đang đọc báo cáo.')
      ) : !backends.length ? (
        tr('No lab results are recorded on this machine yet.', 'Máy này chưa ghi nhận kết quả lab nào.')
      ) : pass.length ? (
        <Translated
          vi={
            <>
              <code>launchLinear</code> của chính bạn đạt các bài test của lab 15.6 trên {pass.map(name).join(' và ')} ở máy
              này: mốc dự án M3 đã có ở đó. Nó so với M2 và thư viện thế nào là việc của benchmark.
            </>
          }
        >
          <>
            Your own <code>launchLinear</code> passes lab 15.6&rsquo;s tests on {pass.map(name).join(' and ')} on this
            machine: milestone M3 is in place there. How it compares with M2 and the library is the benchmark&rsquo;s to
            say.
          </>
        </Translated>
      ) : (
        <Translated
          vi={
            <>
              <code>launchLinear</code> của chính bạn chưa đạt các bài test của lab 15.6 trên backend nào của máy này. Cho tới
              khi đạt, lớp tuyến tính của dự án là M2.
            </>
          }
        >
          <>
            Your own <code>launchLinear</code> does not pass lab 15.6&rsquo;s tests on any backend of this machine yet.
            Until it does, the project&rsquo;s linear layer is M2.
          </>
        </Translated>
      )}
    </p>
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
              Bảy lab, mỗi lab trong thư mục riêng dưới <code>labs/ch15/</code>. Lab 15.1 đến 15.5 là một chiếc thang theo
              cách bố trí của sách, C = A B với cả hai đầu vào theo thứ tự hàng (row-major): mỗi lab bắt đầu từ lời giải của lab
              trước với một phần bị lấy đi, và mỗi benchmark đo bước của bạn bên cạnh kernel của lab 5.3, bước trước nó và GEMM
              của nhà cung cấp. Lab 15.6 là mốc dự án M3, cùng kernel với lớp tuyến tính của dự án. Lab 15.7 giấu ba lỗi trong
              đó để bạn tìm. Mã bạn không sửa, các bài test và benchmark dùng chung và các lời gọi tới nhà cung cấp, nằm trong{' '}
              <code>labs/ch15/shared/</code> và trong thư mục <code>provided/</code> của lab 15.6.
            </>
          }
        >
          <>
            Seven labs, each in its own directory under <code>labs/ch15/</code>. Labs 15.1 to 15.5 are one ladder in the
            book&rsquo;s layout, C = A B with both inputs row-major: each starts from the previous lab&rsquo;s solution with
            one part taken out, and each benchmark times your stage beside lab 5.3&rsquo;s kernel, the stage before it and the
            vendor&rsquo;s GEMM. Lab 15.6 is milestone M3, the same kernel as the project&rsquo;s linear layer. Lab 15.7
            hides three defects in it for you to find. Code you do not edit, the shared tests and benchmark and the vendor
            calls, sits in <code>labs/ch15/shared/</code> and in lab 15.6&rsquo;s <code>provided/</code> folder.
          </>
        </Translated>
      </LabsIntro>
      <RunningLabs
        example="15.1"
        alsoShownIn={
          <Translated
            vi={
              <>
                <a href={href('ch15', 'start')}>trang bắt đầu</a> và các mục <a href={href('ch15', '15.7')}>15.7</a> và{' '}
                <a href={href('ch15', '15.9')}>15.9</a>
              </>
            }
          >
            <>
              the <a href={href('ch15', 'start')}>start page</a> and sections <a href={href('ch15', '15.7')}>15.7</a> and{' '}
              <a href={href('ch15', '15.9')}>15.9</a>
            </>
          </Translated>
        }
      />
      <BackendsAndEmulator />
      <Translated
        vi={
          <p>
            Mọi lab ở đây cũng được build cho biến thể thứ hai của trình giả lập (emulator), <code>race</code>, chạy các luồng
            của một khối cùng lúc dưới ThreadSanitizer và nêu tên hai dòng mã nguồn của bất kỳ cặp truy cập không đồng bộ nào.
            Yêu cầu nó bằng <code>--backend race</code>. Các hình dạng của nó nhỏ, một đến bốn khối, vì một khối 256 luồng dưới
            ThreadSanitizer chạy chậm. Các bảng trạng thái ở trên cho thấy biến thể thứ nhất; bảng này cho thấy biến thể thứ
            hai.
          </p>
        }
      >
        <p>
          Every lab here is also built for the emulator&rsquo;s second flavour, <code>race</code>, which runs one block&rsquo;s
          threads at the same time under ThreadSanitizer and names the two source lines of any unsynchronised pair of
          accesses. Ask for it with <code>--backend race</code>. Its shapes are small, one to four blocks, because a
          256-thread block under ThreadSanitizer is slow. The status tables above show the first flavour; this one shows
          the second.
        </p>
      </Translated>
      <RaceTable />
      <Translated
        vi={
          <p>
            Những gì trình giả lập không thấy quan trọng ở chương này hơn mọi chương trước. Lab 15.2 đến 15.5 không thay đổi
            điều gì mà một bài test quan sát được: phân ô thanh ghi, mẫu ghi, đệm và pipeline đều cho cùng C. Chúng chỉ khác
            nhau về tốc độ, nên trên trình giả lập chúng chỉ là kiểm tra tính đúng, và dự đoán cho mỗi lab được GPU phân xử.
            Còn một khe hở về tính đúng: CPU không bận tâm đến một <code>float4</code> lệch căn chỉnh, nên một truy cập vector
            thiếu điều kiện căn chỉnh vẫn đạt trên <code>emu</code> và thất bại trên GPU.
          </p>
        }
      >
        <p>
          What the emulators cannot see matters more in this chapter than in any before it. Labs 15.2 to 15.5 change
          nothing a test can observe: register tiling, the store pattern, padding and pipelining all give the same C.
          They differ only in speed, so on the emulators they are correctness checks and nothing else, and the prediction
          for each is settled by a GPU. One correctness gap too: a CPU does not mind a misaligned <code>float4</code>, so a
          vector access without its alignment condition passes on <code>emu</code> and fails on a GPU.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Các benchmark đo GEMM của nhà cung cấp bên cạnh của bạn: <code>cublasSgemm</code> trên CUDA,{' '}
            <code>hipblasSgemm</code> trên HIP, <code>MPSMatrixMultiplication</code> trên Metal, mỗi cái được đối chiếu với
            tham chiếu CPU trước. Cả ba tính bằng FP32 trừ khi được yêu cầu khác, và benchmark in ra chế độ tính toán. Các
            lời gọi cuBLAS và hipBLAS chưa bao giờ được build trên máy CUDA hay ROCm (unverified). Nếu chúng không build được,
            hãy cấu hình một lần với <code>cmake -S labs -B labs/build</code> và <code>-DPMPP_CH15_VENDOR=OFF</code> cho lab
            15.1 đến 15.5, <code>-DPMPP_CH15_VENDOR_BLAS=OFF</code> cho lab 15.6, hoặc{' '}
            <code>-DPMPP_CH15_OCCUPANCY_API=OFF</code> nếu truy vấn occupancy là thứ thất bại. CMake giữ thiết lập này cho mọi
            lần chạy <code>pmpp.py</code> về sau.
          </p>
        }
      >
        <p>
          The benchmarks time the vendor&rsquo;s GEMM beside yours: <code>cublasSgemm</code> on CUDA,{' '}
          <code>hipblasSgemm</code> on HIP, <code>MPSMatrixMultiplication</code> on Metal, each checked against the CPU
          reference first. All three compute in FP32 unless told otherwise, and the benchmark prints the math mode. The
          cuBLAS and hipBLAS calls have never been built on a CUDA or ROCm machine (unverified). If they do not build,
          configure once with <code>cmake -S labs -B labs/build</code> and <code>-DPMPP_CH15_VENDOR=OFF</code> for labs
          15.1 to 15.5, <code>-DPMPP_CH15_VENDOR_BLAS=OFF</code> for lab 15.6, or{' '}
          <code>-DPMPP_CH15_OCCUPANCY_API=OFF</code> if the occupancy query is what fails. CMake keeps the setting for
          every later <code>pmpp.py</code> run.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch15-facts.md A10, B6, C6)', '2026-11-20 (notes/today/ch15-facts.md, mục A10, B6, C6)')}>
        <Translated
          vi={
            <p>
              cuBLAS và hipBLAS theo thứ tự cột (column-major), nên một tích theo thứ tự hàng được yêu cầu dưới dạng Cᵀ = Bᵀ
              Aᵀ. TF32 và BF16x9 (cuBLAS), XF32 (hipBLAS) và độ chính xác giảm (MPS) đều phải chủ động bật.
            </p>
          }
        >
          <p>
            cuBLAS and hipBLAS are column-major, so a row-major product is asked for as Cᵀ = Bᵀ Aᵀ. TF32 and BF16x9 (cuBLAS),
            XF32 (hipBLAS) and reduced precision (MPS) are all opt-in.
          </p>
        </Translated>
      </Today>
      <VerificationStatus />

      {/* ------------------------------------------------------------ 15.1 */}
      <h2>
        <Translated vi="Lab 15.1: kernel gộp việc 128 × 128">Lab 15.1: the 128 × 128 coarsened kernel</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn viết kernel ô lớn của sách cho C[M × N] = A[M × K] B[K × N], thứ tự hàng, kích thước bất kỳ. Một khối 256 luồng
            tính một ô 128 × 128 của C. Mỗi luồng sở hữu một mảng 8 × 8 trong đó và giữ mảng trong thanh ghi (register), trong
            khi chiều trong chảy qua bộ nhớ chia sẻ (shared memory) mỗi lần 8 cột của A và 8 hàng của B. Bốn hàm trợ giúp,{' '}
            <code>clear</code>, <code>loadTile</code>, <code>mm</code> và <code>writeTile</code>, rồi đến kernel và lần khởi
            chạy của nó. Mục <a href={href('ch15', '15.3')}>15.3</a> đi qua từng cái.
          </p>
        }
      >
        <p>
          You write the book&rsquo;s large-tile kernel for C[M × N] = A[M × K] B[K × N], row-major, any sizes. A block of
          256 threads computes a 128 × 128 tile of C. Each thread owns an 8 × 8 patch of it and keeps the patch in
          registers, while the inner dimension streams through shared memory 8 columns of A and 8 rows of B at a time. Four
          helpers, <code>clear</code>, <code>loadTile</code>, <code>mm</code> and <code>writeTile</code>, then the kernel
          and its launch. Section <a href={href('ch15', '15.3')}>15.3</a> walks through each of them.
        </p>
      </Translated>
      <Source section="15.3" pages="352–357" figs={tr('Figs. 15.3 to 15.7', 'Figs. 15.3 đến 15.7')} />
      <Translated
        vi={
          <p>
            Những chỗ lab khác trang sách, một cách cố ý: khối là một chiều, 256 luồng (phần văn nói 16 × 16, nhưng Fig. 15.3
            chỉ đọc <code>threadIdx.x</code>); <code>A_s</code> và <code>B_s</code> được khai báo một lần, ở đầu kernel; kích
            thước là <code>int</code> và các đối số là <code>(A, B, C, M, K, N)</code>, theo thứ tự của lab 5.3, để các chỗ gọi
            của mốc dự án M2 chạy không cần đổi; và các hàm trợ giúp là <code>static</code>, vì các lab sau liên kết lời giải
            này vào benchmark của chúng và trình liên kết của trình giả lập sẽ lặng lẽ gộp hai hàm inline không static có cùng
            tên.
          </p>
        }
      >
        <p>
          Where the lab differs from the page, on purpose: the block is one-dimensional, 256 threads (the prose says 16 ×
          16, but Fig. 15.3 reads only <code>threadIdx.x</code>); <code>A_s</code> and <code>B_s</code> are declared once,
          at the top of the kernel; sizes are <code>int</code> and the arguments are <code>(A, B, C, M, K, N)</code>, lab
          5.3&rsquo;s order, so milestone M2&rsquo;s call sites work unchanged; and the helpers are <code>static</code>,
          because later labs link this solution into their benchmark and the emulator&rsquo;s linker would silently merge
          two non-static inline functions with one name.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              <code>__launch_bounds__(256)</code> là của chúng tôi: không có nó HIP biên dịch cho khối lớn nhất mà thiết bị
              chấp nhận và có thể ép số thanh ghi (ch15-facts B3). Trên Metal các hàm trợ giúp là <code>METAL_FUNC</code>,
              macro trong header của Apple, không có trong đặc tả (C4).
            </p>
          }
        >
          <p>
            <code>__launch_bounds__(256)</code> is ours: without it HIP compiles for the largest block a device accepts
            and may squeeze registers (ch15-facts B3). On Metal the helpers are <code>METAL_FUNC</code>, Apple&rsquo;s
            header macro, not in the specification (C4).
          </p>
        </Translated>
      </Beyond>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <Translated
              vi={<><code>clear</code>, <code>loadTile</code>, <code>mm</code>, <code>writeTile</code>, <code>gemmCoarseKernel</code> và <code>launchGemmCoarse</code></>}
            >
              <><code>clear</code>, <code>loadTile</code>, <code>mm</code>, <code>writeTile</code>, <code>gemmCoarseKernel</code> and <code>launchGemmCoarse</code></>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated vi={<><code>gemm_coarse</code> và cùng bốn hàm trợ giúp</>}>
              <><code>gemm_coarse</code> and the same four helpers</>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated vi={<><code>encodeGemmCoarse</code>: các threadgroup và các binding</>}>
              <><code>encodeGemmCoarse</code>: the threadgroups and the bindings</>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 15.1</code>, bắt đầu với <code>--backend emu</code>. Mỗi hình dạng phá một
            thứ: một phần tử, một khối với một cặp ô, ba cặp ô, ô dở dang theo mọi hướng với K và N không phải bội số của 4,
            một hàng decode ngang qua ba khối cột, K = 1, bốn khối đầy đủ, một N hẹp, và K = 0, một tổng rỗng mà C phải ra
            toàn số 0. Một GPU thêm một hình vuông 1024 và ba hình dạng lớn kỳ quặc. C bắt đầu là NaN và được theo sau bởi
            một dải bảo vệ 128 hàng chứa một giá trị canh, nên một phần tử bạn không bao giờ ghi và một lần ghi quá hàng cuối
            đều thất bại, và thông báo nói là cái nào. A và B chứa đúng số float của chúng, nên trên <code>emu</code>{' '}
            AddressSanitizer nêu tên dòng của bất kỳ lần đọc nào vượt quá một trong hai. Trên Metal, chương trình điều khiển
            (driver) trước tiên kiểm tra pipeline chấp nhận 256 luồng mỗi threadgroup, vì Metal có thể đặt giới hạn thấp hơn
            mức kernel yêu cầu.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 15.1</code>, starting with <code>--backend emu</code>. The shapes each break
          one thing: a single element, one block with one tile pair, three tile pairs, partial tiles in every direction
          with K and N not multiples of 4, one decode row across three column blocks, K = 1, four full blocks, a narrow N,
          and K = 0, an empty sum whose C must come out all zeros. A GPU adds a 1024 square and three large awkward
          shapes. C starts as NaN and is followed by a guard band of 128
          rows holding a sentinel, so an element you never write and a store past the last row both fail, and the message
          says which. A and B hold exactly their floats, so on <code>emu</code> AddressSanitizer names the line of any
          read past either. On Metal the driver first checks that the pipeline accepts 256 threads per threadgroup, since
          Metal may set a lower limit than the kernel asks for.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Với <code>--bench</code>, lời giải tham chiếu của lab 5.3, kernel của bạn và GEMM của nhà cung cấp lần lượt chạy
            trên các hình vuông 1024, 2048 và 4096, lớp 1024 → 4096 ở batch 1, 16, 256 và 1024, và hình dạng K ngắn 4096 ×
            128 × 4096. Hai dự đoán trước.
          </p>
        }
      >
        <p>
          With <code>--bench</code>, lab 5.3&rsquo;s reference solution, your kernel and the vendor&rsquo;s GEMM take turns
          on squares of 1024, 2048 and 4096, the 1024 → 4096 layer at batch 1, 16, 256 and 1024, and the short-K shape
          4096 × 128 × 4096. Two predictions first.
        </p>
      </Translated>
      <LabPrediction
        id="ch15/labs/p-lab1-speedup"
        prompt={
          <Translated
            vi={
              <>
                Ở {shapeText(CUBE)}, bạn sẽ đo được kernel của bạn nhanh hơn của lab 5.3 bao nhiêu lần? Cân nhắc ba điều. Mô
                hình đi từ 4 FLOP/B với ô 16 × 16 của lab 5.3 lên 32 với 128 × 128: lưu lượng toàn cục được yêu cầu ít hơn tám
                lần. Vòng lặp trong của Fig. 15.6 vẫn đọc hai giá trị từ bộ nhớ chia sẻ cho mỗi phép nhân-cộng, như của lab
                5.3. Và bộ nhớ đệm (cache) đã phục vụ phần lớn lưu lượng của lab 5.3 trên máy mà khóa học này được chạy thử.
              </>
            }
          >
            <>
              At {shapeText(CUBE)}, what speed-up of your kernel over lab 5.3&rsquo;s will you measure? Weigh three things.
              The model goes from 4 FLOP/B for lab 5.3&rsquo;s 16 × 16 tiles to 32 for 128 × 128: eight times less global
              traffic requested. The inner loop of Fig. 15.6 still reads two values from shared memory for every
              multiply-add, as lab 5.3&rsquo;s does. And caches already served much of lab 5.3&rsquo;s traffic on the
              machine this course was piloted on.
            </>
          </Translated>
        }
        unit={tr('× lab 5.3', '× lab 5.3')}
        measure={stageRatio(l1, 'stage', 'tiled_5_3', CUBE, tr('lab 15.1', 'lab 15.1'), tr('lab 5.3', 'lab 5.3'))}
        reading={
          <Translated
            vi={
              <>
                Hệ số 8 của mô hình đếm số byte được yêu cầu từ bộ nhớ toàn cục, không phải thời gian. Một tỉ số thấp hơn nhiều
                cho thấy có thứ khác ngoài lưu lượng DRAM đã giới hạn lab 5.3 ở đây, hoặc giờ đang giới hạn kernel này: mục{' '}
                <a href={href('ch15', '15.4')}>15.4</a> chỉ ra nghi phạm tiếp theo.
              </>
            }
          >
            <>
              The model&rsquo;s factor of 8 counts bytes requested from global memory, not time. A ratio well under it says
              something other than DRAM traffic was limiting lab 5.3 here, or now limits this kernel: section{' '}
              <a href={href('ch15', '15.4')}>15.4</a> names the next suspect.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch15/labs/p-lab1-batch1"
        prompt={
          <Translated
            vi={
              <>
                Ở batch 1 trên lớp 1024 → 4096 ({shapeText(LAYER_1)}), tốc độ kernel của bạn bằng bao nhiêu lần của lab 5.3.
                Đếm số khối mỗi kernel khởi chạy (benchmark in ra) và so với số SM của thiết bị từ{' '}
                <code>python3 labs/pmpp.py doctor</code>. Dưới 1 nghĩa là chậm hơn.
              </>
            }
          >
            <>
              At batch 1 on the 1024 → 4096 layer ({shapeText(LAYER_1)}), your kernel&rsquo;s speed as a multiple of lab
              5.3&rsquo;s. Count the blocks each one launches (the benchmark prints them) and compare with your
              device&rsquo;s number of SMs from <code>python3 labs/pmpp.py doctor</code>. Below 1 means slower.
            </>
          </Translated>
        }
        unit={tr('× lab 5.3', '× lab 5.3')}
        measure={stageRatio(l1, 'stage', 'tiled_5_3', LAYER_1, tr('lab 15.1', 'lab 15.1'), tr('lab 5.3', 'lab 5.3'))}
        reading={
          <Translated
            vi={
              <>
                Lab 5.3 khởi chạy 256 khối 16 × 16 cho hình dạng này, của bạn 32 khối 128 × 128, mỗi khối có 127 hàng không có
                gì để làm. Metal không báo cáo số SM, nên trên máy Mac số khối là nửa duy nhất của phép so sánh bạn đọc được.
                Mục <a href={href('ch15', '15.2')}>15.2</a> có lý do kia: ở m = 1 không hình dạng ô nào nâng được cường độ tính
                toán.
              </>
            }
          >
            <>
              Lab 5.3 launches 256 blocks of 16 × 16 for this shape, yours 32 blocks of 128 × 128, each with 127 rows of
              nothing to do. Metal reports no SM count, so on a Mac the block count is the only half of the comparison you
              can read. Section <a href={href('ch15', '15.2')}>15.2</a> has the other reason: at m = 1 no tile shape raises
              the intensity.
            </>
          </Translated>
        }
      />
      <LabResults lab={l1} filter={stageFilter} cols={stageCols(false)} />
      <KernelFacts lab={l1} />
      <ReferenceSolution lab={l1} tiers={TIERS_GEMM} />

      {/* ------------------------------------------------------------ 15.2 */}
      <h2>
        <Translated vi="Lab 15.2: phân ô thanh ghi, hoán đổi các vòng lặp">Lab 15.2: register tiling, the loops interchanged</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bản làm của bạn (work) bắt đầu là lời giải của lab 15.1, đổi tên, với thân của <code>mm</code> bị xóa. Viết lại nó
            theo cách Fig. 15.9 làm: vòng lặp chiều trong <code>i</code> trở thành vòng ngoài cùng; với mỗi <code>i</code>,
            chép 8 giá trị của luồng ở cột <code>i</code> trong các hàng của nó của <code>A_s</code> vào <code>a_r</code>, và 8
            giá trị của nó ở hàng <code>i</code> trong các cột của nó của <code>B_s</code> vào <code>b_r</code>; rồi cộng tích
            ngoài vào cả 64 phần tử của ô. Unroll hoàn toàn cả năm vòng lặp, nếu không <code>a_r</code>, <code>b_r</code> và
            ô sẽ rời khỏi thanh ghi.
          </p>
        }
      >
        <p>
          Your work starts as lab 15.1&rsquo;s solution, renamed, with the body of <code>mm</code> removed. Rewrite it the
          way Fig. 15.9 does: the inner-dimension loop <code>i</code> becomes the outermost one; for each <code>i</code>,
          copy the thread&rsquo;s 8 values of column <code>i</code> of its rows of <code>A_s</code> into <code>a_r</code>,
          and its 8 values of row <code>i</code> of its columns of <code>B_s</code> into <code>b_r</code>; then add the
          outer product to all 64 elements of the tile. Unroll all five loops completely, or <code>a_r</code>,{' '}
          <code>b_r</code> and the tile leave the registers.
        </p>
      </Translated>
      <Source section="15.4" pages="357–360" figs={tr('Figs. 15.8 and 15.9', 'Figs. 15.8 và 15.9')} />
      <LabFiles
        lab={l2}
        files={[
          [
            'kernel.cu',
            <Translated vi={<>chỉ <code>mm</code></>}>
              <><code>mm</code> only</>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated vi={<>chỉ <code>mm</code></>}>
              <><code>mm</code> only</>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            <code>work/launch.mm</code> đã hoàn chỉnh: phân ô thanh ghi thay đổi kernel, không thay đổi lần khởi chạy. Các bài
            test là của lab 15.1; mã khởi đầu (starter) chưa sửa build được và thất bại, vì <code>mm</code> của nó không cộng
            gì và C ra toàn số 0. Benchmark thêm lời giải của lab 15.1 làm bước trước, được chép vào bản build khi bạn cấu
            hình.
          </p>
        }
      >
        <p>
          <code>work/launch.mm</code> is complete: register tiling changes the kernel, not the launch. The tests are lab
          15.1&rsquo;s; the untouched starter builds and fails, because its <code>mm</code> adds nothing and C comes out
          all zeros. The benchmark adds lab 15.1&rsquo;s solution as the stage before, copied into the build when you
          configure.
        </p>
      </Translated>
      <LabPrediction
        id="ch15/labs/p-lab2-speedup"
        prompt={
          <Translated
            vi={
              <>
                Ở {shapeText(CUBE)}, mức tăng tốc của bạn so với lab 15.1. Đếm trước: mỗi luồng và mỗi cặp ô, Fig. 15.6 đọc bộ
                nhớ chia sẻ 1024 lần và Fig. 15.9 128 lần. Rồi cân nhắc rằng các lần đọc dải vẫn xung đột trong cách bố trí
                luồng này, trên 32 bank dải A 2 chiều và dải B 4 chiều, và các lệnh ghi chưa thay đổi.
              </>
            }
          >
            <>
              At {shapeText(CUBE)}, your speed-up over lab 15.1. Count first: per thread and tile pair, Fig. 15.6 reads
              shared memory 1024 times and Fig. 15.9 128 times. Then weigh that the strip reads still conflict in this
              thread layout, on 32 banks the A strip 2-way and the B strip 4-way, and that the stores have not changed.
            </>
          </Translated>
        }
        unit={tr('× lab 15.1', '× lab 15.1')}
        measure={stageRatio(l2, 'stage', 'previous', CUBE, tr('lab 15.2', 'lab 15.2'), tr('lab 15.1', 'lab 15.1'))}
        reading={
          <Translated
            vi={
              <>
                Một trình biên dịch unroll hoàn toàn Fig. 15.6 thấy mọi chỉ số là hằng số và có thể tự giữ các lần đọc lặp
                lại trong thanh ghi (suy luận của chúng tôi, không phải phát biểu của nhà cung cấp). Việc hoán đổi làm cho sự
                tái sử dụng trở nên tường minh; nó có thiếu hay không là điều con số này cho bạn biết. Trên Metal, Apple không
                công bố bank lẫn số thanh ghi mỗi luồng, nên không gì được công bố dự đoán được nó.
              </>
            }
          >
            <>
              A compiler that fully unrolls Fig. 15.6 sees every index as a constant and may already keep repeated reads in
              registers on its own (our reasoning, not a vendor statement). The interchange makes the reuse explicit;
              whether it was missing is what this number tells you. On Metal, Apple documents neither banks nor registers
              per thread, so nothing documented predicted it.
            </>
          </Translated>
        }
      />
      <LabResults lab={l2} filter={stageFilter} cols={stageCols(false)} />
      <KernelFacts lab={l2} />
      <ReferenceSolution lab={l2} tiers={TIERS_GEMM} />

      {/* ------------------------------------------------------------ 15.3 */}
      <h2>
        <Translated vi="Lab 15.3: ghi được hợp nhất, bốn ô con 4 × 4 và lệnh ghi float4">
          Lab 15.3: coalesced stores, four 4 × 4 sub-tiles and float4 stores
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            Ở lab 15.1 và 15.2 mỗi luồng ghi một mảng 8 × 8, nên trong một lệnh ghi các luồng kế nhau của một warp ghi các địa
            chỉ cách nhau 8 float. Bản làm của bạn bắt đầu là lời giải của lab 15.2, biến thành một template theo độ rộng warp{' '}
            <code>W</code>. Ánh xạ lại các luồng như mục <a href={href('ch15', '15.5')}>15.5</a> mô tả, và ghi mỗi hàng của một
            ô con bằng một <code>float4</code>. Với W = 32 đó là 8 warp của sách xếp 2 × 4, ô warp 64 × 32, góc phần tư 32 × 16
            và lane xếp 8 × 4; với W = 64 cùng quy tắc cho 4 warp xếp 2 × 2, ô warp 64 × 64 và góc phần tư 32 × 32. Sách không
            đưa mã cho mục này: nó là Bài tập 3 và 2.
          </p>
        }
      >
        <p>
          In labs 15.1 and 15.2 each thread stores an 8 × 8 patch, so in one store instruction neighbouring threads of a
          warp write addresses 8 floats apart. Your work starts as lab 15.2&rsquo;s solution, made a template over the
          warp width <code>W</code>. Re-map the threads as section <a href={href('ch15', '15.5')}>15.5</a> describes, and
          store each row of a sub-tile with one <code>float4</code>. For W = 32 that is the book&rsquo;s 8 warps as 2 × 4,
          warp tiles of 64 × 32, quadrants of 32 × 16 and lanes as 8 × 4; for W = 64 the same rules give 4 warps as 2 × 2,
          warp tiles of 64 × 64 and quadrants of 32 × 32. The book gives no code for this section: it is Exercises 3 and
          2.
        </p>
      </Translated>
      <Source section="15.5" pages="360–361" figs={tr('Fig. 15.10; Exercises 2 and 3, p. 370', 'Fig. 15.10; Bài tập 2 và 3, p. 370')} />
      <Translated
        vi={
          <p>
            Đừng bao giờ viết độ rộng warp thành một con số bên trong kernel. <code>launchGemmRemap</code> đọc nó từ thiết bị
            một lần và gọi instance template tương ứng; <code>launchGemmRemapWidth</code>, mà các bài test gọi với cả hai độ
            rộng, đã được viết sẵn cho bạn. Trên Metal, host biên dịch kernel với <code>SIMD_WIDTH</code> được định nghĩa và
            gắn C lần thứ hai dưới dạng <code>device float4*</code>, như lab 6.2 làm.
          </p>
        }
      >
        <p>
          Never write the warp width as a number inside the kernel. <code>launchGemmRemap</code> reads it from the device
          once and calls the template instance for it; <code>launchGemmRemapWidth</code>, which the tests call with both
          widths, is written for you. On Metal the host compiles the kernel with <code>SIMD_WIDTH</code> defined and binds
          C a second time as <code>device float4*</code>, as lab 6.2 does.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch15-facts B1, C1; ch06-facts C3)', '2026-11-20 (ch15-facts mục B1, C1; ch06-facts mục C3)')}>
        <Translated
          vi={
            <p>
              AMD: warpSize &ldquo;can not be assumed to be a compile-time constant on the host&rdquo; (không thể giả định là
              hằng số lúc biên dịch trên host); cách được khuyến nghị là một instance template do host chọn. Apple: độ rộng chỉ
              đến từ <code>threadExecutionWidth</code> của pipeline.
            </p>
          }
        >
          <p>
            AMD: warpSize &ldquo;can not be assumed to be a compile-time constant on the host&rdquo;; its recommended route
            is a host-chosen template instance. Apple: the width comes only from the pipeline&rsquo;s{' '}
            <code>threadExecutionWidth</code>.
          </p>
        </Translated>
      </Today>
      <LabFiles
        lab={l3}
        files={[
          [
            'kernel.cu',
            <Translated
              vi={<>phép ánh xạ, <code>mm</code> trên bốn ô con, <code>writeTile</code> với lệnh ghi <code>float4</code>, và <code>launchGemmRemap</code></>}
            >
              <>the mapping, <code>mm</code> over the four sub-tiles, <code>writeTile</code> with <code>float4</code> stores, and <code>launchGemmRemap</code></>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated vi={<><code>gemm_remap</code>, với độ rộng là <code>SIMD_WIDTH</code></>}>
              <><code>gemm_remap</code>, with the width as <code>SIMD_WIDTH</code></>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated vi={<><code>encodeGemmRemap</code>, gắn C hai lần</>}>
              <><code>encodeGemmRemap</code>, binding C twice</>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Các bài test là của lab 15.1, mỗi bài chạy với cách bố trí cho W = 32 và cho W = 64 trên mọi backend, kể cả trình
            giả lập: cách bố trí là phép tính chỉ số, nên nó đúng hay không không phụ thuộc vào độ rộng của phần cứng. Trên
            CUDA và HIP, lần khởi chạy đọc độ rộng của thiết bị cũng chạy. Các hình dạng với N = 1, 3, 7, 9, 131 và 511 cần
            nhánh dự phòng vô hướng. Mã khởi đầu chưa sửa build được và thất bại: <code>writeTile</code> của nó không ghi gì,
            nên C vẫn là NaN.
          </p>
        }
      >
        <p>
          The tests are lab 15.1&rsquo;s, each run with the layout for W = 32 and for W = 64 on every backend, the
          emulators included: the layout is index arithmetic, so whether it is right does not depend on the
          hardware&rsquo;s width. On CUDA and HIP the launch that reads the device&rsquo;s width runs too. Shapes with N =
          1, 3, 7, 9, 131 and 511 need the scalar fallback. The untouched starter builds and fails: its{' '}
          <code>writeTile</code> stores nothing, so C stays NaN.
        </p>
      </Translated>
      <LabPrediction
        id="ch15/labs/p-lab3-shortk"
        prompt={
          <Translated
            vi={
              <>
                Trên hình dạng K ngắn {shapeText(SHORT_K)}, mức tăng tốc của bạn so với lab 15.2. Lệnh ghi xảy ra một lần cho
                mỗi đầu ra bất kể K dài bao nhiêu: với K = 128 có 16 cặp ô để nạp và tính cho mỗi ô được ghi, với K = 2048 có
                256.
              </>
            }
          >
            <>
              On the short-K shape {shapeText(SHORT_K)}, your speed-up over lab 15.2. Stores happen once per output however
              long K is: with K = 128 there are 16 tile pairs to load and compute for every tile stored, with K = 2048 there
              are 256.
            </>
          </Translated>
        }
        unit={tr('× lab 15.2', '× lab 15.2')}
        measure={stageRatio(l3, 'stage', 'previous', SHORT_K, tr('lab 15.3', 'lab 15.3'), tr('lab 15.2', 'lab 15.2'))}
        reading={
          <Translated
            vi={
              <>
                Việc ánh xạ lại sửa các lệnh ghi và loại bỏ xung đột bank 4 chiều của dải B, nhưng nâng xung đột của dải A từ 2
                chiều lên 8 chiều trên 32 bank: tám hàng lane đọc các hàng của <code>A_s</code> cách nhau 4 (ch15-facts E1). Lab
                15.4 thêm phần đệm loại bỏ nó. Apple không công bố quy tắc hợp nhất truy cập nào, nên trên máy Mac đây chỉ là
                một phép đo thuần túy.
              </>
            }
          >
            <>
              The remap fixes the stores and removes the B strip&rsquo;s 4-way bank conflict, but raises the A strip&rsquo;s
              from 2-way to 8-way on 32 banks: eight lane rows read rows of <code>A_s</code> 4 apart (ch15-facts E1). Lab
              15.4 adds the padding that removes it. Apple documents no coalescing rule, so on a Mac this is a plain
              measurement.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch15/labs/p-lab3-speedup"
        prompt={
          <Translated vi={<>Và ở {shapeText(CUBE)}, nơi 256 cặp ô chia sẻ mỗi lệnh ghi: mức tăng tốc của bạn so với lab 15.2.</>}>
            <>And at {shapeText(CUBE)}, where 256 tile pairs share each store: your speed-up over lab 15.2.</>
          </Translated>
        }
        unit={tr('× lab 15.2', '× lab 15.2')}
        measure={stageRatio(l3, 'stage', 'previous', CUBE, tr('lab 15.3', 'lab 15.3'), tr('lab 15.2', 'lab 15.2'))}
        reading={
          <Translated
            vi={
              <>
                So sánh hai con số trước khi giải thích con số nào. Một mẫu ghi chỉ có thể quan trọng theo tỉ lệ thời gian dành
                cho việc ghi, và tỉ lệ đó giảm khi K tăng.
              </>
            }
          >
            <>
              Compare the two numbers before you explain either. A store pattern can only matter in proportion to the time
              spent storing, and that share falls as K grows.
            </>
          </Translated>
        }
      />
      <LabResults lab={l3} filter={stageFilter} cols={stageCols(true)} />
      <KernelFacts lab={l3} />
      <ReferenceSolution lab={l3} tiers={TIERS_GEMM} />

      {/* ------------------------------------------------------------ 15.4 */}
      <h2>
        <Translated vi="Lab 15.4: đệm và nạp vector">Lab 15.4: padding and vector loads</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bản làm của bạn bắt đầu là lời giải của lab 15.3 với <code>A_s</code> không đệm và thân của <code>loadTile</code>{' '}
            bị xóa. Hai thay đổi. Thứ nhất, đệm: cho <code>A_s</code> chiều đầu bK + 1 ở mọi nơi nó được đánh chỉ số, tức là
            khai báo của nó, lệnh ghi vào bộ nhớ chia sẻ trong <code>loadTile</code>, bước nhảy hàng của các lần đọc dải trong{' '}
            <code>mm</code>, và độ lệch bắt đầu của luồng trong lời gọi tới <code>mm</code>. Sách nêu ba chỗ đầu mà không nêu
            chỗ cuối (mục <a href={href('ch15', '15.6')}>15.6</a>). Thứ hai, Bài tập 1: một lệnh nạp <code>float4</code> mỗi
            luồng mỗi ô, khi an toàn, và nhánh vô hướng với điền số 0 trong trường hợp còn lại. Ghi vào bộ nhớ chia sẻ từng
            float một: một hàng có đệm của <code>A_s</code> là 9 float, nên một lệnh ghi <code>float4</code> ở đó sẽ không
            được căn chỉnh.
          </p>
        }
      >
        <p>
          Your work starts as lab 15.3&rsquo;s solution with <code>A_s</code> unpadded and the body of{' '}
          <code>loadTile</code> removed. Two changes. First, padding: give <code>A_s</code> the leading dimension bK + 1
          everywhere it is indexed, which is its declaration, the shared store in <code>loadTile</code>, the row stride of
          the strip reads in <code>mm</code>, and the thread&rsquo;s start offset in the call to <code>mm</code>. The book
          names the first three and not the last (section <a href={href('ch15', '15.6')}>15.6</a>). Second, Exercise 1:
          one <code>float4</code> load per thread per tile, when it is safe, and the scalar path with zero fill otherwise.
          Store into shared memory one float at a time: a padded row of <code>A_s</code> is 9 floats, so a{' '}
          <code>float4</code> store there would not be aligned.
        </p>
      </Translated>
      <Source
        section="15.6"
        pages="361–363"
        figs={tr(
          'Fig. 15.11; Exercise 1, p. 370; vector loads at low occupancy, §15.7 p. 364',
          'Fig. 15.11; Bài tập 1, p. 370; nạp vector ở occupancy thấp, §15.7 p. 364',
        )}
      />
      <Translated
        vi={
          <p>
            Sách viết khai báo <code>A_s[bM*(bK + 1)]</code>. Lab đặt tên cho chiều đầu có đệm,{' '}
            <code>#define LDA_S (bK + 1)</code>, viết <code>A_s[bM * LDA_S]</code>, và dùng cùng tên đó trong mọi chỉ số vào{' '}
            <code>A_s</code>, để không chỉ số nào có thể giữ độ rộng cũ.
          </p>
        }
      >
        <p>
          The book writes the declaration <code>A_s[bM*(bK + 1)]</code>. The lab gives the padded leading dimension a
          name, <code>#define LDA_S (bK + 1)</code>, writes <code>A_s[bM * LDA_S]</code>, and uses the same name in every
          index into <code>A_s</code>, so that no index can keep the old width.
        </p>
      </Translated>
      <LabFiles
        lab={l4}
        files={[
          [
            'kernel.cu',
            <Translated vi={<>khai báo có đệm, mọi chỉ số vào <code>A_s</code>, và <code>loadTile</code></>}>
              <>the padded declaration, every index into <code>A_s</code>, and <code>loadTile</code></>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated vi={<><code>gemm_padded</code>, nơi các lệnh nạp vector đi qua các khung nhìn <code>device const float4*</code> của A và B</>}>
              <><code>gemm_padded</code>, where the vector loads go through <code>device const float4*</code> views of A and B</>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated vi={<><code>encodeGemmPadded</code>, với các binding <code>float4</code> bổ sung</>}>
              <><code>encodeGemmPadded</code>, with the extra <code>float4</code> bindings</>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Các bài test là của lab 15.3, ở cả hai độ rộng; các hình dạng có K hoặc N không phải bội số của 4 chạy nhánh dự
            phòng vô hướng. Mã khởi đầu chưa sửa build được và thất bại: <code>loadTile</code> không ghi gì, nên <code>mm</code>{' '}
            nhân bất cứ thứ gì có trong bộ nhớ chia sẻ. Độ lệch bắt đầu bị thiếu được bắt riêng: với khai báo và chiều đầu có
            đệm nhưng <code>&amp;A_s[tRow * bK]</code>, mọi hình dạng có từ 5 hàng trở lên đều thất bại. Các hình dạng có 1, 3
            hoặc 4 hàng đạt, vì mọi hàng của chúng thuộc về các luồng có ô con bắt đầu ở hàng 0 của ô, nơi hai độ lệch trùng
            nhau. Một bài test chỉ dùng ma trận nhỏ sẽ bỏ sót nó.
          </p>
        }
      >
        <p>
          The tests are lab 15.3&rsquo;s, at both widths; shapes with K or N not a multiple of 4 run the scalar fallback.
          The untouched starter builds and fails: <code>loadTile</code> writes nothing, so <code>mm</code> multiplies
          whatever was in shared memory. The missing start offset is caught on its own: with the padded declaration and
          leading dimension but <code>&amp;A_s[tRow * bK]</code>, every shape with 5 rows or more fails. Shapes with 1, 3
          or 4 rows pass, because all their rows belong to threads whose sub-tiles start at row 0 of the tile, where the
          two offsets agree. A test that used only small matrices would have missed it.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Hai thay đổi này nên đem lại gì: trên 32 bank, xung đột dải A của cách bố trí lab 15.3 đi từ 8 chiều về không, vì
            các hàng cách nhau 4 giờ bắt đầu cách nhau 36 word, ở các bank 0, 4, ..., 28. Và nạp vector giảm số lệnh nạp của
            mỗi luồng cho mỗi cặp ô từ 8 xuống 2, cách khắc phục của sách cho một kernel chạy quá ít luồng để giữ bộ nhớ bận.
          </p>
        }
      >
        <p>
          What the two changes should buy: on 32 banks the A-strip conflict of lab 15.3&rsquo;s layout goes from 8-way to
          none, because rows 4 apart now start 36 words apart, in banks 0, 4, ..., 28. And vector loads cut each
          thread&rsquo;s load instructions per tile pair from 8 to 2, the book&rsquo;s remedy for a kernel that runs too
          few threads to keep memory busy.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch15-facts A2, B2, C2, E1)', '2026-11-20 (ch15-facts mục A2, B2, C2, E1)')}>
        <Translated
          vi={
            <p>
              Bất kỳ chiều đầu lẻ nào cũng loại bỏ xung đột này trên 32 bank; trên 64 bank thì bất kỳ chiều đầu nào không chia
              hết cho 4 (phép tính của chúng tôi). MLX đệm các ô Metal của nó 16 byte, một cách làm không có tài liệu nào của
              Apple đứng sau.
            </p>
          }
        >
          <p>
            Any odd leading dimension removes this conflict on 32 banks; on 64 banks any not divisible by 4 does (our
            arithmetic). MLX pads its Metal tiles by 16 bytes, a practice with no Apple documentation behind it.
          </p>
        </Translated>
      </Today>
      <LabPrediction
        id="ch15/labs/p-lab4-speedup"
        prompt={
          <Translated vi={<>Ở {shapeText(CUBE)}, bạn sẽ đo được mức tăng tốc bao nhiêu so với lab 15.3?</>}>
            <>At {shapeText(CUBE)}, what speed-up over lab 15.3 will you measure?</>
          </Translated>
        }
        unit={tr('× lab 15.3', '× lab 15.3')}
        measure={stageRatio(l4, 'stage', 'previous', CUBE, tr('lab 15.4', 'lab 15.4'), tr('lab 15.3', 'lab 15.3'))}
        reading={
          <Translated
            vi={
              <>
                Hai thay đổi trong một bước: nếu con số làm bạn bất ngờ, hãy bỏ một cái ra và đo lại. Trên thiết bị AMD có 64
                bank, các con số xung đột là mô hình của chúng tôi; trên Metal, Apple hoàn toàn không công bố bank của bộ nhớ
                threadgroup, nên đây là một phép đo không có dự đoán nào từ nhà cung cấp.
              </>
            }
          >
            <>
              Two changes in one stage: if the number surprises you, take one out and measure again. On AMD devices with 64
              banks the conflict figures are our model; on Metal, Apple documents no threadgroup-memory banks at all, so
              this is a measurement with no prediction from the vendor.
            </>
          </Translated>
        }
      />
      <LabResults lab={l4} filter={stageFilter} cols={stageCols(true)} />
      <KernelFacts lab={l4} />
      <ReferenceSolution lab={l4} tiers={TIERS_GEMM} />

      {/* ------------------------------------------------------------ 15.5 */}
      <h2>
        <Translated vi="Lab 15.5: đệm kép và pipeline phần mềm">Lab 15.5: double buffering and software pipelining</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bản làm của bạn bắt đầu là lời giải của lab 15.4 với vòng lặp ô bị lấy ra. Đặt nó lại sao cho vòng lặp{' '}
            <code>tile</code> tính với cặp <code>tile</code> trong khi nạp cặp <code>tile + 1</code> vào một buffer thứ hai,
            với một rào chắn mỗi vòng thay vì hai: hai cặp ô chia sẻ, <code>A_s[2][...]</code> và <code>B_s[2][...]</code>;
            một phần mở đầu (prologue) nạp cặp 0 và chờ; một vòng lặp qua <code>tile &lt; numTiles - 1</code> chạy{' '}
            <code>mm</code> trên cặp <code>tile</code>, nạp cặp <code>tile + 1</code> vào các buffer kia, và chờ; và một phần
            kết (epilogue) chạy <code>mm</code> trên cặp cuối.
          </p>
        }
      >
        <p>
          Your work starts as lab 15.4&rsquo;s solution with its tile loop taken out. Put it back so that iteration{' '}
          <code>tile</code> computes with pair <code>tile</code> while it loads pair <code>tile + 1</code> into a second
          set of buffers, with one barrier per iteration instead of two: two pairs of shared tiles,{' '}
          <code>A_s[2][...]</code> and <code>B_s[2][...]</code>; a prologue that loads pair 0 and waits; a loop over{' '}
          <code>tile &lt; numTiles - 1</code> that runs <code>mm</code> on pair <code>tile</code>, loads pair{' '}
          <code>tile + 1</code> into the other buffers, and waits; and an epilogue that runs <code>mm</code> on the last
          pair.
        </p>
      </Translated>
      <Source section="15.8" pages="364–368" figs={tr('Figs. 15.12 to 15.14', 'Figs. 15.12 đến 15.14')} />
      <Translated
        vi={
          <p>
            Fig. 15.14 là một bộ khung: các đối số bị lược bỏ, các buffer không được khai báo, và kiểu con trỏ không khớp với
            các ô một chiều của Fig. 15.3. Sách hoán đổi con trỏ hiện tại và con trỏ kế tiếp mỗi vòng; chọn buffer của một cặp
            theo tính chẵn lẻ, <code>tile &amp; 1</code>, như lab 6.5 làm, cũng là cùng sự luân phiên đó. Hai trường hợp hình
            để lại cho bạn: <code>numTiles - 1</code> làm gì khi K bằng 0 (một bài test có K = 0 và giới hạn 30 giây), và phần
            kết đọc buffer nào. Mục <a href={href('ch15', '15.8')}>15.8</a> chạy từng bước qua cả hai.
          </p>
        }
      >
        <p>
          Fig. 15.14 is a skeleton: its arguments are elided, its buffers undeclared, and its pointer types do not match
          the one-dimensional tiles of Fig. 15.3. The book swaps a current and a next pointer each iteration; choosing a
          pair&rsquo;s buffers by its parity, <code>tile &amp; 1</code>, as lab 6.5 does, is the same alternation. Two
          cases the figure leaves to you: what <code>numTiles - 1</code> does when K is 0 (a test case has K = 0 and a
          30-second limit), and which buffers the epilogue reads. Section <a href={href('ch15', '15.8')}>15.8</a> steps through both.
        </p>
      </Translated>
      <LabFiles
        lab={l5}
        files={[
          [
            'kernel.cu',
            <Translated vi={<>các buffer kép và vòng lặp ô trong <code>gemmPipelinedKernel</code></>}>
              <>the double buffers and the tile loop in <code>gemmPipelinedKernel</code></>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated vi={<>tương tự trong <code>gemm_pipelined</code></>}>
              <>the same in <code>gemm_pipelined</code></>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            <code>work/launch.mm</code> đã hoàn chỉnh. Trước khi viết gì, hãy trả lời từ Fig. 15.12: rào chắn nào trong hai rào
            chắn của lab 15.4 bảo vệ một phụ thuộc thật và rào chắn nào một phụ thuộc giả, và vì sao một luồng chạy trước không
            bao giờ ghi đè lên một cặp mà các luồng lân cận vẫn đang đọc? Sau đó chạy <code>--backend emu</code>, rồi{' '}
            <code>--backend race</code>, rồi phần còn lại. Trên GPU mỗi trường hợp được khởi chạy 10 lần với C được nạp lại
            NaN, và một lần thất bại cho biết kết quả sai trong bao nhiêu lần khởi chạy. Việc bỏ rào chắn trong vòng lặp đã được
            kiểm tra là thất bại riêng lẻ: mọi trường hợp có nhiều hơn một cặp ô đều thất bại trên <code>emu</code> dưới cả ba
            thứ tự luồng, ThreadSanitizer nêu tên lệnh ghi trong <code>loadTile</code> và lệnh đọc trong <code>mm</code>, và
            trên Metal, GPU duy nhất nó được kiểm tra, một số trường hợp chỉ thất bại ở vài trong 10 lần khởi chạy. Mã khởi đầu
            chưa sửa build được và thất bại: không có vòng lặp, C ra 0.
          </p>
        }
      >
        <p>
          <code>work/launch.mm</code> is complete. Before you write anything, answer from Fig. 15.12: which of lab
          15.4&rsquo;s two barriers guards a true dependence and which a false one, and why can a thread that runs ahead
          never overwrite a pair its neighbours are still reading? Then run <code>--backend emu</code>, then{' '}
          <code>--backend race</code>, then the rest. On a GPU every case is launched 10 times with C refilled with NaN, and
          a failure says in how many launches the result was wrong. A removed loop barrier was checked to fail on its own:
          every case with more than one tile pair fails on <code>emu</code> under all three thread orders, ThreadSanitizer
          names the store in <code>loadTile</code> and the read in <code>mm</code>, and on Metal, the only GPU it was
          checked on, some cases failed in only a few of the 10 launches. The untouched starter builds and fails: without the loop, C comes out 0.
        </p>
      </Translated>
      <LabPrediction
        id="ch15/labs/p-lab5-speedup"
        prompt={
          <Translated
            vi={
              <>
                Ở {shapeText(CUBE)}, mức tăng tốc của bạn so với lab 15.4. Trước hết hãy đọc dòng của lab 15.4 ở trên về những
                gì thiết bị nói: nếu hai khối trở lên vừa trên một SM, các khối khác đã chồng lấp các lần nạp của chúng với
                phép tính của khối bạn rồi. Cân nhắc thêm rằng sự chồng lấp là việc của trình biên dịch, và bộ nhớ chia sẻ mỗi
                khối tăng gấp đôi, lên 17,408 byte khi có đệm.
              </>
            }
          >
            <>
              At {shapeText(CUBE)}, your speed-up over lab 15.4. First read lab 15.4&rsquo;s line above about what the
              device said: if two or more blocks fit on an SM, other blocks were already overlapping their loads with your
              block&rsquo;s arithmetic. Weigh too that the overlap is the compiler&rsquo;s to make, and that shared memory per
              block doubles, to 17,408 bytes with the padding.
            </>
          </Translated>
        }
        unit={tr('× lab 15.4', '× lab 15.4')}
        measure={stageRatio(l5, 'stage', 'previous', CUBE, tr('lab 15.5', 'lab 15.5'), tr('lab 15.4', 'lab 15.4'))}
        reading={
          <Translated
            vi={
              <>
                Mã nguồn chỉ đặt các lần nạp của cặp <code>tile + 1</code> và phép tính của cặp <code>tile</code> vào một vòng
                lặp không có rào chắn ở giữa; không gì buộc trình biên dịch xen kẽ chúng. Sau đó thử thứ tự ngược lại trong
                vòng lặp, nạp trước, rồi mới <code>mm</code>. Nó cũng đúng như vậy (các bài test sẽ cho bạn biết) và nó đưa cho
                trình biên dịch mã khác. Đo cả hai. Bất cứ điều gì bạn thấy là sự thật về thiết bị và trình biên dịch của bạn,
                không phải một quy tắc.
              </>
            }
          >
            <>
              The source only puts the loads of pair <code>tile + 1</code> and the arithmetic of pair <code>tile</code> in
              one iteration with no barrier between them; nothing makes the compiler interleave them. Then try the other
              order inside the loop, loads first, then <code>mm</code>. It is just as correct (the tests will tell you) and
              it hands the compiler different code. Measure both. Whatever you find is a fact about your device and your
              compiler, not a rule.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Các thư viện lấy sự chồng lấp khỏi tay trình biên dịch bằng các lần sao chép mà phần cứng chạy ở chế độ nền. Mục{' '}
            <a href={href('ch15', '15.9')}>15.9</a> nói backend nào có chúng.
          </p>
        }
      >
        <p>
          Libraries take the overlap out of the compiler&rsquo;s hands with copies the hardware runs in the background.
          Section <a href={href('ch15', '15.9')}>15.9</a> says which backends have them.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch15-facts A7, A8, A12, B5; ch06-facts C7)', '2026-11-20 (ch15-facts mục A7, A8, A12, B5; ch06-facts mục C7)')}>
        <Translated
          vi={
            <p>
              CUDA có thể chép từ bộ nhớ toàn cục sang bộ nhớ chia sẻ mà không qua thanh ghi (LDGSTS, compute capability 8.0 trở
              về sau) và Hopper thêm TMA; HIP và Metal không công bố cách sao chép bất đồng bộ nào dùng được cho việc này. Vòng
              lặp đồng bộ của lab này là dạng mà mọi backend đều có.
            </p>
          }
        >
          <p>
            CUDA can copy from global to shared memory without registers (LDGSTS, compute capability 8.0 and later) and
            Hopper adds TMA; HIP and Metal document no usable asynchronous copy for this. The synchronous loop of this lab
            is the form every backend has.
          </p>
        </Translated>
      </Today>
      <LabResults lab={l5} filter={stageFilter} cols={stageCols(true)} />
      <KernelFacts lab={l5} />
      <ReferenceSolution lab={l5} tiers={TIERS_GEMM} />

      <h3>
        <Translated vi="Chiếc thang đến giờ">The ladder so far</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Mỗi bậc ở {shapeText(CUBE)}, như benchmark của chính lab đó đo, với tỉ số so với bậc bên dưới lấy từ cùng lần chạy.
            Các lần chạy của các lab khác nhau được thực hiện vào những thời điểm khác nhau, và trên một máy bận, hai lần chạy
            của cùng một kernel có thể chênh nhau nhiều hơn hai bậc kề nhau: hãy đọc cột tỉ số, được đo cạnh nhau, trước cột
            GFLOP/s.
          </p>
        }
      >
        <p>
          Each rung at {shapeText(CUBE)}, as its own lab&rsquo;s benchmark measured it, with the ratio to the rung below
          taken from the same run. The runs of different labs were made at different times, and on a busy machine two
          runs of one kernel can differ by more than neighbouring rungs do: read the ratio column, measured side by side,
          before the GFLOP/s column.
        </p>
      </Translated>
      <Ladder />

      {/* ------------------------------------------------------------ 15.6 */}
      <h2>
        <Translated vi="Lab 15.6: lớp tuyến tính nhanh (mốc dự án M3)">Lab 15.6: the fast linear layer (milestone M3)</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn viết <code>launchLinear(X, W, Y, batch, in, out)</code>, tính Y = X Wᵀ với W lưu theo (out, in) thứ tự hàng,
            như cách <code>nn.Linear</code> và ggml lưu trọng số. Nó đúng chữ ký của lab 6.4, nên kernel này thay thế đúng chỗ
            của M2 (trên Metal, <code>encodeLinear</code> bỏ đối số độ rộng ô của M2). Bên trong là kernel lab 15.5 của bạn
            được điều chỉnh theo cách bố trí của dự án: theo tên của sách A = X (M = batch, K = in), B = Wᵀ (N = out), C = Y.{' '}
            <code>clear</code>, <code>loadTile</code>, <code>mm</code> và <code>writeTile</code> được giữ lại. Ba điều mới:{' '}
            <code>loadTileT</code>, đọc ô W theo cách W được lưu và ghi nó dạng chuyển vị vào <code>B_s[k][n]</code>; bước
            hàng (pitch) của <code>B_s</code>, mà mã khởi đầu đặt bằng <code>bN</code>, đúng nhưng xung đột; và các tên, vì W
            giờ là trọng số, tham số độ rộng là <code>WARP</code>.
          </p>
        }
      >
        <p>
          You write <code>launchLinear(X, W, Y, batch, in, out)</code>, computing Y = X Wᵀ with W stored (out, in)
          row-major, as <code>nn.Linear</code> and ggml store a weight. It is lab 6.4&rsquo;s signature exactly, so this
          kernel drops in where M2 was (on Metal, <code>encodeLinear</code> loses M2&rsquo;s tile-width argument). Inside goes your lab 15.5 kernel adapted to the project&rsquo;s layout: in the
          book&rsquo;s names A = X (M = batch, K = in), B = Wᵀ (N = out), C = Y. <code>clear</code>, <code>loadTile</code>,{' '}
          <code>mm</code> and <code>writeTile</code> carry over. Three things are new: <code>loadTileT</code>, which reads
          the W tile the way W is stored and stores it transposed into <code>B_s[k][n]</code>; the pitch of{' '}
          <code>B_s</code>, which the starter sets to <code>bN</code>, correct and conflicting; and the names, since W is
          now the weight, the width parameter is <code>WARP</code>.
        </p>
      </Translated>
      <Source section="15.3 to 15.8" pages="352–368" figs={tr('Exercises 1 to 3, p. 370', 'Bài tập 1 đến 3, p. 370')} />
      <Translated
        vi={
          <p>
            Mọi thứ điều chỉnh lab 15.5 theo cách bố trí của dự án là của chúng tôi. Lưu W dạng chuyển vị giữ cho lần đọc dải
            B là một hàng liên tiếp, như trong sách; đó là kỹ thuật corner turning của lab 6.4 ở quy mô 128 × 8. Pitch bN + 4
            = 132 làm cho lệnh ghi chuyển vị đó không xung đột trên 32 bank với cả hai cách ánh xạ nạp, điều mà trình khám phá
            của mục <a href={href('ch15', '15.6')}>15.6</a> vẽ ở chế độ thứ ba, giữ các hàng căn chỉnh 16 byte, và là 16 byte
            đệm, đúng lượng MLX dùng cho các ô của nó.
          </p>
        }
      >
        <p>
          Everything that adapts lab 15.5 to the project layout is ours. Storing W transposed keeps the B strip read a
          contiguous row, as in the book; it is lab 6.4&rsquo;s corner turning at the 128 × 8 scale. A pitch of bN + 4 =
          132 makes that transposed store conflict-free on 32 banks for both load mappings, which section{' '}
          <a href={href('ch15', '15.6')}>15.6</a>&rsquo;s explorer draws in its third mode, keeps rows 16-byte aligned, and
          is 16 bytes of padding, the amount MLX gives its tiles.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cách bố trí của dự án là của lab 6.4, từ PyTorch và ggml. Phân tích pitch là phép tính của chúng tôi với quy tắc
              bank của NVIDIA (ch15-facts A2); Apple không công bố bank (C2), nên trên Metal đó là một phép đo.
            </p>
          }
        >
          <p>
            The project layout is lab 6.4&rsquo;s, from PyTorch and ggml. The pitch analysis is our arithmetic with
            NVIDIA&rsquo;s bank rule (ch15-facts A2); Apple documents no banks (C2), so on Metal it is a measurement.
          </p>
        </Translated>
      </Beyond>
      <LabFiles
        lab={l6}
        files={[
          [
            'kernel.cu',
            <Translated vi={<><code>fastLinearKernel&lt;WARP&gt;</code> và các hàm trợ giúp của nó, bao gồm <code>loadTileT</code> và pitch <code>LDB_S</code></>}>
              <><code>fastLinearKernel&lt;WARP&gt;</code> and its helpers, <code>loadTileT</code> and the pitch <code>LDB_S</code> included</>
            </Translated>,
          ],
          ['kernel.metal', <><code>linear_m3</code></>],
          [
            'launch.mm',
            <Translated vi={<><code>encodeLinear</code> đã hoàn chỉnh; nó gắn Y, X và W lần thứ hai, dưới dạng <code>float4</code></>}>
              <><code>encodeLinear</code> is complete; it binds Y, X and W a second time, as <code>float4</code></>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Xử lý <code>in == 0</code>: <code>numTiles - 1</code> của Fig. 15.14 không được quay vòng. Một trường hợp với in =
            0 kiểm tra điều đó: Y phải ra toàn số 0, trong vòng 30 giây. Các hình dạng (batch, in, out) chạy từ một phần tử đến
            (1024, 1024, 1024) và vài token qua một lớp rộng 4096, mỗi cái ở độ rộng warp 32 và 64 trên mọi backend, cộng một
            trường hợp qua <code>launchLinear</code> như dự án gọi nó. Y bắt đầu là NaN và được theo sau bởi một dải bảo vệ;
            tham chiếu là y = x Wᵀ trên CPU bằng double, với dung sai tỉ lệ theo tổng |x w| mỗi đầu ra. Trên GPU, lời gọi tới
            nhà cung cấp được cung cấp sẵn được đối chiếu với cùng tham chiếu: một lần thất bại ở đó là lỗi của khóa học, không
            phải của bạn. Mã khởi đầu chưa sửa build được và thất bại mọi thứ trừ các kiểm tra nhà cung cấp đó.
          </p>
        }
      >
        <p>
          Handle <code>in == 0</code>: Fig. 15.14&rsquo;s <code>numTiles - 1</code> must not wrap. A case with in = 0
          checks it: Y must come out all zeros, within 30 seconds. The shapes (batch, in, out) run from one element to
          (1024, 1024, 1024) and a few tokens through a 4096-wide layer, each at warp widths
          32 and 64 on every backend, plus one case through <code>launchLinear</code> as the project calls it. Y starts as
          NaN and is followed by a guard band; the reference is y = x Wᵀ on the CPU in double, with a tolerance scaled by
          the sum of |x w| per output. On a GPU the provided vendor call is checked against the same reference: a failure
          there is the course&rsquo;s bug, not yours. The untouched starter builds and fails everything except those
          vendor checks.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Với <code>--bench</code>, ba kernel lần lượt chạy: M2 (lời giải tham chiếu của lab 6.4, T = 16, được chép vào bản
            build khi bạn cấu hình), M3 của bạn, và thư viện của nhà cung cấp, trên lớp 1024 → 4096 ở batch 1, 4, 16, 64, 256
            và 1024 và trên một lớp 4096 → 4096 ở batch 1, 16 và 256. Lớp thứ hai là lựa chọn của chúng tôi, kích thước ẩn cỡ
            mô hình 7B. Dự đoán ở batch 256, M3 của bạn tính theo phần trăm của thư viện, nằm trong mục{' '}
            <a href={href('ch15', '15.9')}>15.9</a> dưới &ldquo;Kernel của bạn so với thư viện&rdquo;; nó đọc cùng các dòng.
            Dự đoán ở đây là về batch 1.
          </p>
        }
      >
        <p>
          With <code>--bench</code>, three kernels take turns: M2 (lab 6.4&rsquo;s reference solution, T = 16, copied into
          the build when you configure), your M3, and the vendor library, on the 1024 → 4096 layer at batch 1, 4, 16, 64,
          256 and 1024 and on a 4096 → 4096 layer at batch 1, 16 and 256. The second layer is our choice, a 7B-class
          hidden size. The prediction at batch 256, your M3 as a percentage of the library, is in section{' '}
          <a href={href('ch15', '15.9')}>15.9</a> under &ldquo;Your kernel against the library&rdquo;; it reads the same
          rows. The one here is about batch 1.
        </p>
      </Translated>
      <LabPrediction
        id="ch15/labs/p-lab6-batch1"
        prompt={
          <Translated
            vi={
              <>
                Ở batch 1 trên lớp 1024 → 4096, tốc độ M3 của bạn bằng bao nhiêu lần M2. Cả ba kernel đều phải đọc toàn bộ ma
                trận trọng số một lần; một ô khối 128 hàng tính 128 hàng trong đó 127 là phần đệm. Dưới 1 nghĩa là M3 chậm hơn.
              </>
            }
          >
            <>
              At batch 1 on the 1024 → 4096 layer, your M3&rsquo;s speed as a multiple of M2&rsquo;s. All three kernels must
              read the whole weight matrix once; a 128-row block tile computes 128 rows of which 127 are padding. Below 1
              means M3 is slower.
            </>
          </Translated>
        }
        unit={tr('× M2', '× M2')}
        measure={m3OverM2Batch1}
        reading={
          <Translated
            vi={
              <>
                Sau đó xếp hạng cả ba từ bảng, kể cả thư viện. Một thư viện có thể chọn kernel khác cho mỗi hình dạng: llama.cpp
                chuyển sang kernel ma trận-vector cho tới batch 8, và chính chú thích của PyTorch gọi gemv của nó là &ldquo;bw
                bound&rdquo; (bị giới hạn bởi băng thông) (ch15-facts D1, D4). Cũng không gì đảm bảo M3 thắng M2 ở bất kỳ batch
                nào trên thiết bị của bạn: lập luận của sách được đưa ra cho thanh ghi và bank của NVIDIA, và Apple không công
                bố cái nào (C2, C3).
              </>
            }
          >
            <>
              Then rank all three from the table, library included. A library may pick a different kernel for each shape:
              llama.cpp switches to matrix-vector kernels up to batch 8, and PyTorch&rsquo;s own comment calls its gemv
              &ldquo;bw bound&rdquo; (ch15-facts D1, D4). Nothing guarantees M3 beats M2 at any batch on your device
              either: the book&rsquo;s argument is made for NVIDIA&rsquo;s registers and banks, and Apple documents neither
              (C2, C3).
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l6}
        filter={(r) => r.name === 'm2' || r.name === 'm3' || r.name === 'vendor'}
        cols={[linearKernel, rawCol(tr('Batch', 'Batch'), 'batch'), rawCol(tr('In', 'Vào'), 'in'), rawCol(tr('Out', 'Ra'), 'out'), msCol, numCol('GFLOP/s', 'gflops')]}
      />
      <KernelFacts lab={l6} />
      <ReferenceSolution lab={l6} tiers={TIERS_LINEAR} />
      <h3>
        <Translated vi="Dự án: mốc dự án M3">The project: milestone M3</Translated>
      </h3>
      <Translated
        vi={
          <p>
            M3 giữ các quy ước của M2: tensor <code>float</code> thứ tự hàng với batch đứng đầu, trọng số lưu theo (out, in),
            con trỏ device thô và kích thước tường minh, và <code>launchLinear(X, W, Y, batch, in, out)</code> bên cạnh kernel.
            Mọi mốc dự án sau cần một lớp tuyến tính đều có thể gọi một trong hai.
          </p>
        }
      >
        <p>
          M3 keeps the conventions of M2: row-major <code>float</code> tensors with the batch first, weights stored (out,
          in), raw device pointers and explicit sizes, and <code>launchLinear(X, W, Y, batch, in, out)</code> beside the
          kernel. Every later milestone that needs a linear layer can call either one.
        </p>
      </Translated>
      <MilestoneStatus />

      {/* ------------------------------------------------------------ 15.7 */}
      <h2>
        <Translated vi="Lab 15.7: săn lỗi, ba lỗi trong một lớp tuyến tính nhanh">
          Lab 15.7: bug hunt, three defects in a fast linear layer
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            <code>work/kernel.cu</code> và <code>work/kernel.metal</code> chứa một lớp tuyến tính nhanh theo cách bố trí của dự
            án, thiết kế của lab 15.6, với ba lỗi cài sẵn độc lập, giống nhau ở cả hai tệp. Nó biên dịch được, và một số hình
            dạng đạt. Hãy tìm và sửa cả ba. Đây là những sai lầm mà kernel của chương này dễ dẫn tới, và tìm chúng trong mã bạn
            không viết là kỹ năng bạn sẽ cần cho mã của chính mình.
          </p>
        }
      >
        <p>
          <code>work/kernel.cu</code> and <code>work/kernel.metal</code> hold a fast linear layer in the project&rsquo;s
          layout, lab 15.6&rsquo;s design, with three independent defects, the same in both files. It compiles, and some
          shapes pass. Find and fix all three. These are mistakes this chapter&rsquo;s kernel invites, and finding them in
          code you did not write is the skill you will need for your own.
        </p>
      </Translated>
      <Source section="15.6 and 15.8" pages="361–368" figs={tr('Fig. 15.12; Exercise 2, p. 370', 'Fig. 15.12; Bài tập 2, p. 370')} />
      <LabFiles
        lab={l7}
        files={[
          [
            'kernel.cu',
            <Translated vi={<><code>fastLinearKernel</code> và các hàm trợ giúp của nó, với ba lỗi</>}>
              <><code>fastLinearKernel</code> and its helpers, with the three defects</>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated vi={<><code>linear_m3</code>, với cùng ba lỗi</>}>
              <><code>linear_m3</code>, with the same three</>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated vi={<>được cung cấp sẵn và đúng; không có gì cần đổi</>}>
              <>provided and correct; nothing to change</>
            </Translated>,
          ],
        ]}
      />
      <Translated vi={<p>Phương pháp quan trọng hơn các bản sửa:</p>}>
        <p>The method matters more than the fixes:</p>
      </Translated>
      <ol>
        <Translated
          vi={
            <li>
              Trước khi chạy gì, hãy đọc kernel và ghi lại, với mỗi lỗi bạn nghi ngờ, những backend nào bạn dự đoán sẽ để lộ
              nó: <code>emu</code>, <code>race</code>, Metal, CUDA hay HIP.
            </li>
          }
        >
          <li>
            Before you run anything, read the kernel and write down, for each defect you suspect, which backends you
            expect to show it: <code>emu</code>, <code>race</code>, Metal, CUDA or HIP.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <code>python3 labs/pmpp.py run 15.7 --backend emu</code>. Các thứ tự luồng của nó cố định, nên một lỗi đồng bộ
              hóa thất bại mọi lần, và AddressSanitizer dừng ở lần truy cập đầu tiên ra ngoài vùng cấp phát và nêu tên dòng.
            </li>
          }
        >
          <li>
            <code>python3 labs/pmpp.py run 15.7 --backend emu</code>. Its thread orders are fixed, so a synchronisation
            defect fails every time, and AddressSanitizer stops at the first access outside an allocation and names the
            line.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Sửa một thứ rồi chạy lại; các lỗi thay đổi khi bạn đi tiếp. Đọc xem mỗi lỗi nói gì: hàng nào sai, hình dạng nào
              thất bại và hình dạng nào đạt, dải bảo vệ sau Y có thay đổi không.
            </li>
          }
        >
          <li>
            Fix one thing and run again; the failures change as you go. Read what each failure says: which rows are wrong,
            which shapes fail and which pass, whether the guard band after Y changed.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <code>python3 labs/pmpp.py run 15.7 --backend race</code>. ThreadSanitizer nêu tên hai dòng mã nguồn có truy cập
              va chạm nhau, cho biết chỗ thiếu rào chắn. Nó chỉ thấy những va chạm đã xảy ra trong lần chạy đó, nên một báo cáo
              sạch không phải là bằng chứng.
            </li>
          }
        >
          <li>
            <code>python3 labs/pmpp.py run 15.7 --backend race</code>. ThreadSanitizer names the two source lines whose
            accesses collide, which says where a barrier is missing. It sees only collisions that happened in that run,
            so a clean report is not a proof.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Sau đó trên GPU của bạn, nơi mỗi trường hợp được khởi chạy mười lần với Y được đặt lại về NaN. So sánh với những
              gì bạn đã ghi ở bước 1: GPU của bạn đã che giấu những lỗi nào?
            </li>
          }
        >
          <li>
            Then on your GPU, where each case is launched ten times with Y back to NaN. Compare with what you wrote down
            in step 1: which defects did your GPU hide?
          </li>
        </Translated>
      </ol>
      <Translated
        vi={
          <p>
            Các kiểm tra là của lab 15.6: cùng các hình dạng ở độ rộng warp 32 và 64, X và W được cấp phát đúng số float của
            chúng, và dải bảo vệ sau Y mà không kernel nào được ghi vào. Mỗi lỗi đã được kiểm tra là thất bại riêng lẻ trên
            trình giả lập và trên Metal.
          </p>
        }
      >
        <p>
          The checks are lab 15.6&rsquo;s: the same shapes at warp widths 32 and 64, X and W allocated with exactly their
          floats, and the guard band after Y that no kernel may write. Each defect was checked to fail on its own on the
          emulators and on Metal.
        </p>
      </Translated>
      <LabResults
        lab={l7}
        noBench={tr(
          'This lab has no benchmark. The result is the line above: every test passing on the emulators and on a device. As shipped, the kernel is meant to fail, and on the emulator it may abort.',
          'Lab này không có benchmark. Kết quả là dòng ở trên: mọi bài test đạt trên trình giả lập và trên một thiết bị. Như được phát hành, kernel được thiết kế để thất bại, và trên trình giả lập nó có thể bị dừng đột ngột.',
        )}
      />
      <ReferenceSolution lab={l7} tiers={TIERS_LINEAR} />
    </LabsPage>
  );
}
