import type { Question } from '../../components/Ask';
import { Aside, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
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
  type BenchSet,
  type Col,
} from '../../components/LabKit';
import { fmt, labStatus, num, realBackends, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { BOOK_DEFAULTS, llamaCuda, type LlamaPath } from '../../sim/convlib';
import { expansion, gemmShape, type ConvShape } from '../../sim/unfold';

// The page is built from components/LabKit.tsx; what is here is chapter 19's own content.
// No question on this page: the lab is the exercise. The three predictions are stored, not
// graded, under ch19/labs/p-lab1-late, p-lab1-early and p-lab1-unroll-share.
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch19', [{ num: '19.1', dir: 'lab1_convgemm', name: 'direct and unroll + GEMM', vi: { name: 'trực tiếp và unroll + GEMM' } }]);
const [l1] = LABS;
/** Lab 6.4, whose launchLinear (milestone M2) the unroll + GEMM path calls. */
const LAB64 = 'ch06/lab4_linear';

/** Identifier colours in the reference solution: the tensors live in global memory, indices and sums in registers. */
const TIERS: Record<string, Tier> = {
  X: 'global', F: 'global', Y: 'global', Bt: 'global', Xn: 'global',
  acc: 'reg', m: 'reg', h: 'reg', w: 'reg', n: 'reg', c: 'reg', p: 'reg', q: 'reg', u: 'reg', v: 'reg', r: 'reg', i: 'reg',
};

// ---------------------------------------------------------------- the benchmark's layers
/** The five layers main.cpp times, in its order (Labs.test.ts ties them to the file). Ours, not the book's. */
export const LAB_LAYERS: readonly { id: string; note: string; shape: ConvShape }[] = [
  { id: 'early', get note() { return tr('few channels, a large map', 'ít kênh, map lớn'); }, shape: { N: 8, C: 3, H: 224, W: 224, M: 64, K: 3 } },
  { id: 'middle', get note() { return tr('more channels, a smaller map', 'nhiều kênh hơn, map nhỏ hơn'); }, shape: { N: 8, C: 64, H: 56, W: 56, M: 64, K: 3 } },
  { id: 'late', get note() { return tr('many channels, a small map', 'nhiều kênh, map nhỏ'); }, shape: { N: 8, C: 256, H: 14, W: 14, M: 256, K: 3 } },
  { id: 'pointwise', get note() { return tr('the late layer with 1 × 1 filters', 'lớp late với bộ lọc 1 × 1'); }, shape: { N: 8, C: 256, H: 14, W: 14, M: 256, K: 1 } },
  { id: 'small', get note() { return tr('LeNet C3 at batch 64', 'LeNet C3 ở batch 64'); }, shape: { N: 64, C: 6, H: 14, W: 14, M: 16, K: 5 } },
];
const TILE = 16;

/** What the model says about one layer: computed, never measured. */
export function layerFacts(s: ConvShape) {
  const g = gemmShape(s, { tile: TILE });
  const tileRows = Math.ceil(g.rows / TILE);
  const tileCols = Math.ceil(g.cols / TILE);
  return {
    gemm: `${g.rows} × ${g.inner} × ${g.cols}`,
    expansion: expansion(s),
    tileRows,
    tileCols,
    phases: g.phases,
    /** Width of the last phase's slice of the reduction. */
    lastPhase: g.inner - (g.phases - 1) * TILE,
    launches: s.N,
  };
}

/** llama.cpp's CUDA and HIP choice for each lab layer (ch19-facts D1, shape conditions only). */
export const llamaFor = (s: ConvShape): LlamaPath => llamaCuda({ ...s, ...BOOK_DEFAULTS }).path;
const LLAMA_WORDS: Record<LlamaPath, string> = {
  get cublas() {
    return tr('a cuBLAS GEMM on the input', 'một GEMM của cuBLAS trên đầu vào');
  },
  get direct() {
    return tr('its direct kernel', 'kernel trực tiếp của nó');
  },
  get implicit() {
    return tr('its implicit GEMM', 'GEMM ngầm của nó');
  },
  get 'cublas-or-direct'() {
    return tr(
      'cuBLAS or its direct kernel (the facts do not say which test comes first)',
      'cuBLAS hoặc kernel trực tiếp của nó (các dữ kiện không nói phép kiểm tra nào đứng trước)',
    );
  },
};

// ---------------------------------------------------------------- reading the benchmark rows
/** The row for one path and one layer, from one backend's preferred set. */
export const convRow = (s: BenchSet | undefined, name: string, shape: string): BenchRow | undefined =>
  s?.rows.find((r) => r.name === name && r.shape === shape);
const secs = (r: BenchRow | undefined) => num(r?.seconds);
const ms = (v: number | undefined) => (v ? `${fmt(v * 1000)} ms` : '?'); // i18n-ignore: unit
const source = (r: BenchRow | undefined) => (r?.gemm_source ? `, GEMM: ${String(r.gemm_source)}` : ''); // i18n-ignore: GEMM source name

/** Predictions 1 and 2: direct time over unroll + GEMM time on one layer. Above 1, unroll + GEMM was faster. */
const speedup = (layer: string) => (reports: Reports) =>
  measure(reports, l1, (s) => {
    const d = convRow(s, 'direct', layer);
    const g = convRow(s, 'unroll_gemm', layer);
    const a = secs(d);
    const b = secs(g);
    if (!a || !b) return null;
    return {
      value: a / b,
      detail: tr(`${ms(a)} direct against ${ms(b)} unroll + GEMM${source(g)}`, `${ms(a)} trực tiếp so với ${ms(b)} unroll + GEMM${source(g)}`),
    };
  });

/** Prediction 3: the N unrolls alone, as a percentage of the whole unroll + GEMM layer, on the early layer. */
const unrollShare = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const g = convRow(s, 'unroll_gemm', 'early');
    const whole = secs(g);
    const unroll = num(g?.unroll_seconds);
    if (!whole || !unroll) return null;
    return {
      value: (100 * unroll) / whole,
      detail: tr(
        `${ms(unroll)} for the 8 unrolls alone, ${ms(whole)} for the whole layer, ${ms(num(g?.gemm_seconds))} for the 8 GEMMs alone`,
        `${ms(unroll)} cho riêng 8 lần unroll, ${ms(whole)} cho cả lớp, ${ms(num(g?.gemm_seconds))} cho riêng 8 GEMM`,
      ),
    };
  });

const MISSING = (
  <Translated
    vi={
      <>
        Phần này cần các hàng benchmark của lab 19.1 từ một thiết bị thật. Chạy <code>python3 labs/pmpp.py run 19.1 --bench</code>{' '}
        khi kernel của bạn đã qua (hoặc với <code>--solution</code> để đo thời gian bản tham chiếu). Trình giả lập (emulator)
        không bao giờ được đo hiệu năng.
      </>
    }
  >
    <>
      This needs the benchmark rows of lab 19.1 from a real device. Run <code>python3 labs/pmpp.py run 19.1 --bench</code>{' '}
      once your kernels pass (or with <code>--solution</code> to time the reference). The emulator is never benchmarked.
    </>
  </Translated>
);

// ---------------------------------------------------------------- results table
const PATH: Record<string, string> = {
  get direct() {
    return tr('direct', 'trực tiếp');
  },
  get unroll_gemm() {
    return tr('unroll + GEMM', 'unroll + GEMM');
  },
  get implicit() {
    return tr('implicit GEMM (provided)', 'GEMM ngầm (có sẵn)');
  },
};
// A function so the column heads are read in the current language when the table renders.
const COLS: () => Col[] = () => [
  { head: tr('Layer', 'Lớp'), cell: (r) => String(r.shape ?? '') },
  { head: tr('Path', 'Đường'), cell: (r) => PATH[r.name] ?? r.name },
  msCol,
  numCol('GFLOP/s', 'gflops'),
  { head: tr('Unrolls alone, ms', 'Riêng unroll, ms'), num: true, cell: (r) => (num(r.unroll_seconds) ? fmt(num(r.unroll_seconds)! * 1000) : '') },
  { head: tr('GEMMs alone, ms', 'Riêng GEMM, ms'), num: true, cell: (r) => (num(r.gemm_seconds) ? fmt(num(r.gemm_seconds)! * 1000) : '') },
  {
    head: tr('Unroll share', 'Phần của unroll'),
    num: true,
    cell: (r) => (num(r.unroll_seconds) && num(r.seconds) ? `${fmt((100 * num(r.unroll_seconds)!) / num(r.seconds)!)}%` : ''),
  },
  numCol(tr('Expansion (computed)', 'Độ phình (tính toán)'), 'expansion'),
  numCol(tr('Scratch bytes per sample', 'Byte vùng tạm mỗi mẫu'), 'scratch_bytes'),
  rawCol('GEMM', 'gemm_source'),
];

/** Each path against the direct kernel, per layer and backend, from the preferred rows. */
function AgainstDirect() {
  const { reports, loading } = useLabs();
  if (loading) return null;
  const sets = preferredBench(reports, l1);
  const rows = sets.flatMap((s) => LAB_LAYERS.filter((L) => convRow(s, 'direct', L.id)).map((L) => ({ s, L })));
  if (!rows.length)
    return (
      <Translated
        vi={
          <p className="muted" data-against="none">
            Máy này chưa có hàng benchmark nào cho lab 19.1, nên chưa có gì để so sánh. Chạy{' '}
            <code>python3 labs/pmpp.py run 19.1 --bench</code> trên một thiết bị, rồi tải lại trang.
          </p>
        }
      >
        <p className="muted" data-against="none">
          There are no benchmark rows for lab 19.1 on this machine yet, so there is nothing to compare. Run{' '}
          <code>python3 labs/pmpp.py run 19.1 --bench</code> on a device, then reload.
        </p>
      </Translated>
    );
  const ratio = (s: BenchSet, layer: string, name: string) => {
    const a = secs(convRow(s, 'direct', layer));
    const b = secs(convRow(s, name, layer));
    return a && b ? fmt(a / b) : '';
  };
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto', marginBottom: '1.2rem' }}>
      <table className="data" style={{ margin: 0 }} data-against>
        <thead>
          <tr>
            <th><Translated vi="Backend">Backend</Translated></th>
            <th><Translated vi="Kernel của ai">Whose kernels</Translated></th>
            <th><Translated vi="Lớp">Layer</Translated></th>
            <th className="num"><Translated vi="Trực tiếp, ms">Direct, ms</Translated></th>
            <th className="num"><Translated vi="Thời gian trực tiếp / thời gian unroll + GEMM">Direct time / unroll + GEMM time</Translated></th>
            <th className="num"><Translated vi="Thời gian trực tiếp / thời gian ngầm">Direct time / implicit time</Translated></th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ s, L }) => (
            <tr key={s.backend + L.id} data-backend={s.backend} data-layer={L.id}>
              <td>
                {s.backend} <span className="muted">({s.device})</span>
              </td>
              <td>{s.variant === 'work' ? tr('yours', 'của bạn') : tr('reference solution', 'lời giải tham chiếu')}</td>
              <td>{L.id}</td>
              <td className="num">{fmt(secs(convRow(s, 'direct', L.id))! * 1000)}</td>
              <td className="num" data-col="unroll_gemm">
                {ratio(s, L.id, 'unroll_gemm')}
              </td>
              <td className="num" data-col="implicit">
                {ratio(s, L.id, 'implicit')}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** One line on lab 19.1 on the race backend, which the status table does not show. */
function RaceStatus() {
  const { reports, loading } = useLabs();
  const run = reports.race;
  const line = (variant: 'solution' | 'work') => {
    const rs = (run?.results.filter((x) => x.lab === l1.id && x.variant === variant) ?? []) as (RunResult & { tsan?: unknown[] })[];
    const tests = rs.flatMap((x) => x.tests);
    const found = rs.reduce((n, x) => n + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0);
    return { s: labStatus(run, l1.id, variant), tests, passed: tests.filter((t) => t.pass).length, found };
  };
  const sol = line('solution');
  const mine = line('work');
  return (
    <p data-race-status={loading ? 'loading' : sol.s} data-race-work={loading ? 'loading' : mine.s}>
      {loading ? (
        tr('Reading reports.', 'Đang đọc báo cáo.')
      ) : !run ? (
        tr('No run on the race backend on this machine yet.', 'Máy này chưa có lần chạy nào trên backend race.')
      ) : (
        <>
          {sol.s === 'none'
            ? tr('On this machine the reference solution has no run on the race backend yet.', 'Trên máy này, lời giải tham chiếu chưa có lần chạy nào trên backend race.')
            : sol.s === 'pass'
              ? tr(
                  `On this machine the reference solution passed all ${sol.tests.length} tests on the race backend with ${sol.found ? `${sol.found} ThreadSanitizer report(s)` : 'no race reported'}: emulator-checked on this flavour.`,
                  `Trên máy này, lời giải tham chiếu qua cả ${sol.tests.length} bài kiểm tra trên backend race với ${sol.found ? `${sol.found} báo cáo ThreadSanitizer` : 'không race nào được báo'}: emulator-checked trên biến thể này.`,
                )
              : tr(
                  `On this machine the reference solution failed on the race backend (${sol.passed} of ${sol.tests.length} tests): unverified there.`,
                  `Trên máy này, lời giải tham chiếu thất bại trên backend race (${sol.passed} trên ${sol.tests.length} bài kiểm tra): unverified ở đó.`,
                )}{' '}
          {mine.s === 'none'
            ? tr('Your own kernels have no recorded run there, or are still the untouched starter.', 'Kernel của bạn chưa có lần chạy nào được ghi lại ở đó, hoặc vẫn là mã khởi đầu (starter) chưa động tới.')
            : mine.s === 'pass'
              ? tr(
                  `Your own kernels passed all ${mine.tests.length} tests there with ${mine.found ? `${mine.found} ThreadSanitizer report(s): read them` : 'no race reported'}.`,
                  `Kernel của bạn qua cả ${mine.tests.length} bài kiểm tra ở đó với ${mine.found ? `${mine.found} báo cáo ThreadSanitizer: hãy đọc chúng` : 'không race nào được báo'}.`,
                )
              : tr(
                  `Your own kernels passed ${mine.passed} of ${mine.tests.length} tests there, with ${mine.found} ThreadSanitizer report(s) in the run.`,
                  `Kernel của bạn qua ${mine.passed} trên ${mine.tests.length} bài kiểm tra ở đó, với ${mine.found} báo cáo ThreadSanitizer trong lần chạy.`,
                )}
        </>
      )}
    </p>
  );
}

/** Which GEMM this machine's runs used, and whether lab 6.4's own kernel passes yet. */
function GemmSource() {
  const { reports, loading } = useLabs();
  if (loading) return <p data-gemm-source="loading"><Translated vi="Đang đọc báo cáo.">Reading reports.</Translated></p>;
  const backends = realBackends(reports);
  const sources = [
    ...new Set(
      backends.flatMap((b) =>
        (['work', 'solution'] as const).flatMap((v) =>
          reports[b].results
            .filter((x) => x.lab === l1.id && x.variant === v)
            .flatMap((x) => x.bench)
            .map((r) => r.gemm_source)
            .filter((g): g is string => typeof g === 'string'),
        ),
      ),
    ),
  ];
  const passing = [...backends, 'emu'].filter((b) => labStatus(reports[b], LAB64, 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  return (
    <p data-gemm-source={sources.join(',') || 'none'} data-lab64={passing.join(',') || 'none'}>
      {sources.length
        ? tr(
            `The benchmark rows on this machine were timed with ${sources.map((x) => `"${x}"`).join(' and ')}.`,
            `Các hàng benchmark trên máy này được đo thời gian với ${sources.map((x) => `"${x}"`).join(' và ')}.`,
          )
        : tr(
            'There are no benchmark rows for this lab on this machine yet, so no GEMM source is recorded.',
            'Máy này chưa có hàng benchmark nào cho lab này, nên chưa ghi nhận nguồn GEMM nào.',
          )}{' '}
      {passing.length
        ? tr(
            `Your own lab 6.4 kernel passes its tests on ${passing.map(where).join(' and ')}, so PMPP_CH19_GEMM=work can use it.`,
            `Kernel lab 6.4 của bạn qua các bài kiểm tra trên ${passing.map(where).join(' và ')}, nên PMPP_CH19_GEMM=work có thể dùng nó.`,
          )
        : tr(
            'Your own lab 6.4 kernel has not passed its tests on this machine yet; until it does, PMPP_CH19_GEMM=work makes check (c) fail.',
            'Kernel lab 6.4 của bạn chưa qua các bài kiểm tra trên máy này; cho đến khi qua, PMPP_CH19_GEMM=work làm phép kiểm tra (c) thất bại.',
          )}
    </p>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  const facts = Object.fromEntries(LAB_LAYERS.map((L) => [L.id, layerFacts(L.shape)]));
  const late = facts.late;
  const early = facts.early;
  const small = facts.small;
  return (
    <LabsPage labs={LABS} tiers={TIERS}>
      <LabsIntro>
        <Translated
          vi={
            <>
              Một lab, trong <code>labs/ch19/lab1_convgemm/</code>. Nó dựng lớp của các mục{' '}
              <a href={href('ch19', '19.2')}>19.2</a> và <a href={href('ch19', '19.3')}>19.3</a> theo hai cách, trực tiếp và
              như một bước unfold tường minh theo sau là một GEMM, rồi đo thời gian chúng với nhau và với GEMM ngầm của sách.
              Nó không thêm mốc dự án nào vào dự án xuyên suốt: nó gọi mốc M2, hàm <code>launchLinear</code> của lab 6.4, không
              thay đổi.
            </>
          }
        >
          One lab, in <code>labs/ch19/lab1_convgemm/</code>. It builds the layer of sections{' '}
          <a href={href('ch19', '19.2')}>19.2</a> and <a href={href('ch19', '19.3')}>19.3</a> two ways, directly and as an
          explicit unfold followed by a GEMM, and times them against each other and against the book's implicit GEMM. It adds
          no milestone to the through-line project: it calls milestone M2, lab 6.4's <code>launchLinear</code>, unchanged.
        </Translated>
      </LabsIntro>
      <RunningLabs example="19.1" />
      <BackendsAndEmulator />
      <VerificationStatus />
      <Translated
        vi={
          <p>
            Lab này cũng chạy trên biến thể <code>race</code> của trình giả lập, với các luồng chạy đồng thời dưới
            ThreadSanitizer, vì GEMM của nó và kernel ngầm chia sẻ các ô qua các rào chắn (barrier):{' '}
            <code>python3 labs/pmpp.py run 19.1 --backend race</code>.
          </p>
        }
      >
        <p>
          This lab also runs on the <code>race</code> flavour of the emulator, with the threads concurrent under
          ThreadSanitizer, because its GEMM and the implicit kernel share tiles across barriers:{' '}
          <code>python3 labs/pmpp.py run 19.1 --backend race</code>.
        </p>
      </Translated>
      <RaceStatus />

      <h2>
        <Translated vi="Lab 19.1: một lớp tích chập theo hai cách">Lab 19.1: a convolutional layer two ways</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một lớp tích chập, của sách: bước nhảy (stride) 1, không padding, các tensor <code>float</code> theo thứ tự hàng
            (row-major) và thứ tự NCHW. X là <code>N × C × H × W</code>, F là <code>M × C × K × K</code>, Y là{' '}
            <code>N × M × H_out × W_out</code>, và P = <code>H_out · W_out</code> là số pixel trong một map đầu ra. Bạn tính
            nó theo hai cách.
          </p>
        }
      >
        <p>
          One convolutional layer, the book's: stride 1, no padding, row-major <code>float</code> tensors in NCHW order. X
          is <code>N × C × H × W</code>, F is <code>M × C × K × K</code>, Y is <code>N × M × H_out × W_out</code>, and P ={' '}
          <code>H_out · W_out</code> is the number of pixels in one output map. You compute it two ways.
        </p>
      </Translated>
      <Source section="19.2, 19.3" pages="460–472" figs="Figs. 19.5, 19.7, 19.11; Eqs. 19.4, 19.5" />
      <ol>
        <Translated
          vi={
            <li>
              <b>Trực tiếp.</b> Fig. 19.7, đã sửa: mỗi pixel đầu ra một luồng, các khối <code>TILE_WIDTH × TILE_WIDTH</code>{' '}
              luồng, lưới <code>(M, T, N)</code> với chỉ số ô được tuyến tính hóa theo từng hàng như Fig. 19.5, nhưng với phép
              chia làm tròn lên và kiểm tra biên trên mọi đầu ra. Bạn tự tay tuyến tính hóa <code>X[n, c, h + p, w + q]</code>{' '}
              của trang sách.
            </li>
          }
        >
          <li>
            <b>Direct.</b> Fig. 19.7, corrected: one thread per output pixel, blocks of <code>TILE_WIDTH × TILE_WIDTH</code>{' '}
            threads, grid <code>(M, T, N)</code> with the tile index linearised row by row as in Fig. 19.5, but with ceiling
            division and a bounds check on every output. You linearise the page's <code>X[n, c, h + p, w + q]</code> by hand.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Unroll + GEMM.</b> Một kernel ghi đầu vào đã unroll của một mẫu, ở dạng chuyển vị, thành <code>Bt</code>: P
              hàng, mỗi hàng cho một pixel đầu ra, chứa <code>C·K·K</code> giá trị của vùng thuộc pixel đó theo thứ tự (c, p, q)
              của sách. Sau đó, với mỗi mẫu, một lần unroll vào vùng tạm <code>Bt</code> và một lần gọi GEMM của dự án.
            </li>
          }
        >
          <li>
            <b>Unroll + GEMM.</b> A kernel that writes one sample's unrolled input, transposed, as <code>Bt</code>: P rows,
            one per output pixel, each holding that pixel's <code>C·K·K</code> patch values in the book's (c, p, q) order.
            Then, for each sample, one unroll into a scratch <code>Bt</code> and one call of the project's GEMM.
          </li>
        </Translated>
      </ol>
      <Translated
        vi={
          <p>
            Vì sao cả hai. Mục 19.3 lập luận rằng bước unfold tường minh tốn bộ nhớ và băng thông (p. 467) và rằng dạng GEMM
            có lợi khi các ma trận lớn (p. 472). Ở đây bạn đo cả hai khẳng định trên thiết bị của chính mình, trên các lớp
            được chọn để các ma trận lớn theo những hướng khác nhau và, một lần, nhỏ. Lý do thứ hai là của dự án: lời gọi GEMM
            là <code>launchLinear(X, W, Y, batch, in, out)</code> của lab 6.4, hàm tính <code>Y = X·Wᵀ</code> với W lưu theo
            (out, in). Chọn sáu đối số của nó sao cho Y của mỗi mẫu ra <code>M × P</code>, theo NCHW, mà F không bao giờ bị sao
            chép, chính là phép thử xem chữ ký GEMM của dự án có kết hợp được hay không. Mục 19.3 hỏi điều này dưới dạng câu
            hỏi trước.
          </p>
        }
      >
        <p>
          Why both. Section 19.3 argues that the explicit unfold costs memory and bandwidth (p. 467) and that the GEMM form
          pays when the matrices are large (p. 472). Here you measure both claims on your own device, on layers chosen to
          make the matrices large in different directions and, once, small. The second reason is the project's: the GEMM
          call is lab 6.4's <code>launchLinear(X, W, Y, batch, in, out)</code>, which computes <code>Y = X·Wᵀ</code> with W
          stored (out, in). Choosing its six arguments so that each sample's Y comes out <code>M × P</code>, in NCHW, with F
          never copied, is the test of whether the project's GEMM signature composes. Section 19.3 asks it as a question
          first.
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
                  <code>convDirectKernel</code> và <code>launchConvDirect</code>; <code>unrollKernel</code> và{' '}
                  <code>launchUnroll</code>; <code>launchConvGemm</code>. CUDA thuần chỉ với luồng, khối, bộ nhớ chia sẻ (shared
                  memory) và rào chắn. Cùng tệp đó chạy trên cả hai trình giả lập (emulator-checked), và được thiết lập để build
                  cho HIP qua <code>pmpp/vendors/hip.h</code> (unverified: chưa được biên dịch hay chạy trên phần cứng AMD ở
                  đây).
                </>
              }
            >
              <>
                <code>convDirectKernel</code> and <code>launchConvDirect</code>; <code>unrollKernel</code> and{' '}
                <code>launchUnroll</code>; <code>launchConvGemm</code>. Plain CUDA with only threads, blocks, shared memory and
                barriers. The same file runs on both emulators (emulator-checked), and is set up to build for HIP through{' '}
                <code>pmpp/vendors/hip.h</code> (unverified: not compiled or run on AMD hardware here).
              </>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated vi={<><code>conv_direct</code> và <code>unroll_input</code></>}>
              <><code>conv_direct</code> and <code>unroll_input</code></>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated
              vi={
                <>
                  <code>encodeConvDirect</code>, <code>encodeUnroll</code> và <code>encodeConvGemm</code>. Hàm cuối cùng của nó,{' '}
                  <code>convGemmScratchProblem</code>, có sẵn và đã hoàn chỉnh.
                </>
              }
            >
              <>
                <code>encodeConvDirect</code>, <code>encodeUnroll</code> and <code>encodeConvGemm</code>. Its last function,{' '}
                <code>convGemmScratchProblem</code>, is provided and complete.
              </>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Trên Metal, <code>encodeConvGemm</code> lặp qua các mẫu bên trong một compute encoder và gọi hàm có sẵn{' '}
            <code>encodeLinearAt</code>, hàm gắn kernel <code>linear</code> của lab 6.4 với một offset byte vào mỗi buffer. Một
            vùng tạm phục vụ mọi mẫu vì một compute encoder thông thường chạy các dispatch theo thứ tự và các buffer được theo
            dõi hazard. Trước khi chương trình điều khiển (driver) cấp phát vùng tạm đó, <code>convGemmScratchProblem</code>{' '}
            kiểm tra <code>Bt</code> của một mẫu so với <code>maxBufferLength</code> của thiết bị và thất bại kèm thông báo nếu
            nó không vừa.
          </p>
        }
      >
        <p>
          On Metal, <code>encodeConvGemm</code> loops over the samples inside one compute encoder and calls the provided{' '}
          <code>encodeLinearAt</code>, which binds lab 6.4's <code>linear</code> kernel with a byte offset into each buffer.
          One scratch serves every sample because a plain compute encoder runs its dispatches in order and the buffers are
          hazard-tracked. Before the driver allocates that scratch, <code>convGemmScratchProblem</code> checks one sample's{' '}
          <code>Bt</code> against the device's <code>maxBufferLength</code> and fails with a message if it does not fit.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md C9; ch19-facts.md C2)" /* i18n-ignore: citation */>
        <Translated
          vi={
            <p>
              Một compute encoder mặc định chạy các dispatch theo thứ tự, có theo dõi hazard. Theo tài liệu của Apple,{' '}
              <code>maxBufferLength</code> ít nhất là 256 MB; thiết bị tự báo giá trị của nó.
            </p>
          }
        >
          <p>
            A default compute encoder runs dispatches in order, with hazard tracking. <code>maxBufferLength</code> is at least
            256 MB by Apple's documentation; the device reports its own.
          </p>
        </Translated>
      </Today>

      <h3>
        <Translated vi="Những chỗ mã khác với trang sách">Where the code differs from the page</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Lab khác với các hình của sách ở bốn chỗ. Ba chỗ là các mục "Unclear in the PDF" của ghi chú cho pp. 461, 462 và
            471: trình khám phá "From blockIdx to pixel" của mục 19.2 cho thấy hai chỗ đầu thất bại và phần săn errata của mục
            19.3 cho thấy chỗ thứ ba. Chỗ thứ tư lấp một khoảng trống sách để lại.
          </p>
        }
      >
        <p>
          The lab departs from the book's figures in four places. Three are the notes' "Unclear in the PDF" items for pp.
          461, 462 and 471: section 19.2's "From blockIdx to pixel" explorer shows the first two failing and section 19.3's erratum hunt the
          third. The fourth fills a gap the book leaves.
        </p>
      </Translated>
      <ul>
        <Translated
          vi={
            <li>
              Fig. 19.5 chia <code>W_out / TILE_WIDTH</code>, khiến 12 hàng và 12 cột của đầu ra 28 × 28 của LeNet C1 không được
              tính. Lab làm tròn lên, và các luồng không có pixel nào thì không làm gì.
            </li>
          }
        >
          <li>
            Fig. 19.5 divides <code>W_out / TILE_WIDTH</code>, which leaves 12 rows and 12 columns of LeNet C1's 28 × 28
            output uncomputed. The lab rounds up, and the threads with no pixel do nothing.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Fig. 19.7 không có kiểm tra biên, và chữ ký của nó thiếu H, W và M, những thứ các chỉ số tuyến tính hóa cần đến.
              Trang sách cũng đặt tên các biến <code>dim3</code> là <code>blockDim</code> và <code>gridDim</code>; lab dùng{' '}
              <code>dimBlock</code> và <code>dimGrid</code>, như lab 6.4, để chúng không che các biến dựng sẵn cùng tên.
            </li>
          }
        >
          <li>
            Fig. 19.7 has no bounds check, and its signature lacks H, W and M, which the linearised indices need. The page
            also names its <code>dim3</code> variables <code>blockDim</code> and <code>gridDim</code>; the lab uses{' '}
            <code>dimBlock</code> and <code>dimGrid</code>, as lab 6.4 does, so they do not hide the built-in variables of
            those names.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Sách không đưa ra kernel unroll nào. Nó mô tả unroll tường minh và chi phí của nó (p. 467) rồi chuyển sang phiên
              bản ngầm. Kernel unroll và cách ánh xạ luồng của nó là của khóa học này.
            </li>
          }
        >
          <li>
            The book gives no unroll kernel. It describes explicit unrolling and its cost (p. 467) and moves on to the
            implicit version. The unroll kernel and its thread mapping are ours.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Fig. 19.11, có sẵn trong <code>provided/implicit_gemm.*</code>: số pha của nó được làm tròn lên (
              <code>(C*K*K)/TILE_WIDTH</code> của trang sách chạy một pha cho 25 số hạng và bỏ mất 9), việc nạp F, gom X và ghi
              kết quả đều được kiểm tra biên với giá trị lấp bằng 0, và lần khởi chạy mà sách không trình bày dùng{' '}
              <code>(ceil(P/16), ceil(M/16), N)</code> khối.
            </li>
          }
        >
          <li>
            Fig. 19.11, provided in <code>provided/implicit_gemm.*</code>: its phase count rounds up (the page's{' '}
            <code>(C*K*K)/TILE_WIDTH</code> runs one phase for 25 terms and drops 9), the F load, the X gather and the store
            are bounds-checked with zero fill, and the launch the book does not show uses{' '}
            <code>(ceil(P/16), ceil(M/16), N)</code> blocks.
          </li>
        </Translated>
      </ul>

      <h3>
        <Translated vi="Có sẵn, không chỉnh sửa">Provided, not edited</Translated>
      </h3>
      <ul>
        <Translated
          vi={
            <li>
              <code>provided/gemm.cmake</code> chọn kernel lab 6.4 nào mà đường GEMM của bạn gọi (bên dưới).
            </li>
          }
        >
          <li>
            <code>provided/gemm.cmake</code> picks which lab 6.4 kernel your GEMM path calls (below).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <code>provided/linear_at.mm</code>: <code>encodeLinearAt</code>, lần khởi chạy Metal của lab 6.4 với offset byte.
            </li>
          }
        >
          <li>
            <code>provided/linear_at.mm</code>: <code>encodeLinearAt</code>, lab 6.4's Metal launch with byte offsets.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <code>provided/implicit_gemm.cu</code>, <code>.metal</code> và <code>_launch.mm</code>: Fig. 19.11 đã sửa, được
              đo thời gian bên cạnh hai đường của bạn. Mục <a href={href('ch19', '19.3')}>19.3</a> trình bày nó.
            </li>
          }
        >
          <li>
            <code>provided/implicit_gemm.cu</code>, <code>.metal</code> and <code>_launch.mm</code>: the corrected Fig.
            19.11, timed beside your two paths. Section <a href={href('ch19', '19.3')}>19.3</a> shows it.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <code>main.cpp</code>: chương trình điều khiển. Bản tham chiếu CPU của nó, <code>cpuConvLayer</code>, là Fig. 19.4
              với các chỉ số đã tuyến tính hóa và tổng được giữ ở double.
            </li>
          }
        >
          <li>
            <code>main.cpp</code>: the driver. Its CPU reference, <code>cpuConvLayer</code>, is Fig. 19.4 with the indices
            linearised and the sum kept in double.
          </li>
        </Translated>
      </ul>

      <h3>
        <Translated vi="GEMM nào">Which GEMM</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Mặc định, đường unroll + GEMM gọi lời giải tham chiếu (solution) của lab 6.4. Để gọi kernel lab 6.4 của chính bạn,
            chạy với <code>PMPP_CH19_GEMM=work</code>; <code>PMPP_CH19_GEMM=m2</code>, hoặc không đặt gì, thì quay lại. Lựa chọn
            được đọc khi các lab được cấu hình, việc mà <code>pmpp.py</code> làm trước mỗi lần chạy, và mỗi lần chạy kiểm tra và
            mỗi hàng benchmark đều ghi GEMM nào đã được dùng, <code>"m2 reference"</code> hay <code>"m2 yours"</code>. Với{' '}
            <code>work</code>, một lỗi biên dịch trong tệp lab 6.4 của bạn cũng làm hỏng bản build của lab này.
          </p>
        }
      >
        <p>
          By default the unroll + GEMM path calls lab 6.4's reference solution. To call your own lab 6.4 kernel, run with{' '}
          <code>PMPP_CH19_GEMM=work</code>; <code>PMPP_CH19_GEMM=m2</code>, or nothing, goes back. The choice is read when
          the labs are configured, which <code>pmpp.py</code> does before every run, and every test run and benchmark row
          says which GEMM it used, <code>"m2 reference"</code> or <code>"m2 yours"</code>. With <code>work</code>, a compile
          error in your lab 6.4 file breaks this lab's build too.
        </p>
      </Translated>
      <GemmSource />
      <Aside>
        <Translated
          vi={
            <p>
              Một kết quả đã ghi có thể còn lại sau khi bạn chuyển đổi: phép kiểm tra nguồn của báo cáo không bao gồm tệp GEMM,
              nên hãy xem cột GEMM của bảng kết quả.
            </p>
          }
        >
          <p>
            A recorded result can outlive a switch: the reports' source check does not cover the GEMM file, so look at the
            GEMM column of the results table.
          </p>
        </Translated>
      </Aside>

      <h3>
        <Translated vi="Các bài kiểm tra kiểm tra gì">What the tests check</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 19.1 --backend emu</code> trước. Nó chạy dưới AddressSanitizer, và X, F và{' '}
            <code>Bt</code> chứa đúng số float của chúng, nên một lần đọc hay ghi vượt quá bất kỳ mảng nào sẽ dừng lần chạy kèm
            dòng mã nguồn. Rồi đến <code>--backend race</code>, rồi <code>python3 labs/pmpp.py run 19.1</code> trên GPU của bạn.
            Mỗi kích thước có bốn phép kiểm tra:
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 19.1 --backend emu</code> first. It runs under AddressSanitizer, and X, F and{' '}
          <code>Bt</code> hold exactly their floats, so a read or write past any of them stops the run with the source line.
          Then <code>--backend race</code>, then <code>python3 labs/pmpp.py run 19.1</code> on your GPU. Every shape gets four
          checks:
        </p>
      </Translated>
      <ul>
        <Translated
          vi={
            <li>
              <b>(a) Riêng bước unroll.</b> Mọi phần tử của <code>Bt</code> của mọi mẫu được so sánh chính xác với một bản unroll
              trên CPU. Một lỗi chỉ số được báo là lỗi unroll, kèm hàng, cột và phần tử X mà lẽ ra nó phải là, chứ không phải
              là một GEMM sai.
            </li>
          }
        >
          <li>
            <b>(a) The unroll alone.</b> Every element of every sample's <code>Bt</code> is compared exactly with a CPU
            unroll. An index mistake is reported as an unroll error, with the row, the column and the X element it should
            have been, and not as a wrong GEMM.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>(b) Đường trực tiếp</b> và <b>(c) đường unroll + GEMM</b>, so với <code>cpuConvLayer</code> với dung sai được
              nhân theo tổng <code>|x·f|</code> của mỗi đầu ra. Nếu (a) qua mà (c) thất bại, thông báo chỉ vào các đối số trong
              lời gọi GEMM của bạn.
            </li>
          }
        >
          <li>
            <b>(b) The direct path</b> and <b>(c) the unroll + GEMM path</b>, against <code>cpuConvLayer</code> with a
            tolerance scaled by the sum of <code>|x·f|</code> for each output. If (a) passed and (c) fails, the message
            points at the arguments of your GEMM call.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>(d) GEMM ngầm có sẵn.</b> Thất bại ở đó là lỗi của khóa học, và thông báo nói rõ như vậy.
            </li>
          }
        >
          <li>
            <b>(d) The provided implicit GEMM.</b> A failure there is the course's bug, and the message says so.
          </li>
        </Translated>
      </ul>
      <Translated
        vi={
          <p>
            Các kích thước, dạng (N, C, H, W, M, K): ví dụ của sách (1, 3, 3, 3, 2, 2) với các số nguyên của Fig. 19.2, có Y phải
            đúng bằng <code>14 20 15 24 / 12 24 17 26</code> không có dung sai, vì số nguyên nhỏ được biểu diễn chính xác trong
            float; cùng ví dụ đó với batch 2; LeNet C1 (2, 1, 32, 32, 6, 5), trong đó C·K·K = 25, đầu ra 28 × 28 và M = 6 đều
            không phải bội của 16; LeNet C3 (2, 6, 14, 14, 16, 5); một lớp 1 × 1 (2, 8, 7, 9, 5, 1); một pixel đầu ra (1, 4, 5,
            5, 3, 5); một hình chữ nhật (1, 2, 9, 17, 3, 3); và C·K·K là bội của 16 (1, 16, 6, 6, 16, 2). Trên GPU có thêm ba lớp
            lớn hơn. Y bắt đầu là NaN và theo sau là một dải bảo vệ gồm các float canh gác phải còn nguyên, và trên GPU mỗi
            phép kiểm tra chạy hai lần với Y được đặt lại ở giữa. Biến thể race chạy ba trong số các kích thước nhỏ.
          </p>
        }
      >
        <p>
          The shapes, as (N, C, H, W, M, K): the book's example (1, 3, 3, 3, 2, 2) with Fig. 19.2's integers, whose Y must be
          exactly <code>14 20 15 24 / 12 24 17 26</code> with no tolerance, since small integers are exact in float; the same
          twice; LeNet C1 (2, 1, 32, 32, 6, 5), where C·K·K = 25, the 28 × 28 output and M = 6 are none of them multiples of
          16; LeNet C3 (2, 6, 14, 14, 16, 5); a 1 × 1 layer (2, 8, 7, 9, 5, 1); one output pixel (1, 4, 5, 5, 3, 5); a
          rectangle (1, 2, 9, 17, 3, 3); and C·K·K a multiple of 16 (1, 16, 6, 6, 16, 2). On a GPU three larger layers are
          added. Y starts as NaN and is followed by a guard band of sentinel floats that must survive, and on a GPU each
          check runs twice with Y reset in between. The race flavour runs three of the small shapes.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Mã khởi đầu (starter) chưa động tới vẫn build được và thất bại: các kernel rỗng của nó để <code>Bt</code> và Y là NaN.
          </p>
        }
      >
        <p>The untouched starter builds and fails: its empty kernels leave <code>Bt</code> and Y as NaN.</p>
      </Translated>

      <h3>
        <Translated vi="Trước khi đo hiệu năng, hãy dự đoán">Before you benchmark, predict</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Benchmark chỉ chạy trên thiết bị thật, trên năm kích thước lớp do khóa học này chọn, ghi bên dưới theo tên gọi của
            chương (M map đầu ra, K là cạnh của bộ lọc). Với mỗi lớp, ba đường lần lượt chạy và lấy kết quả tốt nhất của vài
            vòng: kernel trực tiếp của bạn; unroll + GEMM của bạn, đo thời gian cho cả lớp, rồi riêng N lần unroll, rồi riêng N
            GEMM trên một vùng tạm đã được lấp đầy; và GEMM ngầm có sẵn. Mỗi đường trước tiên được kiểm tra trên hai kích thước
            khó, vì thời gian của một kernel sai chẳng có ý nghĩa gì.
          </p>
        }
      >
        <p>
          The benchmark runs on real devices only, on five layer shapes of our choosing, given below in the chapter's names
          (M output maps, K the filter's side). For each layer, three paths take
          turns and the best of several rounds counts: your direct kernel; your unroll + GEMM, timed as the whole layer,
          then its N unrolls alone, then its N GEMMs alone on a filled scratch; and the provided implicit GEMM. Every path is
          first checked on two awkward shapes, because a timing of a wrong kernel means nothing.
        </p>
      </Translated>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto', marginBottom: '1.2rem' }}>
        <table className="data" style={{ margin: 0 }} data-table="layers">
          <thead>
            <tr>
              <th><Translated vi="Lớp">Layer</Translated></th>
              <th>(N, C, H, W, M, K)</th>
              <th><Translated vi="GEMM mỗi mẫu, M × C·K·K × P">Per-sample GEMM, M × C·K·K × P</Translated></th>
              <th className="num"><Translated vi="Độ phình, Eq. 19.4">Expansion, Eq. 19.4</Translated></th>
              <th className="num"><Translated vi="Số ô 16 × 16 mỗi GEMM">16 × 16 tiles per GEMM</Translated></th>
              <th className="num"><Translated vi="Số pha">Phases</Translated></th>
            </tr>
          </thead>
          <tbody>
            {LAB_LAYERS.map((L) => {
              const f = facts[L.id];
              const s = L.shape;
              return (
                <tr key={L.id} data-layer={L.id}>
                  <td>
                    {L.id} <span className="muted">({L.note})</span>
                  </td>
                  <td>
                    ({s.N}, {s.C}, {s.H}, {s.W}, {s.M}, {s.K})
                  </td>
                  <td>{f.gemm}</td>
                  <td className="num" data-col="expansion">
                    {f.expansion.toFixed(2)}
                  </td>
                  <td className="num" data-col="tiles">
                    {f.tileRows} × {f.tileCols}
                  </td>
                  <td className="num" data-col="phases">
                    {f.phases}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <Translated
        vi={
          <p>
            Mọi con số trong bảng này đều được tính từ kích thước, không con số nào là đo được. Hãy ghi ra ba con số trước khi
            chạy <code>python3 labs/pmpp.py run 19.1 --bench</code>.
          </p>
        }
      >
        <p>
          Every number in this table is computed from the shapes, none measured. Write down three numbers before you run{' '}
          <code>python3 labs/pmpp.py run 19.1 --bench</code>.
        </p>
      </Translated>
      <LabPrediction
        id="ch19/labs/p-lab1-late"
        prompt={
          <Translated
            vi={
              <>
                Lớp late. Thời gian trực tiếp chia cho thời gian unroll + GEMM trên thiết bị của bạn: trên 1 nghĩa là unroll +
                GEMM nhanh hơn, dưới 1 là chậm hơn.
              </>
            }
          >
            <>
              The late layer. Direct time divided by unroll + GEMM time on your device: above 1 means unroll + GEMM is
              faster, below 1 slower.
            </>
          </Translated>
        }
        unit={tr('(direct time / unroll + GEMM time)', '(thời gian trực tiếp / thời gian unroll + GEMM)')}
        measure={speedup('late')}
        missing={MISSING}
        reading={
          <Translated
            vi={
              <>
                Mỗi mẫu, GEMM là {late.gemm}: {late.tileRows} × {late.tileCols} ô, mỗi ô chạy {late.phases} pha, nên mỗi ô của
                F và của <code>Bt</code> mà nó nạp được dùng lại 16 lần. Kernel trực tiếp không chia sẻ gì giữa các luồng, và
                yêu cầu 0.25 FLOP/B, nhưng các luồng lân cận của nó đọc những vùng chồng lên nhau mà bộ nhớ đệm có thể phục vụ.
                Hãy so cả hàng của GEMM ngầm: nó làm cùng phép tính mà không ghi <code>Bt</code>.
              </>
            }
          >
            <>
              Per sample the GEMM is {late.gemm}: {late.tileRows} × {late.tileCols} tiles, each running {late.phases}{' '}
              phases, so every tile of F and of <code>Bt</code> it loads is reused 16 times. The direct kernel shares nothing
              between threads, and requests 0.25 FLOP/B, but its neighbouring threads read overlapping patches that caches
              may serve. Compare the implicit GEMM's row too: it does the same arithmetic without writing{' '}
              <code>Bt</code>.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch19/labs/p-lab1-early"
        prompt={
          <Translated vi={<>Cùng tỉ lệ đó trên lớp early: thời gian trực tiếp chia cho thời gian unroll + GEMM.</>}>
            <>The same ratio on the early layer: direct time divided by unroll + GEMM time.</>
          </Translated>
        }
        unit={tr('(direct time / unroll + GEMM time)', '(thời gian trực tiếp / thời gian unroll + GEMM)')}
        measure={speedup('early')}
        missing={MISSING}
        reading={
          <Translated
            vi={
              <>
                Mỗi mẫu, GEMM là {early.gemm}: {early.tileRows} hàng ô và {early.tileCols.toLocaleString('en-US')} cột ô, nhưng
                chỉ {early.phases} pha, pha thứ hai rộng {early.lastPhase} trên 16. Phép rút gọn (reduction) ngắn, nên mỗi ô ít
                được dùng lại, và <code>Bt</code> lớn gấp {early.expansion.toFixed(2)} lần mẫu mà nó sinh ra từ đó. Hãy đặt tỉ
                lệ của hai lớp cạnh nhau trước khi giải thích lớp nào.
              </>
            }
          >
            <>
              Per sample the GEMM is {early.gemm}: {early.tileRows} tile rows and {early.tileCols.toLocaleString('en-US')}{' '}
              tile columns, but only {early.phases} phases, the second {early.lastPhase} of 16 wide. The reduction is short,
              so each tile is reused little, and <code>Bt</code> is {early.expansion.toFixed(2)} times the size of the
              sample it came from. Set the two layers' ratios side by side before you explain either.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch19/labs/p-lab1-unroll-share"
        prompt={
          <Translated
            vi={
              <>
                Trên lớp early, bao nhiêu phần trăm thời gian của unroll + GEMM dành cho riêng 8 lần unroll? (Benchmark đo thời
                gian các lần unroll và các GEMM riêng rẽ, nên hai phần không nhất thiết cộng lại bằng tổng.)
              </>
            }
          >
            <>
              On the early layer, what percentage of unroll + GEMM's time goes to the 8 unrolls alone? (The benchmark times
              the unrolls and the GEMMs separately, so the two need not add up to the whole.)
            </>
          </Translated>
        }
        unit={tr("% of the layer's time", '% thời gian của lớp')}
        measure={unrollShare}
        missing={MISSING}
        reading={
          <Translated
            vi={
              <>
                Bước unroll di chuyển byte và không làm phép tính số học nào: nó đọc X, ghi <code>Bt</code>, và GEMM đọc lại{' '}
                <code>Bt</code>. Nó còn làm phép chia nguyên và lấy dư cho mỗi phần tử nó ghi, trong khi kernel trực tiếp làm
                một phép chia và một phép lấy dư cho mỗi luồng trước các vòng lặp. Nếu tỉ lệ này làm bạn bất ngờ, hãy chạy lại
                benchmark: một lần chạy của một kernel ngắn có thể gây hiểu nhầm.
              </>
            }
          >
            <>
              The unroll moves bytes and does no arithmetic: it reads X, writes <code>Bt</code>, and the GEMM reads{' '}
              <code>Bt</code> back. It also does integer divides and remainders for every element it writes, where the
              direct kernel does one divide and one remainder per thread before its loops. If the share surprises you, run
              the benchmark again: one run of a short kernel can mislead.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Những điều cần cân nhắc, cho cả ba. Không điều nào cho ra đáp án, và vài điều kéo về hai hướng ngược nhau. Những gì
            các nhà sản xuất nói về phép chia nguyên, và quy tắc của llama.cpp cho các lớp nhỏ, đã được kiểm tra vào ngày ghi
            ở lề; phần còn lại là phép tính của chương và của khóa học này.
          </p>
        }
      >
        <p>
          Things to weigh, for all three. None of them gives the answer, and some pull in opposite directions. What the
          vendors say about integer division, and llama.cpp's rule for small layers, were checked on the date in the
          margin; the rest is the chapter's arithmetic and ours.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch19-facts A8, B5, C6, D1)" /* i18n-ignore: citation */>
        <Translated
          vi={
            <p>
              CUDA Best Practices Guide; tài liệu HIP; WWDC16 session 606. llama.cpp: <code>small_conv</code> trong{' '}
              <code>ggml-cuda/conv2d.cu</code>.
            </p>
          }
        >
          <p>
            CUDA Best Practices Guide; HIP documentation; WWDC16 session 606. llama.cpp: <code>small_conv</code> in{' '}
            <code>ggml-cuda/conv2d.cu</code>.
          </p>
        </Translated>
      </Today>
      <ul>
        <Translated
          vi={
            <li>
              Lưu lượng được yêu cầu không phải là lưu lượng thật. Các luồng lân cận yêu cầu những vùng chồng lên nhau, và bộ
              nhớ đệm (cache) có thể phục vụ một yêu cầu lặp lại mà không cần thêm một chuyến tới bộ nhớ. Kernel trực tiếp ở
              đây có cùng 0.25 FLOP/B như matmul ngây thơ của chương 5: hãy so kết quả ngây thơ của lab 5.1 của bạn với trần đó
              trên roofline ở <a href={href('ch05', '5.1')}>mục 5.1</a> (đánh dấu &ldquo;show my measurements&rdquo;).
            </li>
          }
        >
          <li>
            Requested traffic is not traffic. Neighbouring threads ask for overlapping patches, and caches can serve a
            repeated request without another trip to memory. The direct kernel here has the same 0.25 FLOP/B as chapter
            5&rsquo;s naive matmul: compare your own lab 5.1 naive result with that ceiling on the roofline in{' '}
            <a href={href('ch05', '5.1')}>section 5.1</a> (tick &ldquo;show my measurements&rdquo;).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Lớp small, LeNet C3 ở batch 64, thực hiện {small.launches} lần khởi chạy unroll và {small.launches} lần khởi chạy
              GEMM, mỗi GEMM {small.tileRows} hàng ô nhân {small.tileCols} cột ô. Sách nêu chi phí khởi chạy là một cái giá của
              các phép nhân nhỏ (p. 472).
            </li>
          }
        >
          <li>
            The small layer, LeNet C3 at batch 64, makes {small.launches} unroll launches and {small.launches} GEMM
            launches, each GEMM {small.tileRows} tile row by {small.tileCols} tile columns. The book names launch overhead
            as a cost of small multiplications (p. 472).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Phép chia nguyên tốn kém. NVIDIA ghi trong tài liệu rằng nó được biên dịch thành tới 20 lệnh, AMD nói nó tốn
              nhiều chu kỳ hơn hẳn một phép nhân, còn phát biểu duy nhất của Apple là một bài nói năm 2016 cho biết phần lớn
              GPU không có lệnh chia nguyên.
            </li>
          }
        >
          <li>
            Integer division is costly. NVIDIA documents it as compiling to up to 20 instructions, AMD as taking many more
            cycles than a multiplication, and Apple's only statement is a 2016 talk saying most GPUs have no integer divide
            instruction.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Bản unroll tham chiếu giao cho mỗi luồng một phần tử của <code>Bt</code> theo thứ tự của chính <code>Bt</code>,
              nên các luồng lân cận ghi các float lân cận và đọc X dọc theo một hàng của vùng. Đánh số luồng theo pixel thay vào
              đó thì đọc X liền mạch và ghi <code>Bt</code> với bước nhảy C·K·K. Bạn không thể có cả hai nếu không đi qua bộ nhớ
              chia sẻ. Bạn đã chọn cách nào?
            </li>
          }
        >
          <li>
            The reference unroll gives each thread one element of <code>Bt</code> in <code>Bt</code>'s own order, so
            neighbouring threads write neighbouring floats and read X along a patch row. Numbering the threads along the
            pixels instead reads X contiguously and writes <code>Bt</code> with a stride of C·K·K. You cannot have both
            without staging through shared memory. Which did you pick?
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Backend CUDA và HIP của llama.cpp gửi các lớp có C·K·K dưới 64 và ít hơn 512 pixel đầu ra tới một kernel trực
              tiếp. Quy tắc đó sẽ gửi lớp nào trong năm lớp tới đó? Đó là một lựa chọn thực tế cho các kernel của llama.cpp,
              không phải một dự đoán cho kernel của bạn.
            </li>
          }
        >
          <li>
            llama.cpp's CUDA and HIP backend sends layers with C·K·K below 64 and fewer than 512 output pixels to a direct
            kernel. Which of the five layers would that rule send there? It is a production choice for llama.cpp's
            kernels, not a prediction for yours.
          </li>
        </Translated>
      </ul>
      <LabResults lab={l1} cols={COLS()} />
      <Translated
        vi={
          <p>
            "Phần của unroll" là riêng N lần unroll chia cho cả lớp unroll + GEMM. "Độ phình" là Eq. 19.4 cho lớp đó, do chương
            trình điều khiển tính, và vùng tạm là <code>Bt</code> của một mẫu. GFLOP/s đếm <code>2·N·M·C·K²·P</code> cho mọi
            đường, nên ba đường của một lớp được so trên cùng một khối lượng công việc.
          </p>
        }
      >
        <p>
          "Unroll share" is the N unrolls alone over the whole unroll + GEMM layer. "Expansion" is Eq. 19.4 for the layer,
          computed by the driver, and the scratch is one sample's <code>Bt</code>. GFLOP/s counts <code>2·N·M·C·K²·P</code>{' '}
          for every path, so the three paths of one layer are compared on the same work.
        </p>
      </Translated>
      <Translated vi={<p>Cùng các hàng đó, mỗi đường so với kernel trực tiếp. Trên 1 nghĩa là đường đó nhanh hơn trực tiếp.</p>}>
        <p>The same rows, each path against the direct kernel. Above 1, the path was faster than direct.</p>
      </Translated>
      <AgainstDirect />
      <p data-llama>
        <Translated vi="Đáp án cho câu hỏi cuối trong danh sách: theo quy tắc kích thước của llama.cpp như đã đọc vào 2026-11-20,">
          The answer to the last question in the list: by llama.cpp's shape rule as read on 2026-11-20,
        </Translated>{' '}
        {LAB_LAYERS.map((L, i) => (
          <span key={L.id} data-llama-layer={L.id} data-llama-path={llamaFor(L.shape)}>
            {i ? (i === LAB_LAYERS.length - 1 ? tr(', and ', ', và ') : ', ') : ''}
            <Translated vi={<>{L.id} sẽ đi tới {LLAMA_WORDS[llamaFor(L.shape)]}</>}>
              {L.id} would go to {LLAMA_WORDS[llamaFor(L.shape)]}
            </Translated>
          </span>
        ))}
        . {LAB_LAYERS.some((L) => llamaFor(L.shape) === 'direct') ? '' : tr('None of the five goes to its direct kernel. ', 'Không lớp nào trong năm lớp đi tới kernel trực tiếp của nó. ')}
        <Translated
          vi={
            <>
              Phép rút gọn của lớp small, C·K·K = {LAB_LAYERS[4].shape.C * LAB_LAYERS[4].shape.K ** 2}, đã vượt ngưỡng 64.
            </>
          }
        >
          The small layer's reduction, C·K·K = {LAB_LAYERS[4].shape.C * LAB_LAYERS[4].shape.K ** 2}, is already above the
          cut-off of 64.
        </Translated>
      </p>
      <Simplification>
        <Translated
          vi={
            <p>
              Chỉ các điều kiện về kích thước, như đã đọc vào 2026-11-20 (ch19-facts D1): kiểu dữ liệu và điều kiện split-K
              không được mô hình hóa. Năm lớp là của khóa học này, được chọn để trải từ early, middle, late, 1 × 1 đến small, và
              không thuộc mạng cụ thể nào.
            </p>
          }
        >
          <p>
            Shape conditions only, as read on 2026-11-20 (ch19-facts D1): data types and the split-K condition are not
            modelled. The five layers are ours, chosen to span early, middle, late, 1 × 1 and small, and are no particular
            network.
          </p>
        </Translated>
      </Simplification>

      <h3>
        <Translated vi="Đi xa hơn">Go further</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Khi kernel của bạn đã qua, có năm điều để thử. Điều đầu tiên là gợi ý của chính sách; K hằng số và GEMM cho cả
            batch đến từ cách mã thực tế xử lý lớp này.
          </p>
        }
      >
        <p>
          Once your kernels pass, five things to try. The first is the book's own suggestion; the constant K and the
          whole-batch GEMM come from how production code handles this layer.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Của khóa học này, từ mã của llama.cpp và PyTorch như đã đọc vào 2026-11-20 (ch19-facts D1, D2, D4, E2).</p>}>
          <p>Ours, from llama.cpp's and PyTorch's code as read on 2026-11-20 (ch19-facts D1, D2, D4, E2).</p>
        </Translated>
      </Beyond>
      <ul>
        <Translated
          vi={
            <li>
              Sách để việc phân ô (tiling) bằng bộ nhớ chia sẻ cho kernel trực tiếp lại cho bạn (p. 463). Các ô có vùng biên
              (halo) của chương 7, mục <a href={href('ch07', '7.5')}>7.5</a> và lab 7.2, là kỹ thuật cần dùng. Phân ô một map đầu
              vào, đo, rồi tự hỏi ô đó không thể cho bạn điều gì: sự dùng lại giữa các map đầu ra, thứ mà mục 19.3 có được từ
              dạng GEMM.
            </li>
          }
        >
          <li>
            The book leaves shared-memory tiling of the direct kernel to you (p. 463). Chapter 7's halo tiles, section{' '}
            <a href={href('ch07', '7.5')}>7.5</a> and lab 7.2, are the technique. Tile one input map, measure, then ask what
            the tile cannot give you: reuse across output maps, which section 19.3 gets from the GEMM form.
          </li>
        </Translated>
        <li>
          <Translated vi="Đảo cách ánh xạ luồng của bản unroll và đo thời gian riêng các lần unroll, trên lớp early và lớp late.">
            Flip your unroll's thread mapping and time the unrolls alone, on the early and the late layer.
          </Translated>
        </li>
        <Translated
          vi={
            <li>
              Biến <code>K</code> thành hằng số lúc biên dịch, một macro trong CUDA hoặc một function constant trong Metal, để{' '}
              <code>/K</code> và <code>%K</code> chia cho một hằng số cụ thể, rồi đo lại thời gian các lần unroll. llama.cpp làm
              điều này cho bộ lọc 3 × 3 và 1 × 1.
            </li>
          }
        >
          <li>
            Make <code>K</code> a compile-time constant, a macro in CUDA or a function constant in Metal, so that{' '}
            <code>/K</code> and <code>%K</code> divide by a literal, and time the unrolls again. llama.cpp does this for 3 ×
            3 and 1 × 1 filters.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Unroll cả batch thành một ma trận <code>C·K·K × N·P</code> và gọi GEMM một lần. Kết quả ra{' '}
              <code>M × N × P</code> và cần một phép hoán vị để thành NCHW: hãy đo thời gian cả phép hoán vị.
            </li>
          }
        >
          <li>
            Unroll the whole batch into one <code>C·K·K × N·P</code> matrix and make one GEMM call. The result comes out{' '}
            <code>M × N × P</code> and needs a permute to become NCHW: time the permute with it.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Chạy với <code>PMPP_CH19_GEMM=work</code> khi kernel lab 6.4 của bạn đã qua: cùng lớp đó, GEMM của chính bạn.
            </li>
          }
        >
          <li>
            Run with <code>PMPP_CH19_GEMM=work</code> once your lab 6.4 kernel passes: the same layer, your own GEMM.
          </li>
        </Translated>
      </ul>
      <Translated
        vi={
          <p>
            Nếu bạn so với <code>torch.nn.functional.conv2d</code> trên GPU NVIDIA, một lời gọi float32 có thể chạy ở TF32 trừ
            khi bạn tắt nó đi. Hãy so theo một dung sai và in ra thiết lập đó. Mục <a href={href('ch19', '19.4')}>19.4</a> hỏi
            cần kiểm tra gì trước tiên khi hai bên không khớp.
          </p>
        }
      >
        <p>
          If you compare with <code>torch.nn.functional.conv2d</code> on an NVIDIA GPU, a float32 call may run in TF32
          unless you turn that off. Compare with a tolerance and print the setting. Section{' '}
          <a href={href('ch19', '19.4')}>19.4</a> asks what to check first when the two disagree.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch19-facts A6)" /* i18n-ignore: citation */>
        <Translated
          vi={
            <p>
              <code>cudnn.allow_tf32</code> mặc định là True; PyTorch dự định thay nó bằng{' '}
              <code>cudnn.conv.fp32_precision</code>, mà giá trị mặc định thì không được nêu.
            </p>
          }
        >
          <p>
            <code>cudnn.allow_tf32</code> defaults to True; PyTorch plans to replace it with{' '}
            <code>cudnn.conv.fp32_precision</code>, whose default it does not state.
          </p>
        </Translated>
      </Today>
      <ReferenceSolution lab={l1} />

      <h3>
        <Translated vi="Dự án">The project</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Chương này không thêm mốc dự án nào. Lab dùng lại M2 nguyên trạng: <code>launchLinear</code> của lab 6.4, gọi một
            lần cho mỗi mẫu với F làm ma trận đầu vào và <code>Bt</code> làm ma trận trọng số. Không có gì trong GEMM phải thay
            đổi, và đó chính là ý nghĩa các quy ước của dự án: <code>float</code> theo thứ tự hàng, con trỏ device thô và kích
            thước tường minh. Phần <a href={href('ch19', 'wrap')}>tóm tắt</a> quay lại chuyện điều đó có nghĩa gì cho các kernel
            sắp tới.
          </p>
        }
      >
        <p>
          This chapter adds no milestone. The lab reuses M2 as it is: lab 6.4's <code>launchLinear</code>, called once per
          sample with F as the input matrix and <code>Bt</code> as the weight matrix. Nothing in the GEMM had to change,
          which is the point of the project's conventions: row-major <code>float</code>, raw device pointers and explicit
          sizes. The <a href={href('ch19', 'wrap')}>summary</a> returns to what that means for the kernels to come.
        </p>
      </Translated>
    </LabsPage>
  );
}
