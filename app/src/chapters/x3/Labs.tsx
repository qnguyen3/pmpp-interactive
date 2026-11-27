import { useState, type ReactNode } from 'react';
import { Predict, type Question } from '../../components/Ask';
import { Beyond, Today, type Tier } from '../../components/notes';
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
  benchSets,
  defineLabs,
  measure,
  msCol,
  numCol,
  rawCol,
  useLabs,
  type BenchSet,
  type Col,
  type Lab,
} from '../../components/LabKit';
import { Translated, tr } from '../../lib/i18n';
import { fmt, labStatus, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';
import { LAB_GPU } from '../../sim/ckptformat';
import { JOB_LABEL, launchCount, stepLaunches, type Config } from '../../sim/llamastep';

// The page is built from components/LabKit.tsx; what is here is module X3's own content.
// No question on this page: the labs are the exercise. Its five predictions are stored, not
// graded: x3/labs/p-lab1-port-emu, p-lab1-rmsnorm-bw, p-lab1-rmsnorm-bw-256, p-lab2-bwfraction,
// p-lab2-attn-share.
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('x3', [
  { num: 'X3.1', dir: 'lab1_llama_pieces', name: 'RMSNorm, RoPE, SwiGLU and the port', vi: { name: 'RMSNorm, RoPE, SwiGLU và bản port' } },
  { num: 'X3.2', dir: 'lab2_tinyllama', name: 'the M10 Llama step', vi: { name: 'bước Llama của M10' } },
]);
const [l1, l2] = LABS;

/** Identifier colours in lab X3.1's reference solution: where each value lives. */
const tiers = (global: string[], shared: string[], reg: string[]): Record<string, Tier> => ({
  ...Object.fromEntries(global.map((k) => [k, 'global' as Tier])),
  ...Object.fromEntries(shared.map((k) => [k, 'shared' as Tier])),
  ...Object.fromEntries(reg.map((k) => [k, 'reg' as Tier])),
});
const TIERS_1 = tiers(
  ['x', 'w', 'y', 'in', 'out', 'pos', 'cosT', 'sinT', 'gate', 'up', 'v', 'x4', 'w4', 'y4', 'xr', 'yr'],
  ['scratch', 's_sum'],
  ['s', 'scale', 'sum', 'val', 'x0', 'x1', 'g', 'lane', 'warp', 'tid'],
);

// ---------------------------------------------------------------- reading the reports
const missingBench = (labNum: string, extra?: ReactNode) => (
  <Translated
    vi={
      <>
        Phần này cần các hàng benchmark của lab {labNum} từ một thiết bị thật. Hãy chạy{' '}
        <code>python3 labs/pmpp.py run {labNum} --bench</code> khi các kernel của bạn đã qua kiểm thử (hoặc thêm{' '}
        <code>--solution</code> để đo thời gian lời giải tham chiếu). Trình giả lập không bao giờ được đo hiệu năng.
        {extra ? <> {extra}</> : null}
      </>
    }
  >
    <>
      This needs the benchmark rows of lab {labNum} from a real device. Run{' '}
      <code>python3 labs/pmpp.py run {labNum} --bench</code> once your kernels pass (or with <code>--solution</code> to time
      the reference). The emulator is never benchmarked.{extra ? <> {extra}</> : null}
    </>
  </Translated>
);
const NEED_51 = (
  <Translated
    vi={
      <>
        Tỉ lệ này còn cần băng thông của lab 5.1: hãy chạy <code>python3 labs/pmpp.py run 5.1 --bench</code> trước.
      </>
    }
  >
    <>
      The fraction also needs lab 5.1's bandwidth: run <code>python3 labs/pmpp.py run 5.1 --bench</code> first.
    </>
  </Translated>
);

/** Lab X3.1: launchRmsNorm's counted bandwidth over lab 5.1's, at d = 4096, as a percentage. */
const rmsFraction = (rows: number) => (reports: Reports) =>
  measure(reports, l1, (s: BenchSet) => {
    const r = benchRow(s, 'rmsnorm', { rows, d: 4096 });
    const bw = num(r?.gb_per_s);
    const ref = num(r?.lab51_gb_per_s);
    if (bw == null || !ref) return null;
    return {
      value: (bw / ref) * 100,
      detail: tr(
        `${fmt(bw)} GB/s counted against lab 5.1's ${fmt(ref)} GB/s, rows ${rows}, d 4096`,
        `${fmt(bw)} GB/s tính theo số byte, so với ${fmt(ref)} GB/s của lab 5.1, rows ${rows}, d 4096`,
      ),
    };
  });

/** Lab X3.2: tokens per second at batch 1, position 16, as a percentage of the weight-bandwidth ceiling. */
const ceilingFraction = (reports: Reports) =>
  measure(reports, l2, (s: BenchSet) => {
    const r = benchRow(s, 'ceiling');
    const f = num(r?.fraction);
    if (f == null) return null;
    return {
      value: f * 100,
      detail: tr(
        `${fmt(num(r!.tokens_per_s) ?? NaN)} tokens/s at batch 1, position 16, against a ceiling of ${fmt(num(r!.ceiling_tokens_per_s) ?? NaN)}: lab 5.1's ${fmt(num(r!.lab51_gb_per_s) ?? NaN)} GB/s over ${fmt((num(r!.weight_bytes_per_token) ?? NaN) / 1e6)} MB of weights per token`,
        `${fmt(num(r!.tokens_per_s) ?? NaN)} token/s ở batch 1, vị trí 16, so với trần ${fmt(num(r!.ceiling_tokens_per_s) ?? NaN)}: ${fmt(num(r!.lab51_gb_per_s) ?? NaN)} GB/s của lab 5.1 chia cho ${fmt((num(r!.weight_bytes_per_token) ?? NaN) / 1e6)} MB trọng số mỗi token`,
      ),
    };
  });

/** A piece row of lab X3.2: one group's time at a batch and position. */
const piece = (s: BenchSet, batch: number, position: number, group: string) =>
  s.rows.find((r) => r.name === 'piece' && num(r.batch) === batch && num(r.position) === position && r.group === group);

/** Lab X3.2: attention's share of the summed pieces at position 512, batch 1; position 16 alongside. */
const attentionShare = (reports: Reports) =>
  measure(reports, l2, (s: BenchSet) => {
    const late = piece(s, 1, 512, 'attention');
    const v = num(late?.share);
    if (v == null) return null;
    const early = num(piece(s, 1, 16, 'attention')?.share);
    return {
      value: v * 100,
      detail: tr(
        `batch 1, the pieces timed alone with the ${GEMM[String(late!.gemm)] ?? String(late!.gemm)} for the matrix products; at position 16 the share was ${early == null ? 'not recorded' : `${fmt(early * 100)} %`}`,
        `batch 1, mỗi phần được đo riêng, với ${GEMM[String(late!.gemm)] ?? String(late!.gemm)} cho các phép nhân ma trận; ở vị trí 16 tỉ phần là ${early == null ? 'không được ghi lại' : `${fmt(early * 100)} %`}`,
      ),
    };
  });

// ---------------------------------------------------------------- results tables
const KERNEL_1: Record<string, string> = {
  rmsnorm: 'launchRmsNorm',
  get rope() {
    return tr('launchRope, mode 0', 'launchRope, chế độ 0');
  },
  swiglu: 'launchSwiGLU',
};
const usCol: Col = {
  get head() {
    return tr('Time, µs', 'Thời gian, µs');
  },
  num: true,
  cell: (r) => {
    const v = num(r.seconds);
    return v == null ? '' : fmt(v * 1e6);
  },
};
const pctOf51: Col = {
  get head() {
    return tr('% of lab 5.1', '% của lab 5.1');
  },
  num: true,
  cell: (r) => {
    const bw = num(r.gb_per_s);
    const ref = num(r.lab51_gb_per_s);
    return bw == null || !ref ? '' : fmt((bw / ref) * 100);
  },
};
const headsCol: Col = {
  get head() {
    return tr('Heads × size', 'Số head × kích thước');
  },
  num: true,
  cell: (r) => (num(r.heads) != null ? `${r.heads} × ${r.head_dim}` : ''),
};

const GEMM: Record<string, string> = {
  get gemv() {
    return tr('lab 22.1 gather', 'lab 22.1, kiểu gather');
  },
  m2: 'lab 6.4 (M2)', // i18n-ignore: lab number and milestone
  m3: 'lab 15.6 (M3)', // i18n-ignore: lab number and milestone
};
const ATTN: Record<string, string> = {
  get grouped() {
    return tr('lab 20.5, grouped', 'lab 20.5, theo nhóm');
  },
  decode: 'lab 20.4', // i18n-ignore: lab number
};
const big = (v: unknown) => (num(v) != null ? num(v)!.toLocaleString('en-US') : '?');
const mb = (v: unknown) => (num(v) != null ? fmt(num(v)! / 1e6) : '');
const STEP_COLS: () => Col[] = () => [
  { head: tr('Matrix products', 'Phép nhân ma trận'), cell: (r) => GEMM[String(r.gemm)] ?? String(r.gemm ?? '') },
  { head: tr('Attention', 'Attention'), cell: (r) => ATTN[String(r.attention)] ?? String(r.attention ?? '') },
  rawCol(tr('Batch', 'Batch'), 'batch'),
  rawCol(tr('Position', 'Vị trí'), 'position'),
  msCol,
  numCol(tr('Tokens/s', 'Token/s'), 'tokens_per_s'),
  numCol(tr('Weight GB/s', 'GB/s trọng số'), 'weight_gb_per_s'),
  { head: tr('Weights per token, MB', 'Trọng số mỗi token, MB'), num: true, cell: (r) => mb(r.weight_bytes_per_token) },
  { head: tr('KV read, MB', 'KV đọc, MB'), num: true, cell: (r) => mb(r.kv_bytes) },
];

/** Lab X3.2: the per-piece split, one row per (backend, kernel, batch, position), groups as columns. */
const GROUPS: () => [string, string][] = () => [
  ['matmuls', tr('matrix products', 'phép nhân ma trận')],
  ['attention', tr('append + attention', 'append + attention')],
  ['norms+rope+swiglu+residual', tr('norms, RoPE, SwiGLU, residual', 'chuẩn hóa, RoPE, SwiGLU, residual')],
  ['embed+lm head+argmax+advance', tr('embed, LM head, argmax, advance', 'embed, LM head, argmax, tiến bước')], // i18n-ignore: group key in the benchmark rows
];
function PieceTable() {
  const { reports, loading } = useLabs();
  const rows = benchSets(reports, l2).flatMap((s) => {
    const keys = [...new Set(s.rows.filter((r) => r.name === 'piece').map((r) => `${r.gemm}|${r.batch}|${r.position}`))];
    return keys.map((k) => {
      const [gemm, batch, position] = k.split('|');
      const at = (g: string) => piece(s, Number(batch), Number(position), g) as BenchRow | undefined;
      return { s, gemm, batch, position, at };
    });
  });
  if (loading)
    return (
      <p className="muted">
        <Translated vi="Đang đọc báo cáo.">Reading reports.</Translated>
      </p>
    );
  if (!rows.length)
    return (
      <p className="muted" data-pieces-empty="X3.2">
        <Translated
          vi={
            <>
              Máy này chưa có số đo thời gian theo từng phần: chúng có được khi chạy{' '}
              <code>python3 labs/pmpp.py run X3.2 --bench</code>.
            </>
          }
        >
          No per-piece timings on this machine yet: they come with <code>python3 labs/pmpp.py run X3.2 --bench</code>.
        </Translated>
      </p>
    );
  return (
    <div className="wide" style={{ maxWidth: '62rem', overflowX: 'auto' }}>
      <table className="data" data-table="pieces">
        <thead>
          <tr>
            <th>
              <Translated vi="Backend">Backend</Translated>
            </th>
            <th>
              <Translated vi="Kernel">Kernel</Translated>
            </th>
            <th>
              <Translated vi="Phép nhân ma trận">Matrix products</Translated>
            </th>
            <th className="num">
              <Translated vi="Batch">Batch</Translated>
            </th>
            <th className="num">
              <Translated vi="Vị trí">Position</Translated>
            </th>
            {GROUPS().map(([, h]) => (
              <th key={h} className="num">
                {h}
                <Translated vi=", ms (tỉ phần, %)">, ms (share, %)</Translated>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ s, gemm, batch, position, at }, i) => (
            <tr key={i}>
              <td>{s.backend}</td>
              <td>{s.variant === 'work' ? tr('yours', 'của bạn') : tr('reference solution', 'lời giải tham chiếu')}</td>
              <td>{GEMM[gemm] ?? gemm}</td>
              <td className="num">{batch}</td>
              <td className="num">{position}</td>
              {GROUPS().map(([g]) => {
                const r = at(g);
                const t = num(r?.seconds);
                const sh = num(r?.share);
                return (
                  <td key={g} className="num">
                    {t == null ? '' : `${fmt(t * 1000)} (${sh == null ? '?' : fmt(sh * 100)})`}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Lab X3.2: the shape of the benchmark model, as the driver reported it. */
function ModelLine() {
  const { reports, loading } = useLabs();
  const found = benchSets(reports, l2).flatMap((s) => {
    const r = benchRow(s, 'model');
    return r ? [{ s, r }] : [];
  });
  return (
    <p data-model={loading ? 'loading' : found.length ? found.map((f) => f.s.backend + ':' + f.s.variant).join(',') : 'none'}>
      {loading
        ? tr('Reading reports.', 'Đang đọc báo cáo.')
        : !found.length
          ? tr(
              'No benchmark model is recorded on this machine yet; the driver prints its shape when it runs the benchmark.',
              'Máy này chưa ghi lại mô hình benchmark nào; chương trình điều khiển (driver) in ra kích thước của nó khi chạy benchmark.',
            )
          : found.map((f, i) => (
              <span key={i}>
                <Translated
                  vi={
                    <>
                      {i ? ' ' : ''}Trên {f.s.backend} ({f.s.variant === 'work' ? 'lần chạy của bạn' : 'lần chạy của lời giải tham chiếu'}),
                      mô hình benchmark có d {big(f.r.d)} với {big(f.r.heads)} head truy vấn trên {big(f.r.kv_heads)} head KV, hidden{' '}
                      {big(f.r.hidden)}, {big(f.r.layers)} lớp, từ vựng {big(f.r.vocab)} và độ dài cache {big(f.r.seq_len)}:{' '}
                      {mb(f.r.weight_bytes)} MB trọng số float32.
                    </>
                  }
                >
                  {i ? ' ' : ''}On {f.s.backend} ({f.s.variant === 'work' ? 'your run' : 'the reference solution’s run'}) the benchmark
                  model was d {big(f.r.d)} with {big(f.r.heads)} query heads over {big(f.r.kv_heads)} KV heads, hidden{' '}
                  {big(f.r.hidden)}, {big(f.r.layers)} layers, vocabulary {big(f.r.vocab)} and cache length {big(f.r.seq_len)}:{' '}
                  {mb(f.r.weight_bytes)} MB of float32 weights.
                </Translated>
              </span>
            ))}
    </p>
  );
}

// ---------------------------------------------------------------- the race backend, lab X3.1
/** One line on lab X3.1's runs on the race backend, which the status table does not show. */
function RaceLine({ lab }: { lab: Lab }) {
  const { reports, loading } = useLabs();
  const run = reports.race;
  const line = (variant: 'solution' | 'work') => {
    const rs = (run?.results.filter((x) => x.lab === lab.id && x.variant === variant) ?? []) as (RunResult & { tsan?: unknown[] })[];
    const tests = rs.flatMap((x) => x.tests);
    const found = rs.reduce((k, x) => k + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0);
    return { s: labStatus(run, lab.id, variant), tests, passed: tests.filter((t) => t.pass).length, found };
  };
  const sol = line('solution');
  const mine = line('work');
  return (
    <p data-race={lab.num} data-race-status={loading ? 'loading' : sol.s} data-race-work={loading ? 'loading' : mine.s}>
      {loading ? (
        tr('Reading reports.', 'Đang đọc báo cáo.')
      ) : !run ? (
        tr('No run on the race backend on this machine yet.', 'Máy này chưa có lần chạy nào trên backend race.')
      ) : (
        <>
          {sol.s === 'none'
            ? tr(
                `The reference solution of lab ${lab.num} has no run on the race backend on this machine.`,
                `Lời giải tham chiếu của lab ${lab.num} chưa có lần chạy nào trên backend race ở máy này.`,
              )
            : sol.s === 'pass'
              ? tr(
                  `On the race backend the reference solution of lab ${lab.num} is emulator-checked: all ${sol.tests.length} tests passed, with ${sol.found ? `${sol.found} ThreadSanitizer report(s)` : 'no race reported'}.`,
                  `Trên backend race, lời giải tham chiếu của lab ${lab.num} là emulator-checked: cả ${sol.tests.length} kiểm thử đều qua, với ${sol.found ? `${sol.found} báo cáo ThreadSanitizer` : 'không có race nào được báo'}.`,
                )
              : tr(
                  `On the race backend the reference solution of lab ${lab.num} is unverified: failed on this machine (${sol.passed} of ${sol.tests.length} tests passed).`,
                  `Trên backend race, lời giải tham chiếu của lab ${lab.num} là unverified: thất bại trên máy này (${sol.passed} trên ${sol.tests.length} kiểm thử qua).`,
                )}{' '}
          {mine.s === 'none'
            ? tr(
                'Your own kernels have no recorded run there, or are still the untouched starter.',
                'Các kernel của bạn chưa có lần chạy nào được ghi lại ở đó, hoặc vẫn là mã khởi đầu (starter) chưa sửa.',
              )
            : mine.s === 'pass'
              ? tr(
                  `Your own kernels passed all ${mine.tests.length} tests there with ${mine.found ? `${mine.found} ThreadSanitizer report(s): read them` : 'no race reported'}.`,
                  `Các kernel của bạn qua cả ${mine.tests.length} kiểm thử ở đó, với ${mine.found ? `${mine.found} báo cáo ThreadSanitizer: hãy đọc chúng` : 'không có race nào được báo'}.`,
                )
              : tr(
                  `Your own kernels passed ${mine.passed} of ${mine.tests.length} tests there, with ${mine.found} ThreadSanitizer report(s) in the run.`,
                  `Các kernel của bạn qua ${mine.passed} trên ${mine.tests.length} kiểm thử ở đó, với ${mine.found} báo cáo ThreadSanitizer trong lần chạy.`,
                )}
        </>
      )}
    </p>
  );
}

/** Lab X3.1's port prediction: how many of the three defects emu reports in the code as shipped. */
function PortReveal({ committed }: { committed: number }): ReactNode {
  // The count and the reason come at once; which defect is which waits until the reader says
  // the review is over, so the reveal does not hand over the answers before the first run.
  const [fixed, setFixed] = useState(false);
  return (
    <div data-reveal="p-lab1-port-emu">
      <Translated
        vi={
          <p>
            Khi lab này được xây dựng, <code>emu</code> báo <b>2</b> trong ba lỗi của bản port như được giao, ở cả ba thứ tự
            luồng của nó. {committed === 2 ? 'Đó đúng là con số của bạn.' : `Bạn đã dự đoán ${committed}.`}
          </p>
        }
      >
        <p>
          When this lab was built, <code>emu</code> reported <b>2</b> of the three in the port as shipped, in all three of
          its thread orders. {committed === 2 ? 'That is your number.' : `You predicted ${committed}.`}
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Lỗi nó không báo là một rào chắn (barrier) bị thiếu. Trình giả lập chạy mỗi luồng tới điểm đồng bộ hóa kế tiếp
            của nó (một rào chắn hoặc một warp shuffle) rồi mới chạy luồng tiếp theo, nên một lần ghi xảy ra sau điểm đồng bộ
            hóa kế tiếp của chính luồng ghi không bao giờ vượt được lần đọc của một luồng chậm hơn. Nó chỉ lộ ra rào chắn bị
            thiếu khi một luồng ghi vào buffer chia sẻ trước điểm đó. Backend race chạy các luồng cùng lúc và chỉ ra cả hai
            dòng.
          </p>
        }
      >
        <p>
          The one it did not report is a missing barrier. The emulator runs each thread up to its next synchronisation point
          (a barrier or a warp shuffle) before the next thread runs, so a write that comes after the writer's own next
          synchronisation point can never overtake a slower thread's read. It shows a missing barrier only when a thread
          writes the shared buffer before that point. The race backend runs the threads at once and names both lines.
        </p>
      </Translated>
      {fixed ? (
        <Translated
          vi={
            <p data-reveal-detail="p-lab1-port-emu">
              Từng lỗi một: (a), giai đoạn thứ hai đọc những ô mà không warp nào ghi, biến các trường hợp d 64 thành NaN, vì
              một khối 256 luồng có ít warp hơn số ô mà giai đoạn đó đọc, và trình giả lập lấp bộ nhớ chia sẻ bằng{' '}
              <code>0xCD</code>. (c), đường <code>float4</code> được chọn khi d không phải bội của 4, làm lần chạy dừng với
              một báo cáo AddressSanitizer tại lần đọc vector. (b), không có rào chắn trước khi hàng tiếp theo ghi đè buffer,
              qua mọi trường hợp của <code>emu</code> cho đến khi (a) được sửa bằng cách xóa các ô về 0 trước: việc xóa về 0
              là một lần ghi trước điểm đồng bộ hóa kế tiếp của luồng ghi, và khi đó các trường hợp nhiều hàng thất bại trên{' '}
              <code>emu</code> ở hai trong các thứ tự của nó. Backend race chỉ ra (b) trong mã như được giao, tại lần ghi ô và lần
              đọc hệ số scale. Trên GPU Metal nơi lab này được viết, chỉ (c) thất bại; (a) và (b) qua mọi trường hợp ở đó.
            </p>
          }
        >
          <p data-reveal-detail="p-lab1-port-emu">
            Defect by defect: (a), the second stage reading slots no warp wrote, turned the d 64 cases to NaN, because a block
            of 256 has fewer warps than the stage reads slots and the emulator fills shared memory with <code>0xCD</code>.
            (c), the <code>float4</code> path taken when d is not a multiple of 4, stopped the run with an AddressSanitizer
            report at the vector read. (b), no barrier before the next row overwrites the buffer, passed every{' '}
            <code>emu</code> case until (a) was fixed by zeroing the slots first: the zeroing is a write before the writer's
            next synchronisation point, and then <code>emu</code> fails the several-rows cases in two of its orders. The race
            backend named (b) in the shipped code, at the slot write and the scale read. On the Metal GPU this lab was written
            on, only (c) failed; (a) and (b) passed every case there.
          </p>
        </Translated>
      ) : (
        <p>
          <button type="button" onClick={() => setFixed(true)}>
            <Translated vi="Tôi đã sửa cả ba: cho xem backend nào thấy lỗi nào">
              I have fixed all three: show which backend saw which
            </Translated>
          </button>
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- lab X3.2's launch order, from the tested model
const GPU_MODEL: Config = {
  d: LAB_GPU.dim,
  hidden: LAB_GPU.hidden_dim,
  layers: LAB_GPU.n_layers,
  heads: LAB_GPU.n_heads,
  kvHeads: LAB_GPU.n_kv_heads,
  vocab: LAB_GPU.vocab_size,
  head: LAB_GPU.dim / LAB_GPU.n_heads,
  seqLen: LAB_GPU.seq_len,
};
function LaunchTable() {
  const all = stepLaunches(GPU_MODEL, 0);
  const L = GPU_MODEL.layers;
  const head = all.filter((l) => l.layer === null || l.layer === 0);
  const firstTail = head.findIndex((l) => l.layer === null && l.n > 0);
  const row = (l: (typeof all)[number]) => (
    <tr key={l.n} data-launch={l.job}>
      <td className="num">{l.n}</td>
      <td>
        {l.layer === 0 ? tr('layer 0: ', 'lớp 0: ') : ''}
        {JOB_LABEL[l.job]}
      </td>
      <td>
        <code>{l.fn}</code>
      </td>
      <td>{l.exists ? tr(l.lab, l.lab.replace(' (or ', ' (hoặc ').replace(' (yours)', ' (của bạn)')) : <b>{tr(l.lab, l.lab.replace(' (or ', ' (hoặc ').replace(' (yours)', ' (của bạn)'))}</b>}</td>
    </tr>
  );
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
      <table className="data" data-table="launches">
        <thead>
          <tr>
            <th className="num">#</th>
            <th>
              <Translated vi="Phép toán">Operation</Translated>
            </th>
            <th>
              <Translated vi="Lần khởi chạy">Launch</Translated>
            </th>
            <th>
              <Translated vi="Từ">From</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          {head.slice(0, firstTail).map(row)}
          <tr data-launch="layers">
            <td className="num">
              17–{16 * L}
            </td>
            <td colSpan={3} className="muted">
              <Translated
                vi={<>các lớp 1 đến {L - 1}: cùng mười sáu lần khởi chạy, mỗi lớp với trọng số và cache của riêng nó</>}
              >
                layers 1 to {L - 1}: the same sixteen launches, each with its own layer's weights and cache
              </Translated>
            </td>
          </tr>
          {head.slice(firstTail).map(row)}
        </tbody>
      </table>
      <Translated
        vi={
          <p className="muted" style={{ marginTop: '0.4rem' }}>
            Một bước của mô hình kiểm thử GPU trong lab X3.2 ({L} lớp, {all.length} lần khởi chạy), lấy từ{' '}
            <code>app/src/sim/llamastep.ts</code>, mô hình mà mục <a href={href('x3', 'X3.7')}>X3.7</a> chạy từng bước; unit
            test của nó kiểm tra thứ tự so với <code>llama.h</code>. Nói chung một bước gồm {launchCount(1) - 16} + 16 × số
            lớp lần khởi chạy. In đậm: mới trong module này.
          </p>
        }
      >
        <p className="muted" style={{ marginTop: '0.4rem' }}>
          One step of lab X3.2's GPU test model ({L} layers, {all.length} launches), from <code>app/src/sim/llamastep.ts</code>
          , the model section <a href={href('x3', 'X3.7')}>X3.7</a> steps through; its unit test checks the order against{' '}
          <code>llama.h</code>. In general a step is {launchCount(1) - 16} + 16 × layers launches. Bold: new in this module.
        </p>
      </Translated>
    </div>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS} tiers={TIERS_1}>
      <LabsIntro>
        <Translated
          vi={
            <>
              Hai lab, mỗi lab trong thư mục riêng dưới <code>labs/x3/</code>. Lab thứ nhất viết ba kernel mà một bước decode
              của Llama cần nhưng các mốc dự án M0 đến M8 chưa có (RMSNorm, RoPE và SwiGLU), và review một RMSNorm được port
              theo cấu trúc các kernel của llama.cpp, có ba lỗi cài sẵn. Lab thứ hai là mốc dự án M10: nó ghép các lời giải
              tham chiếu của những mốc trước thành một bước decode greedy của một mô hình kiểu Llama và kiểm tra từng logit,
              từng token so với một bản tham chiếu CPU.
            </>
          }
        >
          Two labs, each in its own directory under <code>labs/x3/</code>. The first writes the three kernels a Llama
          decode step needs that milestones M0 to M8 lack (RMSNorm, RoPE and SwiGLU) and reviews an RMSNorm ported in the
          structure of llama.cpp's kernels, with three defects planted in it. The second is milestone M10: it composes
          the earlier milestones' reference solutions into one greedy decode step of a Llama-style model and checks each logit and each token against a
          CPU reference.
        </Translated>
      </LabsIntro>
      <Translated
        vi={
          <p>
            <code>pmpp.py</code> đặt tên các lab của module này giống như các lab của sách:{' '}
            <code>python3 labs/pmpp.py run X3.1</code> chạy lab X3.1, và <code>X3.2</code> là mốc dự án M10.
          </p>
        }
      >
        <p>
          <code>pmpp.py</code> names this module's labs as it names the book's: <code>python3 labs/pmpp.py run X3.1</code>{' '}
          runs lab X3.1, and <code>X3.2</code> is milestone M10.
        </p>
      </Translated>
      <Beyond title={tr('All of this is ours', 'Tất cả phần này là của khóa học này')}>
        <Translated
          vi={
            <p>
              Module X3 không có trong sách. Các hợp đồng (contract), kiểm thử và mô hình trên trang này là của khóa học này.
              Những gì các kernel tính lấy từ các dự án được đọc ngày 2026-11-20 tại các commit mà mục{' '}
              <a href={href('x3', 'start')}>Trước khi bắt đầu</a> liệt kê; phép rút gọn (reduction) là của lab 10.3, từ mục
              10.8 của sách.
            </p>
          }
        >
          <p>
            Module X3 is not in the book. The contracts, tests and models on this page are ours. What the kernels compute
            comes from the projects read on 2026-11-20 at the commits section <a href={href('x3', 'start')}>Before you
            start</a> lists; the reduction is lab 10.3's, from the book's section 10.8.
          </p>
        </Translated>
      </Beyond>
      <RunningLabs example="X3.1" />
      <BackendsAndEmulator />
      <Translated
        vi={
          <p>
            Lab X3.1 cũng được build cho biến thể race của trình giả lập, vì bản port của nó dùng chung một buffer giữa hai
            phép rút gọn và qua nhiều hàng. <code>python3 labs/pmpp.py run X3.1 --backend race</code> chạy các luồng của một
            khối cùng lúc dưới ThreadSanitizer, công cụ báo hai lần truy cập vào cùng một vị trí mà không có gì sắp thứ tự
            chúng, kèm cả hai dòng mã nguồn. Một lần chạy sạch là bằng chứng, không phải chứng minh: nó chỉ thấy những lịch
            chạy đã xảy ra. Lab X3.2 thì không: mã của bạn ở đó khởi chạy các phần và không ghi bộ nhớ chia sẻ nào, và mỗi
            phần đã được kiểm tra race trong lab riêng của nó.
          </p>
        }
      >
        <p>
          Lab X3.1 is also built for the emulator's race flavour, because its port shares one buffer between two reductions
          and across rows. <code>python3 labs/pmpp.py run X3.1 --backend race</code> runs a block's threads at once under
          ThreadSanitizer, which reports two accesses to one location with nothing ordering them, with both source lines. A
          clean run is evidence, not proof: it sees only the schedules that happened. Lab X3.2 is not: your code there
          launches pieces and writes no shared memory, and each piece was race-tested in its own lab.
        </p>
      </Translated>
      <VerificationStatus />
      <Translated
        vi={
          <p>
            Lab X3.2 chạy lời giải tham chiếu của các lab khác, những thứ bạn không sửa. <code>provided/pieces.cmake</code>{' '}
            sao chép chúng vào thư mục build lúc configure: các lab 2.3, 6.4, 10.3, 20.4, 22.2 và X3.1 là bắt buộc, còn 15.6,
            20.5, 22.1, X1.3 và X2.4 (cùng các GEMV của X2.1 và X2.2) được dùng khi có mặt. Nó đổi tên những gì sẽ trùng và
            biến các hàm trợ giúp device một dòng thành <code>static</code>, như lab 22.2 làm. Một phần bắt buộc bị thiếu,
            hoặc một phần có chữ ký khác với chữ ký mà <code>llama.h</code> khai báo, sẽ khiến lab bị bỏ qua kèm một ghi chú
            nêu tên nó. Vì vậy M10 chạy các phần tham chiếu, không phải phiên bản của riêng bạn: một thất bại trong bước kiểm
            tra "pieces alone" là lỗi của khóa học, không phải của bạn.
          </p>
        }
      >
        <p>
          Lab X3.2 runs other labs' reference solutions, which you do not edit. <code>provided/pieces.cmake</code> copies them
          into the build directory at configure time: labs 2.3, 6.4, 10.3, 20.4, 22.2 and X3.1 are required, and 15.6, 20.5,
          22.1, X1.3 and X2.4 (with X2.1's and X2.2's GEMVs) are used when present. It renames what would clash and makes one-line device helpers{' '}
          <code>static</code>, as lab 22.2's does. A missing required piece, or one whose signature differs from the one{' '}
          <code>llama.h</code> declares, skips the lab with a note naming it. So M10 runs the reference pieces, not your own
          versions of them: a failure in its "pieces alone" check is the course's, not yours.
        </p>
      </Translated>

      {/* ------------------------------------------------------------ X3.1 */}
      <h2>
        <Translated vi="Lab X3.1: những phần Llama khóa học chưa xây dựng, và một lần review bản port">
          Lab X3.1: the Llama pieces the course has not built, and a port review
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            Phần A là ba kernel ở dạng nhỏ nhất, float32 theo thứ tự hàng (row-major) với các token đứng trước. Mục{' '}
            <a href={href('x3', 'X3.3')}>X3.3</a> đọc các kernel RMSNorm của các dự án và mục{' '}
            <a href={href('x3', 'X3.7')}>X3.7</a> đặt cả ba vào bước decode.
          </p>
        }
      >
        <p>
          Part A is the three kernels in their smallest form, row-major float32 with the tokens first. Section{' '}
          <a href={href('x3', 'X3.3')}>X3.3</a> reads the projects' RMSNorm kernels and section{' '}
          <a href={href('x3', 'X3.7')}>X3.7</a> places all three in the step.
        </p>
      </Translated>
      <ul>
        <Translated
          vi={
            <li>
              <code>launchRmsNorm(x, w, y, rows, d, eps)</code>: mỗi hàng được nhân với 1 / sqrt(mean(x²) + eps), rồi nhân
              với trọng số. Mỗi hàng một khối 256 luồng với phép tổng hai giai đoạn của lab 10.3: một lane mask có kiểu,{' '}
              <code>warpSize</code>, giai đoạn hai có kiểm tra điều kiện. <code>y</code> có thể là <code>x</code>.
            </li>
          }
        >
          <li>
            <code>launchRmsNorm(x, w, y, rows, d, eps)</code>: each row scaled by 1 / sqrt(mean(x²) + eps), times the weight.
            One block of 256 threads per row with lab 10.3's two-stage sum: a typed lane mask, <code>warpSize</code>, the
            second stage guarded. <code>y</code> may be <code>x</code>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <code>launchRope(x, pos, cosT, sinT, rows, heads, headDim, mode)</code>: tại chỗ, hàng r ở vị trí{' '}
              <code>pos[r]</code>, các bảng được tính trên host bằng double để kernel không phải làm phép lượng giác nào. Chế
              độ 0 xoay các cặp (2i, 2i+1), kiểu NORM của llama.cpp và của llama2.c; chế độ 1 xoay (i, i + headDim/2), cách
              ghép cặp NeoX của <code>rotate_half</code> trong Hugging Face.
            </li>
          }
        >
          <li>
            <code>launchRope(x, pos, cosT, sinT, rows, heads, headDim, mode)</code>: in place, row r at position{' '}
            <code>pos[r]</code>, the tables built on the host in double so the kernel does no trigonometry. Mode 0 rotates
            the pairs (2i, 2i+1), llama.cpp's NORM and llama2.c's; mode 1 rotates (i, i + headDim/2), the NeoX pairing of
            Hugging Face's <code>rotate_half</code>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <code>launchSwiGLU(gate, up, out, n)</code>: gate / (1 + exp(−gate)) × up, tức SiLU của llama.cpp nhân với phép
              chiếu up. <code>out</code> có thể là <code>gate</code>.
            </li>
          }
        >
          <li>
            <code>launchSwiGLU(gate, up, out, n)</code>: gate / (1 + exp(−gate)) × up, llama.cpp's SiLU times the up
            projection. <code>out</code> may be <code>gate</code>.
          </li>
        </Translated>
      </ul>
      <Translated
        vi={
          <p>
            Trên Metal: <code>rms_norm_rows</code> (mỗi hàng một threadgroup, <code>simd_sum</code>, độ rộng SIMD group đọc từ
            pipeline), <code>rope_rows</code> và <code>swiglu</code>, được mã hóa vào encoder của bên gọi. Các kiểm thử đưa
            cho SwiGLU một gate bằng −100, nơi exp(100) bị tràn số. Phép chia IEEE mặc định của CUDA biến −100 / vô cực thành
            −0; dưới fast math của Metal, hành vi của vô cực là không xác định, nên kernel Metal phải tính cùng hàm đó mà
            không tạo ra vô cực.
          </p>
        }
      >
        <p>
          On Metal: <code>rms_norm_rows</code> (one threadgroup per row, <code>simd_sum</code>, the SIMD-group width read from
          the pipeline), <code>rope_rows</code> and <code>swiglu</code>, encoded into the caller's encoder. The tests feed
          SwiGLU a gate of −100, where exp(100) overflows. CUDA's default IEEE division turns −100 / infinity into −0; under
          Metal's fast math the behaviour of infinity is undefined, so the Metal kernel must compute the same function
          without forming it.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch05-facts.md C7; ch10-facts.md C3; ch21-facts.md A10)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              MSL mặc định dùng các tối ưu dấu phẩy động không theo IEEE (các nguồn của Apple không thống nhất chính xác là
              chế độ nào); dưới fast math, cách xử lý vô cực và NaN là không xác định. Phép chia và <code>sqrtf</code> của
              CUDA mặc định được làm tròn đúng.
            </p>
          }
        >
          <p>
            MSL defaults to non-IEEE floating-point optimisations (Apple's sources differ on exactly which mode); under fast
            math the handling of infinity and NaN is undefined. CUDA's division and <code>sqrtf</code> are correctly rounded
            by default.
          </p>
        </Translated>
      </Today>
      <LabFiles
        lab={l1}
        files={[
          ['kernel.cu', <Translated vi={<><code>rmsNormRowsKernel</code>, <code>ropeKernel</code>, <code>swigluKernel</code> và các hàm khởi chạy của chúng; <code>rmsNormPortKernel</code> và <code>launchRmsNormPort</code> của phần B</>}><><code>rmsNormRowsKernel</code>, <code>ropeKernel</code>, <code>swigluKernel</code> and their launchers; part B's <code>rmsNormPortKernel</code> and <code>launchRmsNormPort</code></></Translated>],
          ['kernel.metal', <Translated vi={<><code>rms_norm_rows</code>, <code>rope_rows</code>, <code>swiglu</code>; <code>rms_norm_port</code> của phần B</>}><><code>rms_norm_rows</code>, <code>rope_rows</code>, <code>swiglu</code>; part B's <code>rms_norm_port</code></></Translated>],
          ['launch.mm', <Translated vi={<><code>encodeRmsNorm</code>, <code>encodeRope</code>, <code>encodeSwiGLU</code>; <code>encodeRmsNormPort</code> của phần B, hàm chọn đường <code>float4</code> trên host như llama.cpp làm</>}><><code>encodeRmsNorm</code>, <code>encodeRope</code>, <code>encodeSwiGLU</code>; part B's <code>encodeRmsNormPort</code>, which picks the <code>float4</code> path on the host as llama.cpp does</></Translated>],
        ]}
      />
      <h3>
        <Translated vi="Phần B: bản port">Part B: the port</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Một đồng nghiệp đã port một RMSNorm theo cấu trúc các kernel của llama.cpp: 256 luồng cho các hàng dưới 1,024
            phần tử và 1,024 luồng cho hàng lớn hơn, một đường <code>float4</code> khi lần khởi chạy đánh giá là an toàn, và
            một phép rút gọn hai giai đoạn qua một buffer chia sẻ 64 ô, trong đó ô 0 chuyển hệ số scale cho cả khối. Một lưới
            cố định 4 khối đi qua các hàng, nên một khối dùng lại buffer của nó hết hàng này sang hàng khác (bản thân
            llama.cpp chạy mỗi hàng một khối). Nó biên dịch được và qua một số kiểm thử. Nó có ba lỗi độc lập, cùng ba lỗi đó
            trong cả hai file. Mỗi lỗi là thứ mà các kernel của llama.cpp có phòng ngừa.
          </p>
        }
      >
        <p>
          A colleague ported an RMSNorm in the structure of llama.cpp's kernels: 256 threads for rows under 1,024 elements and
          1,024 above, a <code>float4</code> path when the launch judges it safe, and a two-stage reduction through one
          shared buffer of 64 slots whose slot 0 hands the scale to the block. A fixed grid of 4 blocks walks the rows, so a
          block reuses its buffer row after row (llama.cpp itself runs one block per row). It compiles and passes some tests.
          It has three independent defects, the same three in both files. Each is something llama.cpp's kernels guard
          against.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            README liệt kê những gì không phải lỗi: lane mask có kiểu và <code>warpSize</code>, phép ép kiểu{' '}
            <code>float*</code> sang <code>float4*</code> của sách, việc bind một buffer Metal lần thứ hai dưới dạng{' '}
            <code>device float4*</code>, 64 ô, lưới 4 khối, quy tắc kích thước khối, <code>1 / sqrtf</code> và việc đọc hệ
            số scale từ ô 0.
          </p>
        }
      >
        <p>
          The README lists what is not a defect: the typed lane mask and <code>warpSize</code>, the book's{' '}
          <code>float*</code> to <code>float4*</code> cast, binding a Metal buffer a second time as{' '}
          <code>device float4*</code>, the 64 slots, the grid of 4, the block-size rule, <code>1 / sqrtf</code> and reading
          the scale from slot 0.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/infra-facts.md A6, C7; x3-facts.md B2; ch10-facts.md D1)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              Không nhà cung cấp nào ghi trong tài liệu phép ép kiểu con trỏ này; Metal ghi trong tài liệu việc khai báo
              buffer là <code>device float4*</code>. llama.cpp kiểm tra điều kiện ở giai đoạn hai và kiểm tra{' '}
              <code>ncols % 4</code>.
            </p>
          }
        >
          <p>
            No vendor documents the pointer cast; Metal documents declaring the buffer <code>device float4*</code>. llama.cpp
            guards its second stage and checks <code>ncols % 4</code>.
          </p>
        </Translated>
      </Today>
      <ol>
        <Translated
          vi={
            <li>
              Đọc <code>work/kernel.cu</code> trước và ghi lại các lỗi bạn thấy, cùng với backend bạn nghĩ sẽ báo từng lỗi:{' '}
              <code>emu</code>, <code>race</code>, GPU của bạn, hoặc không backend nào.
            </li>
          }
        >
          <li>
            Read <code>work/kernel.cu</code> first and write down the defects you see, and for each the backend you expect to
            report it: <code>emu</code>, <code>race</code>, your GPU, or none.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Chạy <code>python3 labs/pmpp.py run X3.1 --backend emu</code>. Các thứ tự luồng của nó là cố định, bộ nhớ chia
              sẻ khởi đầu bằng byte <code>0xCD</code>, và AddressSanitizer dừng lần chạy ở lần đọc đầu tiên vượt quá một vùng
              cấp phát và chỉ ra dòng đó. Các trường hợp đọc vượt quá vùng cấp phát được để sau cùng.
            </li>
          }
        >
          <li>
            Run <code>python3 labs/pmpp.py run X3.1 --backend emu</code>. Its thread orders are fixed, shared memory starts as
            the byte <code>0xCD</code>, and AddressSanitizer stops the run at the first read past an allocation and names the
            line. The cases that read past an allocation come last.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Sửa một chỗ rồi chạy lại. Sau đó chạy <code>python3 labs/pmpp.py run X3.1 --backend race</code>.
            </li>
          }
        >
          <li>
            Fix one thing and run again. Then run <code>python3 labs/pmpp.py run X3.1 --backend race</code>.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Sửa y như vậy trong <code>kernel.metal</code> và <code>launch.mm</code> rồi chạy trên GPU của bạn:{' '}
              <code>python3 labs/pmpp.py run X3.1</code>. Ghi lại những lỗi nó cho bạn thấy và những lỗi nó che đi.
            </li>
          }
        >
          <li>
            Make the same fixes in <code>kernel.metal</code> and <code>launch.mm</code> and run on your GPU:{' '}
            <code>python3 labs/pmpp.py run X3.1</code>. Note which defects it showed you and which it hid.
          </li>
        </Translated>
      </ol>
      <Predict
        id="x3/labs/p-lab1-port-emu"
        unit={tr('defects', 'lỗi')}
        prompt={<Translated vi={<>Sau khi đọc bản port và trước khi chạy nó: trong ba lỗi của nó, <code>emu</code> sẽ báo bao nhiêu lỗi trong mã như được giao?</>}><>After reading the port and before running it: of its three defects, how many will <code>emu</code> report in the code as shipped?</></Translated>}
      >
        {(c) => <PortReveal committed={c} />}
      </Predict>
      <Translated
        vi={
          <p>
            Dù GPU của bạn cho thấy gì, hãy đọc nó đối chiếu với những gì các nhà cung cấp ghi trong tài liệu. Bộ nhớ chia sẻ
            hay bộ nhớ threadgroup mà không ai ghi thì chứa gì cũng được, và không file dữ kiện (facts) nào chúng tôi đã kiểm tra nói
            GPU đặt gì vào đó. Một trường hợp qua mọi lần chạy trên GPU của bạn chỉ cho bạn thấy các lịch chạy của thiết bị
            đó, không phải một kernel đúng.
          </p>
        }
      >
        <p>
          Whatever your GPU showed, read it against what the vendors document. Shared or threadgroup memory nobody wrote holds
          whatever it holds, and no facts file we checked says what a GPU puts there. A case that passed every run on your GPU
          showed you that device's schedules, not a correct kernel.
        </p>
      </Translated>

      <h3>
        <Translated vi="Các kiểm thử kiểm tra gì">What the tests check</Translated>
      </h3>
      <ul>
        <Translated
          vi={
            <li>
              RMSNorm: rows 1, 3 và 17 với d 1, 7, 64, 128, 576 (độ rộng của SmolLM2-135M), 1,000 và 4,097, tại chỗ và không
              tại chỗ, các hàng có thang giá trị khác nhau trong cùng một batch, một hàng toàn số 0 mà đầu ra phải là 0, và
              eps 1e-5. Đầu ra khởi đầu là NaN, các giá trị canh sau <code>y</code> phải không bị đụng tới, và đầu vào phải
              trở về nguyên vẹn.
            </li>
          }
        >
          <li>
            RMSNorm: rows 1, 3 and 17 by d 1, 7, 64, 128, 576 (SmolLM2-135M's width), 1,000 and 4,097, in place and out of
            place, rows of different scales in one batch, an all-zero row whose output must be 0, and eps 1e-5. Outputs
            start as NaN, guard values after <code>y</code> must be untouched, and the inputs must come back unchanged.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              RoPE: các vị trí gồm cả 0 và hàng cuối của bảng; 1, 2 và 4 head; kích thước head 8, 64 và 128; cả hai chế độ so
              với bản tham chiếu. Vị trí 0 phải trả lại chính xác đầu vào, hai chế độ phải khác nhau, và phép hoán vị hàng của
              llama.cpp phải biến chế độ 1 trên (W x) thành chế độ 0 trên (P W x), cho Q và cho K dưới grouped-query
              attention. Hoán vị K theo số head truy vấn phải làm hỏng điều đó.
            </li>
          }
        >
          <li>
            RoPE: positions that include 0 and the last table row; 1, 2 and 4 heads; head sizes 8, 64 and 128; both modes
            against the reference. Position 0 must return the input exactly, the two modes must differ, and llama.cpp's row
            permutation must turn mode 1 on (W x) into mode 0 on (P W x), for Q and for K under grouped-query attention.
            Permuting K with the query head count must break it.
          </li>
        </Translated>
        <Translated
          vi={
            <li>SwiGLU: n 1, 31, 1,000 và 4,099, không tại chỗ và tại chỗ, với các gate −100 và +100 cho kết quả hữu hạn.</li>
          }
        >
          <li>SwiGLU: n 1, 31, 1,000 and 4,099, out of place and in place, with gates of −100 and +100 giving finite results.</li>
        </Translated>
        <Translated
          vi={
            <li>
              Bản port: rows 1 và 3 (mỗi khối một hàng) và nhiều hàng hơn số khối; d 64 và 4,096, và d 577, 1,001 và 4,097,
              không phải bội của 4. Trên các thiết bị và trên <code>race</code>, sau đầu vào là một dải canh giá trị 1000.0,
              nên một lần đọc vượt quá cuối của chúng sẽ làm thay đổi kết quả.
            </li>
          }
        >
          <li>
            The port: rows 1 and 3 (one row per block) and more rows than blocks; d 64 and 4,096, and d 577, 1,001 and 4,097,
            which are not multiples of 4. On devices and on <code>race</code> the inputs are followed by a guard band of
            1000.0, so a read past their end changes the result.
          </li>
        </Translated>
      </ul>
      <Translated
        vi={
          <p>
            Mọi kết quả được so với một bản tham chiếu CPU độ chính xác kép trong phạm vi k × FLT_EPSILON × (1 + |ref|), với
            k = 32 cho RMSNorm, 8 cho RoPE và 16 cho SwiGLU. README suy ra từng k từ độ dài của phép rút gọn và sai số được
            ghi trong tài liệu của từng phép toán; nó giống nhau trên mọi backend.
          </p>
        }
      >
        <p>
          Every result is compared with a double-precision CPU reference within k × FLT_EPSILON × (1 + |ref|), with k = 32
          for RMSNorm, 8 for RoPE and 16 for SwiGLU. The README derives each k from the length of the reduction and the
          documented error of each operation; it is the same on every backend.
        </p>
      </Translated>
      <RaceLine lab={l1} />

      <h3>
        <Translated vi="Trước khi đo hiệu năng">Before you benchmark</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Một lần gọi ở rows 1 và d 4,096 di chuyển khoảng 48 KiB: hàng vào, hàng ra và trọng số. Hãy quyết định nó đạt bao
            nhiêu phần băng thông lab 5.1 của bạn, rồi 256 hàng đạt bao nhiêu phần. Cân nhắc ba lực: một hàng là một khối, và
            một khối chạy trên một lõi (core) của GPU trong khi các lõi còn lại chờ; mỗi lần khởi chạy tốn một khoảng thời gian
            cố định bất kể nó di chuyển bao nhiêu; và benchmark lặp lại lần gọi trên cùng các buffer, nên dữ liệu vừa trong
            các bộ nhớ đệm (cache) không cần lấy từ bộ nhớ sau lần gọi đầu tiên, trong khi lab 5.1 truyền nhiều hơn hẳn những
            gì bất kỳ cache nào chứa được.
          </p>
        }
      >
        <p>
          One call at rows 1 and d 4,096 moves about 48 KiB: the row in, the row out and the weight. Decide what fraction of
          your lab 5.1 bandwidth that reaches, and then what fraction 256 rows reach. Weigh three forces: one row is one
          block, and one block runs on one of the GPU's cores while the rest wait; every launch costs a fixed time whatever
          it moves; and the benchmark repeats the call on the same buffers, so data that fits in the caches need not come
          from memory after the first call, where lab 5.1 streamed far more than any cache holds.
        </p>
      </Translated>
      <LabPrediction
        id="x3/labs/p-lab1-rmsnorm-bw"
        prompt={<Translated vi={<>Ở rows = 1, d = 4,096, <code>launchRmsNorm</code> sẽ đạt bao nhiêu phần trăm băng thông lab 5.1 của bạn?</>}><>At rows = 1, d = 4,096, what percentage of your lab 5.1 bandwidth will <code>launchRmsNorm</code> reach?</></Translated>}
        unit={tr("% of lab 5.1's bandwidth", '% băng thông của lab 5.1')}
        measure={rmsFraction(1)}
        missing={missingBench('X3.1', NEED_51)}
        reading={
          <Translated
            vi={
              <>
                Số GB/s là tính theo số byte, không phải đo: số byte mà lần gọi phải di chuyển, một lần, chia cho thời gian của
                nó. Một phần trăm nhỏ cho biết lần gọi quá ngắn để bị chi phối bởi băng thông.
              </>
            }
          >
            <>
              The GB/s are counted, not measured: the bytes the call must move, once, divided by its time. A small
              percentage says the call is too short to be about bandwidth.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="x3/labs/p-lab1-rmsnorm-bw-256"
        prompt={<Translated vi={<>Còn ở rows = 256, d = 4,096?</>}><>And at rows = 256, d = 4,096?</></Translated>}
        unit={tr("% of lab 5.1's bandwidth", '% băng thông của lab 5.1')}
        measure={rmsFraction(256)}
        missing={missingBench('X3.1', NEED_51)}
        reading={
          <Translated
            vi={
              <>
                Trên 100 % nghĩa là không phải mọi byte được tính đều đến từ bộ nhớ: 256 hàng 4,096 float, vào và ra, đủ nhỏ
                để cache phục vụ một lần gọi lặp lại. Con số của lab 5.1 là cho một tập dữ liệu làm việc không cache nào chứa
                nổi.
              </>
            }
          >
            <>
              Above 100 % means the counted bytes did not all come from memory: 256 rows of 4,096 floats, in and out, are
              small enough for caches to serve a repeated call. Lab 5.1's figure is for a working set no cache holds.
            </>
          </Translated>
        }
      />
      <LabResults lab={l1} cols={[{ head: tr('Launch', 'Lần khởi chạy'), cell: (r) => KERNEL_1[r.name] ?? r.name }, rawCol(tr('Rows', 'Số hàng'), 'rows'), rawCol('d', 'd'), headsCol, usCol, numCol(tr('Counted GB/s', 'GB/s tính theo byte'), 'gb_per_s'), pctOf51]} />
      <Translated
        vi={
          <p>
            Cách benchmark hoạt động: mỗi kernel ở rows (token) 1, 16 và 256 với d 576 và 4,096. RoPE chạy trên các head 128
            khi d là bội của 128, nếu không thì các head 64 (576 là 9 × 64, kích thước của SmolLM2-135M); SwiGLU trên rows ×
            4d phần tử. Số byte được tính là các lần đọc và ghi duy nhất của mỗi lần gọi, một lần. Mỗi thời gian là tốt nhất
            trong 5 lần chạy sau một lần không đo, mỗi lần chạy lặp lại lần gọi cho tới khi kéo dài ít nhất 2 ms, và mọi kết
            quả được đo đều được kiểm tra trước.
          </p>
        }
      >
        <p>
          How the benchmark works: each kernel at rows (tokens) 1, 16 and 256 for d 576 and 4,096. RoPE runs over heads of
          128 where d is a multiple of 128, else heads of 64 (576 is 9 × 64, SmolLM2-135M's shape); SwiGLU over rows × 4d
          elements. The counted bytes are each call's unique reads and writes, once. Each time is the best of 5 runs after an
          untimed one, each run repeating the call until it lasts at least 2 ms, and every timed result is checked first.
        </p>
      </Translated>
      <ReferenceSolution lab={l1} />

      {/* ------------------------------------------------------------ X3.2 */}
      <h2>
        <Translated vi="Lab X3.2: mốc dự án M10, một mô hình kiểu Llama chạy trọn vẹn trên các kernel của khóa học">
          Lab X3.2: milestone M10, a Llama-style model end to end on the course's kernels
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mọi kernel của một bước decode Llama giờ đã có trong khóa học này. Lab này ghép chúng thành một bước greedy và
            chạy một mô hình do một script trong repository viết ra từ một seed cố định. Bạn không viết kernel số học nào. Bạn
            quyết định thứ tự các phần, đối số của chúng và nơi mỗi kết quả nằm, điều chiếm phần lớn việc file mô hình của một
            framework làm, và viết một kernel nhỏ giữ vòng lặp sinh token trên device.
          </p>
        }
      >
        <p>
          Every kernel of a Llama decode step now exists in this course. This lab composes them into one greedy step and runs
          a model that a script in the repository writes from a fixed seed. You write no arithmetic kernel. You decide the
          order of the pieces, their arguments and where each result lives, which is most of what a framework's model file
          does, and one small kernel that keeps the generation loop on the device.
        </p>
      </Translated>
      <ul>
        <Translated
          vi={
            <li>
              <code>launchLlamaStep(m, s)</code> (<code>encodeLlamaStep</code> trên Metal, mọi dispatch trong một compute
              encoder và một command buffer mỗi bước): một bước cho cả batch, mỗi phép toán một lần khởi chạy, không có gì chờ
              GPU hay sao chép về host.
            </li>
          }
        >
          <li>
            <code>launchLlamaStep(m, s)</code> (<code>encodeLlamaStep</code> on Metal, every dispatch in one compute encoder
            and one command buffer per step): one step for the whole batch, one launch per operation, nothing that waits for
            the GPU or copies to the host.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <code>launchAdvancePositions</code> (<code>advance_positions</code> trên Metal): mỗi chuỗi một luồng, ở cuối mỗi
              bước. Nó ghi lại argmax của bước, đưa vào token prompt kế tiếp khi chuỗi vẫn còn trong prompt, và đẩy vị trí lên.
              Trước khi viết nó, hãy nói host sẽ phải làm gì giữa hai bước nếu không có nó.
            </li>
          }
        >
          <li>
            <code>launchAdvancePositions</code> (<code>advance_positions</code> on Metal): one thread per sequence, at the end
            of each step. It records the step's argmax, feeds the next prompt token while the sequence is still in its prompt,
            and moves the position on. Before you write it, say what the host would have to do between two steps without
            it.
          </li>
        </Translated>
      </ul>
      <LaunchTable />
      <Translated
        vi={
          <p>
            Hai câu hỏi cần giải quyết trước khi chạy: attention phải thấy độ dài nào để nó đọc hàng bạn vừa append và không
            gì sau đó, và các token kế tiếp phải đi đâu để bước tiếp theo bắt đầu trên device?{' '}
            <code>llama.h</code> khai báo mọi phần với thứ tự đối số và kích thước của mọi buffer.
          </p>
        }
      >
        <p>
          Two questions to settle before you run it: which length must the attention see so that it reads the row you just
          appended and nothing after it, and where must the next tokens go so that the next step starts on the device?{' '}
          <code>llama.h</code> declares every piece with its argument order and every buffer's shape.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <code>provided/make_model.py</code> viết mô hình theo định dạng <code>.bin</code> phiên bản 0 của llama2.c, định
            dạng mà mục <a href={href('x3', 'X3.8')}>X3.8</a> giải mã: một header 28 byte, rồi các trọng số float32 theo thứ
            tự của llama2.c. Nó chỉ dùng thư viện chuẩn của Python, với bộ sinh số riêng và cách đóng gói little-endian, nên
            mọi máy đều viết ra cùng các byte, và CMake chạy nó lúc configure. CMake, script, chương trình điều khiển hay các
            kiểm thử đều không tải gì về. Prompt đi qua bước decode từng token một, như cách llama2.c xử lý nó: attention
            prefill của M6 không được ghép vào, vì bố cục của nó không phải bố cục cache mà M7 đọc.
          </p>
        }
      >
        <p>
          <code>provided/make_model.py</code> writes the model in llama2.c's version 0 <code>.bin</code> format, the one
          section <a href={href('x3', 'X3.8')}>X3.8</a> decodes: a 28-byte header, then float32 weights in llama2.c's order.
          It uses the Python standard library only, with its own generator and little-endian packing, so every machine writes
          the same bytes, and CMake runs it at configure time. Nothing is downloaded by CMake, the script, the driver or the
          tests. The prompt goes through the step one token at a time, as llama2.c processes it: M6's prefill attention is not
          composed, because its layout is not the cache layout M7 reads.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Của khóa học này: mô hình là <code>forward</code> của llama2.c (RMSNorm với eps 1e-5, grouped-query attention
              với RoPE cơ số 10,000, một feed-forward có cổng SiLU, bộ phân loại dùng chung trọng số với embedding) với kích
              thước head 128, kích thước mà các kernel decode của M7 được biên dịch cho. Trọng số của nó là các giá trị đều
              với thang chúng tôi chọn để các bước greedy có khoảng cách top-two lớn. Chúng không phải một mô hình đã huấn
              luyện.
            </p>
          }
        >
          <p>
            Ours: the model is llama2.c's <code>forward</code> (RMSNorm with eps 1e-5, grouped-query attention with RoPE base
            10,000, a SiLU-gated feed-forward, the classifier tied to the embedding) with head size 128, the size M7's decode
            kernels are compiled for. Its weights are uniform values with scales we chose so that greedy steps have large
            top-two margins. They are not a trained model.
          </p>
        </Translated>
      </Beyond>
      <LabFiles
        lab={l2}
        files={[
          ['kernel.cu', <Translated vi={<><code>advancePositionsKernel</code>, <code>launchAdvancePositions</code> và <code>launchLlamaStep</code></>}><><code>advancePositionsKernel</code>, <code>launchAdvancePositions</code> and <code>launchLlamaStep</code></></Translated>],
          ['kernel.metal', <><code>advance_positions</code></>],
          ['launch.mm', <Translated vi={<><code>encodeAdvancePositions</code> và <code>encodeLlamaStep</code></>}><><code>encodeAdvancePositions</code> and <code>encodeLlamaStep</code></></Translated>],
        ]}
      />

      <h3>
        <Translated vi="Các kiểm thử kiểm tra gì">What the tests check</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Hãy chạy <code>python3 labs/pmpp.py run X3.2 --backend emu</code> trước: nó chạy mọi luồng của mọi lần khởi chạy
            của mọi bước theo ba thứ tự, nên hãy chờ vài phút. Chương trình điều khiển in ra các kích thước: trên các trình giả
            lập là d 256 với 2 head truy vấn trên 1 head KV, hidden 352, 2 lớp, từ vựng 300; trên GPU là d 512 với 4 trên 2,
            hidden 1,376, 4 lớp, từ vựng 4,099 và 60 token được sinh. Mọi kích thước đều khác nhau, nên một kích thước truyền
            vào sai đối số sẽ thất bại.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run X3.2 --backend emu</code> first: it runs every thread of every launch of every
          step in three orders, so expect minutes. The driver prints the shapes: on the emulators d 256 with 2 query heads
          over 1 KV head, hidden 352, 2 layers, vocabulary 300; on a GPU d 512 with 4 over 2, hidden 1,376, 4 layers,
          vocabulary 4,099 and 60 generated tokens. Every size differs from every other, so a size passed in the wrong
          argument fails.
        </p>
      </Translated>
      <ul>
        <Translated
          vi={
            <li>
              <b>File mô hình.</b> Kích thước file bằng công thức của mục X3.8 cho header của nó, và một checksum của các trọng
              số đầu và cuối bằng với giá trị script đã in, nên một bộ sinh viết ra byte khác trên một máy khác sẽ thất bại ở
              đây.
            </li>
          }
        >
          <li>
            <b>Model files.</b> The file's size equals section X3.8's formula for its header, and a checksum of its first and
            last weights equals what the script printed, so a generator that wrote other bytes on another machine fails here.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Từng phần riêng lẻ.</b> Mỗi phần được cung cấp chạy một lần so với một bản tham chiếu CPU trước khi mã của bạn
              chạy, rồi tới phần advance của bạn chạy riêng.
            </li>
          }
        >
          <li>
            <b>Pieces alone.</b> Each provided piece once against a CPU reference before your code runs, then your advance
            alone.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Logits với teacher forcing.</b> Cả hai phía được đưa vào token của bản tham chiếu ở mọi bước, nên logits của
              mọi bước đều so sánh được. Chương trình điều khiển in NMSE của từng bước, thước đo của llama.cpp, và sai số lớn
              nhất của nó. Một logit qua nếu nằm trong 4e-5 × (1 + mag), trong đó mag là độ lớn các số hạng của tích vô hướng ở
              LM head: một dung sai cho mọi backend, dung sai của README.
            </li>
          }
        >
          <li>
            <b>Teacher-forced logits.</b> Both sides are fed the reference's token at every step, so every step's logits can be
            compared. The driver prints each step's NMSE, llama.cpp's metric, and its largest error. A logit passes within
            4e-5 × (1 + mag), where mag is the size of the terms of the LM head's dot product: one tolerance for every
            backend, the README's.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Greedy chạy tự do.</b> Các token phải bằng của bản tham chiếu ở mọi bước có khoảng cách top-two vượt quá tổng
              dung sai của hai logit. Một bước dưới mức đó khiến phần còn lại thành "inconclusive from step k", được in ra,
              không bao giờ âm thầm tính là qua. Mục <a href={href('x3', 'X3.9')}>X3.9</a> cho thấy vì sao một lần đổi ở đó làm
              thay đổi mọi token sau đó.
            </li>
          }
        >
          <li>
            <b>Free-running greedy.</b> The tokens must equal the reference's at every step whose top-two margin exceeds the
            two logits' tolerances added. A step below that makes the rest "inconclusive from step k", printed, never passed
            silently. Section <a href={href('x3', 'X3.9')}>X3.9</a> shows why a flip there changes every later token.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Bất biến theo batch.</b> Các token của mỗi chuỗi trong một batch bằng các token của nó khi được decode riêng.
            </li>
          }
        >
          <li>
            <b>Batch invariance.</b> Each sequence's tokens in a batch equal its tokens decoded alone.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Cách ghép cặp RoPE.</b> Mô hình được viết với <code>--permute-qk</code>, phép hoán vị hàng của llama.cpp, chạy
              ở chế độ 0, cho kết quả giống bản gốc ở chế độ 1.
            </li>
          }
        >
          <li>
            <b>RoPE pairing.</b> The model written with <code>--permute-qk</code>, llama.cpp's row permutation, run in mode 0,
            gives what the original gives in mode 1.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>M9 trong M10.</b> Khi có lời giải của lab X2.4, bước của bạn chạy lại, không đổi, với lớp lượng tử hóa của M9
              làm lớp tuyến tính: mọi phép chiếu và head ở Q8_0, rồi ở Q4_0. Các token của nó phải bằng các token của bản tham
              chiếu CPU chạy trên cùng các trọng số đã lượng tử hóa, chạy tự do, theo cùng quy tắc khoảng cách như trên (với
              seed được cung cấp, một chuỗi Q4_0 là inconclusive từ một bước trở đi), và trên GPU mọi logit cũng được kiểm tra
              với teacher forcing. Lần chạy cũng in ra nơi các token greedy của mô hình lượng tử hóa lần đầu rời khỏi các token
              của mô hình float32; đó không phải một kiểm thử.
            </li>
          }
        >
          <li>
            <b>M9 in M10.</b> When lab X2.4's solution is present, your step runs again, unchanged, with M9's quantised layer
            as its linear layer: every projection and the head in Q8_0, then in Q4_0. Its tokens must equal those of the CPU
            reference run on the same quantised weights, free-running, under the same margin rule as above (with the provided
            seed one Q4_0 sequence is inconclusive from one step on), and on a GPU every logit is checked teacher-forced too.
            The run also prints where the quantised model's greedy tokens first leave the float32 model's; that is not a test.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Bộ lấy mẫu.</b> Trên GPU, khi có lời giải của lab X1.3: M5 ở nhiệt độ 0 cho ra các token greedy, và hai lần
              chạy ở cùng một seed cố định và nhiệt độ 0.8 cho các token giống hệt nhau. Ở đây các token được lấy mẫu không được
              so sánh giữa các backend.
            </li>
          }
        >
          <li>
            <b>The sampler.</b> On a GPU, when lab X1.3's solution is present: M5 at temperature 0 gives the greedy tokens, and
            two runs at a fixed seed and temperature 0.8 give identical tokens. Sampled tokens are not compared across
            backends here.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Giá trị độc và dải canh.</b> Activation, logits và mọi hàng cache khởi đầu là NaN, nên một giá trị không ai
              ghi, hoặc một hàng cache mà attention không nên đọc, sẽ biến logits thành NaN; sau mọi đầu ra đều có dải canh.
            </li>
          }
        >
          <li>
            <b>Poison and guards.</b> Activations, logits and every cache row start as NaN, so a value nobody wrote, or a
            cache row the attention should not read, turns the logits to NaN; guard bands follow every output.
          </li>
        </Translated>
      </ul>
      <Today
        checked={tr(
          '2026-11-20 in the facts pass (notes/today/x3-facts.md D1, D7, D9)',
          '2026-11-20 trong lượt kiểm tra dữ kiện (notes/today/x3-facts.md D1, D7, D9)',
        )}
      >
        <Translated
          vi={
            <p>
              <code>test-backend-ops</code> của llama.cpp so sánh từng backend với CPU bằng NMSE, mặc định 1e-7 cho mỗi op.
              Bước greedy của llama2.c giữ giá trị lớn nhất đầu tiên, và M4 cũng vậy.
            </p>
          }
        >
          <p>
            llama.cpp's <code>test-backend-ops</code> compares each backend with the CPU by NMSE, 1e-7 per op by default.
            llama2.c's greedy step keeps the first maximum, and so does M4.
          </p>
        </Translated>
      </Today>

      <h3>
        <Translated vi="Trước khi đo hiệu năng">Before you benchmark</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Benchmark in số byte trọng số mỗi token và băng thông của lab 5.1 cạnh nhau; thương của chúng là số token mỗi giây
            tối đa mà riêng trọng số cho phép ở batch 1. Hãy quyết định một bước đạt bao nhiêu phần của trần đó, và, khi đã
            biết tỉ phần của attention trong bước ở vị trí 16, tỉ phần đó thành bao nhiêu ở vị trí 512. Cân nhắc bốn lực. Ở
            batch 1 mọi phép chiếu là một phép nhân ma trận với vector, và kiểu gather của lab 22.1 đọc mỗi trọng số một lần
            trong khi các ô (tile) của lab 6.4 được xây dựng cho ma trận (mục <a href={href('x3', 'X3.4')}>X3.4</a> cho thấy
            llama.cpp đổi kernel ở 8 token). Một bước gồm mười sáu lần khởi chạy mỗi lớp cộng năm, mỗi lần đều ngắn ở batch 1.
            Số trọng số đọc mỗi token là cố định, trong khi cache mà attention đọc lớn dần theo vị trí, khoảng 30 lần từ vị
            trí 16 đến 512. Và trọng số của một mô hình nhỏ có thể nằm một phần trong cache của thiết bị.
          </p>
        }
      >
        <p>
          The benchmark prints the weight bytes per token and lab 5.1's bandwidth side by side; their quotient is the most
          tokens per second the weights alone allow at batch 1. Decide what fraction of that ceiling one step reaches, and,
          once you know the attention's share of the step at position 16, what it becomes at position 512. Weigh four
          forces. At batch 1 every projection is a matrix-vector product, and lab 22.1's gather reads each weight once where
          lab 6.4's tiles were built for matrices (section <a href={href('x3', 'X3.4')}>X3.4</a> shows llama.cpp switching
          kernels at 8 tokens). A step is sixteen launches per layer plus five, each short at batch 1. The weights read per
          token are fixed, while the cache the attention reads grows with the position, about 30 times from position 16 to
          512. And a small model's weights may sit partly in the device's caches.
        </p>
      </Translated>
      <LabPrediction
        id="x3/labs/p-lab2-bwfraction"
        prompt={<Translated vi={<>Ở batch 1, vị trí 16, một bước decode sẽ đạt bao nhiêu phần trăm của trần băng thông trọng số, tính theo token mỗi giây?</>}><>At batch 1, position 16, what percentage of the weight-bandwidth ceiling will one decode step reach in tokens per second?</></Translated>}
        unit={tr('% of the ceiling', '% của trần')}
        measure={ceilingFraction}
        missing={missingBench('X3.2', NEED_51)}
        reading={
          <Translated
            vi={
              <>
                Trần này giả định mọi trọng số đến từ bộ nhớ một lần mỗi bước với tốc độ của lab 5.1 và không gì khác tốn thời
                gian. Các hàng step bên dưới cho biết con số đó là của kernel nhân ma trận nào và các kernel khác đạt bao nhiêu.
              </>
            }
          >
            <>
              The ceiling assumes every weight comes from memory once per step at lab 5.1's rate and nothing else costs time.
              The step rows below show which matrix-product kernel the figure is for and what the others reached.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="x3/labs/p-lab2-attn-share"
        prompt={<Translated vi={<>Ở batch 1, vị trí 512, attention (phần append và kernel attention) sẽ chiếm bao nhiêu phần trăm tổng thời gian theo từng phần?</>}><>At batch 1, position 512, what percentage of the summed per-piece time will attention (the append and the attention kernel) take?</></Translated>}
        unit={tr("% of the pieces' summed time", '% tổng thời gian của các phần')}
        measure={attentionShare}
        missing={missingBench('X3.2')}
        reading={
          <Translated
            vi={
              <>
                Tỉ phần này là trên tổng thời gian của bốn nhóm, mỗi phần được đo riêng trong lần khởi chạy hay command buffer
                của riêng nó, nên tổng có thể vượt quá một bước. Hãy so nó với tỉ phần ở vị trí 16 trong phần chi tiết: cache
                lớn lên khoảng 30 lần, còn trọng số thì không đổi.
              </>
            }
          >
            <>
              The share is of the four groups' summed times, each piece timed alone in its own launch or command buffer, so
              the sum can exceed a step. Compare it with the share at 16 in the detail: the cache grew about 30 times, the
              weights not at all.
            </>
          </Translated>
        }
      />
      <LabResults lab={l2} cols={STEP_COLS()} filter={(r) => r.name === 'step'} />
      <ModelLine />
      <Translated
        vi={<p>Mỗi phần được đo riêng, nhân với số lần một bước chạy nó, cộng lại thành bốn nhóm:</p>}
      >
        <p>Each piece timed alone, times how often a step runs it, summed into four groups:</p>
      </Translated>
      <PieceTable />
      <Translated
        vi={
          <p>
            Cách benchmark hoạt động: cùng script đó viết một mô hình lớn hơn lúc benchmark (d 2,048 với 16 head truy vấn trên
            4 head KV, hidden 5,632, 8 lớp, từ vựng 32,000: kích thước do chúng tôi chọn), giảm một nửa d, số head và độ rộng
            hidden cho tới khi trọng số chiếm tối đa một phần tư bộ nhớ mà thiết bị báo. Giá trị của nó lặp lại một khối ngẫu
            nhiên, nên file được viết trong vài giây và các token của nó không có ý nghĩa gì. Các bước chạy ở batch 1 và 4 và
            các vị trí 16 và 512, các hàng cache trước vị trí được điền như sau một prompt. Metal đo một command buffer mỗi
            bước, CUDA và HIP đo một cặp event; mỗi con số là tốt nhất trong vài lần chạy sau một lần không đo.
          </p>
        }
      >
        <p>
          How the benchmark works: the same script writes a larger model at bench time (d 2,048 with 16 query heads over 4
          KV heads, hidden 5,632, 8 layers, vocabulary 32,000: our shape), halving d, the heads and the hidden width until the
          weights take at most a quarter of the memory the device reports. Its values repeat one random block, so the file
          is written in seconds and its tokens mean nothing. Steps run at batch 1 and 4 and positions 16 and 512, the cache
          rows before the position filled as after a prompt. Metal times one command buffer per step, CUDA and HIP one pair
          of events; each figure is the best of several runs after an untimed one.
        </p>
      </Translated>

      <h3>
        <Translated vi="Một checkpoint thật do bạn tự tải về">A real checkpoint you download yourself</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Bộ nạp đọc định dạng của llama2.c, nên checkpoint thật đầu tiên nên thử là một trong các tinyllamas của Karpathy,
            và bước kiểm tra kích thước của nó tái tạo được kích thước của cả hai file từ header của chúng. Chúng sẽ không chạy
            được nguyên trạng: kích thước head của chúng là 8 và 48, còn các kernel decode của M7 được biên dịch cho 128, nên
            attention trước hết phải nhận kích thước head làm tham số. Một Llama của Hugging Face như SmolLM2-135M cần thêm
            bốn phép chuyển đổi của mục <a href={href('x3', 'X3.8')}>X3.8</a>. Không phần nào trong đó được chạy khi lab được
            viết: README mô tả công việc, unverified.
          </p>
        }
      >
        <p>
          The loader reads llama2.c's format, so the first real checkpoint to try is one of Karpathy's tinyllamas, and its
          size check reproduces both files' sizes from their headers. They will not run as they are: their head sizes are 8
          and 48, and M7's decode kernels are compiled for 128, so the attention must first take the head size as a
          parameter. A Hugging Face Llama such as SmolLM2-135M needs the four conversions of section{' '}
          <a href={href('x3', 'X3.8')}>X3.8</a> on top. None of this was run when the lab was written: the README describes
          the work, unverified.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 in the facts pass (notes/today/x3-facts.md D2, D4, E2)',
          '2026-11-20 trong lượt kiểm tra dữ kiện (notes/today/x3-facts.md D2, D4, E2)',
        )}
      >
        <Translated
          vi={
            <p>
              stories260K: kích thước head 8, 8 head truy vấn trên 4. stories15M: 48. SmolLM2-135M: 64, bfloat16, cách ghép
              cặp NeoX. Không cái nào có 128.
            </p>
          }
        >
          <p>
            stories260K: head size 8, 8 query heads over 4. stories15M: 48. SmolLM2-135M: 64, bfloat16, the NeoX pairing.
            None has 128.
          </p>
        </Translated>
      </Today>

      <h3>
        <Translated vi="Mốc dự án">The project milestone</Translated>
      </h3>
      <Translated
        vi={
          <p>
            <code>launchLlamaStep</code> hoàn thành mốc dự án M10: một decoder kiểu Llama chạy trọn vẹn trên chính các kernel
            của khóa học, toàn bộ vòng lặp sinh token nằm trên device, và mọi logit cùng mọi token quyết định đều được kiểm tra
            so với một bản tham chiếu CPU. Nó theo các quy ước của dự án: float32 theo thứ tự hàng với batch đứng trước, con
            trỏ device thô và kích thước tường minh, một hàm host <code>launch</code> cạnh mỗi kernel, và bản tham chiếu CPU
            trong <code>main.cpp</code>. M10 trong kế hoạch nêu một checkpoint thật cỡ nhỏ; lab chạy một checkpoint được sinh
            ra, vì không có checkpoint thật cỡ nhỏ nào có kích thước head 128 và bản build không tải gì về.
          </p>
        }
      >
        <p>
          <code>launchLlamaStep</code> completes milestone M10: a Llama-style decoder runs end to end on the course's own
          kernels, the whole generation loop stays on the device, and every logit and every decisive token is checked
          against a CPU reference. It follows the project's conventions: row-major float32 with the batch first, raw device
          pointers and explicit sizes, a <code>launch</code> host function beside each kernel, and the CPU reference in{' '}
          <code>main.cpp</code>. The plan's M10 names a small real checkpoint; the lab runs a generated one, because no small
          real checkpoint has head size 128 and the build downloads nothing.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            M10 cũng chạy với lớp lượng tử hóa của M9. Lớp gián tiếp <code>LinearFn</code> trong <code>llama.h</code> là nơi
            một lớp tuyến tính lượng tử hóa cắm vào, nhưng <code>launchQLinear</code> của lab X2.4 nhận một descriptor ma trận
            lượng tử hóa, không phải con trỏ float và kích thước của <code>LinearFn</code>. Một adapter được cung cấp,{' '}
            <code>provided/m9_linear</code>, nối hai bên mà không đổi giao diện của bạn. Chương trình điều khiển lượng tử hóa
            mọi phép chiếu và embedding dùng chung theo quy tắc của llama.cpp, như chương trình điều khiển của lab X2.4 làm,
            giữ các giá trị đã giải lượng tử hóa trong các buffer float của mô hình, và gắn mỗi buffer float với bản sao lượng
            tử hóa của nó. Adapter là một <code>LinearFn</code> tra cứu <code>W</code> nó được đưa và gọi{' '}
            <code>launchQLinear</code> với bản sao lượng tử hóa. Nó không bao giờ đọc <code>W</code> float.
          </p>
        }
      >
        <p>
          M10 also runs with M9's quantised layer. The <code>LinearFn</code> indirection in <code>llama.h</code> is where a
          quantised linear layer plugs in, but lab X2.4's <code>launchQLinear</code> takes a quantised matrix descriptor, not{' '}
          <code>LinearFn</code>'s float pointer and sizes. A provided adapter, <code>provided/m9_linear</code>, joins them
          without changing your interface. The driver quantises every projection and the tied embedding with llama.cpp's
          rules, as lab X2.4's driver does, keeps the dequantised values in the model's float buffers, and binds each float
          buffer to its quantised copy. The adapter is a <code>LinearFn</code> that looks up the <code>W</code> it is handed
          and calls <code>launchQLinear</code> with the quantised copy. It never reads the float <code>W</code>.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Adapter có cái giá của nó, và chúng tôi phải nói ra. Các bản sao float vẫn nằm trong bộ nhớ device cạnh các bản
            lượng tử hóa, nên lần chạy giữ nhiều byte hơn riêng mô hình float32, và embedding đọc hàng của nó từ các float đó
            trong khi bước của chính M9 giải lượng tử hóa hàng nó cần. Điều đó ổn cho một kiểm thử tính đúng; một bước được xây
            dựng để tiết kiệm bộ nhớ sẽ truyền chính descriptor, như <code>launchQDecodeStep</code> của M9 làm.
          </p>
        }
      >
        <p>
          The adapter has a price, and it is ours to name. The float copies stay in device memory beside the quantised ones,
          so the run holds more bytes than the float32 model alone, and the embedding reads its row from those floats where
          M9's own step dequantises the row it needs. That is fine for a test of correctness; a step built to save memory
          would pass the descriptor itself, as M9's <code>launchQDecodeStep</code> does.
        </p>
      </Translated>
      <ReferenceSolution lab={l2} />
    </LabsPage>
  );
}
