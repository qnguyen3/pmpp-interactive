import { useState, type ReactNode } from 'react';
import { Predict, type Question } from '../../components/Ask';
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
  rawCol,
  useLabs,
  type BenchSet,
  type Col,
  type Lab,
} from '../../components/LabKit';
import { fmt, labStatus, num, realBackends, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import { byteRatio, measuredBandwidth } from '../../sim/qtraffic';

// Module X2's labs page, built from components/LabKit.tsx. No graded question here: the labs
// are the exercise. Its nine predictions are stored, not graded: x2/labs/p-lab1-speedup,
// p-lab1-batch8, p-lab2-fused, p-lab2-unfused, p-lab3-emu, p-lab3-path, p-lab4-speedup,
// p-lab4-head, p-lab4-agree. pmpp.py knows these labs as X2.1 to X2.4.
export const questionsLabs: Record<string, Question> = {};

const FACTS = 'notes/today/x2-facts.md';

const LABS = defineLabs('x2', [
  { num: 'X2.1', dir: 'lab1_q8gemv', name: 'int8-weight GEMV', vi: { name: 'GEMV với trọng số int8' } },
  { num: 'X2.2', dir: 'lab2_q4gemv', name: 'GEMV on Q4_0 blocks', vi: { name: 'GEMV trên các khối Q4_0' } },
  { num: 'X2.3', dir: 'lab3_qhunt', name: 'integer-path bug hunt', vi: { name: 'săn lỗi trên đường số nguyên' } },
  { num: 'X2.4', dir: 'lab4_m9', name: 'the M9 decode step', vi: { name: 'bước decode M9' } },
]);
const [l1, l2, l3, l4] = LABS;

/** Identifier colours in the reference solutions: where each value lives. */
const tiers = (global: string[], shared: string[], reg: string[]): Record<string, Tier> => ({
  ...Object.fromEntries(global.map((k) => [k, 'global' as Tier])),
  ...Object.fromEntries(shared.map((k) => [k, 'shared' as Tier])),
  ...Object.fromEntries(reg.map((k) => [k, 'reg' as Tier])),
});
const TIERS_1 = tiers(['X', 'Wq', 'Wd', 'Y', 'q', 'd', 'x'], [], ['sum', 'four', 'scale', 'q0', 'q1', 'q2', 'q3', 'total', 'lane', 'j']);
const TIERS_2 = tiers(['X', 'W', 'Y', 'row', 'qs16', 'x'], [], ['sum', 'four', 'lo', 'hi', 'qx', 'sx', 'total', 'lane', 'j', 's']);
const TIERS_3 = tiers(['W', 'Xq', 'Y', 'X', 'w', 'q16', 'xb', 'u'], ['partial'], ['sum', 'sumi', 'v', 'vi0', 'vi1', 'dw', 't', 'lane', 'warp']);
const TIERS_4 = tiers(['E', 'q8', 'q8d', 'q4', 'tokens', 'H', 'blk'], [], ['i', 'r', 'j', 'k', 't', 'code']);

// ---------------------------------------------------------------- reading the reports
const ms = (s: number | undefined) => (s ? `${fmt(s * 1000)} ms` : '?'); // i18n-ignore: unit
const secs = (r: BenchRow | undefined) => num(r?.seconds);
/** The first row of a set with this name whose fields equal `where`, strings and numbers alike. */
const row = benchRow;
const qgemv = (s: BenchSet, kernel: string, batch: number, inW = 4096, outW = 4096) => row(s, 'qgemv', { kernel, batch, in: inW, out: outW });
const qstep = (s: BenchSet, layers: string, head: string, batch: number) => row(s, 'qstep', { format_layers: layers, format_head: head, batch });
/** Lab 5.1's bandwidth on a backend, as a percentage of which a row's weight GB/s is read. */
const ofBandwidth = (reports: Reports, backend: string, gbs: number | undefined) => {
  const bw = measuredBandwidth(reports, backend);
  return bw && gbs ? (100 * gbs) / bw : undefined;
};
const BYTES_Q8 = byteRatio('q8_0');
const BYTES_Q4 = byteRatio('q4_0');
const BYTES_Q4_Q8 = byteRatio('q4_0', 'q8_0');

/** Lab X2.1: Q8_0 over F32 at batch 1, 4096 to 4096, with lab 5.1's bandwidth alongside. */
const q8Speedup = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const f = qgemv(s, 'gather', 1);
    const q = qgemv(s, 'q8_0', 1);
    if (!secs(f) || !secs(q)) return null;
    const pct = ofBandwidth(reports, s.backend, num(q!.weight_gb_per_s));
    return {
      value: secs(f)! / secs(q)!,
      detail:
        tr(
          `F32 ${ms(secs(f))} against Q8_0 ${ms(secs(q))} per launch, 4096 to 4096; the byte ratio is ${fmt(BYTES_Q8)}`,
          `F32 ${ms(secs(f))} so với Q8_0 ${ms(secs(q))} mỗi lần khởi chạy, 4096 sang 4096; tỉ lệ byte là ${fmt(BYTES_Q8)}`,
        ) +
        (pct != null
          ? tr(`; Q8_0 read its weights at ${fmt(pct)} % of lab 5.1's bandwidth`, `; Q8_0 đọc trọng số ở ${fmt(pct)} % băng thông của lab 5.1`)
          : tr('; no lab 5.1 run to compare its bandwidth with', '; chưa có lần chạy lab 5.1 nào để so sánh băng thông')),
    };
  });

/** Lab X2.1: the same speed-up at batch 8, with batch 1's beside it. */
const q8Batch8 = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const at = (b: number) => {
      const f = secs(qgemv(s, 'gather', b));
      const q = secs(qgemv(s, 'q8_0', b));
      return f && q ? f / q : undefined;
    };
    const eight = at(8);
    if (eight == null) return null;
    const one = at(1);
    return {
      value: eight,
      detail:
        tr(`4096 to 4096 at batch 8`, `4096 sang 4096 ở batch 8`) +
        (one != null ? tr(`; at batch 1 it was ${fmt(one)}`, `; ở batch 1 là ${fmt(one)}`) : tr('; no batch-1 row', '; không có hàng batch 1')),
    };
  });

/** Lab X2.2: fused Q4_0 over F32 at batch 1, 4096 to 4096, and over Q8_0 where that row exists. */
const q4Fused = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const f = secs(qgemv(s, 'gather', 1));
    const q4 = secs(qgemv(s, 'q4_0', 1));
    if (!f || !q4) return null;
    const q8 = secs(qgemv(s, 'q8_0', 1));
    return {
      value: f / q4,
      detail:
        tr(
          `F32 ${ms(f)} against fused Q4_0 ${ms(q4)}, 4096 to 4096; the byte ratio is ${fmt(BYTES_Q4)}`,
          `F32 ${ms(f)} so với Q4_0 gộp ${ms(q4)}, 4096 sang 4096; tỉ lệ byte là ${fmt(BYTES_Q4)}`,
        ) +
        (q8
          ? tr(
              `; over lab X2.1's Q8_0 it is ${fmt(q8 / q4)}, against a byte ratio of ${fmt(BYTES_Q4_Q8)}`,
              `; so với Q8_0 của lab X2.1 là ${fmt(q8 / q4)}, trong khi tỉ lệ byte là ${fmt(BYTES_Q4_Q8)}`,
            )
          : tr('; no Q8_0 row', '; không có hàng Q8_0')),
    };
  });

/** Lab X2.2: the provided unfused path over F32 at batch 1, 4096 to 4096. */
const q4Unfused = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const f = secs(qgemv(s, 'gather', 1));
    const u = secs(qgemv(s, 'q4_0_unfused', 1));
    if (!f || !u) return null;
    const q4 = secs(qgemv(s, 'q4_0', 1));
    return {
      value: f / u,
      detail:
        tr(`F32 ${ms(f)} against dequantise-then-gather ${ms(u)}, 4096 to 4096`, `F32 ${ms(f)} so với giải lượng tử hóa rồi gather ${ms(u)}, 4096 sang 4096`) +
        (q4 ? tr(`; the fused kernel took ${ms(q4)}`, `; kernel gộp mất ${ms(q4)}`) : ''),
    };
  });

/** Lab X2.3: the integer path's time over the float path's, quantise pass included. */
const pathRatio = (reports: Reports) =>
  measure(reports, l3, (s) => {
    const at = (b: number) => {
      const i = row(s, 'qhunt', { path: 'int8_q8_1', batch: b, in: 4096, out: 4096 });
      const f = row(s, 'qhunt', { path: 'float_q4_0', batch: b, in: 4096, out: 4096 });
      const ti = num(i?.seconds_total);
      const tf = num(f?.seconds_total);
      return ti && tf ? { ti, tf, quant: num(i?.seconds_quantise) } : null;
    };
    const one = at(1);
    if (!one) return null;
    const eight = at(8);
    return {
      value: one.ti / one.tf,
      detail:
        tr(`integer path ${ms(one.ti)}`, `đường số nguyên ${ms(one.ti)}`) +
        (one.quant != null ? tr(` (of which the quantise pass ${ms(one.quant)})`, ` (trong đó lượt lượng tử hóa ${ms(one.quant)})`) : '') +
        tr(` against the float path ${ms(one.tf)}, 4096 to 4096, batch 1`, ` so với đường số thực ${ms(one.tf)}, 4096 sang 4096, batch 1`) +
        (eight
          ? tr(`; at batch 8 the ratio is ${fmt(eight.ti / eight.tf)}`, `; ở batch 8 tỉ lệ là ${fmt(eight.ti / eight.tf)}`)
          : tr('; no batch-8 rows', '; không có hàng batch 8')),
    };
  });

/** Lab X2.4: a format pair's step over the F32 step at batch 1. */
const stepSpeedup = (layers: string, head: string, model: string) => (reports: Reports) =>
  measure(reports, l4, (s) => {
    const f = secs(qstep(s, 'f32', 'f32', 1));
    const q = qstep(s, layers, head, 1);
    if (!f || !secs(q)) return null;
    const pct = ofBandwidth(reports, s.backend, num(q!.weight_gb_per_s));
    return {
      value: f / secs(q)!,
      detail:
        tr(
          `F32 step ${ms(f)} against ${ms(secs(q))}, d ${String(q!.d ?? '?')}, vocab ${num(q!.vocab)?.toLocaleString('en-US') ?? '?'}, batch 1; the bytes model says ${model}`,
          `bước F32 ${ms(f)} so với ${ms(secs(q))}, d ${String(q!.d ?? '?')}, vocab ${num(q!.vocab)?.toLocaleString('en-US') ?? '?'}, batch 1; mô hình byte cho ${model}`,
        ) +
        (pct != null ? tr(`; the step read its weights at ${fmt(pct)} % of lab 5.1's bandwidth`, `; bước này đọc trọng số ở ${fmt(pct)} % băng thông của lab 5.1`) : ''),
    };
  });

/** Lab X2.4: greedy steps in agreement with F32, at Q4_0, with Q8_0's alongside. */
const agreement = (reports: Reports) =>
  measure(reports, l4, (s) => {
    const q4 = qstep(s, 'q4_0', 'q4_0', 1);
    const a4 = num(q4?.agree_steps);
    if (a4 == null) return null;
    const q8 = qstep(s, 'q8_0', 'q8_0', 1);
    const a8 = num(q8?.agree_steps);
    return {
      value: a4,
      detail:
        tr(
          `of 32 steps at batch 1, Q4_0 everywhere; first-step logits off by ${fmt(100 * (num(q4!.logits_rel_rms) ?? NaN))} % relative RMS`,
          `trên 32 bước ở batch 1, Q4_0 ở mọi nơi; logits của bước đầu lệch ${fmt(100 * (num(q4!.logits_rel_rms) ?? NaN))} % RMS tương đối`,
        ) +
        (a8 != null
          ? tr(
              `; Q8_0 agreed for ${a8} steps, its logits off by ${fmt(100 * (num(q8!.logits_rel_rms) ?? NaN))} %`,
              `; Q8_0 khớp trong ${a8} bước, logits của nó lệch ${fmt(100 * (num(q8!.logits_rel_rms) ?? NaN))} %`,
            )
          : ''),
    };
  });

const missingBench = (labNum: string) => (
  <Translated
    vi={
      <>
        Phần này cần các hàng benchmark của lab {labNum} từ một thiết bị thật. Hãy chạy{' '}
        <code>python3 labs/pmpp.py run {labNum} --bench</code> khi các kernel của bạn đã qua (hoặc với <code>--solution</code>{' '}
        để đo thời gian bản tham chiếu). Trình giả lập không bao giờ được benchmark.
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
const KERNEL: Record<string, string> = {
  get gather() { return tr('F32 gather (lab 22.1)', 'gather F32 (lab 22.1)'); },
  get q8_0() { return tr('Q8_0 (lab X2.1)', 'Q8_0 (lab X2.1)'); },
  get q4_0_unfused() { return tr('Q4_0, dequantise then gather (provided)', 'Q4_0, giải lượng tử hóa rồi gather (có sẵn)'); },
  get q4_0() { return tr('Q4_0 fused (lab X2.2)', 'Q4_0 gộp (lab X2.2)'); },
};
const PATH: Record<string, string> = {
  get int8_q8_1() { return tr('quantise to Q8_1, then the integer GEMV', 'lượng tử hóa về Q8_1, rồi GEMV số nguyên'); },
  get float_q4_0() { return tr('fused float Q4_0 GEMV', 'GEMV Q4_0 số thực gộp'); },
};
const FMT: Record<string, string> = { f32: 'F32', q8_0: 'Q8_0', q4_0: 'Q4_0' };
const kernelCol: Col = { get head() { return tr('What ran', 'Đã chạy gì'); }, cell: (r) => KERNEL[String(r.kernel)] ?? String(r.kernel ?? '') };
const pathCol: Col = {
  get head() {
    return tr('What ran', 'Đã chạy gì');
  },
  cell: (r) => (r.path === 'float_q4_0' ? `${PATH.float_q4_0}, ${String(r.source ?? '')}` : PATH[String(r.path)] ?? String(r.path ?? '')),
};
const pairCol: Col = { get head() { return tr('Layers / head', 'Các lớp / head'); }, cell: (r) => `${FMT[String(r.format_layers)] ?? r.format_layers} / ${FMT[String(r.format_head)] ?? r.format_head}` };
const msOf = (head: string, key: string): Col => ({ head, num: true, cell: (r) => (num(r[key]) == null ? '' : fmt(num(r[key])! * 1000)) });
const mbCol: Col = { get head() { return tr('Weights, MB', 'Trọng số, MB'); }, num: true, cell: (r) => (num(r.weight_bytes) == null ? '' : fmt(num(r.weight_bytes)! / 1e6)) };
const pctCol = (head: string, key: string): Col => ({ head, num: true, cell: (r) => (num(r[key]) == null ? '' : fmt(num(r[key])! * 100)) });

/** "% of lab 5.1": a row's weight GB/s over lab 5.1's measured bandwidth on the row's backend. */
function useBandwidthCol(): Col {
  const { reports } = useLabs();
  return {
    head: tr('% of lab 5.1', '% của lab 5.1'),
    num: true,
    cell: (r) => {
      const p = ofBandwidth(reports, String(r.backend), num(r.weight_gb_per_s));
      return p == null ? '' : fmt(p);
    },
  };
}

function GemvResults({ lab }: { lab: Lab }) {
  const bw = useBandwidthCol();
  return (
    <LabResults
      lab={lab}
      cols={[kernelCol, rawCol('Batch', 'batch'), rawCol(tr('In', 'Vào'), 'in'), rawCol(tr('Out', 'Ra'), 'out'), msCol, numCol(tr('Weight GB/s', 'GB/s trọng số'), 'weight_gb_per_s'), bw, numCol(tr('vs F32', 'so với F32'), 'speedup_vs_f32')]}
    />
  );
}
function StepResults() {
  const bw = useBandwidthCol();
  return (
    <LabResults
      lab={l4}
      cols={[pairCol, rawCol('Batch', 'batch'), msCol, numCol(tr('Tokens/s', 'Token/s'), 'tokens_per_s'), mbCol, numCol(tr('Weight GB/s', 'GB/s trọng số'), 'weight_gb_per_s'), bw, numCol(tr('vs F32', 'so với F32'), 'speedup_vs_f32'), pctCol(tr('Logits error, %', 'Sai số logits, %'), 'logits_rel_rms'), rawCol(tr('Agree', 'Khớp'), 'agree_steps')]}
    />
  );
}

/** Lab 5.1's measured bandwidth on each device: the 100 % of the "% of lab 5.1" columns. */
function BandwidthLine() {
  const { reports, loading } = useLabs();
  const found = realBackends(reports).flatMap((b) => {
    const bw = measuredBandwidth(reports, b);
    return bw ? [{ b, bw }] : [];
  });
  return (
    <p data-measured={loading ? 'loading' : found.length ? found.map((f) => `${f.b}:${fmt(f.bw)}`).join(',') : 'none'}>
      {loading
        ? tr('Reading reports.', 'Đang đọc báo cáo.')
        : found.length
          ? tr(
              `On this machine, lab 5.1 measured ${found.map((f) => `${fmt(f.bw)} GB/s on ${f.b}`).join(' and ')}. That is the 100 % of the "% of lab 5.1" columns below.`,
              `Trên máy này, lab 5.1 đo được ${found.map((f) => `${fmt(f.bw)} GB/s trên ${f.b}`).join(' và ')}. Đó là mức 100 % của các cột "% của lab 5.1" bên dưới.`,
            )
          : tr(
              'No run of lab 5.1 on this machine yet, so the "% of lab 5.1" columns stay empty. Run python3 labs/pmpp.py run 5.1 --bench first.',
              'Chưa có lần chạy lab 5.1 nào trên máy này, nên các cột "% của lab 5.1" để trống. Hãy chạy python3 labs/pmpp.py run 5.1 --bench trước.',
            )}
    </p>
  );
}

// ---------------------------------------------------------------- the race backend, for lab X2.3
/** One line on lab X2.3's runs on the race backend, which the status table does not show. */
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
        tr('No run on the race backend on this machine yet.', 'Chưa có lần chạy nào trên backend race ở máy này.')
      ) : (
        <>
          {sol.s === 'none'
            ? tr(
                `On this machine the reference solution of lab ${lab.num} has no run on the race backend yet.`,
                `Trên máy này, lời giải tham chiếu của lab ${lab.num} chưa có lần chạy nào trên backend race.`,
              )
            : sol.s === 'pass'
              ? tr(
                  `On this machine the reference solution of lab ${lab.num} passed all ${sol.tests.length} tests on the race backend with ${sol.found ? `${sol.found} ThreadSanitizer report(s)` : 'no race reported'}: emulator-checked on this flavour.`,
                  `Trên máy này, lời giải tham chiếu của lab ${lab.num} qua cả ${sol.tests.length} bài kiểm tra trên backend race với ${sol.found ? `${sol.found} báo cáo ThreadSanitizer` : 'không có race nào được báo'}: emulator-checked trên biến thể này.`,
                )
              : tr(
                  `On this machine the reference solution of lab ${lab.num} failed on the race backend (${sol.passed} of ${sol.tests.length} tests): unverified there.`,
                  `Trên máy này, lời giải tham chiếu của lab ${lab.num} thất bại trên backend race (${sol.passed} trên ${sol.tests.length} bài kiểm tra): unverified ở đó.`,
                )}{' '}
          {mine.s === 'none'
            ? tr(
                'Your own kernels have no recorded run there, or are still the untouched starter.',
                'Các kernel của bạn chưa có lần chạy nào được ghi lại ở đó, hoặc vẫn là mã khởi đầu (starter) chưa sửa.',
              )
            : mine.s === 'pass'
              ? tr(
                  `Your own kernels passed all ${mine.tests.length} tests there with ${mine.found ? `${mine.found} ThreadSanitizer report(s): read them` : 'no race reported'}.`,
                  `Các kernel của bạn qua cả ${mine.tests.length} bài kiểm tra ở đó với ${mine.found ? `${mine.found} báo cáo ThreadSanitizer: hãy đọc chúng` : 'không có race nào được báo'}.`,
                )
              : tr(
                  `Your own kernels passed ${mine.passed} of ${mine.tests.length} tests there, with ${mine.found} ThreadSanitizer report(s) in the run.`,
                  `Các kernel của bạn qua ${mine.passed} trên ${mine.tests.length} bài kiểm tra ở đó, với ${mine.found} báo cáo ThreadSanitizer trong lần chạy.`,
                )}
        </>
      )}
    </p>
  );
}

/** Lab X2.3's first prediction: how many of the four defects emu reports. Not measured from the reports. */
function HuntReveal({ committed }: { committed: number }): ReactNode {
  // The count and the reason come at once; which defect showed where waits until the reader
  // says the hunt is over, so the reveal does not hand over the answers before the first run.
  const [fixed, setFixed] = useState(false);
  return (
    <div data-reveal="p-lab3-emu">
      <Translated
        vi={
        <p>
          Khi lab này được xây dựng, mỗi lỗi được cài riêng vào lời giải tham chiếu, và <code>emu</code> báo{' '}
          <b>4</b> trên bốn lỗi. {committed === 4 ? 'Đó là con số của bạn.' : `Bạn đã dự đoán ${committed}.`}
        </p>
        }
      >
      <p>
        When this lab was built, each defect was planted alone into the reference solution, and <code>emu</code> reported{' '}
        <b>4</b> of the four. {committed === 4 ? 'That is your number.' : `You predicted ${committed}.`}
      </p>
      </Translated>
      <Translated
        vi={
        <p>
          Hai trong số đó chỉ lộ ra với những đầu vào hay thứ tự luồng cụ thể, và đó là lý do các bài kiểm tra chứa những trường
          hợp như vậy. Một là lỗi trong số học số nguyên mà chỉ một khối được dựng sẵn chạm tới; bỏ trường hợp đó đi thì nó qua ở
          mọi nơi. Một lỗi khác là lỗi đồng bộ hóa: <code>emu</code> lấp đầy bộ nhớ chia sẻ (shared memory) bằng một mẫu byte, nên
          một lần đọc diễn ra quá sớm sẽ trả về một giá trị vô lý ở một số thứ tự luồng. Backend race chỉ ra hai dòng của lỗi đó,
          và thiết bị Metal dùng để xây dựng lab đã qua nó ở mọi trường hợp.
        </p>
        }
      >
      <p>
        Two of them show only on particular inputs or thread orders, which is why the tests hold the cases they do. One
        is a defect in integer arithmetic that only one constructed block reaches; take that case away and it passes
        everywhere. Another is a synchronisation defect: <code>emu</code> fills shared memory with a byte pattern, so a
        read that comes too early returns an absurd value in some of its thread orders. The race backend named the two
        lines of that one, and the Metal device the lab was built on passed it on every case.
      </p>
      </Translated>
      {fixed ? (
        <Translated
          vi={
          <p data-reveal-detail="p-lab3-emu">
            Từng lỗi một, mỗi lỗi riêng trên <code>emu</code>: lỗi 1 (các word activation 2i và 2i + 1) làm thất bại mọi trường
            hợp của đường số nguyên trừ khối xấu nhất, nơi mọi trọng số bằng nhau, nên cặp ghép nào cũng cho cùng tổng. Lỗi 2 (tổng
            khối kiểu <code>short</code>) chỉ làm thất bại khối xấu nhất, có tổng 60,960 vượt 32,767 của short. Lỗi 3 (8 nhân d ở
            chỗ lẽ ra là s của Q8_1) làm thất bại mọi trường hợp của đường số nguyên. Lỗi 4 (thiếu rào chắn) thất bại dưới thứ tự
            luồng tuần tự và thứ tự xáo trộn; race báo lần ghi{' '}
            <code>partial[warp][b]</code> và lần đọc <code>partial[lane][b]</code>; Metal qua cả 8 trường hợp.
          </p>
          }
        >
        <p data-reveal-detail="p-lab3-emu">
          Defect by defect, each alone on <code>emu</code>: defect 1 (activation words 2i and 2i + 1) failed every
          integer-path case except the worst-case block, whose weights are all equal, so any pairing gives the same sum.
          Defect 2 (the <code>short</code> block sum) failed only the worst-case block, whose sum of 60,960 is past a
          short's 32,767. Defect 3 (8 times d where Q8_1's s belongs) failed every integer-path case. Defect 4 (the missing
          barrier) failed under the in-order and shuffled thread orders; race reported the write of{' '}
          <code>partial[warp][b]</code> and the read of <code>partial[lane][b]</code>; Metal passed all 8 cases.
        </p>
        </Translated>
      ) : (
        <p>
          <button type="button" onClick={() => setFixed(true)}>
            <Translated vi="Tôi đã sửa cả bốn: cho xem mỗi backend thấy gì">I have fixed all four: show what each backend saw</Translated>
          </button>
        </p>
      )}
    </div>
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
              Bốn lab, mỗi lab trong thư mục riêng dưới <code>labs/x2/</code>. Lab thứ nhất lưu trọng số của một GEMV decode
              dưới dạng mã int8 với một hệ số tỉ lệ cho mỗi khối 32, rồi đo nó so với gather float32 của lab 22.1. Lab thứ hai
              đọc chính các khối Q4_0 18 byte của llama.cpp. Lab thứ ba săn bốn lỗi trên đường số nguyên của llama.cpp. Lab thứ
              tư là mốc dự án M9: bước decode của lab 22.2 với trọng số đã lượng tử hóa, đo tốc độ tăng và sai số của nó trên
              thiết bị của bạn.
            </>
          }
        >
          <>
            Four labs, each in its own directory under <code>labs/x2/</code>. The first stores a decode GEMV's weights as
            int8 codes with a scale per block of 32 and measures it against lab 22.1's float32 gather. The second reads
            llama.cpp's own 18-byte Q4_0 blocks. The third hunts four defects in llama.cpp's integer path. The fourth is
            milestone M9: lab 22.2's decode step with quantised weights, its speed-up and its error measured on your device.
          </>
        </Translated>
      </LabsIntro>
      <Beyond title={tr("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
        <Translated
          vi={
            <p>
              Sách không có lượng tử hóa số nguyên và không có kernel lượng tử hóa nào (§A.5 của phụ lục A chỉ có hai đoạn).
              Các định dạng là của llama.cpp; các hợp đồng, bố cục của lab X2.1, các bài kiểm tra và các lỗi là của khóa học
              này. Các thành phần là của sách: cường độ tính toán của chương 5, phép rút gọn trong warp của chương 10, gather
              của chương 22.
            </p>
          }
        >
          <p>
            The book has no integer quantisation and no quantised kernel (appendix A's §A.5 is two paragraphs). The formats
            are llama.cpp's; the contracts, layouts of lab X2.1, tests and defects are ours. The parts are the book's: chapter
            5's intensity, chapter 10's warp reduction, chapter 22's gather.
          </p>
        </Translated>
      </Beyond>
      <Source section="A.5" pages="592–593" />
      <Translated
        vi={
          <p>
            Module này không phải một chương của sách, nhưng <code>pmpp.py</code> biết các lab của nó theo số:{' '}
            <code>python3 labs/pmpp.py run X2.1</code> chạy lab X2.1, và cứ thế đến X2.4.
          </p>
        }
      >
        <p>
          This module is not a chapter of the book, but <code>pmpp.py</code> knows its labs by their numbers:{' '}
          <code>python3 labs/pmpp.py run X2.1</code> runs lab X2.1, and so on to X2.4.
        </p>
      </Translated>
      <RunningLabs
        example="X2.1"
        alsoShownIn={
          <Translated
            vi={
              <>
                các bảng số đo của <a href={href('x2', 'X2.1')}>X2.1</a> và <a href={href('x2', 'X2.7')}>X2.7</a>
              </>
            }
          >
          <>
            the measured panels of <a href={href('x2', 'X2.1')}>X2.1</a> and <a href={href('x2', 'X2.7')}>X2.7</a>
          </>
          </Translated>
        }
      />
      <BackendsAndEmulator />
      <Translated
        vi={
          <p>
            Lab X2.3 cũng được build cho biến thể race của trình giả lập, vì kernel của nó chuyển các tổng riêng phần của
            từng warp cho một warp qua bộ nhớ chia sẻ (shared memory). <code>python3 labs/pmpp.py run X2.3 --backend race</code>{' '}
            chạy các luồng của một khối cùng lúc dưới ThreadSanitizer, công cụ báo hai lần truy cập vào một vị trí mà không có
            gì sắp thứ tự giữa chúng, kèm cả hai dòng mã nguồn. Một lần chạy sạch là bằng chứng, không phải chứng minh: nó chỉ
            thấy những lịch chạy đã xảy ra. Ba lab còn lại không có cập nhật chung: mọi luồng hay warp đều ghi đầu ra của riêng
            nó.
          </p>
        }
      >
        <p>
          Lab X2.3 is also built for the emulator's race flavour, because its kernel hands per-warp partial sums to one warp
          through shared memory. <code>python3 labs/pmpp.py run X2.3 --backend race</code> runs a block's threads at once
          under ThreadSanitizer, which reports two accesses to one location with nothing ordering them, with both source
          lines. A clean run is evidence, not proof: it sees only the schedules that happened. The other three labs have no
          shared updates: every thread or warp writes its own outputs.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Có một lỗi không trình giả lập nào thấy được: một lần nạp 4 byte từ địa chỉ không phải bội số của 4. CPU của host
            cho phép điều đó và AddressSanitizer không kiểm tra căn chỉnh, nên nó qua <code>emu</code>, <code>race</code> và có
            thể cả GPU của bạn, và nó là hành vi không xác định trên mọi backend. Các lab X2.2 và X2.3 đọc các khối 18 byte của
            llama.cpp, vốn chỉ căn chỉnh 2 byte. Hãy kiểm tra các kernel đó bằng cách đọc: mọi lần nạp từ một khối đều rộng 1
            hoặc 2 byte.
          </p>
        }
      >
        <p>
          One defect no emulator can see: a 4-byte load from an address that is not a multiple of 4. The host CPU allows it
          and AddressSanitizer does not check alignment, so it passes <code>emu</code>, <code>race</code> and perhaps your
          GPU, and it is undefined on every backend. Labs X2.2 and X2.3 read llama.cpp's 18-byte blocks, which are only
          2-byte aligned. Check those kernels by reading: every load from a block is 1 or 2 bytes wide.
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${FACTS} A5, C5)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${FACTS}, mục A5, C5)`)}>
        <Translated
          vi={
            <p>
              CUDA: <code>char4</code> có kích thước 4 và căn chỉnh 4. llama.cpp đọc các mã Q4_0 bằng hai lần nạp 16 bit, "assume
              at least 2 byte alignment". Trên Metal, <code>half</code> căn chỉnh 2 byte.
            </p>
          }
        >
          <p>
            CUDA: <code>char4</code> has size 4 and alignment 4. llama.cpp reads Q4_0 codes with two 16-bit loads, "assume at
            least 2 byte alignment". On Metal, <code>half</code> is 2-byte aligned.
          </p>
        </Translated>
      </Today>
      <VerificationStatus />
      <Translated
        vi={
          <p>
            Các lab X2.3 và X2.4 gọi các GEMV của lab X2.1 và X2.2. Bản build của chúng chép lời giải tham chiếu (solution) của
            các lab đó vào thư mục build (<code>labs/x2/shared-g12/qpieces.cmake</code>,{' '}
            <code>labs/x2/lab4_m9/provided/pieces.cmake</code>), với mọi hàm trợ giúp một dòng được đặt <code>static</code>, vì
            file của lab nào cũng tên là <code>kernel.cu</code> và hai bản sao của một hàm trợ giúp trong cùng một chương trình
            giả lập sẽ xung đột. Nếu một lời giải bị thiếu hoặc có chữ ký khác, một bản thay thế từ{' '}
            <code>labs/x2/shared-g12/</code> sẽ thế chỗ, và dòng đầu tiên của lần chạy cho biết bản nào đã chạy. Các bộ lượng tử
            hóa tham chiếu trên CPU nằm trong <code>labs/x2/shared/qformat.h</code> (lab X2.1 và X2.2) và{' '}
            <code>labs/x2/shared-g12/qformats.h</code> (lab X2.3 và X2.4): hai bản sao các quy tắc <code>_ref</code> của
            llama.cpp, mỗi bản được kiểm tra trên các khối tính tay trước khi bất kỳ kernel nào chạy.
          </p>
        }
      >
        <p>
          Labs X2.3 and X2.4 call the GEMVs of labs X2.1 and X2.2. Their builds copy those labs' reference solutions into
          the build directory (<code>labs/x2/shared-g12/qpieces.cmake</code>, <code>labs/x2/lab4_m9/provided/pieces.cmake</code>),
          with any one-line helper made <code>static</code>, because every lab's file is called <code>kernel.cu</code> and
          two copies of one helper in one emulator program would clash. If a solution is missing or has another signature,
          a stand-in from <code>labs/x2/shared-g12/</code> takes its place, and the first line of the run says which one ran.
          The CPU reference quantisers live in <code>labs/x2/shared/qformat.h</code> (labs X2.1 and X2.2) and{' '}
          <code>labs/x2/shared-g12/qformats.h</code> (labs X2.3 and X2.4): two copies of llama.cpp's <code>_ref</code> rules,
          each checked on hand-derived blocks before any kernel runs.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Các bản sao, các bản thay thế và các khối fixture tính tay là của khóa học này.</p>}>
          <p>The copies, the stand-ins and the hand-derived fixture blocks are ours.</p>
        </Translated>
      </Beyond>
      <h2>
        <Translated vi="Mức sử dụng băng thông">Bandwidth utilisation</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Giới hạn tốc độ của một GEMV decode là băng thông của thiết bị, nên câu hỏi hữu ích về một thời gian đo được là nó
            đến gần giới hạn đó bao nhiêu. Mỗi hàng benchmark bên dưới báo số byte trọng số chia cho thời gian của nó, và các
            bảng chia con số đó cho những gì lab 5.1 đo được trên cùng thiết bị: PyTorch gọi đây là mức sử dụng băng thông của
            mô hình (model bandwidth utilisation). Nó chỉ đếm trọng số, nên bỏ qua activation, đầu ra và mọi lưu lượng nháp, và
            nó có thể ra cao hơn một chút so với mức mà phép sao chép đơn giản của lab 5.1 đạt được.
          </p>
        }
      >
        <p>
          A decode GEMV's speed limit is the device's bandwidth, so the useful question about a measured time is how close
          it came. Every benchmark row below reports its weight bytes over its time, and the tables divide that by what lab
          5.1 measured on the same device: PyTorch calls this model bandwidth utilisation. It counts only the weights, so it
          leaves out the activations, the outputs and any scratch traffic, and it can come out a little above what lab 5.1's
          simple copy reached.
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${FACTS} A1)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${FACTS}, mục A1)`)}>
        <Translated
          vi={
            <p>
              MBU của PyTorch: kích thước mô hình nhân số lần suy luận mỗi giây, chia cho băng thông đỉnh. Khóa học chia cho
              băng thông đo được của lab 5.1, không phải đỉnh trên datasheet.
            </p>
          }
        >
          <p>
            PyTorch's MBU: model size times inferences per second, divided by peak bandwidth. The course divides by lab
            5.1's measured bandwidth, not a datasheet peak.
          </p>
        </Translated>
      </Today>
      <BandwidthLine />

      {/* ------------------------------------------------------------ X2.1 */}
      <h2>
        <Translated vi="Lab X2.1: một GEMV trọng số int8 cho decode batch 1">Lab X2.1: an int8-weight GEMV for batch-1 decode</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hãy viết <code>launchGemvQ8_0(X, Wq, Wd, Y, batch, in, out)</code>: Y[b][j] là tổng theo k của X[b][k] nhân
            d[j][k / 32] nhân q[j][k], cho batch đến 8. Quy tắc là Q8_0 của llama.cpp: với mỗi khối 32 trọng số, d là độ lớn
            lớn nhất chia 127, mỗi mã là <code>roundf(w / d)</code>, và d được lưu dạng FP16. Bố cục là của khóa học này: các
            mã nằm trong một mảng <code>Wq [out × in]</code>, các hệ số tỉ lệ FP16 dưới dạng bit <code>uint16_t</code> trong
            mảng thứ hai <code>Wd [out × in/32]</code>, giống cách MLX để các hệ số tỉ lệ riêng. Tách như vậy, mọi hàng mã đều
            căn chỉnh 4 byte, và một lane có thể đọc bốn mã như một word 32 bit.
          </p>
        }
      >
        <p>
          Write <code>launchGemvQ8_0(X, Wq, Wd, Y, batch, in, out)</code>: Y[b][j] is the sum over k of X[b][k] times
          d[j][k / 32] times q[j][k], for batches up to 8. The rule is llama.cpp's Q8_0: per block of 32 weights, d is the
          largest magnitude over 127, each code is <code>roundf(w / d)</code>, and d is stored as FP16. The layout is ours:
          codes in one array <code>Wq [out × in]</code>, the FP16 scales as <code>uint16_t</code> bits in a second array{' '}
          <code>Wd [out × in/32]</code>, as MLX keeps its scales apart. Split that way, every row of codes is 4-byte aligned,
          and a lane can read four codes as one 32-bit word.
        </p>
      </Translated>
      <Source section="A.5, 5.1, 22.2" pages="593, 93–97, 532–536" />
      <Translated
        vi={
          <p>
            Cách ánh xạ là gather của lab 22.1: một warp cho mỗi đầu ra, các lane đi dọc hàng mỗi lần một word, một tổng trong
            thanh ghi cho mỗi hàng batch, bốn tích của mỗi word được nhân với d của khối chúng một lần, một phép cộng trong
            warp, lane 0 ghi. Số 32 duy nhất trong kernel là <code>QK8_0</code>, kích thước khối, và nó có tên: bước nhảy là{' '}
            <code>warpSize</code>. Lab X2.4 liên kết hàm này, nên chữ ký của nó cố định.
          </p>
        }
      >
        <p>
          The mapping is lab 22.1's gather: one warp per output, the lanes striding along the row a word at a time, one
          register sum per batch row, each word's four products scaled by their block's d once, a warp sum, lane 0 writes.
          The only 32 in the kernel is <code>QK8_0</code>, the block size, and it has a name: the stride is{' '}
          <code>warpSize</code>. Lab X2.4 links this function, so its signature is fixed.
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${FACTS} D1, D5)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${FACTS}, mục D1, D5)`)}>
        <Translated
          vi={
            <p>
              llama.cpp: <code>d = amax / ((1 &lt;&lt; 7) - 1)</code>, các mã bằng <code>roundf</code>, d lưu bằng{' '}
              <code>GGML_FP32_TO_FP16</code>. MLX lưu hệ số tỉ lệ và độ lệch (bias) tách khỏi các mã đã đóng gói.
            </p>
          }
        >
          <p>
            llama.cpp: <code>d = amax / ((1 &lt;&lt; 7) - 1)</code>, codes by <code>roundf</code>, d stored with{' '}
            <code>GGML_FP32_TO_FP16</code>. MLX stores scales and biases apart from its packed codes.
          </p>
        </Translated>
      </Today>
      <LabFiles
        lab={l1}
        files={[
          ['kernel.cu', <Translated vi={<><code>codeOf</code>, <code>gemvQ8_0Kernel</code> và <code>launchGemvQ8_0</code>; <code>warpSum</code> với lane mask có kiểu, <code>deviceWarpSize</code> và <code>halfBitsToFloat</code> được cung cấp sẵn</>}><><code>codeOf</code>, <code>gemvQ8_0Kernel</code> and <code>launchGemvQ8_0</code>; <code>warpSum</code> with its typed lane mask, <code>deviceWarpSize</code> and <code>halfBitsToFloat</code> are provided</></Translated>],
          ['kernel.metal', <Translated vi={<><code>gemv_q8_0</code>, với các mã dạng <code>device const char4*</code> và các hệ số tỉ lệ dạng <code>device const half*</code></>}><><code>gemv_q8_0</code>, with the codes as <code>device const char4*</code> and the scales as <code>device const half*</code></></Translated>],
          ['launch.mm', <Translated vi={<>dispatch trong <code>encodeGemvQ8_0</code>, vào encoder được truyền cho nó; phần gắn buffer được cung cấp sẵn</>}><>the dispatch in <code>encodeGemvQ8_0</code>, into the encoder it is given; the buffer binding is provided</></Translated>],
        ]}
      />
      <h3>
        <Translated vi="Các bài kiểm tra kiểm tra gì">What the tests check</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Trước hết là các phần của chính chương trình điều khiển (driver): phép chuyển FP16 của nó trên các giá trị tính
            tay và trên cả 65,536 mẫu bit, và bộ lượng tử hóa của nó trên một khối tính tay có các mã gồm cả các trường hợp
            hòa 2.5 và −2.5. Sau đó là các kích thước từ (1, 32, 1) đến (5, 4096, 3) trên trình giả lập, và đến (8, 4096,
            4096) và (4, 14336, 4096) trên GPU. Mỗi khối của một hàng được nhân với lũy thừa của hai riêng từ 2⁻¹² đến 2¹², nên
            một hệ số tỉ lệ đọc từ khối bên cạnh sai ít nhất 32 lần; mọi khối đều chứa mã 127 và −127; một hàng toàn số 0; các
            hệ số tỉ lệ nhỏ nhất là số subnormal FP16. Y bắt đầu là NaN với một dải bảo vệ phía sau, mỗi trường
            hợp khởi chạy hai lần, và kết quả được so với một tích vô hướng độ chính xác kép trên các trọng số đã giải lượng
            tử hóa trong phạm vi 1e-4 × (1 + Σ|x w|), không bao giờ so từng bit.
          </p>
        }
      >
        <p>
          The driver's own parts first: its FP16 conversion on hand-derived values and on all 65,536 patterns, and its
          quantiser on a hand-derived block whose codes include the ties 2.5 and −2.5. Then shapes from (1, 32, 1) to (5,
          4096, 3) on the emulator, and up to (8, 4096, 4096) and (4, 14336, 4096) on a GPU. Each block of a row is scaled
          by its own power of two from 2⁻¹² to 2¹², so a scale read from the neighbouring block is at least 32 times wrong; every
          block holds codes 127 and −127; one row is all zero; the smallest scales are FP16 subnormals. Y starts as NaN with
          a guard band after it, every case launches twice, and results are compared with a double-precision dot on the
          dequantised weights within 1e-4 × (1 + Σ|x w|), never bit for bit.
        </p>
      </Translated>
      <h3>
        <Translated vi="Trước khi benchmark">Before you benchmark</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Trần của mô hình byte là tỉ lệ byte được lưu: 4 byte mỗi trọng số cho F32 so với 1.0625 cho Q8_0, {fmt(BYTES_Q8)}.
            Mục <a href={href('x2', 'X2.1')}>X2.1</a> nói vì sao đó là một trần chứ không phải một lời hứa. Hãy quyết định bao
            nhiêu phần của nó còn lại trên thiết bị của bạn, rồi tỉ lệ đó tăng hay giảm ở batch 8. Hãy cân nhắc hai điều: số
            byte activation tăng theo batch và như nhau cho cả hai định dạng, còn công việc biến một mã thành số thực được làm
            một lần cho mỗi trọng số và dùng chung cho mọi hàng của batch.
          </p>
        }
      >
        <p>
          The bytes model's ceiling is the ratio of bytes stored: 4 bytes per weight for F32 against 1.0625 for Q8_0, {fmt(BYTES_Q8)}.
          Section <a href={href('x2', 'X2.1')}>X2.1</a> says why that is a ceiling and not a promise. Decide how much of it
          survives on your device, then whether the ratio grows or shrinks at batch 8. Weigh two things: the activation
          bytes grow with the batch and are the same for both formats, and the work of turning a code into a float is done
          once per weight and shared by all the rows of the batch.
        </p>
      </Translated>
      <LabPrediction
        id="x2/labs/p-lab1-speedup"
        prompt={
          <Translated vi={<>Ở batch 1 trên lớp 4096 sang 4096, kernel Q8_0 của bạn sẽ nhanh gấp bao nhiêu lần gather F32 của lab 22.1?</>}>
            <>At batch 1 on the 4096 to 4096 layer, how many times faster than lab 22.1's F32 gather will your Q8_0 kernel be?</>
          </Translated>
        }
        unit={tr("× the F32 gather's speed", '× tốc độ của gather F32')}
        measure={q8Speedup}
        missing={missingBench('X2.1')}
        reading={
          <Translated
            vi={
              <>
                Một số đo cao hơn tỉ lệ byte không có nghĩa mô hình byte sai về số byte. Hai kernel khác nhau ở nhiều thứ hơn
                trọng số: kernel này đọc X bốn số thực mỗi lần còn kernel của lab 22.1 đọc một. Hãy đọc tỉ lệ băng thông trước:
                nó cho biết còn lại bao nhiêu khoảng trống.
              </>
            }
          >
          <>
            A measurement above the byte ratio does not mean the bytes model is wrong about bytes. The two kernels differ in
            more than their weights: this one reads X four floats at a time and lab 22.1's reads one. Read the bandwidth
            share first: it says how much room was left.
          </>
          </Translated>
        }
      />
      <LabPrediction
        id="x2/labs/p-lab1-batch8"
        prompt={
          <Translated vi={<>Còn ở batch 8, cùng lớp đó: nhanh gấp bao nhiêu lần F32?</>}>
            <>And at batch 8, same layer: how many times faster than F32?</>
          </Translated>
        }
        unit={tr("× the F32 gather's speed", '× tốc độ của gather F32')}
        measure={q8Batch8}
        missing={missingBench('X2.1')}
        reading={
          <Translated
            vi={
              <>
                Cả hai kernel đọc trọng số một lần mỗi lần khởi chạy bất kể batch, nên một tỉ lệ giảm xuống cho biết có thứ
                khác ngoài trọng số đã bắt đầu tốn thời gian: activation, phép tính số học, hoặc cả hai. Bảng có batch 4 và
                hai kích thước MLP.
              </>
            }
          >
          <>
            Both kernels read their weights once per launch whatever the batch, so a falling ratio says something other
            than the weights started to cost time: the activations, the arithmetic, or both. The table has batch 4 and
            the two MLP shapes.
          </>
          </Translated>
        }
      />
      <GemvResults lab={l1} />
      <Translated
        vi={
          <p>
            Cách benchmark hoạt động: các lớp 4096 sang 4096, 4096 sang 14336 và 14336 sang 4096 (hai lớp sau là các kích thước
            MLP của khóa học này), batch 1, 4 và 8. Một lần nộp được đo thời gian khởi chạy kernel một lần cho mỗi bản sao
            trọng số của nó, các bản sao cộng lại chiếm ít nhất 256 MB cho mỗi định dạng, nên không lần khởi chạy nào thấy
            trọng số của nó ở chỗ lần trước để lại. Hai kernel thay phiên nhau: một vòng không đo, rồi năm vòng, và mỗi kernel
            giữ kết quả tốt nhất. Thời gian trên Metal là thời gian GPU của command buffer; CUDA và HIP đo bằng đồng hồ của
            host.
          </p>
        }
      >
        <p>
          How the benchmark works: layers 4096 to 4096, 4096 to 14336 and 14336 to 4096 (the last two are our MLP shapes),
          batches 1, 4 and 8. One timed submission launches the kernel once per copy of its weights, copies that together
          hold at least 256 MB per format, so no launch finds its weights where the previous one left them. The two kernels
          take turns: one untimed round, then five, and each keeps its best. Metal times are the command buffer's GPU time;
          CUDA and HIP time on the host clock.
        </p>
      </Translated>
      <ReferenceSolution lab={l1} tiers={TIERS_1} />

      {/* ------------------------------------------------------------ X2.2 */}
      <h2>
        <Translated vi="Lab X2.2: một GEMV 4 bit trên các khối Q4_0 của llama.cpp">Lab X2.2: a 4-bit GEMV on llama.cpp's Q4_0 blocks</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hãy viết <code>launchGemvQ4_0(X, W, Y, batch, in, out)</code> trên các khối của llama.cpp đúng như{' '}
            <code>ggml-common.h</code> định nghĩa: <code>struct BlockQ4_0 {'{'} uint16_t d; uint8_t qs[16]; {'}'}</code>, 18
            byte, byte j giữ mã j ở nibble thấp và mã j + 16 ở nibble cao, mỗi giá trị là (q − 8) × d. Đây là định dạng bạn gặp
            trong một file GGUF. W là <code>[out × in/32]</code> khối; X và Y là float32, batch đứng trước.
          </p>
        }
      >
        <p>
          Write <code>launchGemvQ4_0(X, W, Y, batch, in, out)</code> on llama.cpp's blocks exactly as{' '}
          <code>ggml-common.h</code> defines them: <code>struct BlockQ4_0 {'{'} uint16_t d; uint8_t qs[16]; {'}'}</code>, 18
          bytes, byte j holding code j in its low nibble and code j + 16 in its high nibble, each value (q − 8) × d. This is
          the format you meet in a GGUF file. W is <code>[out × in/32]</code> blocks; X and Y are float32, batch first.
        </p>
      </Translated>
      <Source section="A.5, 22.2, 10.7" pages="592–593, 532–536, 237–241" />
      <Translated
        vi={
          <p>
            Hãy đọc các mã dưới dạng byte hoặc word 16 bit, không bao giờ qua <code>int</code> hay <code>char4</code>: mục{' '}
            <a href={href('x2', 'X2.3')}>X2.3</a> chỉ ra những khối nào sẽ bị lệch căn chỉnh. Giải nén bằng mặt nạ bit. Việc đưa
            −8 ra khỏi vòng lặp trong qua tổng x của khối (mục <a href={href('x2', 'X2.5')}>X2.5</a>) và việc giao cho một khối
            nhiều lane (trình chạy từng bước của mục <a href={href('x2', 'X2.4')}>X2.4</a>) là lựa chọn của bạn. Đường không gộp,
            giải lượng tử hóa W vào một ma trận nháp float32 rồi chạy gather của lab 22.1, được cung cấp sẵn và đúng, chỉ dùng
            cho benchmark.
          </p>
        }
      >
        <p>
          Read the codes as bytes or 16-bit words, never through <code>int</code> or <code>char4</code>: section{' '}
          <a href={href('x2', 'X2.3')}>X2.3</a> shows which blocks would be misaligned. Unpack with masks. Folding the −8 out
          of the inner loop through the block's sum of x (section <a href={href('x2', 'X2.5')}>X2.5</a>) and giving a block
          several lanes (section <a href={href('x2', 'X2.4')}>X2.4</a>'s stepper) are your choices. The unfused path,
          dequantise W into a float32 scratch matrix and then run lab 22.1's gather, is provided and correct, for the
          benchmark only.
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${FACTS} D1, C2)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${FACTS}, mục D1, C2)`)}>
        <Translated
          vi={
            <p>
              <code>block_q4_0</code> của llama.cpp: <code>ggml_half d</code> và <code>uint8_t qs[QK4_0 / 2]</code>.{' '}
              <code>int4b_format</code> mới của MSL không có thứ tự nibble được tài liệu hóa, và cả llama.cpp lẫn MLX đều không
              dùng nó.
            </p>
          }
        >
          <p>
            llama.cpp's <code>block_q4_0</code>: <code>ggml_half d</code> and <code>uint8_t qs[QK4_0 / 2]</code>. MSL's new{' '}
            <code>int4b_format</code> has no documented nibble order, and neither llama.cpp nor MLX uses it.
          </p>
        </Translated>
      </Today>
      <LabFiles
        lab={l2}
        files={[
          ['kernel.cu', <Translated vi={<><code>gemvQ4_0Kernel</code> và <code>launchGemvQ4_0</code>; đường cơ sở không gộp nằm trong <code>provided/unfused.cu</code></>}><><code>gemvQ4_0Kernel</code> and <code>launchGemvQ4_0</code>; the unfused baseline is in <code>provided/unfused.cu</code></></Translated>],
          ['kernel.metal', <Translated vi={<><code>gemv_q4_0</code>, trên <code>struct block_q4_0 {'{'} half d; uchar qs[16]; {'}'}</code> đúng như llama.cpp khai báo trên Metal</>}><><code>gemv_q4_0</code>, on <code>struct block_q4_0 {'{'} half d; uchar qs[16]; {'}'}</code> as llama.cpp declares it on Metal</></Translated>],
          ['launch.mm', <><code>encodeGemvQ4_0</code></>],
        ]}
      />
      <h3>
        <Translated vi="Các bài kiểm tra kiểm tra gì">What the tests check</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Trước hết là bộ lượng tử hóa trên CPU, trên ba khối tính tay: một khối có cực trị âm (d = +1/8, bit{' '}
            <code>0x3000</code>), một khối có cực trị dương (d = −1/4, <code>0xB400</code>), và một khối toàn số 0, có d là −0
            (<code>0x8000</code>) với mọi byte là <code>0x88</code>. Sau đó là các kích thước của lab X2.1 và thêm một, (1, 32 ×
            33, 3): 33 khối mỗi hàng, nên các hàng bắt đầu ở cả khối chẵn lẫn khối lẻ. Mỗi khối có một đỉnh dương và một đỉnh
            âm, nên cả hai dấu của d đều xuất hiện và đỉnh còn lại bị cắt về mã 15; mỗi hàng có một khối với 16 trọng số đầu
            dương và 16 trọng số cuối âm, nên đọc một byte như hai phần tử kề nhau sẽ thất bại; một số khối toàn số 0. Đường
            không gộp được cung cấp cũng được kiểm tra, để benchmark so sánh các kernel đúng. Dung sai, khởi đầu NaN, dải bảo
            vệ và hai lần khởi chạy như lab X2.1.
          </p>
        }
      >
        <p>
          The CPU quantiser first, on three hand-derived blocks: one whose extreme is negative (d = +1/8, bits{' '}
          <code>0x3000</code>), one whose extreme is positive (d = −1/4, <code>0xB400</code>), and an all-zero block, whose d
          is −0 (<code>0x8000</code>) with every byte <code>0x88</code>. Then lab X2.1's shapes and one more, (1, 32 × 33, 3):
          33 blocks per row, so rows start at both even and odd blocks. Each block has a positive and a negative peak, so
          both signs of d occur and the other peak clips to code 15; one block per row has its first 16 weights positive
          and its last 16 negative, so reading a byte as two neighbours fails; some blocks are all zero. The provided
          unfused path is tested too, so the benchmark compares correct kernels. Tolerance, NaN start, guard band and two
          launches as lab X2.1.
        </p>
      </Translated>
      <h3>
        <Translated vi="Trước khi benchmark">Before you benchmark</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Ba tỉ lệ byte: Q4_0 gộp so với F32 là 4 / 0.5625 = {fmt(BYTES_Q4)}; so với Q8_0 là 1.0625 / 0.5625 ={' '}
            {fmt(BYTES_Q4_Q8)}. Đường không gộp đọc 0.5625 byte mỗi trọng số, ghi 4 và đọc lại 4, so với 4 của F32: mô hình
            byte nói nó nhanh chừng một nửa so với không lượng tử hóa gì cả (các pipeline của mục{' '}
            <a href={href('x2', 'X2.1')}>X2.1</a>). Ở chỗ kernel gộp không đạt tỉ lệ của nó, hãy hỏi nó trả thêm gì cho mỗi
            trọng số mà F32 không phải trả: việc giải nén, hệ số tỉ lệ, phép −8.
          </p>
        }
      >
        <p>
          Three byte ratios: fused Q4_0 against F32 is 4 / 0.5625 = {fmt(BYTES_Q4)}; against Q8_0, 1.0625 / 0.5625 ={' '}
          {fmt(BYTES_Q4_Q8)}. The unfused path reads 0.5625 bytes per weight, writes 4 and reads 4 again, against F32's 4:
          the bytes model says it is about half as fast as not quantising at all (section{' '}
          <a href={href('x2', 'X2.1')}>X2.1</a>'s pipelines). Where the fused kernel falls short of its ratio, ask what it
          pays per weight that F32 does not: the unpacking, the scale, the −8.
        </p>
      </Translated>
      <LabPrediction
        id="x2/labs/p-lab2-fused"
        prompt={
          <Translated vi={<>Ở batch 1 trên 4096 sang 4096, kernel Q4_0 gộp của bạn sẽ nhanh gấp bao nhiêu lần gather F32?</>}>
            <>At batch 1 on 4096 to 4096, how many times faster than the F32 gather will your fused Q4_0 kernel be?</>
          </Translated>
        }
        unit={tr("× the F32 gather's speed", '× tốc độ của gather F32')}
        measure={q4Fused}
        missing={missingBench('X2.2')}
        reading={
          <Translated
            vi={
              <>
                Phần chi tiết cũng so Q4_0 với Q8_0 của lab X2.1. Nếu Q4_0 giữ được ít phần tỉ lệ của nó hơn Q8_0, thì công
                việc thêm trên mỗi byte (hai mã mỗi byte, một phép −8, nhiều hệ số tỉ lệ hơn trên mỗi byte đọc) đang bắt đầu lộ
                ra.
              </>
            }
          >
          <>
            The detail compares Q4_0 with lab X2.1's Q8_0 too. If Q4_0 keeps less of its ratio than Q8_0 did, the extra
            work per byte (two codes per byte, a −8, more scales per byte read) is starting to show.
          </>
          </Translated>
        }
      />
      <LabPrediction
        id="x2/labs/p-lab2-unfused"
        prompt={
          <Translated vi={<>Còn đường không gộp được cung cấp, giải lượng tử hóa rồi gather: tốc độ của nó gấp bao nhiêu lần F32? (Dưới 1 nghĩa là chậm hơn.)</>}>
            <>And the provided unfused path, dequantise then gather: its speed as a multiple of F32's? (Below 1 means slower.)</>
          </Translated>
        }
        unit={tr("× the F32 gather's speed", '× tốc độ của gather F32')}
        measure={q4Unfused}
        missing={missingBench('X2.2')}
        reading={
          <Translated
            vi={
              <>
                Ma trận nháp của nó không được nhân bản như trọng số, nên bộ nhớ đệm (cache) có thể giúp nó; cột GB/s trọng số
                của bảng bỏ qua lưu lượng đó. Dù thế nào, đây cũng là trường hợp PyTorch đã đo: trọng số lượng tử hóa chỉ giúp
                decode khi bước giải lượng tử hóa được gộp vào phép nhân.
              </>
            }
          >
          <>
            Its scratch matrix is not copied like the weights, so a cache may help it; the table's weight GB/s leaves that
            traffic out. Either way it is the case PyTorch measured: quantised weights help decode only when the
            dequantise is fused into the product.
          </>
          </Translated>
        }
      />
      <GemvResults lab={l2} />
      <Translated
        vi={
          <p>
            Cách benchmark hoạt động: như lab X2.1, với bốn kernel thay phiên nhau: gather F32 của lab 22.1, lời giải Q8_0 của
            lab X2.1 khi có, đường không gộp được cung cấp và kernel của bạn. Phần "go further" của README xây đường prefill,
            một ô (tile) của W được giải lượng tử hóa vào bộ nhớ chia sẻ trước GEMM của lab 15.6, đó là cách{' '}
            <code>mul_mm</code> của llama.cpp xử lý nhiều token.
          </p>
        }
      >
        <p>
          How the benchmark works: as lab X2.1, with four kernels taking turns: lab 22.1's F32 gather, lab X2.1's Q8_0
          solution when it is present, the provided unfused path and yours. The README's "go further" builds the prefill
          path, a tile of W dequantised into shared memory before lab 15.6's GEMM, which is how llama.cpp's{' '}
          <code>mul_mm</code> handles many tokens.
        </p>
      </Translated>
      <ReferenceSolution lab={l2} tiers={TIERS_2} />

      {/* ------------------------------------------------------------ X2.3 */}
      <h2>
        <Translated vi="Lab X2.3: săn lỗi trên đường số nguyên">Lab X2.3: bug hunt on the integer path</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn được giao một GEMV Q4_0 × Q8_1 theo dạng CUDA của llama.cpp, bằng CUDA và bằng Metal. Activation trước hết được
            lượng tử hóa về Q8_1 (mã int8 cho mỗi 32 phần tử, một hệ số tỉ lệ d và s = d × Σq tính sẵn); sau đó mỗi khối là
            một tích vô hướng số nguyên của các nibble không dấu với các byte có dấu, được co giãn lại một lần thành d_w (sumi
            × d_x − 8 × s_x). Một khối 4 warp cho mỗi hàng đầu ra; các warp chia nhau các khối của hàng; mỗi warp rút gọn bằng
            shuffle và chuyển tổng riêng phần của nó cho warp 0 qua bộ nhớ chia sẻ. Nó biên dịch được, và nó có bốn lỗi độc lập.
            Hãy tìm và sửa cả bốn.
          </p>
        }
      >
        <p>
          You are handed a Q4_0 × Q8_1 GEMV in llama.cpp's CUDA shape, in CUDA and in Metal. The activations are first
          quantised to Q8_1 (int8 codes per 32, a scale d and the precomputed s = d × Σq); each block is then an integer dot
          product of unsigned nibbles against signed bytes, rescaled once as d_w (sumi × d_x − 8 × s_x). One block of 4 warps
          per output row; the warps split the row's blocks; each warp reduces with shuffles and hands its partial sums to
          warp 0 through shared memory. It compiles, and it has four independent defects. Find and fix all four.
        </p>
      </Translated>
      <Source section="10.7, 5.4" pages="237–241, 108–112" />
      <Today checked={tr(`2026-11-20 in the facts pass (${FACTS} D3, A2, B1, C1)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${FACTS}, mục D3, A2, B1, C1)`)}>
        <Translated
          vi={
            <p>
              <code>mmvq</code> của llama.cpp lượng tử hóa activation về Q8_1 cho mỗi token. <code>__dp4a</code> có trên CUDA
              từ sm_61; HIP không có <code>__dp4a</code>; MSL không có tích vô hướng số nguyên.
            </p>
          }
        >
          <p>
            llama.cpp's <code>mmvq</code> quantises the activations to Q8_1 for every token. <code>__dp4a</code> exists on CUDA
            from sm_61; HIP has no <code>__dp4a</code>; MSL has no integer dot product.
          </p>
        </Translated>
      </Today>
      <LabFiles
        lab={l3}
        files={[
          ['kernel.cu', <Translated vi={<><code>gemvQ4_0Q8_1Kernel</code>, với bốn lỗi; bộ lượng tử hóa, <code>dot4_i8</code> (chỉ gọi <code>__dp4a</code> trên CUDA sm_61 trở lên) và <code>launchGemvQ4_0Q8_1</code> đều đúng</>}><><code>gemvQ4_0Q8_1Kernel</code>, with the four defects; the quantiser, <code>dot4_i8</code> (which calls <code>__dp4a</code> only on CUDA sm_61 and later) and <code>launchGemvQ4_0Q8_1</code> are correct</></Translated>],
          ['kernel.metal', <Translated vi={<><code>gemv_q4_0_q8_1</code>, với cùng bốn lỗi đó</>}><><code>gemv_q4_0_q8_1</code>, with the same four</></Translated>],
          ['launch.mm', <Translated vi={<>được cung cấp sẵn và đúng; không cần sửa gì</>}><>provided and correct; nothing to change</></Translated>],
        ]}
      />
      <Translated
        vi={
          <p>
            README liệt kê những gì không phải lỗi: các lần nạp 16 bit của trọng số và các lần nạp 32 bit của các khối Q8_1 36
            byte, các nibble không dấu đưa vào phép nhân byte có dấu, lane mask có kiểu và <code>warpSize</code>. Phương pháp
            quan trọng hơn các bản sửa:
          </p>
        }
      >
        <p>
          The README lists what is not a defect: the 16-bit loads of the weights and the 32-bit loads of the 36-byte Q8_1
          blocks, the unsigned nibbles fed to a signed byte product, the typed lane mask and <code>warpSize</code>. The
          method matters more than the fixes:
        </p>
      </Translated>
      <ol>
        <li>
          <Translated
            vi={
              <>
                Trước khi chạy bất cứ thứ gì, hãy đọc <code>work/kernel.cu</code> và ghi ra những lỗi bạn thấy được, và với mỗi
                lỗi là backend bạn nghĩ sẽ báo nó: <code>emu</code>, <code>race</code>, GPU của bạn, hay không backend nào, và ở
                trường hợp nào.
              </>
            }
          >
          <>
          Before running anything, read <code>work/kernel.cu</code> and write down the defects you can see, and for each the
          backend you expect to report it: <code>emu</code>, <code>race</code>, your GPU, or none, and on which case.
          </>
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Chạy trình giả lập: <code>python3 labs/pmpp.py run X2.3 --backend emu</code>. Các thứ tự luồng của nó cố định,
                nên cái gì nó báo thì lần nào nó cũng báo. Hãy dùng điểm chung của các trường hợp thất bại: kích thước nào, loại
                dữ liệu nào, thứ tự luồng nào.
              </>
            }
          >
          <>
          Run the emulator: <code>python3 labs/pmpp.py run X2.3 --backend emu</code>. Its thread orders are fixed, so what it
          reports, it reports every time. Use what the failing cases have in common: which shape, which kind of data,
          which thread order.
          </>
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Sửa từng lỗi một rồi chạy lại. Sau đó chạy <code>python3 labs/pmpp.py run X2.3 --backend race</code>, lệnh này
                chỉ ra hai dòng va chạm nhau, dù có giá trị nào ra sai hay không.
              </>
            }
          >
          <>
          Fix one defect at a time and run again. Then <code>python3 labs/pmpp.py run X2.3 --backend race</code>, which names
          the two lines that collide, whether or not a value came out wrong.
          </>
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                Sửa y như vậy trong <code>kernel.metal</code> và chạy trên thiết bị: <code>python3 labs/pmpp.py run X2.3</code>.
                Ghi lại những lỗi nào nó cho bạn thấy và những lỗi nào nó che đi.
              </>
            }
          >
          <>
          Make the same fixes in <code>kernel.metal</code> and run on the device: <code>python3 labs/pmpp.py run X2.3</code>.
          Note which defects it showed you and which it hid.
          </>
          </Translated>
        </li>
      </ol>
      <Predict
        id="x2/labs/p-lab3-emu"
        unit={tr('defects', 'lỗi')}
        prompt={
          <Translated vi={<>Sau khi đọc kernel và trước khi chạy nó: trong bốn lỗi, mỗi lỗi được cài riêng, <code>emu</code> sẽ báo bao nhiêu lỗi?</>}>
            <>After reading the kernel and before running it: of the four defects, each planted alone, how many will <code>emu</code> report?</>
          </Translated>
        }
      >
        {(c) => <HuntReveal committed={c} />}
      </Predict>
      <h3>
        <Translated vi="Các bài kiểm tra kiểm tra gì">What the tests check</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Các kích thước (1, 32, 1), (1, 256, 3), (2, 512, 5) và (8, 1024, 9) với trọng số thật qua quy tắc Q4_0 của
            llama.cpp. Ba trường hợp được dựng sẵn: khối xấu nhất, mọi mã trọng số là 15 với các activation đều lượng tử hóa
            thành 127, nên tổng số nguyên của mọi khối là 15 × 127 × 32 = 60,960 (mục <a href={href('x2', 'X2.6')}>X2.6</a>);
            các khối có nibble thấp là 15 và nibble cao là 0, hoặc ngược lại, nên ghép một nibble với nhầm activation không thể
            qua; và các hệ số tỉ lệ khối là lũy thừa của hai. Bản tham chiếu là cùng đường số nguyên đó trên CPU, tổng khối số
            nguyên chính xác rồi phép co giãn số thực bằng double, trong phạm vi 1e-5 × (1 + độ lớn của các khối). Mỗi trường
            hợp cũng in ra, để tham khảo, đường số nguyên cách đường số thực bao xa: đó là cái giá của việc làm tròn activation
            về int8.
          </p>
        }
      >
        <p>
          Shapes (1, 32, 1), (1, 256, 3), (2, 512, 5) and (8, 1024, 9) with real weights through llama.cpp's Q4_0 rule.
          Three constructed cases: the worst-case block, every weight code 15 against activations that all quantise to 127,
          so every block's integer sum is 15 × 127 × 32 = 60,960 (section <a href={href('x2', 'X2.6')}>X2.6</a>); blocks whose
          low nibbles are 15 and high nibbles 0, or the other way round, so pairing a nibble with the wrong activation cannot
          pass; and power-of-two block scales. The reference is the same integer path on the CPU, exact integer block sums
          then the float rescale in double, within 1e-5 × (1 + the block magnitudes). Each case also prints, as information,
          how far the integer path is from the float path: that is the price of rounding the activations to int8.
        </p>
      </Translated>
      <RaceLine lab={l3} />
      <Translated
        vi={
          <p>
            Dù GPU của bạn cho thấy gì, hãy đọc nó đối chiếu với những gì các hãng tài liệu hóa: các warp tiến triển độc lập
            giữa các rào chắn (barrier), và đọc và ghi một vị trí mà không đồng bộ hóa là hành vi không xác định. Một trường hợp
            qua mười lần trên GPU của bạn đã cho bạn thấy mười lịch chạy của một thiết bị, không phải một kernel đúng.
          </p>
        }
      >
        <p>
          Whatever your GPU showed, read it against what the vendors document: warps make independent progress between
          barriers, and reading and writing one location without synchronisation is undefined. A case that passed ten times
          on your GPU showed you ten schedules of one device, not a correct kernel.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch04-facts.md A9; infra-facts.md C5; ch10-facts.md A1)', '2026-11-20 (notes/today/ch04-facts.md, mục A9; infra-facts.md, mục C5; ch10-facts.md, mục A1)')}>
        <Translated
          vi={
            <p>
              NVIDIA: các warp chạy độc lập. Apple: các SIMD-group "make independent forward progress ... in the absence of
              threadgroup barrier operations". CUDA: truy cập không đồng bộ vào một vị trí là hành vi không xác định.
            </p>
          }
        >
          <p>
            NVIDIA: warps run independently. Apple: SIMD-groups "make independent forward progress ... in the absence of
            threadgroup barrier operations". CUDA: unsynchronised access to one location is undefined behaviour.
          </p>
        </Translated>
      </Today>
      <h3>
        <Translated vi="Trước khi benchmark">Before you benchmark</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Khi đã sửa các lỗi, đường nào nhanh hơn cho một token decode, tính cả lượt lượng tử hóa: (a) lượng tử hóa activation
            về Q8_1, rồi GEMV số nguyên này; hay (b) GEMV số thực gộp của lab X2.2? Cả hai đọc cùng các byte trọng số Q4_0. Hãy
            cân nhắc mỗi backend cho đường số nguyên những gì: CUDA có lệnh tích vô hướng byte từ sm_61, bản build HIP dùng hàm
            trợ giúp bốn tích thông thường, còn Metal hoàn toàn không có tích vô hướng số nguyên, đó là lý do decode trên Metal
            của chính llama.cpp vẫn dùng số thực (mục <a href={href('x2', 'X2.5')}>X2.5</a>).
          </p>
        }
      >
        <p>
          With the defects fixed, which path is faster for one token of decode, the quantise pass included: (a) quantise the
          activations to Q8_1, then this integer GEMV; or (b) lab X2.2's fused float GEMV? Both read the same Q4_0 weight
          bytes. Weigh what each backend gives the integer path: CUDA has a byte dot instruction from sm_61, the HIP build
          takes the plain four-product helper, and Metal has no integer dot at all, which is why llama.cpp's own Metal decode
          stays in float (section <a href={href('x2', 'X2.5')}>X2.5</a>).
        </p>
      </Translated>
      <LabPrediction
        id="x2/labs/p-lab3-path"
        prompt={
          <Translated
            vi={
              <>
                Ở batch 1 trên 4096 sang 4096, tính cả lượt lượng tử hóa, thời gian của đường số nguyên chia cho thời gian của
                đường số thực là bao nhiêu? (Dưới 1 nghĩa là đường số nguyên thắng.)
              </>
            }
          >
          <>
            At batch 1 on 4096 to 4096, quantise pass included, what is the integer path's time divided by the float path's?
            (Below 1 means the integer path wins.)
          </>
          </Translated>
        }
        unit={tr("× the float path's time", '× thời gian của đường số thực')}
        measure={pathRatio}
        missing={missingBench('X2.3')}
        reading={
          <Translated
            vi={
              <>
                Cách ánh xạ của lab này giao cho mỗi luồng trọn các khối và nạp lại activation của một khối cho mọi hàng của
                batch, nên hãy đọc tỉ lệ ở batch 8 như một nhận định về kernel này, không phải về tích vô hướng số nguyên. Trên
                Metal, "tích vô hướng số nguyên" là bốn phép nhân byte bằng các lệnh thông thường.
              </>
            }
          >
          <>
            This lab's mapping gives each thread whole blocks and reloads a block's activations for every row of the batch,
            so read the batch-8 ratio as a statement about this kernel, not about integer dot products. On Metal the
            "integer dot" is four byte products in ordinary instructions.
          </>
          </Translated>
        }
      />
      <LabResults
        lab={l3}
        cols={[pathCol, rawCol('Batch', 'batch'), rawCol(tr('In', 'Vào'), 'in'), rawCol(tr('Out', 'Ra'), 'out'), msOf(tr('Quantise, ms', 'Lượng tử hóa, ms'), 'seconds_quantise'), msOf(tr('GEMV, ms', 'GEMV, ms'), 'seconds_gemv'), msOf(tr('Total, ms', 'Tổng, ms'), 'seconds_total')]}
      />
      <Translated
        vi={
          <p>
            Cách benchmark hoạt động: 4096 sang 4096 và 14336 sang 4096, batch 1 và 8, các khối Q4_0 ngẫu nhiên. Các lần chạy
            đo thời gian liên tiếp đọc các bản sao trọng số khác nhau, tổng cộng khoảng 512 MB hoặc một phần tư bộ nhớ của thiết
            bị nếu con số đó nhỏ hơn: đây là lựa chọn của khóa học này, vì Apple không tài liệu hóa kích thước bộ nhớ đệm nào.
            Mỗi con số là kết quả tốt nhất của ít nhất 10 lần chạy đo thời gian sau một lần không đo.
          </p>
        }
      >
        <p>
          How the benchmark works: 4096 to 4096 and 14336 to 4096, batches 1 and 8, random Q4_0 blocks. Successive timed runs
          read different copies of the weights, about 512 MB in all or a quarter of the device's memory if that is less: our
          choice, since Apple documents no cache sizes. Each figure is the best of at least 10 timed runs after an untimed
          one.
        </p>
      </Translated>
      <ReferenceSolution lab={l3} tiers={TIERS_3} />

      {/* ------------------------------------------------------------ X2.4 */}
      <h2>
        <Translated vi="Lab X2.4: mốc dự án M9, bước decode lượng tử hóa">Lab X2.4: milestone M9, the quantised decode step</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hãy bắt đầu từ bước decode của lab 22.2 (M8) và lưu trọng số của nó ở Q8_0 hoặc Q4_0. Viết ba hàm phía host.{' '}
            <code>launchEmbedQ(E, tokens, H, batch)</code> là một gather có giải lượng tử hóa: một luồng cho mỗi phần tử của H,
            đọc mã của nó và hệ số tỉ lệ của khối nó, khớp từng bit với hàng đã giải lượng tử hóa của CPU.{' '}
            <code>launchQLinear(X, W, Y, batch)</code> dispatch theo <code>W.fmt</code> sang <code>launchGemvGather</code> của
            lab 22.1, <code>launchGemvQ8_0</code> của lab X2.1 hoặc <code>launchGemvQ4_0</code> của lab X2.2, cho batch đến
            8. <code>launchQDecodeStep(m, s)</code> là bước của lab 22.2 với hai hàm đó thay cho phép nhúng (embedding) và các
            lớp tuyến tính của nó: nhúng, rồi với mỗi lớp là một lớp tuyến tính, bias và residual, rồi LM head với E, softmax và
            argmax, một lần khởi chạy cho mỗi phép toán cho cả batch và không có gì phải chờ.
          </p>
        }
      >
        <p>
          Start from lab 22.2's decode step (M8) and store its weights in Q8_0 or Q4_0. Write three host functions.{' '}
          <code>launchEmbedQ(E, tokens, H, batch)</code> is a dequantising gather: one thread per element of H, reading its
          code and its block's scale, bit for bit the CPU's dequantised row. <code>launchQLinear(X, W, Y, batch)</code>{' '}
          dispatches on <code>W.fmt</code> to lab 22.1's <code>launchGemvGather</code>, lab X2.1's{' '}
          <code>launchGemvQ8_0</code> or lab X2.2's <code>launchGemvQ4_0</code>, for batches up to 8.{' '}
          <code>launchQDecodeStep(m, s)</code> is lab 22.2's step with those two in place of its embedding and its linear
          layers: embed, then per layer a linear layer, bias and residual, then the LM head with E, softmax and argmax, one
          launch per operation for the whole batch and nothing waiting.
        </p>
      </Translated>
      <Source section="22.5, 20.6" pages="539–540, 504–506" />
      <Translated
        vi={
          <p>
            Trọng số vẫn ở dạng lượng tử hóa trong bộ nhớ thiết bị. Không chỗ nào trong bước này ghi ra một bản sao đã giải lượng
            tử hóa của ma trận trọng số: việc đó sẽ di chuyển nhiều byte hơn bước float32. Chương trình điều khiển lượng tử hóa
            mô hình đồ chơi có seed của lab 22.2 lúc nạp bằng các quy tắc <code>_ref</code> của llama.cpp, và chạy bốn cặp định
            dạng (các lớp, rồi embedding và LM head dùng chung trọng số): (F32, F32), (Q8_0, Q8_0), (Q4_0, Q4_0) và (Q4_0, F32).
          </p>
        }
      >
        <p>
          The weights stay quantised in device memory. Nothing in the step writes a dequantised copy of a weight matrix:
          that would move more bytes than the float32 step does. The driver quantises lab 22.2's seeded toy model at load
          with llama.cpp's <code>_ref</code> rules, and runs four pairs of formats (layers, then the tied embedding and LM
          head): (F32, F32), (Q8_0, Q8_0), (Q4_0, Q4_0) and (Q4_0, F32).
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 (${FACTS} A1; notes/today/ch15-facts.md D1)`, `2026-11-20 (${FACTS}, mục A1; notes/today/ch15-facts.md, mục D1)`)}>
        <Translated
          vi={
            <p>
              PyTorch đo được một đường int8 không gộp chậm hơn FP16. llama.cpp dùng các kernel ma trận-vector cho đến 8 token.
            </p>
          }
        >
          <p>
            PyTorch measured an unfused int8 path slower than FP16. llama.cpp uses matrix-vector kernels up to 8 tokens.
          </p>
        </Translated>
      </Today>
      <LabFiles
        lab={l4}
        files={[
          ['kernel.cu', <Translated vi={<>hai kernel embedding, <code>launchEmbedQ</code>, <code>launchQLinear</code> và <code>launchQDecodeStep</code>; <code>qdecode.h</code> khai báo <code>QMatrix</code> và <code>QDecodeModel</code></>}><>the two embedding kernels, <code>launchEmbedQ</code>, <code>launchQLinear</code> and <code>launchQDecodeStep</code>; <code>qdecode.h</code> declares <code>QMatrix</code> and <code>QDecodeModel</code></></Translated>],
          ['kernel.metal', <Translated vi={<><code>embed_rows_q8_0</code> và <code>embed_rows_q4_0</code>; <code>embed_rows_f32</code> được cung cấp sẵn</>}><><code>embed_rows_q8_0</code> and <code>embed_rows_q4_0</code>; <code>embed_rows_f32</code> is provided</></Translated>],
          ['launch.mm', <Translated vi={<><code>encodeEmbedQ</code>, <code>encodeQLinear</code> và <code>encodeQDecodeStep</code>, mọi dispatch nằm trong encoder của bên gọi: một command buffer cho mỗi bước</>}><><code>encodeEmbedQ</code>, <code>encodeQLinear</code> and <code>encodeQDecodeStep</code>, every dispatch in the caller's encoder: one command buffer per step</></Translated>],
        ]}
      />
      <h3>
        <Translated vi="Các bài kiểm tra kiểm tra gì">What the tests check</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Một mô hình đồ chơi trên trình giả lập: d 64, từ vựng 96, 2 lớp, batch 1, 3 và 8, mọi cặp định dạng. Trước hết là
            riêng phần embedding, cho E ở mỗi định dạng, trên một batch chứa token đầu tiên, token cuối cùng và một token hai
            lần: mọi hàng của H khớp từng bit với của CPU. Sau mỗi bước, H và logits được so với một bước CPU bằng double trên
            các trọng số đã giải lượng tử hóa, trong dung sai của lab 22.2; token kế tiếp phải bằng token của bản tham chiếu ở
            mọi chỗ mà hai logits tốt nhất của bản tham chiếu cách nhau xa hơn tổng dung sai của chúng. Sau đó là tính bất biến
            theo batch: mỗi chuỗi trong batch lớn nhất decode y hệt như khi chạy một mình. Trọng số không đổi sau lần chạy, mọi
            đầu ra bắt đầu là NaN và có dải bảo vệ. Trên GPU còn có kích thước mặc định của lab 22.2, d 4096, 2 lớp, từ vựng
            32,000. Lần chạy trên trình giả lập mất vài phút.
          </p>
        }
      >
        <p>
          A toy model on the emulator: d 64, vocabulary 96, 2 layers, batches 1, 3 and 8, every format pair. The embedding
          alone first, for E in each format, on a batch holding the first token, the last and one token twice: every row of
          H bit for bit the CPU's. After every step, H and the logits against a CPU step in double on the dequantised
          weights, within lab 22.2's tolerance; the next token must equal the reference's wherever the reference's two best
          logits are further apart than their tolerances together. Then batch invariance: each sequence of the largest batch
          decodes exactly as it does alone. The weights are unchanged after the run, every output starts as NaN and has a
          guard band. On a GPU also lab 22.2's default shape, d 4096, 2 layers, vocabulary 32,000. The emulator run takes
          minutes.
        </p>
      </Translated>
      <h3>
        <Translated vi="Trước khi benchmark">Before you benchmark</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Trình khám phá của mục <a href={href('x2', 'X2.7')}>X2.7</a> tính tỉ lệ byte của cả bước. Với mọi ma trận ở Q4_0,
            nó gần với {fmt(BYTES_Q4)} của GEMV; với head 32,000 × 4,096 để ở F32, nó là 1.21, vì head chiếm phần lớn số byte
            của bước. Về sai số, mô hình đồ chơi là ngẫu nhiên và không có attention: mức khớp của nó với F32 không nói gì về
            mức khớp của một mô hình đã huấn luyện.
          </p>
        }
      >
        <p>
          Section <a href={href('x2', 'X2.7')}>X2.7</a>'s explorer computes the byte ratio of the whole step. With every
          matrix in Q4_0 it is close to the GEMV's {fmt(BYTES_Q4)}; with the 32,000 × 4,096 head left in F32 it is 1.21,
          because the head is most of the step's bytes. For the error, the toy model is random and has no attention: its
          agreement with F32 says nothing about a trained model's.
        </p>
      </Translated>
      <LabPrediction
        id="x2/labs/p-lab4-speedup"
        prompt={
          <Translated vi={<>Ở batch 1, bước (Q4_0, Q4_0) sẽ nhanh gấp bao nhiêu lần bước (F32, F32)?</>}>
            <>At batch 1, how many times faster than the (F32, F32) step will the (Q4_0, Q4_0) step be?</>
          </Translated>
        }
        unit={tr("× the F32 step's speed", '× tốc độ của bước F32')}
        measure={stepSpeedup('q4_0', 'q4_0', tr(`about ${fmt(BYTES_Q4)}`, `khoảng ${fmt(BYTES_Q4)}`))}
        missing={missingBench('X2.4')}
        reading={
          <Translated
            vi={
              <>
                Một bước gồm nhiều lần khởi chạy, và những lần nhỏ (bias, residual, softmax, argmax) không co lại theo trọng
                số. Bao nhiêu phần tốc độ tăng của GEMV từ lab X2.2 còn lại cho biết chúng chiếm bao nhiêu phần của bước.
              </>
            }
          >
          <>
            A step is several launches, and the small ones (bias, residual, softmax, argmax) do not shrink with the weights.
            How much of the GEMV's speed-up from lab X2.2 survives says how much of the step they are.
          </>
          </Translated>
        }
      />
      <LabPrediction
        id="x2/labs/p-lab4-head"
        prompt={
          <Translated vi={<>Còn bước (Q4_0, F32), chỉ các lớp được lượng tử hóa: nhanh gấp bao nhiêu lần F32?</>}>
            <>And the (Q4_0, F32) step, with only the layers quantised: how many times faster than F32?</>
          </Translated>
        }
        unit={tr("× the F32 step's speed", '× tốc độ của bước F32')}
        measure={stepSpeedup('q4_0', 'f32', '1.21')}
        missing={missingBench('X2.4')}
        reading={
          <Translated vi={<>Nếu head chiếm phần lớn số byte của bước, nó cũng chiếm phần lớn thời gian, trừ khi có thứ khác giới hạn bước.</>}>
            <>If the head dominates the step's bytes, it dominates its time too, unless something else limits the step.</>
          </Translated>
        }
      />
      <LabPrediction
        id="x2/labs/p-lab4-agree"
        prompt={
          <Translated vi={<>Trong 32 bước greedy từ các token khởi đầu cố định, bước Q4_0 sẽ chọn cùng token với F32 ở bao nhiêu bước?</>}>
            <>Of 32 greedy steps from fixed start tokens, for how many will the Q4_0 step choose the same tokens as F32?</>
          </Translated>
        }
        unit={tr('steps', 'bước')}
        measure={agreement}
        missing={missingBench('X2.4')}
        reading={
          <Translated
            vi={
              <>
                Số đếm dừng ở bước đầu tiên mà có chuỗi nào chọn token khác. Hãy đọc nó cạnh sai số logits: logits của một mô
                hình ngẫu nhiên có thể cách nhau xa, nên sai số lớn vẫn có thể để nguyên mọi argmax. Trình khám phá mô hình đồ
                chơi của mục <a href={href('x2', 'X2.7')}>X2.7</a> cho thấy trường hợp ngược lại.
              </>
            }
          >
          <>
            The count stops at the first step where any sequence chose another token. Read it beside the logits error: a
            random model's logits can be far apart, so a large error may still leave every argmax in place. Section{' '}
            <a href={href('x2', 'X2.7')}>X2.7</a>'s toy explorer shows the opposite case.
          </>
          </Translated>
        }
      />
      <StepResults />
      <Translated
        vi={
          <p>
            Cách benchmark hoạt động: kích thước và seed mặc định của lab 22.2, batch 1, 2, 4 và 8, cả ba định dạng của mọi ma
            trận cùng nằm trên thiết bị. Một bước là một command buffer trên Metal và một cặp event trên CUDA và HIP, lấy kết
            quả tốt nhất của vài lần chạy đo thời gian sau một lần không đo. Số byte trọng số đếm những gì một bước phải đọc: E
            một lần cho head, W của mọi lớp và các bias. Sai số logits là RMS tương đối của bước đầu tiên so với F32; "Khớp" là
            số bước greedy, tính từ bước đầu tiên, mà trong đó mọi chuỗi đều chọn token của F32.
          </p>
        }
      >
        <p>
          How the benchmark works: lab 22.2's default shape and seeds, batches 1, 2, 4 and 8, all three formats of every
          matrix on the device at once. One step is one command buffer on Metal and one pair of events on CUDA and HIP, best
          of several timed runs after an untimed one. Weight bytes count what one step must read: E once for the head, every
          layer's W and the biases. The logits error is the first step's relative RMS against F32; "Agree" is the number of
          greedy steps, from the first, in which every sequence chose F32's token.
        </p>
      </Translated>

      <h3>
        <Translated vi="Mốc dự án">The project milestone</Translated>
      </h3>
      <Translated
        vi={
          <p>
            <code>launchQDecodeStep</code> hoàn thành mốc dự án M9: bước decode với trọng số int8 và 4 bit, tốc độ tăng đo
            trên thiết bị của bạn và sai số đầu ra được báo so với float32, theo logits và theo token greedy. Nó theo các quy
            ước của dự án để các mốc sau có thể gọi nó: activation float32 theo thứ tự hàng (row-major) với batch đứng trước,
            con trỏ device thô và kích thước tường minh, một hàm host <code>launch</code> bên cạnh mỗi kernel, bản tham chiếu CPU
            trong <code>main.cpp</code>, và các GEMV của lab X2.1 và X2.2 với chữ ký cố định. Mốc M10 của module X3 gọi{' '}
            <code>launchQLinear</code> của lời giải tham chiếu qua một bộ chuyển đổi nhỏ,{' '}
            <code>labs/x3/lab2_tinyllama/provided/m9_linear</code>, gắn mỗi con trỏ trọng số float của mô hình Llama với một{' '}
            <code>QMatrix</code>: khi đó cùng bước Llama đó chạy với trọng số Q8_0 và Q4_0, và các token của nó được kiểm tra
            so với một bản tham chiếu CPU trên cùng các trọng số đã lượng tử hóa.
          </p>
        }
      >
        <p>
          <code>launchQDecodeStep</code> completes milestone M9: the decode step with int8 and 4-bit weights, its speed-up
          measured on your device and its output error reported against float32, by logits and by greedy tokens. It follows
          the project's conventions so later milestones can call it: row-major float32 activations with the batch first, raw
          device pointers and explicit sizes, a <code>launch</code> host function beside each kernel, the CPU reference in{' '}
          <code>main.cpp</code>, and the GEMVs of labs X2.1 and X2.2 with their fixed signatures. Module X3's milestone M10
          calls the reference solution's <code>launchQLinear</code> through a small adapter,{' '}
          <code>labs/x3/lab2_tinyllama/provided/m9_linear</code>, which binds each float weight pointer of its Llama model to a{' '}
          <code>QMatrix</code>: the same Llama step then runs with Q8_0 and Q4_0 weights, and its tokens are checked against a
          CPU reference on the same quantised weights.
        </p>
      </Translated>
      <ReferenceSolution lab={l4} tiers={TIERS_4} />
    </LabsPage>
  );
}
