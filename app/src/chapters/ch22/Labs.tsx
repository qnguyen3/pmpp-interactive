import { useState } from 'react';
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
  defineLabs,
  measure,
  msCol,
  numCol,
  preferredBench,
  rawCol,
  useLabs,
  whose,
  type BenchSet,
} from '../../components/LabKit';
import { fmt, num, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import { M8_GEMMS, kneeOf, measuredDevices, thresholdOf, type ReportsLike } from '../../sim/batching';

// The page is built from components/LabKit.tsx; what is here is chapter 22's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch22', [
  { num: '22.1', dir: 'lab1_gemv', name: 'gather and split-K', vi: { name: 'gather và split-K' } },
  { num: '22.2', dir: 'lab2_batching', name: 'batching study (M8)', vi: { name: 'nghiên cứu batching (M8)' } },
]);
const [l1, l2] = LABS;

/** Identifier colours in the reference solutions: where each value lives. */
const TIERS_GEMV: Record<string, Tier> = {
  X: 'global',
  W: 'global',
  Y: 'global',
  w: 'global',
  sum: 'reg',
  total: 'reg',
  wk: 'reg',
  lane: 'reg',
  j: 'reg',
  g: 'reg',
  s: 'reg',
  k: 'reg',
  k0: 'reg',
  k1: 'reg',
};
const TIERS_M8: Record<string, Tier> = { E: 'global', tokens: 'global', H: 'global', i: 'reg', r: 'reg', j: 'reg' };

// ---------------------------------------------------------------- shared readers
const secs = (r: BenchRow | undefined) => num(r?.seconds) ?? 0;
const us = (r: BenchRow | undefined) => (secs(r) ? `${fmt(secs(r) * 1e6)} µs` : '?');
const msOf = (r: BenchRow | undefined) => (secs(r) ? `${fmt(secs(r) * 1e3)} ms` : '?'); // i18n-ignore: unit
const n = (v: number) => v.toLocaleString('en-US');
const asLike = (reports: Reports) => reports as unknown as ReportsLike;

// ---------------------------------------------------------------- lab 22.1: what the predictions are measured against
interface Layer {
  in: number;
  out: number;
}
const NARROW: Layer = { in: 4096, out: 16 };
const SQUARE: Layer = { in: 4096, out: 4096 };
const DOWN: Layer = { in: 14336, out: 4096 };
const LAYERS = [SQUARE, NARROW, DOWN];

const gemvRows = (s: BenchSet, l: Layer, batch: number) =>
  s.rows.filter((r) => r.name === 'gemv' && num(r.in) === l.in && num(r.out) === l.out && num(r.batch) === batch);

/** Gather's time over the best split-K time for one layer and batch: above 1, split-K was faster. */
function splitGain(s: BenchSet, l: Layer, batch: number) {
  const rows = gemvRows(s, l, batch);
  const gather = rows.find((r) => r.kernel === 'gather' && secs(r));
  const splits = rows.filter((r) => r.kernel === 'splitk' && secs(r)).sort((a, b) => secs(a) - secs(b));
  if (!gather || !splits.length) return null;
  const best = splits[0];
  return { gather, best, count: splits.length, ratio: secs(gather) / secs(best) };
}

const gainAt = (l: Layer) => (reports: Reports) =>
  measure(reports, l1, (s) => {
    const g = splitGain(s, l, 1);
    if (!g) return null;
    return {
      value: g.ratio,
      detail: tr(
        `gather ${us(g.gather)} against split-K at S = ${String(g.best.splits)}, ${us(g.best)}, the fastest of the ${g.count} split counts timed`,
        `gather ${us(g.gather)} so với split-K ở S = ${String(g.best.splits)}, ${us(g.best)}, nhanh nhất trong ${g.count} số lần chia đã đo`,
      ),
    };
  });

const MISSING_221 = (
  <Translated
    vi={
      <>
        Phần này cần các dòng benchmark của lab 22.1. Hãy chạy <code>python3 labs/pmpp.py run 22.1 --bench</code> trên một máy có
        GPU khi các kernel của bạn đã qua (hoặc thêm <code>--solution</code> để đo lời giải tham chiếu).
      </>
    }
  >
    <>
      This needs lab 22.1's benchmark rows. Run <code>python3 labs/pmpp.py run 22.1 --bench</code> on a machine with a GPU
      once your kernels pass (or with <code>--solution</code> to time the reference).
    </>
  </Translated>
);

/** Lab 22.1's best weight bandwidth per layer at batch 1, beside lab 5.1's measured bandwidth. */
function SpeedOfLight() {
  const { reports, loading } = useLabs();
  const devices = measuredDevices(asLike(reports));
  const sets = preferredBench(reports, l1);
  if (loading) return null;
  if (!sets.length)
    return (
      <p className="muted" data-sol="none">
        <Translated vi="Chưa có dòng benchmark nào của lab 22.1 trên máy này, nên chưa có gì để đặt cạnh băng thông của lab 5.1.">
          No benchmark rows for lab 22.1 on this machine yet, so there is nothing to set beside lab 5.1's bandwidth.
        </Translated>
      </p>
    );
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
      <table className="data" data-table="speed-of-light">
        <thead>
          <tr>
            <th>Backend</th>{/* i18n-ignore: kept in English */}
            <th><Translated vi="Lớp">Layer</Translated></th>
            <th><Translated vi="Nhanh nhất ở batch 1">Fastest at batch 1</Translated></th>
            <th className="num">W GB/s</th>{/* i18n-ignore: unit */}
            <th className="num"><Translated vi="Lab 5.1, GB/s">Lab 5.1, GB/s</Translated></th>
            <th className="num"><Translated vi="Tỷ lệ">Share</Translated></th>
          </tr>
        </thead>
        <tbody>
          {sets.flatMap((s) => {
            const dev = devices.find((d) => d.backend === s.backend);
            return LAYERS.flatMap((l) => {
              const rows = gemvRows(s, l, 1).filter((r) => num(r.weight_gb_per_s));
              if (!rows.length) return [];
              const best = rows.reduce((a, b) => (secs(b) < secs(a) ? b : a));
              const gbs = num(best.weight_gb_per_s)!;
              const label = best.kernel === 'gather' ? 'gather' : `split-K, S = ${String(best.splits)}`;
              return [
                <tr key={s.backend + l.in + '-' + l.out} data-layer={`${l.in}-${l.out}`} data-backend={s.backend} data-gbs={fmt(gbs)}>
                  <td>
                    {s.backend} <span className="muted">({s.device}, {s.variant === 'work' ? tr('yours', 'của bạn') : tr('reference', 'tham chiếu')})</span>
                  </td>
                  <td>
                    {n(l.in)} → {n(l.out)}
                  </td>
                  <td>{label}</td>
                  <td className="num">{fmt(gbs)}</td>
                  <td className="num">{dev ? fmt(dev.bw) : <span className="muted"><Translated vi="chưa chạy lab 5.1">no lab 5.1 run</Translated></span>}</td>
                  <td className="num">{dev ? `${fmt((100 * gbs) / dev.bw)}%` : ''}</td>
                </tr>,
              ];
            });
          })}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------- lab 22.2: what the predictions are measured against
const stepRows = (s: BenchSet, gemm: string) =>
  s.rows
    .filter((r) => r.name === 'step' && r.gemm === gemm && num(r.batch) && secs(r))
    .sort((a, b) => num(a.batch)! - num(b.batch)!);
const pointsOf = (rows: BenchRow[]) => rows.map((r) => ({ batch: num(r.batch)!, seconds: secs(r) }));
const tps = (r: BenchRow) => num(r.batch)! / secs(r);

const kneeM3 = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const rows = stepRows(s, 'm3');
    const knee = kneeOf(pointsOf(rows));
    if (knee == null) return null;
    const at = rows.find((r) => num(r.batch) === knee);
    const before = rows.filter((r) => num(r.batch)! < knee).at(-1);
    return {
      value: knee,
      detail: tr(
        `${msOf(rows[0])} per step at batch ${String(rows[0].batch)}, ${msOf(before)} at batch ${String(before?.batch)}, ${msOf(at)} at batch ${knee}`,
        `${msOf(rows[0])} mỗi bước ở batch ${String(rows[0].batch)}, ${msOf(before)} ở batch ${String(before?.batch)}, ${msOf(at)} ở batch ${knee}`,
      ),
    };
  });

const gain64 = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const rows = stepRows(s, 'm3');
    const b1 = rows.find((r) => num(r.batch) === 1);
    const b64 = rows.find((r) => num(r.batch) === 64);
    if (!b1 || !b64) return null;
    return {
      value: tps(b64) / tps(b1),
      detail: tr(
        `${fmt(tps(b1))} tokens per second at batch 1 (${msOf(b1)} per step), ${fmt(tps(b64))} at batch 64 (${msOf(b64)} per step)`,
        `${fmt(tps(b1))} token mỗi giây ở batch 1 (${msOf(b1)} mỗi bước), ${fmt(tps(b64))} ở batch 64 (${msOf(b64)} mỗi bước)`,
      ),
    };
  });

/** What to say when there is no M3 knee to compare with: no rows, or rows without a knee. */
function KneePrediction() {
  const { reports } = useLabs();
  const ranges = preferredBench(reports, l2).flatMap((s) => {
    const rows = stepRows(s, 'm3');
    return rows.length && kneeOf(pointsOf(rows)) == null ? [tr(`${s.backend} up to batch ${String(rows.at(-1)!.batch)}`, `${s.backend} tới batch ${String(rows.at(-1)!.batch)}`)] : [];
  });
  return (
    <LabPrediction
      id="ch22/labs/p-knee-m3"
      prompt={
        <Translated
          vi={
            <>
              Với M3 làm GEMM của bước, ở batch nào (1, 2, 4, 8, ...) thời gian mỗi bước lần đầu đạt 1.5 lần thời gian ở batch 1? Batch
              đó là điểm gãy: dưới nó, thêm chuỗi gần như đi kèm miễn phí.
            </>
          }
        >
          <>
            With M3 as the step's GEMM, at which batch (1, 2, 4, 8, ...) will the time per step first reach 1.5 times its
            time at batch 1? That batch is the knee: below it, more sequences ride along almost for free.
          </>
        </Translated>
      }
      unit={tr('(the measured knee, as a batch size)', '(điểm gãy đo được, tính theo kích thước batch)')}
      measure={kneeM3}
      missing={
        ranges.length ? (
          <Translated
            vi={
              <>
                Không có điểm gãy trong khoảng đã đo ({ranges.join('; ')}): thời gian mỗi bước chưa bao giờ đạt 1.5 lần giá trị ở batch
                1. Đó là một kết quả, không phải thiếu kết quả; benchmark dừng khi một bước mất hơn 2 s.
              </>
            }
          >
            <>
              No knee in the measured range ({ranges.join('; ')}): the time per step never reached 1.5 times its batch-1
              value. That is a result, not a missing one; the benchmark stops when a step takes more than 2 s.
            </>
          </Translated>
        ) : (
          <Translated
            vi={
              <>
                Phần này cần các dòng <code>step</code> của lab 22.2 cho M3, chỉ có khi lời giải tham chiếu của lab 15.6 nằm trong kho
                mã lúc configure. Hãy chạy <code>python3 labs/pmpp.py run 22.2 --bench</code> trên một máy có GPU (hoặc thêm{' '}
                <code>--solution</code>).
              </>
            }
          >
            <>
              This needs lab 22.2's <code>step</code> rows for M3, which exist only when lab 15.6's reference solution is in
              the repository at configure time. Run <code>python3 labs/pmpp.py run 22.2 --bench</code> on a machine with a
              GPU (or with <code>--solution</code>).
            </>
          </Translated>
        )
      }
      reading={
        <Translated
          vi={
            <>
              Mô hình ở <a href={href('ch22', '22.5')}>mục 22.5</a> đặt điểm gãy của một phép chiếu float32 ở gần gấp đôi ngưỡng của
              bạn. M3 tính 128 hàng của batch cho mỗi khối bất kể batch là bao nhiêu, nên chỉ riêng lý do đó đã khiến thời gian GEMM
              của nó phẳng tới 128 (ch22-facts E3): trong các giá trị 1, 2, 4, ..., điểm gãy của nó chỉ có thể đến trước 256 nếu riêng
              các phần việc theo từng hàng đã cộng thêm nửa bước. Hãy so với điểm gãy của M2 trong bảng bên dưới, có ô 16 hàng. Định
              nghĩa điểm gãy là của chúng tôi.
            </>
          }
        >
          <>
            The model of <a href={href('ch22', '22.5')}>section 22.5</a> puts a float32 projection's knee near twice your
            threshold. M3 computes 128 rows of the batch per block whatever the batch, so its GEMM time is flat up to 128 for
            that reason alone (ch22-facts E3): among 1, 2, 4, ..., its knee can come before 256 only if the per-row pieces
            alone add half a step. Compare with M2's knee in the table below, whose tiles are 16 rows. The definition of the
            knee is ours.
          </>
        </Translated>
      }
    />
  );
}

/** The reader's lab 5.1 threshold, the input to the M8 predictions. */
function Threshold() {
  const { reports, loading } = useLabs();
  const devices = measuredDevices(asLike(reports));
  if (loading) return null;
  return devices.length ? (
    <>
      {devices.map((d) => (
        <Translated
          vi={
            <p key={d.backend} data-threshold={fmt(thresholdOf(d))} data-backend={d.backend}>
              Ngưỡng của bạn từ lab 5.1 trên {d.name} ({d.backend}), lần chạy ngày {d.date}: <b>{fmt(thresholdOf(d))} FLOP/B</b> (
              {fmt(d.peak)} GFLOP/s chia cho {fmt(d.bw)} GB/s). Trình khám phá ở <a href={href('ch22', '22.5')}>mục 22.5</a> biến nó
              thành một batch: hãy chọn thiết bị của bạn ở đó.
            </p>
          }
        >
          <p key={d.backend} data-threshold={fmt(thresholdOf(d))} data-backend={d.backend}>
            Your threshold from lab 5.1 on {d.name} ({d.backend}), run of {d.date}: <b>{fmt(thresholdOf(d))} FLOP/B</b> (
            {fmt(d.peak)} GFLOP/s over {fmt(d.bw)} GB/s). <a href={href('ch22', '22.5')}>Section 22.5</a>'s explorer turns it
            into a batch: select your device there.
          </p>
        </Translated>
      ))}
    </>
  ) : (
    <Translated
      vi={
        <p className="muted" data-threshold="none">
          Chưa chạy lab 5.1 trên máy này, nên chưa có ngưỡng của bạn để bắt đầu. Hãy chạy{' '}
          <code>python3 labs/pmpp.py run 5.1 --bench</code> trước; trên Metal đó là ngưỡng duy nhất bạn có, vì Apple không công bố
          số liệu đỉnh nào (infra-facts C13).
        </p>
      }
    >
      <p className="muted" data-threshold="none">
        No lab 5.1 run on this machine, so there is no threshold of yours to start from. Run{' '}
        <code>python3 labs/pmpp.py run 5.1 --bench</code> first; on Metal it is the only threshold you have, since Apple
        publishes no peak figures (infra-facts C13).
      </p>
    </Translated>
  );
}

/** Per backend and GEMM: the knee, batch 1 against the best batch, and the batched step against one-by-one steps. */
function BatchingSummary() {
  const { reports, loading } = useLabs();
  if (loading) return null;
  const rows = preferredBench(reports, l2).flatMap((s) =>
    Object.keys(M8_GEMMS).flatMap((gemm) => {
      const steps = stepRows(s, gemm);
      if (!steps.length) return [];
      const knee = kneeOf(pointsOf(steps));
      const best = steps.reduce((a, b) => (tps(b) > tps(a) ? b : a));
      // The book's comparison (p. 539): one batched step against that many batch-1 steps.
      const ones = s.rows.filter((r) => r.name === 'one_by_one' && r.gemm === gemm && secs(r));
      const common = ones
        .map((o) => ({ o, st: steps.find((x) => num(x.batch) === num(o.batch)) }))
        .filter((x) => x.st && num(x.o.batch)! > 1)
        .sort((a, b) => num(b.o.batch)! - num(a.o.batch)!)[0];
      return [{ s, gemm, steps, knee, best, common }];
    }),
  );
  if (!rows.length)
    return (
      <Translated
        vi={
          <p className="muted" data-batching="none">
            Chưa có dòng <code>step</code> nào của lab 22.2 trên máy này. Hãy chạy{' '}
            <code>python3 labs/pmpp.py run 22.2 --bench</code> trên một máy có GPU.
          </p>
        }
      >
        <p className="muted" data-batching="none">
          No <code>step</code> rows for lab 22.2 on this machine yet. Run <code>python3 labs/pmpp.py run 22.2 --bench</code>{' '}
          on a machine with a GPU.
        </p>
      </Translated>
    );
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
      <table className="data" data-table="batching-summary">
        <thead>
          <tr>
            <th>Backend</th>{/* i18n-ignore: kept in English */}
            <th>GEMM</th>{/* i18n-ignore: kept in English */}
            <th className="num"><Translated vi="Điểm gãy">Knee</Translated></th>
            <th className="num"><Translated vi="Token/s ở batch 1">Tokens/s at batch 1</Translated></th>
            <th className="num"><Translated vi="Token/s cao nhất (batch)">Most tokens/s (batch)</Translated></th>
            <th><Translated vi="Một bước có batch so với các bước riêng lẻ">One batched step against separate steps</Translated></th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ s, gemm, steps, knee, best, common }) => (
            <tr key={s.backend + gemm} data-gemm={gemm} data-backend={s.backend} data-knee={knee ?? 'none'}>
              <td>
                {s.backend} <span className="muted">({s.device})</span>
              </td>
              <td>{M8_GEMMS[gemm] ?? gemm}</td>
              <td className="num">{knee ?? <span className="muted"><Translated vi={<>không có tới {String(steps.at(-1)!.batch)}</>}>none up to {String(steps.at(-1)!.batch)}</Translated></span>}</td>
              <td className="num">{num(steps[0].batch) === 1 ? fmt(tps(steps[0])) : ''}</td>
              <td className="num">
                {fmt(tps(best))} ({String(best.batch)})
              </td>
              <td data-one-by-one={common ? fmt(secs(common.o) / secs(common.st)) : 'none'}>
                {common ? (
                  <Translated
                    vi={
                      <>
                        {String(common.o.batch)} chuỗi: {msOf(common.st)} khi gom batch, {msOf(common.o)} khi chạy lần lượt,{' '}
                        {fmt(secs(common.o) / secs(common.st))}×
                      </>
                    }
                  >
                    <>
                      {String(common.o.batch)} sequences: {msOf(common.st)} batched, {msOf(common.o)} one after another,{' '}
                      {fmt(secs(common.o) / secs(common.st))}×
                    </>
                  </Translated>
                ) : (
                  <span className="muted"><Translated vi="không có dòng one_by_one">no one_by_one rows</Translated></span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <Translated
        vi={
          <p className="status-line" style={{ margin: '-0.8rem 0 1.4rem' }}>
            Từ {rows.map((r) => whose(r.s)).filter((w, i, a) => a.indexOf(w) === i).join('; ')}. Điểm gãy là batch đầu tiên có thời
            gian mỗi bước ít nhất bằng 1.5 lần thời gian ở batch nhỏ nhất (định nghĩa của chúng tôi).
          </p>
        }
      >
        <p className="status-line" style={{ margin: '-0.8rem 0 1.4rem' }}>
          From {rows.map((r) => whose(r.s)).filter((w, i, a) => a.indexOf(w) === i).join('; ')}. The knee is the first batch
          whose time per step is at least 1.5 times the time at the smallest batch (our definition).
        </p>
      </Translated>
    </div>
  );
}

const PIECES = ['embed', 'linear', 'bias', 'residual', 'lm_head', 'softmax', 'argmax'];
const PIECE_LABEL: Record<string, string> = {
  get embed() { return tr('embed (yours)', 'embed (của bạn)'); },
  get linear() { return tr('one layer: linear', 'một lớp: linear'); },
  get bias() { return tr('one layer: bias', 'một lớp: bias'); },
  get residual() { return tr('one layer: residual', 'một lớp: residual'); },
  get lm_head() { return tr('LM head', 'LM head'); },
  get softmax() { return tr('softmax', 'softmax'); },
  get argmax() { return tr('argmax', 'argmax'); },
};

/** Each piece of the step timed alone, at the batches the benchmark recorded. */
function PieceTable() {
  const { reports, loading } = useLabs();
  const sets = preferredBench(reports, l2);
  const gemms = Object.keys(M8_GEMMS).filter((g) => sets.some((s) => s.rows.some((r) => r.name === 'piece' && r.gemm === g)));
  const [pick, setPick] = useState<string>('m3');
  const gemm = gemms.includes(pick) ? pick : gemms[0];
  if (loading || !gemm) return null;
  return (
    <div className="wide" style={{ maxWidth: '58rem' }} data-pieces={gemm}>
      <div className="seg" role="group" aria-label={tr('GEMM', 'GEMM')} style={{ flexWrap: 'wrap', marginBottom: '0.6rem' }}>
        {gemms.map((g) => (
          <button key={g} aria-pressed={g === gemm} onClick={() => setPick(g)}>
            {g.toUpperCase()}
          </button>
        ))}
      </div>
      {sets.map((s) => {
        const rows = s.rows.filter((r) => r.name === 'piece' && r.gemm === gemm && secs(r));
        const batches = [...new Set(rows.map((r) => num(r.batch)!))].sort((a, b) => a - b);
        if (!batches.length) return null;
        const at = (p: string, b: number) => rows.find((r) => r.piece === p && num(r.batch) === b);
        return (
          <div key={s.backend} style={{ overflowX: 'auto' }}>
            <table className="data" data-table="pieces" data-backend={s.backend}>
              <thead>
                <tr>
                  <th>
                    <Translated vi={<>{s.backend}, {gemm.toUpperCase()}: phần việc, ms</>}>
                      {s.backend}, {gemm.toUpperCase()}: piece, ms
                    </Translated>
                  </th>
                  {batches.map((b) => (
                    <th key={b} className="num">
                      <Translated vi={<>batch {b}</>}>batch {b}</Translated>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {PIECES.map((p) => (
                  <tr key={p} data-piece={p}>
                    <td>{PIECE_LABEL[p]}</td>
                    {batches.map((b) => {
                      const r = at(p, b);
                      return (
                        <td key={b} className="num">
                          {r ? fmt(secs(r) * 1e3) : ''}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS} tiers={TIERS_GEMV}>
      <LabsIntro>
        <Translated
          vi={
            <>
              Hai lab, trong <code>labs/ch22/</code>. Chương này không in đoạn mã nào, nên cả hai đều là của khóa học này. Lab 22.1 cắt
              phép nhân ma trận-vector nằm ở trung tâm của một bước decode theo hai cách, hai cách phân rã của{' '}
              <a href={href('ch22', '22.2')}>mục 22.2</a>, và đo xem thiết bị của bạn ưa cách nào hơn. Lab 22.2 là mốc dự án M8: trọn
              một bước decode greedy được xây từ các mảnh của dự án, đo ở các kích thước batch từ 1 trở lên, để đo sự đánh đổi của{' '}
              <a href={href('ch22', '22.5')}>mục 22.5</a>.
            </>
          }
        >
          Two labs, in <code>labs/ch22/</code>. The chapter prints no code, so both are the course's own. Lab 22.1 cuts the
          matrix-vector product at the heart of a decode step two ways, the two decompositions of{' '}
          <a href={href('ch22', '22.2')}>section 22.2</a>, and measures which your device prefers. Lab 22.2 is milestone M8:
          a whole greedy decode step built from the project's pieces, timed at batch sizes from 1 upward, which measures the
          trade of <a href={href('ch22', '22.5')}>section 22.5</a>.
        </Translated>
      </LabsIntro>
      <RunningLabs
        example="22.1"
        alsoShownIn={
          <Translated
            vi={
              <>
                trình khám phá batching ở <a href={href('ch22', '22.5')}>trang 22.5</a>, nơi vẽ thời gian bước của lab 22.2 cạnh mô hình
              </>
            }
          >
            <>
              the batching explorer on <a href={href('ch22', '22.5')}>page 22.5</a>, which draws lab 22.2's step times beside
              the model
            </>
          </Translated>
        }
      />
      <BackendsAndEmulator />
      <VerificationStatus />

      {/* ------------------------------------------------------------ 22.1 */}
      <h2><Translated vi="Lab 22.1: hai cách cắt một phép nhân ma trận-vector lúc decode">Lab 22.1: two ways to cut a decode-time matrix-vector product</Translated></h2>
      <Translated
        vi={
          <p>
            Lúc decode, một nhúm token đi qua mỗi lớp tuyến tính: <code>Y = X W^T</code>, với <code>X</code> kích thước{' '}
            <code>batch × in</code>, <code>W</code> lưu dạng <code>(out, in)</code> theo thứ tự hàng (row-major), và batch tối đa là
            8. Đó là bố cục và thứ tự tham số của <code>launchLinear</code> trong dự án (lab 6.4 và 15.6). Chi phí của nó là đọc{' '}
            <code>W</code> một lần.
          </p>
        }
      >
        <p>
          At decode time a handful of tokens passes through each linear layer: <code>Y = X W^T</code>, with{' '}
          <code>X</code> of <code>batch × in</code>, <code>W</code> stored <code>(out, in)</code> row-major, and a batch of at
          most 8. That is the layout and argument order of the project's <code>launchLinear</code> (labs 6.4 and 15.6). Its
          cost is reading <code>W</code> once.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Bạn viết cách cắt lấy đầu ra làm trung tâm. Một warp sở hữu đầu ra <code>j</code>: các lane của nó cùng đi theo bước nhảy
            dọc hàng <code>j</code> của <code>W</code>, mỗi lane giữ một tổng tích lũy cho mỗi hàng batch trong thanh ghi
            (register), một phép rút gọn bằng shuffle kết hợp các lane, và lane 0 ghi <code>Y[b][j]</code>. Không ai khác ghi phần tử
            đó, nên không cần atomic nào. Bạn viết phần việc của warp một lần, dưới tên <code>warpRowDots</code>: các tích vô hướng
            của một đoạn trong một hàng của <code>W</code> với mọi hàng batch. Kernel split-K được cung cấp gọi cùng hàm đó trên
            phần của nó: nó cắt <code>in</code> thành <code>S</code> phần và giao một warp cho mỗi cặp (phần, đầu ra), nên{' '}
            <code>S</code> warp cùng cộng vào mỗi đầu ra. Nó lấy đầu vào làm trung tâm dọc theo <code>in</code>, một scatter, và nó
            mang một lỗi cài sẵn: các tổng riêng phần được kết hợp bằng một lệnh <code>+=</code> thông thường. Bạn tìm ra nó, sửa
            nó, rồi đo sự đánh đổi mà sách nêu ở tr. 533: split-K bộc lộ mức song song gấp <code>S</code> lần và trả{' '}
            <code>S</code> phép atomic cho mỗi đầu ra.
          </p>
        }
      >
        <p>
          You write the output-centric cut. One warp owns output <code>j</code>: its lanes stride along row{' '}
          <code>j</code> of <code>W</code> together, each lane keeps one running sum per batch row in registers, a shuffle
          reduction combines the lanes, and lane 0 writes <code>Y[b][j]</code>. Nobody else writes that element, so nothing
          needs an atomic. You write the warp's part once, as <code>warpRowDots</code>: the dot products of a stretch of one
          row of <code>W</code> with every batch row. The provided split-K kernel calls the same function on its chunk: it
          cuts <code>in</code> into <code>S</code> chunks and gives one warp to each (chunk, output) pair, so{' '}
          <code>S</code> warps add into every output. It is input-centric along{' '}
          <code>in</code>, a scatter, and it carries one planted defect: the partial sums are combined with a plain{' '}
          <code>+=</code>. You find it, fix it, then measure the trade the book names on p. 533: split-K exposes{' '}
          <code>S</code> times the parallelism and pays <code>S</code> atomics per output.
        </p>
      </Translated>
      <Source section="22.2" pages="532–536" />
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <Translated
              vi={
                <>
                  viết <code>warpRowDots</code>, <code>gemvGatherKernel</code> và <code>launchGemvGather</code>; tìm và sửa lỗi trong{' '}
                  <code>gemvSplitKKernel</code> (phần khởi chạy của nó, vốn đặt <code>Y</code> về 0 bằng <code>cudaMemset</code>, đã được
                  cung cấp)
                </>
              }
            >
              <>
                write <code>warpRowDots</code>, <code>gemvGatherKernel</code> and <code>launchGemvGather</code>; find
                and fix the defect in <code>gemvSplitKKernel</code> (its launch, which zeroes <code>Y</code> with{' '}
                <code>cudaMemset</code>, is provided)
              </>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated
              vi={
                <>
                  viết <code>simdRowDots</code> và <code>gemv_gather</code>; sửa <code>gemv_splitk</code>, có tham số <code>Y</code> phải
                  đổi thành <code>device atomic_float*</code>
                </>
              }
            >
              <>
                write <code>simdRowDots</code> and <code>gemv_gather</code>; fix <code>gemv_splitk</code>, whose{' '}
                <code>Y</code> parameter must become <code>device atomic_float*</code>
              </>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated
              vi={
                <>
                  viết phần dispatch trong <code>encodeGemvGather</code>: một SIMD group cho mỗi đầu ra, độ rộng lấy từ{' '}
                  <code>threadExecutionWidth</code> của pipeline. <code>encodeGemvSplitK</code>, cùng dispatch đặt về 0 của nó trong cùng
                  encoder tuần tự, đã được cung cấp
                </>
              }
            >
              <>
                write the dispatch in <code>encodeGemvGather</code>: one SIMD-group per output, the width from the
                pipeline's <code>threadExecutionWidth</code>. <code>encodeGemvSplitK</code>, with its zeroing dispatch in the
                same serial encoder, is provided
              </>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Tên và chữ ký hàm là cố định, vì lab 22.2 liên kết gather của bạn làm đường cho batch nhỏ khi có lời giải của lab này.
            Đừng bao giờ viết hằng số 32 cho độ rộng warp: dùng <code>warpSize</code> trong kernel, thuộc tính của device trong
            phần khởi chạy (<code>deviceWarpSize()</code> đã được cung cấp) và <code>threadExecutionWidth</code> của pipeline trên
            Metal. <code>warpSum</code> của chương 10, với kiểu lane-mask riêng của nó, đã được cung cấp.
          </p>
        }
      >
        <p>
          The names and signatures are fixed, because lab 22.2 links your gather as its small-batch path when this lab's
          solution is present. Never write a literal 32 for the warp width: <code>warpSize</code> in the kernel, the device's
          property in the launcher (<code>deviceWarpSize()</code> is provided) and the pipeline's{' '}
          <code>threadExecutionWidth</code> on Metal. Chapter 10's <code>warpSum</code>, with its own lane-mask type, is
          provided.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 22.1 --backend emu</code> trước, rồi <code>--backend race</code>, rồi không có{' '}
            <code>--backend</code> cho GPU của bạn. Mọi trường hợp đều so <code>Y</code> với <code>x W^T</code> tính bằng double
            trong <code>main.cpp</code>, trong phạm vi 1e-4 nhân với một cộng tổng |x·w| cho đầu ra đó, không bao giờ so từng bit:
            các atomic cộng các tổng riêng phần theo thứ tự bất kỳ. Các kích thước đi từ 1 × 1 × 1 đến 8 × 257 × 130, và trên GPU
            tới 4 × 14336 × 4096. Các số lần chia gồm 1, 2, 3, 7, 32 và một phần cho mỗi độ rộng warp của đầu vào, nên có phần rỗng
            và có số lần chia không chia hết <code>in</code>. <code>X</code> và <code>W</code> chứa đúng số float của chúng, nên
            trên <code>emu</code> một lần đọc vượt quá một trong hai là một báo cáo AddressSanitizer chỉ rõ dòng. <code>Y</code> bắt
            đầu là NaN và theo sau là các giá trị canh (sentinel) phải còn nguyên. Mọi trường hợp khởi chạy hai lần mà không đặt lại{' '}
            <code>Y</code>, nên một lần khởi chạy split-K không tự đặt <code>Y</code> về 0 sẽ thất bại. Mã khởi đầu (starter) chưa
            sửa thất bại vì gather không ghi gì và split-K cộng các số 0.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 22.1 --backend emu</code> first, then <code>--backend race</code>, then without{' '}
          <code>--backend</code> for your GPU. Every case compares <code>Y</code> with <code>x W^T</code> computed in double
          in <code>main.cpp</code>, within 1e-4 times one plus the sum of |x·w| for that output, never bit for bit: atomics
          add the partial sums in any order. The shapes run from 1 × 1 × 1 to 8 × 257 × 130, and on a GPU up to 4 × 14336 ×
          4096. The split counts include 1, 2, 3, 7, 32 and one per warp-width of input, so some chunks are empty and some
          counts do not divide <code>in</code>. <code>X</code> and <code>W</code> hold exactly their floats, so on{' '}
          <code>emu</code> a read past either is an AddressSanitizer report naming the line. <code>Y</code> starts as NaN
          and is followed by sentinels that must survive. Every case launches twice without resetting <code>Y</code>, so a
          split-K launch that does not zero <code>Y</code> itself fails. The untouched starter fails because the gather
          writes nothing and split-K adds zeros.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Backend nào thấy được lỗi là bài học từ trình chạy từng bước ở <a href={href('ch22', '22.2')}>mục 22.2</a>.{' '}
            <code>emu</code> chạy một luồng giả lập cho tới lệnh shuffle hoặc rào chắn kế tiếp của nó, nên việc lane 0 đọc, cộng và
            ghi <code>Y[b][j]</code> không bao giờ bị ngắt: lệnh <code>+=</code> thông thường qua được ở đó, lần nào cũng vậy.{' '}
            <code>race</code> chạy đồng thời các luồng của một khối, và nhiều khối, dưới ThreadSanitizer, và báo dòng chứa{' '}
            <code>+=</code> hai lần, mỗi lần cho một trong hai truy cập va chạm. Nó không so được hai khối đã chạy lần lượt trên cùng
            một luồng OS, vì thế split-K đánh số các warp sao cho <code>S</code> warp của một đầu ra nằm cạnh nhau trong một khối.
            ThreadSanitizer báo một cặp dòng một lần mỗi lần chạy, nên trường hợp race đầu tiên là trường hợp thất bại kèm báo cáo.
            Trên Metal không có gì phát hiện race ở đây: một cập nhật bị mất chỉ lộ ra thành một tổng sai, và chỉ khi hai SIMD group
            va chạm về thời gian. Một lần chạy sạch ở đó không phải bằng chứng về một kernel đúng.
          </p>
        }
      >
        <p>
          Which backend can see the defect is the lesson of <a href={href('ch22', '22.2')}>section 22.2</a>'s stepper.{' '}
          <code>emu</code> runs one emulated thread until its next shuffle or barrier, so lane 0's read, add and write of{' '}
          <code>Y[b][j]</code> is never interrupted: the plain <code>+=</code> passes there, every time. <code>race</code>{' '}
          runs the threads of a block, and several blocks, at once under ThreadSanitizer, and reports the line of the{' '}
          <code>+=</code> twice, once for each of the two colliding accesses. It cannot compare two blocks that ran one after
          the other on one OS thread, which is why split-K numbers its warps so that the <code>S</code> warps of one output
          are neighbours in one block. ThreadSanitizer reports a pair of lines once per run, so the first racing case is the
          one that fails with a report. On Metal nothing detects races here: a lost update shows only as a wrong sum, and only
          when two SIMD-groups collide in time. A clean run there is not evidence of a correct kernel.
        </p>
      </Translated>
      <RaceStatus lab={l1} />
      <Translated
        vi={
          <p>
            Cách sửa khác nhau theo backend. Trên CUDA và HIP là <code>atomicAdd(&amp;Y[...], total)</code>. Trên Metal một hàm
            atomic nhận một kiểu atomic, nên <code>Y</code> trở thành <code>device atomic_float*</code> và phép cộng là{' '}
            <code>atomic_fetch_add_explicit(..., memory_order_relaxed)</code>; phần khởi chạy gắn cùng một buffer bất kể kiểu của nó
            trong kernel. Và không có gì trên Mac kiểm tra warp 64 lane: cả hai trình giả lập đều mô hình hóa warp 32 lane, nên một
            hằng số 32 có thể qua được ở mọi nơi tại đây. Hãy đọc vòng lặp shuffle của bạn: nó bắt đầu ở <code>warpSize / 2</code>,
            mask có kiểu riêng, bước nhảy là <code>warpSize</code>.
          </p>
        }
      >
        <p>
          The fix differs by backend. On CUDA and HIP it is <code>atomicAdd(&amp;Y[...], total)</code>. On Metal an atomic
          function takes an atomic type, so <code>Y</code> becomes <code>device atomic_float*</code> and the add is{' '}
          <code>atomic_fetch_add_explicit(..., memory_order_relaxed)</code>; the launch binds the same buffer whatever its type
          in the kernel. And nothing on a Mac checks a 64-lane warp: both emulators model 32-lane warps, so a literal 32 can
          pass everywhere here. Read your shuffle loop: it starts at <code>warpSize / 2</code>, the mask has its own type,
          the stride is <code>warpSize</code>.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/infra-facts.md B1, B3, C1, C2; ch05-facts B5)', '2026-11-20 (notes/today/infra-facts.md, mục B1, B3, C1, C2; ch05-facts, mục B5)')}>
        <Translated
          vi={
            <p>
              <code>atomicAdd</code> cho float của HIP biên dịch thành một dạng an toàn, có thể là một vòng lặp compare-and-swap, trừ
              khi build với <code>-munsafe-fp-atomics</code>; bản build của khóa học không truyền cờ này. Các intrinsic{' '}
              <code>_sync</code> của HIP cần mask 64 bit. Wavefront của AMD rộng 64 hoặc 32. Atomic float của Metal cần{' '}
              <code>atomic_float</code>, Metal 3 và GPU Apple7 trở lên.
            </p>
          }
        >
          <p>
            HIP's float <code>atomicAdd</code> compiles to a safe form, possibly a compare-and-swap loop, unless built with{' '}
            <code>-munsafe-fp-atomics</code>; the course's build does not pass it. HIP's <code>_sync</code> intrinsics need a
            64-bit mask. AMD's wavefront is 64 or 32 wide. Metal float atomics need <code>atomic_float</code>, Metal 3 and an
            Apple7 GPU or later.
          </p>
        </Translated>
      </Today>

      <h3><Translated vi="Thí nghiệm">The experiment</Translated></h3>
      <Translated
        vi={
          <p>
            Khi các kernel của bạn đã qua, <code>--bench</code> đo ba lớp ở batch 1, 4 và 8: 4096 → 4096; 4096 → 16, một lớp hẹp như
            router; và 14336 → 4096, một phép chiếu xuống (down-projection) của MLP (hai kích thước sau là của chúng tôi). Gather chạy
            một lần, split-K ở <code>S</code> = 2, 4, 8, ... tới <code>in</code> / (8 × độ rộng warp). Các kernel chạy luân phiên và
            mỗi kernel giữ vòng tốt nhất của nó, và mỗi lần khởi chạy đọc một bản sao khác của <code>W</code>, nên không lần khởi chạy
            nào thấy trọng số của nó ở chỗ lần trước để lại. Thời gian của split-K bao gồm cả việc đặt về 0. Hãy ghi lại hai dự đoán
            trước khi chạy.
          </p>
        }
      >
        <p>
          Once your kernels pass, <code>--bench</code> times three layers at batches 1, 4 and 8: 4096 → 4096; 4096 → 16, a
          narrow layer such as a router; and 14336 → 4096, an MLP down-projection (the last two shapes are ours). Gather
          runs once, split-K at <code>S</code> = 2, 4, 8, ... up to <code>in</code> / (8 × the warp width). The kernels take
          turns and each keeps its best round, and every launch reads a different copy of <code>W</code>, so no launch finds
          its weights where the previous one left them. Split-K's time includes its zeroing. Write down two predictions
          before you run it.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Hãy cân nhắc bốn điều. Gather có một warp cho mỗi đầu ra, tổng cộng 16 warp trên lớp hẹp; split-K có <code>S</code> ×
            16. Thiết bị của bạn giữ thường trú được bao nhiêu warp? <code>python3 labs/pmpp.py doctor</code> báo các số liệu trên
            CUDA và HIP; Metal không có truy vấn occupancy. Split-K thực hiện <code>S</code> phép atomic cho mỗi đầu ra. Số byte của{' '}
            <code>W</code> được đọc là như nhau với mọi <code>S</code>.
          </p>
        }
      >
        <p>
          Weigh four things. Gather has one warp per output, 16 warps in all on the narrow layer; split-K has{' '}
          <code>S</code> × 16. How many warps can your device keep resident? <code>python3 labs/pmpp.py doctor</code> reports
          the figures on CUDA and HIP; Metal has no occupancy query. Split-K makes <code>S</code> atomics per output. The bytes
          of <code>W</code> read are the same for every <code>S</code>.
        </p>
      </Translated>
      <Source section="22.2" pages="533" />
      <LabPrediction
        id="ch22/labs/p-narrow"
        prompt={
          <Translated
            vi={
              <>
                Trên lớp 4096 → 16 ở batch 1, split-K nhanh nhất sẽ chạy nhanh hơn gather của bạn bao nhiêu lần? (Dưới 1 nghĩa là gather
                thắng.)
              </>
            }
          >
            <>
              On the 4096 → 16 layer at batch 1, how many times faster than your gather will the fastest split-K run? (Below 1
              means gather wins.)
            </>
          </Translated>
        }
        unit={tr('× gather', '× so với gather')}
        measure={gainAt(NARROW)}
        missing={MISSING_221}
        reading={
          <Translated
            vi={
              <>
                Trên 1, cách chia tốt nhất nhanh hơn. Sách cân nhắc mức song song được bộc lộ với atomic (tr. 533) nhưng không đưa ra con
                số nào cho bên nào, và cũng không nhà cung cấp nào làm vậy (ch22-facts A1). Phần chi tiết nêu số lần chia đã thắng; trên
                một lớp hẹp, đó là thước đo số warp mà thiết bị của bạn cần để giữ bộ nhớ luôn bận.
              </>
            }
          >
            <>
              Above 1, the best split was faster. The book weighs exposed parallelism against atomics (p. 533) but puts no
              number on either, and neither does any vendor (ch22-facts A1). The detail names the split count that won; on a
              narrow layer that is a measure of how many warps your device needed to keep its memory busy.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch22/labs/p-square"
        prompt={
          <Translated vi="Trên lớp 4096 → 4096 ở batch 1, cùng tỷ số đó: tốc độ của split-K nhanh nhất so với gather.">
            <>On the 4096 → 4096 layer at batch 1, the same ratio: the fastest split-K's speed over gather's.</>
          </Translated>
        }
        unit={tr('× gather', '× so với gather')}
        measure={gainAt(SQUARE)}
        missing={MISSING_221}
        reading={
          <Translated
            vi={
              <>
                Với 4,096 đầu ra, riêng gather đã khởi chạy 4,096 warp. Điều đó đã lấp đầy thiết bị của bạn hay chưa là thứ tỷ số này cho
                bạn biết. Một giá trị gần 1 nghĩa là mức song song thêm vào mua được rất ít và atomic tốn rất ít; phần chi tiết cho thấy
                cách chia nào thắng.
              </>
            }
          >
            <>
              With 4,096 outputs, gather alone launches 4,096 warps. Whether that already fills your device is what this
              ratio tells you. A value near 1 means the extra parallelism bought little and the atomics cost little; the
              detail shows which split won.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l1}
        filter={(r) => r.name === 'gemv'}
        cols={[
          { head: tr('Cut', 'Cách cắt'), cell: (r) => (r.kernel === 'gather' ? 'gather' : 'split-K') },
          rawCol('S', 'splits'),
          rawCol(tr('Batch', 'Batch'), 'batch'),
          { head: tr('Layer', 'Lớp'), cell: (r) => `${n(num(r.in) ?? 0)} → ${n(num(r.out) ?? 0)}` },
          { head: tr('Time, µs', 'Thời gian, µs'), num: true, cell: (r) => (secs(r) ? fmt(secs(r) * 1e6) : '') },
          numCol('W GB/s', 'weight_gb_per_s'),
        ]}
      />
      <Translated
        vi={
          <p>
            <code>W GB/s</code> là số byte trọng số, 4 × <code>in</code> × <code>out</code>, chia cho thời gian: thứ mà một phép nhân
            ma trận-vector bắt buộc phải đọc, không phải mọi thứ hệ thống bộ nhớ đã di chuyển. Lab 5.1 đã đo băng thông mà thiết bị
            của bạn duy trì được, đó là giới hạn tốc độ cho một kernel đọc mỗi trọng số một lần. Bảng đặt hai thứ đó cạnh nhau.
          </p>
        }
      >
        <p>
          <code>W GB/s</code> is the weight bytes, 4 × <code>in</code> × <code>out</code>, over the time: what a
          matrix-vector product must read, not everything the memory system moved. Lab 5.1 measured the bandwidth your device
          sustains, which is the speed limit for a kernel that reads each weight once. The table sets the two side by side.
        </p>
      </Translated>
      <SpeedOfLight />
      <Translated
        vi={
          <p>
            Rồi đọc số liệu của bạn đối chiếu với tr. 533. Ở đâu việc gather không có atomic đã thắng, và ở đâu mức song song thêm
            của split-K thắng? Kích thước nào làm đảo ngược câu trả lời? Đây là các phép đo thiết bị của bạn ở những kích thước này,
            không phải thuộc tính của các kernel. Một backend thật vạch cùng đường ranh đó bằng phép đo: llama.cpp giữ các kernel ma
            trận-vector riêng cho tối đa 8 token, với giới hạn thấp hơn cho một số kiểu trọng số trên những GPU cụ thể (
            <a href={href('ch22', '22.1')}>mục 22.1</a>).
          </p>
        }
      >
        <p>
          Then read your numbers against p. 533. Where did gather's lack of atomics win, and where did split-K's extra
          parallelism? Which shape turned the answer around? These are measurements of your device at these shapes, not
          properties of the kernels. A real backend draws the same line by measurement: llama.cpp keeps separate
          matrix-vector kernels for up to 8 tokens, with a lower limit for some weight types on particular GPUs (
          <a href={href('ch22', '22.1')}>section 22.1</a>).
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch22-facts D6)', '2026-11-20 (ch22-facts, mục D6)')}>
        <Translated
          vi={
            <p>
              llama.cpp: <code>mmvf</code> và <code>mmvq</code> cho tối đa 8 token trên CUDA và HIP, <code>mul_mv_ext</code> cho 2 đến
              8 trên Metal. Đọc từ nhánh mặc định của nó; nó thay đổi thường xuyên.
            </p>
          }
        >
          <p>
            llama.cpp: <code>mmvf</code> and <code>mmvq</code> up to 8 tokens on CUDA and HIP, <code>mul_mv_ext</code> for 2
            to 8 on Metal. Read from its default branch; it changes often.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Để đi xa hơn: giao cho mỗi đầu ra trọn một khối gồm <code>S</code> warp và kết hợp các tổng riêng phần trong bộ nhớ chia
            sẻ (shared memory) bằng phép rút gọn hai giai đoạn của chương 10. Đó là cùng mức song song cho <code>S</code> tới số warp
            trong một khối, không cần atomic và không cần đặt về 0. Và đếm số kết quả khác nhau qua 20 lần chạy split-K, như lab 10.1
            đã làm: với atomic, các bit cuối của <code>Y</code> có thể thay đổi giữa các lần chạy, còn gather chỉ có một thứ tự.
          </p>
        }
      >
        <p>
          To go further: give each output a whole block of <code>S</code> warps and combine the partial sums in shared memory
          with chapter 10's two-stage reduction. That is the same parallelism for <code>S</code> up to the warps in a block,
          with no atomic and no zeroing. And count distinct results over 20 runs of split-K, as lab 10.1 did: with atomics
          the last bits of <code>Y</code> can change from run to run, and gather has one order.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>Cả hai kernel, lỗi cài sẵn, các kích thước, biến thể dùng bộ nhớ chia sẻ và phương pháp benchmark đều là của chúng tôi.</p>
          }
        >
          <p>Both kernels, the defect, the shapes, the shared-memory variant and the benchmark method are ours.</p>
        </Translated>
      </Beyond>
      <ReferenceSolution lab={l1} tiers={TIERS_GEMV} />

      {/* ------------------------------------------------------------ 22.2 */}
      <h2><Translated vi="Lab 22.2: nghiên cứu batching (mốc dự án M8)">Lab 22.2: the batching study (milestone M8)</Translated></h2>
      <Translated
        vi={
          <p>
            Mục 22.5 nói rằng gom batch các truy vấn của nhiều người dùng biến các phép nhân vector-ma trận của mỗi phép chiếu thành
            một phép nhân ma trận dùng lại trọng số: mỗi truy vấn chờ lâu hơn, cả nhóm xong sớm hơn. Lab này xây một bước decode
            greedy cho <code>batch</code> chuỗi từ chính các mảnh của dự án và đo xem sớm hơn bao nhiêu, trên thiết bị của bạn, và
            đến đâu thì nó hết có lợi.
          </p>
        }
      >
        <p>
          Section 22.5 says that batching the queries of many users turns each projection's vector-matrix products into one
          matrix product that reuses the weights: each query waits longer, the group finishes sooner. This lab builds one
          greedy decode step for <code>batch</code> sequences from the project's own pieces and measures how much sooner, on
          your device, and where it stops paying.
        </p>
      </Translated>
      <Source section="22.5" pages="539" />
      <Translated vi={<p>Bạn viết hai thứ. Phần còn lại của bước là các mốc dự án bạn đã có, chép từ lời giải tham chiếu của chúng:</p>}>
        <p>You write two things. The rest of the step is milestones you already have, copied from their reference solutions:</p>
      </Translated>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-table="step">
          <thead>
            <tr>
              <th><Translated vi="Phép toán, cho cả batch">Operation, for the whole batch</Translated></th>
              <th><Translated vi="Mảnh">Piece</Translated></th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <code>H = E[tokens]</code>
              </td>
              <td>
                <Translated vi={<>của bạn: <code>launchEmbed</code>, một luồng cho mỗi phần tử của <code>H</code></>}>
                  yours: <code>launchEmbed</code>, one thread per element of <code>H</code>
                </Translated>
              </td>
            </tr>
            <tr>
              <td>
                <Translated vi={<>mỗi lớp: <code>T = H W_l^T</code></>}>
                  per layer: <code>T = H W_l^T</code>
                </Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      GEMM mà chương trình điều khiển (driver) chọn: M2 (<a href={href('ch06', 'labs')}>lab 6.4</a>), M3 (
                      <a href={href('ch15', 'labs')}>lab 15.6</a>) hoặc gather của lab 22.1
                    </>
                  }
                >
                  the GEMM the driver chooses: M2 (<a href={href('ch06', 'labs')}>lab 6.4</a>), M3 (
                  <a href={href('ch15', 'labs')}>lab 15.6</a>) or lab 22.1's gather
                </Translated>
              </td>
            </tr>
            <tr>
              <td>
                <Translated vi={<>mỗi lớp: <code>T += b_l</code>, rồi <code>H += T</code></>}>
                  per layer: <code>T += b_l</code>, then <code>H += T</code>
                </Translated>
              </td>
              <td>
                M0 (<a href={href('ch02', 'labs')}>lab 2.3</a>){/* i18n-ignore: milestone and lab number */}
              </td>
            </tr>
            <tr>
              <td>
                <code>logits = H E^T</code>
              </td>
              <td><Translated vi="cùng GEMM đó; trọng số của LM head chính là bảng embedding">the same GEMM; the LM head's weight is the embedding table</Translated></td>
            </tr>
            <tr>
              <td>
                <code>probs = softmax(logits / temperature)</code>, <code>tokens = argmax(logits)</code>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      M4 (<a href={href('ch10', 'labs')}>lab 10.3</a>); argmax ghi các token kế tiếp đè lên những token mà
                      bước đã đọc
                    </>
                  }
                >
                  M4 (<a href={href('ch10', 'labs')}>lab 10.3</a>); the argmax writes the next tokens over the ones the step
                  read
                </Translated>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <Translated
        vi={
          <p>
            Embedding lấy đầu ra làm trung tâm: luồng sở hữu <code>H[i]</code> gom (gather) đúng một float mà nó cần, và hai chuỗi có
            cùng token chỉ đọc cùng một hàng. Trước khi viết, hãy tự hỏi phiên bản lấy đầu vào làm trung tâm sẽ ra sao, một luồng cho
            mỗi hàng của <code>E</code>, và tiêu chí nào của sách loại bỏ nó. Bước này là một lần khởi chạy cho mỗi phép toán cho cả
            batch, không có gì phải chờ bên trong, và các id token không bao giờ rời device, nên bước tiếp theo bắt đầu mà không cần
            sao chép về host rồi quay lại. Đó là ý của mục 22.3 áp dụng cho decode: bất cứ thứ gì ở lại trên host đều giới hạn tốc độ
            tăng.
          </p>
        }
      >
        <p>
          The embedding is output-centric: the thread that owns <code>H[i]</code> gathers the one float it needs, and two
          sequences on the same token only read the same row. Before you write it, ask what the input-centric version would
          be, one thread per row of <code>E</code>, and which of the book's criteria rules it out. The step is one launch per
          operation for the whole batch with nothing waiting inside it, and the token ids never leave the device, so the next
          step starts with no copy to the host and back. That is section 22.3's point applied to decoding: whatever stays on
          the host bounds the speed-up.
        </p>
      </Translated>
      <Source section="22.2, 22.3" pages="532–533, 537–538" />
      <LabFiles
        lab={l2}
        files={[
          [
            'kernel.cu',
            <Translated
              vi={
                <>
                  <code>embedGatherKernel</code>, <code>launchEmbed</code> và <code>launchDecodeStep</code>
                </>
              }
            >
              <>
                <code>embedGatherKernel</code>, <code>launchEmbed</code> and <code>launchDecodeStep</code>
              </>
            </Translated>,
          ],
          [
            'kernel.metal',
            <>
              <code>embed_rows</code>
            </>,
          ],
          [
            'launch.mm',
            <Translated
              vi={
                <>
                  <code>encodeEmbed</code> và <code>encodeDecodeStep</code>, đặt mọi dispatch vào compute encoder tuần tự của bên gọi: cả
                  bước là một command buffer
                </>
              }
            >
              <>
                <code>encodeEmbed</code> and <code>encodeDecodeStep</code>, which put every dispatch into the caller's serial
                compute encoder: the whole step is one command buffer
              </>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Được cung cấp và không chỉnh sửa: <code>decode.h</code>, khai báo mô hình, trạng thái và mọi mảnh cùng thứ tự tham số của
            chúng, và <code>provided/pieces.cmake</code>, chép lời giải tham chiếu của các lab 2.3, 6.4 và 10.3, cùng 15.6 và 22.1
            khi chúng tồn tại, vào thư mục build, đổi tên ở chỗ tên trùng nhau. Kernel Metal của mỗi mảnh được biên dịch với các hằng
            số mà lab của chính nó dùng. Mỗi lần khởi chạy thấy những gì lần trước đã ghi: các lần khởi chạy không có stream chạy
            theo thứ tự phát lệnh trên CUDA và HIP, và một compute encoder Metal thông thường chạy các dispatch theo thứ tự với các
            buffer được theo dõi xung đột (hazard-tracked).
          </p>
        }
      >
        <p>
          Provided and not edited: <code>decode.h</code>, which declares the model, the state and every piece with its
          argument order, and <code>provided/pieces.cmake</code>, which copies the reference solutions of labs 2.3, 6.4 and
          10.3, and of 15.6 and 22.1 when they exist, into the build directory, renaming where names collide. Each piece's
          Metal kernel is compiled with the constants its own lab uses. Each launch sees what the one before it wrote:
          launches with no stream run in issue order on CUDA and HIP, and a plain Metal compute encoder runs its dispatches in
          order with buffers hazard-tracked.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch02-facts A5, B3, C9; ch22-facts C1)', '2026-11-20 (ch02-facts, mục A5, B3, C9; ch22-facts, mục C1)')}>
        <Translated
          vi={
            <p>
              Một compute encoder thông thường "dispatches its compute commands serially". Trong một encoder đồng thời, bước này sẽ cần
              một rào chắn bộ nhớ sau mỗi dispatch mà dispatch kế tiếp đọc.
            </p>
          }
        >
          <p>
            A plain compute encoder "dispatches its compute commands serially". In a concurrent encoder the step would need
            a memory barrier after each dispatch the next one reads.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 22.2 --backend emu</code> trước; nó chậm, vì nó chạy mọi luồng của mọi lần khởi chạy
            của mọi bước dưới ba thứ tự luồng. Trên trình giả lập, mô hình có <code>d</code> = 48, bộ từ vựng 301 và 2 lớp, ở batch 1,
            2, 3 và 5 trong bốn bước; trên GPU là <code>d</code> = 256, bộ từ vựng 4,099, batch tới 64, sáu bước; mỗi batch chạy với
            mọi GEMM hiện có. Riêng <code>launchEmbed</code> phải chép các hàng chính xác từng bit, kể cả token 0, token cuối và một
            token được dùng hai lần. Sau mỗi bước, token của mỗi chuỗi phải bằng argmax của một bản tham chiếu CPU tính bằng double,
            với <code>H</code>, logits và các xác suất nằm trong dung sai. Mô hình kiểm thử được sinh sao cho hai logit lớn nhất luôn
            cách xa nhau, nên làm tròn float không thể lật một token. Bất biến theo batch: các token của mỗi chuỗi trong một batch
            phải bằng các token của nó khi decode riêng. Các dải canh (guard band) phải còn nguyên và đầu ra bắt đầu là NaN. Lab này
            không chạy trên backend race: mọi luồng trong mã của bạn ghi phần tử riêng của nó. Mã khởi đầu (starter) chưa sửa thất bại
            vì embedding và bước không ghi gì.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 22.2 --backend emu</code> first; it is slow, since it runs every thread of every
          launch of every step under three thread orders. On the emulator the model is <code>d</code> = 48, a vocabulary of
          301 and 2 layers, at batches 1, 2, 3 and 5 for four steps; on a GPU <code>d</code> = 256, a vocabulary of 4,099,
          batches up to 64, six steps; each batch runs with every GEMM present. <code>launchEmbed</code> alone must copy rows
          bit for bit, including token 0, the last token and a token used twice. After every step each sequence's token must
          equal the argmax of a CPU reference in double, with <code>H</code>, the logits and the probabilities within a
          tolerance. The test model is generated so that the top two logits are always far apart, so float rounding cannot
          flip a token. Batch invariance: each sequence's tokens in a batch must equal its tokens decoded alone. Guard bands
          must survive and outputs start as NaN. The lab is not on the race backend: every thread of your code writes its own
          element. The untouched starter fails because the embedding and the step write nothing.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Mô hình là của chúng tôi: embedding, các lớp linear cộng bias cộng residual, LM head gắn với embedding, một bộ lấy mẫu.
              Nó bỏ attention ra vì attention không gom batch được (ch22-facts A6): trong một bước thật nó sẽ nằm cùng các phần có chi
              phí tăng theo batch ngay từ đầu. Decode greedy chỉ đọc argmax; các xác suất được tính vì bộ lấy mẫu của mốc dự án M5 cần
              chúng.
            </p>
          }
        >
          <p>
            The model is ours: embedding, layers of linear plus bias plus residual, the LM head tied to the embedding, a
            sampler. It leaves attention out because attention does not batch (ch22-facts A6): in a real step it would sit
            with the parts whose cost grows with the batch from the start. Greedy decoding reads only the argmax; the
            probabilities are computed because the sampler of milestone M5 needs them.
          </p>
        </Translated>
      </Beyond>

      <h3><Translated vi="Thí nghiệm">The experiment</Translated></h3>
      <Translated
        vi={
          <p>
            <code>--bench</code> đo bước ở <code>d</code> = 4096, 2 lớp và bộ từ vựng 32,000 bằng float32, khoảng 650 MB trọng số
            (giảm một nửa <code>d</code> cho đến khi vừa với thiết bị ít bộ nhớ hơn), ở batch 1, 2, 4, ... tới 1,024, cho mọi GEMM
            hiện có. Một bước là một command buffer trên Metal và một cặp event trên CUDA và HIP. Nó cũng đo cùng số chuỗi đó khi
            decode lần lượt từng chuỗi, tức "sum of all the vector-matrix multiplications" của sách, và từng mảnh của bước khi chạy
            riêng. Hãy bắt đầu từ ngưỡng của bạn:
          </p>
        }
      >
        <p>
          <code>--bench</code> times the step at <code>d</code> = 4096, 2 layers and a vocabulary of 32,000 in float32, about
          650 MB of weights (halving <code>d</code> until it fits a device with less memory), at batches 1, 2, 4, ... up to
          1,024, for every GEMM present. One step is one command buffer on Metal and one pair of events on CUDA and HIP. It also
          times the same number of sequences decoded one at a time, the book's "sum of all the vector-matrix
          multiplications", and each piece of the step alone. Start from your threshold:
        </p>
      </Translated>
      <Threshold />
      <Translated
        vi={
          <p>
            Ba điều cần cân nhắc. Mô hình đặt điểm gãy của một phép chiếu float32 gần gấp đôi ngưỡng của bạn, vì cường độ tính toán
            của nó khoảng B/2 FLOP mỗi byte khi batch còn nhỏ so với kích thước của lớp. Ô khối của M3 là 128 hàng của batch, nên
            thời gian của nó phẳng tới 128 bất kể băng thông, còn của M2 thay đổi theo bậc 16. Và các mảnh theo từng phần tử cùng
            softmax tăng theo batch ngay từ đầu, vì cường độ tính toán của chúng không tăng theo batch. Nếu bộ nhớ đệm (cache) của
            thiết bị chứa được một phần trọng số, batch 1 có thể vượt băng thông của lab 5.1; Apple không công bố kích thước cache
            nào.
          </p>
        }
      >
        <p>
          Three things to weigh. The model puts a float32 projection's knee near twice your threshold, because its intensity
          is about B/2 FLOP per byte while the batch is small next to the layer's sizes. M3's block tile is 128 rows of the
          batch, so its time is flat up to 128 whatever the bandwidth, and M2's moves in steps of 16. And the elementwise
          pieces and the softmax grow with the batch from the start, because their intensity does not rise with it. If your
          device's caches hold part of the weights, batch 1 can beat lab 5.1's bandwidth; Apple documents no cache sizes.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch22-facts E1, E2, E3; infra-facts C13)', '2026-11-20 (ch22-facts, mục E1, E2, E3; infra-facts, mục C13)')}>
        <Translated
          vi={
            <p>
              E1 đến E3 là phép tính của khóa học và kích thước ô của dự án (M2 16, M3 128), đã đối chiếu với mã và mô tả của NVIDIA
              về tile quantization.
            </p>
          }
        >
          <p>
            E1 to E3 are the course's arithmetic and the project's tile sizes (M2 16, M3 128), checked against the code and
            NVIDIA's account of tile quantization.
          </p>
        </Translated>
      </Today>
      <KneePrediction />
      <LabPrediction
        id="ch22/labs/p-gain64"
        prompt={
          <Translated vi="Với M3, batch 64 sẽ cho số token mỗi giây gấp bao nhiêu lần batch 1?">
            <>With M3, how many times the tokens per second of batch 1 will batch 64 give?</>
          </Translated>
        }
        unit={tr('× batch 1', '× so với batch 1')}
        measure={gain64}
        missing={
          <Translated
            vi={
              <>
                Phần này cần các dòng <code>step</code> của lab 22.2 cho M3 ở batch 1 và 64. Hãy chạy{' '}
                <code>python3 labs/pmpp.py run 22.2 --bench</code> trên một máy có GPU, với lời giải tham chiếu của lab 15.6 trong kho mã.
              </>
            }
          >
            <>
              This needs lab 22.2's <code>step</code> rows for M3 at batches 1 and 64. Run{' '}
              <code>python3 labs/pmpp.py run 22.2 --bench</code> on a machine with a GPU, with lab 15.6's reference solution
              in the repository.
            </>
          </Translated>
        }
        reading={
          <Translated
            vi={
              <>
                64 nghĩa là bước mất cùng thời gian ở cả hai batch: 63 chuỗi thêm vào đi kèm miễn phí, và một giá trị lệch một chút về
                hai phía là dao động giữa các lần chạy trên một đường phẳng. Thấp hơn hẳn 64, thời gian của bước đã tăng; phần chi tiết
                cho cả hai thời gian, và phần tăng đó là cái giá về độ trễ mà mỗi người dùng trả cho thông lượng của cả nhóm (tr. 539).
              </>
            }
          >
            <>
              64 would mean the step took the same time at both batches: the 63 extra sequences rode for free, and a value
              a little either side of it is run-to-run variation on a flat line. Well below 64, the step's time grew; the
              detail gives both times, and the growth is what each user paid in latency for the group's throughput (p. 539).
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l2}
        filter={(r) => r.name === 'step'}
        cols={[
          { head: 'GEMM', cell: (r) => String(r.gemm ?? '').toUpperCase() }, // i18n-ignore: kept in English
          rawCol(tr('Batch', 'Batch'), 'batch'),
          msCol,
          numCol(tr('Tokens/s', 'Token/s'), 'tokens_per_s'),
          numCol('W GB/s', 'weight_gb_per_s'), // i18n-ignore: unit
          numCol(tr('Model FLOP/B', 'FLOP/B theo mô hình'), 'model_flop_per_byte'),
        ]}
      />
      <Translated
        vi={
          <p>
            Bảng tóm tắt đọc cùng các dòng đó: điểm gãy của mỗi GEMM rơi vào đâu, số token mỗi giây cao nhất và ở batch nào, và phép
            so sánh của sách giữa một bước có batch với cùng các chuỗi đó decode lần lượt.
          </p>
        }
      >
        <p>
          The summary reads the same rows: where each GEMM's knee fell, the most tokens per second and at which batch, and
          the book's comparison of one batched step with the same sequences decoded one after another.
        </p>
      </Translated>
      <BatchingSummary />
      <Translated
        vi={
          <p>
            Từng mảnh đo riêng cho thấy khi nào các phần tăng theo batch bắt đầu đáng kể. Một bước chạy các mảnh linear, bias và
            residual một lần cho mỗi lớp.
          </p>
        }
      >
        <p>
          Each piece timed alone shows when the parts that grow with the batch start to matter. A step runs the linear, bias
          and residual pieces once per layer.
        </p>
      </Translated>
      <PieceTable />
      <Translated
        vi={
          <p>
            <code>llama-batched-bench</code> đo cùng đại lượng đó trên một mô hình thật, và <a href={href('ch22', '22.5')}>mục 22.5</a>{' '}
            giải thích server làm gì với nó: batching liên tục nạp lại batch khi các yêu cầu hoàn tất, nên bước tiếp tục chạy ở kích
            thước batch mà server đã chọn.
          </p>
        }
      >
        <p>
          <code>llama-batched-bench</code> measures the same quantity on a real model, and{' '}
          <a href={href('ch22', '22.5')}>section 22.5</a> explains what a server does with it: continuous batching refills the
          batch as requests finish, so the step keeps running at the batch size the server chose.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch22-facts D3)', '2026-11-20 (ch22-facts, mục D3)')}>
        <Translated
          vi={
            <p>
              <code>llama-batched-bench</code> decode một token từ mọi chuỗi trong mỗi bước; với <code>-tgs</code> nó decode lần lượt
              từng chuỗi.
            </p>
          }
        >
          <p>
            <code>llama-batched-bench</code> decodes one token from every sequence per step; with <code>-tgs</code> it
            decodes the sequences one at a time.
          </p>
        </Translated>
      </Today>
      <ReferenceSolution lab={l2} tiers={TIERS_M8} />

      <h2><Translated vi="Mốc dự án M8">Milestone M8</Translated></h2>
      <Translated
        vi={
          <p>
            Lab 22.2 là mốc dự án đầu tiên chạy trọn một bước decode thay vì một kernel. Nó ghép M0 (lab 2.3), M2 (lab 6.4), M3 (lab
            15.6), M4 (lab 10.3) và gather của lab 22.1 qua một con trỏ hàm cho GEMM, nên một GEMM về sau có thể được đo trong cùng
            bước bằng cách đặt con trỏ đó. Attention của các mốc dự án M6 và M7 (<a href={href('ch20')}>chương 20</a>) và bộ lấy mẫu
            của M5 (<a href={href('x1')}>module X1</a>) là nơi bước này lớn lên tiếp theo. Mốc dự án M10 (
            <a href={href('x3', 'labs')}>các lab của module X3</a>) ghép attention lúc decode của M7, phần gather embedding và bước của
            lab này, cùng RMSNorm, RoPE và SwiGLU của lab X3.1 thành một bước decode kiểu Llama, với bộ lấy mẫu của M5 trên GPU khi có
            lời giải của lab X1.3; M6, kernel prefill, không thuộc về nó.
          </p>
        }
      >
        <p>
          Lab 22.2 is the first milestone that runs a whole decode step rather than one kernel. It composes M0 (lab 2.3), M2
          (lab 6.4), M3 (lab 15.6), M4 (lab 10.3) and lab 22.1's gather through one function pointer for the GEMM, so a later
          GEMM can be timed in the same step by setting that pointer. The attention of milestones M6 and M7 (
          <a href={href('ch20')}>chapter 20</a>) and the sampler of M5 (<a href={href('x1')}>module X1</a>) are where the step
          grows next. Milestone M10 (<a href={href('x3', 'labs')}>module X3&rsquo;s labs</a>) composes M7&rsquo;s decode
          attention, this lab&rsquo;s embedding gather and step, and lab X3.1&rsquo;s RMSNorm, RoPE and SwiGLU into a
          Llama-style decode step, with M5&rsquo;s sampler on a GPU when lab X1.3&rsquo;s solution is present; M6, the
          prefill kernel, is not part of it.
        </p>
      </Translated>
    </LabsPage>
  );
}
