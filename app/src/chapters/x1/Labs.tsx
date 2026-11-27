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
  RaceStatus,
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
} from '../../components/LabKit';
import { Translated, tr } from '../../lib/i18n';
import { fmt, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is module X1's own content.
// No question on this page: the labs are the exercise. Its six predictions are stored, not
// graded: x1/labs/p-lab1-sort, p-lab2-fused, p-lab2-sort, p-lab3-cpu, p-lab3-batch, p-lab4-emu.
// pmpp.py resolves "N.M" only for chapters, so the labs are numbered by path, as appC's are.
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('x1', [
  { num: 'x1/lab1', dir: 'lab1_topk', name: 'top-k by radix select (X1.1)', vi: { name: 'top-k bằng radix select (X1.1)' } },
  { num: 'x1/lab2', dir: 'lab2_filters', name: 'top-p and min-p, composed and fused (X1.2)', vi: { name: 'top-p và min-p, ghép từ các lab và gộp thành một kernel (X1.2)' } },
  { num: 'x1/lab3', dir: 'lab3_sampler', name: 'the M5 sampler (X1.3)', vi: { name: 'sampler M5 (X1.3)' } },
  { num: 'x1/lab4', dir: 'lab4_filterhunt', name: 'filter hunt (X1.4)', vi: { name: 'săn lỗi bộ lọc (X1.4)' } },
]);
const [l1, l2, l3, l4] = LABS;

/** Identifier colours in the reference solutions: where each value lives. */
const tiers = (global: string[], shared: string[], reg: string[]): Record<string, Tier> => ({
  ...Object.fromEntries(global.map((k) => [k, 'global' as Tier])),
  ...Object.fromEntries(shared.map((k) => [k, 'shared' as Tier])),
  ...Object.fromEntries(reg.map((k) => [k, 'reg' as Tier])),
});
const TIERS_1 = tiers(
  ['logitBits', 'scratch', 'outIdx', 'outBits', 'outVal', 'logits', 'row', 'idxRow', 'valRow', 'thr'],
  ['hist_s', 'pick_s', 'warpTies_s', 'warpKeeps_s', 'hist', 'pick', 'group_ties', 'group_keeps'],
  ['prefix', 'mask', 'need', 'key', 'kStar', 'tiesBefore', 'keptBefore', 'slot', 'keep', 'above', 'tie'],
);
const TIERS_2 = tiers(
  ['sortedLogits', 'sortedProbs', 'candLogits', 'sortedPos', 'keptTopP', 'n', 'rowLogits', 'rowProbs'],
  ['stage_s', 'warpSums_s', 'warpCounts_s', 'warpMinP_s', 'warpTopP_s', 'simdSums_s', 'simdCounts_s', 'simdMinP_s', 'simdTopP_s', 'myStage'],
  ['l0', 'threshold', 'total', 'inv', 'carry', 'prob', 'keptMinP', 's'],
);
const TIERS_3 = tiers(
  ['logits', 'counts', 'seeds', 'streams', 'steps', 'tokens', 'uOut', 'uIn', 'workspace', 'sortedLogits', 'sortedPos', 'candIdx', 'n'],
  ['sums_s', 'chunk_s', 'hit_s', 'sums'],
  ['u', 'target', 'total', 'aggregate', 'winner', 'part', 'run', 'l0', 'w'],
);

// ---------------------------------------------------------------- reading the reports
const VOCAB = 151936;
const ms = (s: number | undefined) => (s ? `${fmt(s * 1000)} ms` : '?'); // i18n-ignore: unit
const big = (v: unknown) => (num(v) != null ? num(v)!.toLocaleString('en-US') : '');
const secs = (r: BenchRow | undefined) => num(r?.seconds);

/** Lab X1.1: the full sort's time over the select's, 64 × 151,936, k = 40; batch 1 alongside. */
const sortOverSelect = (reports: Reports) =>
  measure(reports, l1, (s: BenchSet) => {
    const at = (rows: number) => {
      const sel = secs(benchRow(s, 'topk_select', { rows, cols: VOCAB, k: 40 }));
      const sort = secs(benchRow(s, 'full_sort_then_first_k', { rows, cols: VOCAB, k: 40 }));
      return sel && sort ? { sel, sort } : null;
    };
    const m = at(64);
    if (!m) return null;
    const one = at(1);
    return {
      value: m.sort / m.sel,
      detail: tr(
        `the select ${ms(m.sel)} against the sort ${ms(m.sort)} per call, 64 × 151,936, k = 40` +
          (one ? `; at 1 × 151,936 the ratio is ${fmt(one.sort / one.sel)}` : '; no batch-1 row'),
        `phép chọn ${ms(m.sel)} so với phép sắp xếp ${ms(m.sort)} mỗi lần gọi, 64 × 151,936, k = 40` +
          (one ? `; ở 1 × 151,936 tỉ lệ là ${fmt(one.sort / one.sel)}` : '; không có hàng batch 1'),
      ),
    };
  });

/** Lab X1.2: the composed chain after the sort over the fused kernel, 64 × 151,936. */
const fusedSpeedup = (reports: Reports) =>
  measure(reports, l2, (s: BenchSet) => {
    const a = benchRow(s, 'filters_unfused', { rows: 64, cols: VOCAB });
    const b = benchRow(s, 'filters_fused', { rows: 64, cols: VOCAB });
    const ta = secs(a);
    const tb = secs(b);
    if (!ta || !tb) return null;
    return {
      value: ta / tb,
      detail: tr(
        `composed ${ms(ta)} at ${fmt(num(a!.gb_per_s) ?? NaN)} GB/s counted, fused ${ms(tb)} at ${fmt(num(b!.gb_per_s) ?? NaN)} GB/s counted, 64 × 151,936`,
        `chuỗi ghép ${ms(ta)} ở ${fmt(num(a!.gb_per_s) ?? NaN)} GB/s tính theo số byte, kernel gộp ${ms(tb)} ở ${fmt(num(b!.gb_per_s) ?? NaN)} GB/s tính theo số byte, 64 × 151,936`,
      ),
    };
  });

/** Lab X1.2: the sort's share of the composed chain, sort included, 64 × 151,936; K = 40 alongside. */
const sortShare = (reports: Reports) =>
  measure(reports, l2, (s: BenchSet) => {
    const share = (cols: number) => num(benchRow(s, 'chain_composed', { rows: 64, cols })?.sort_share);
    const big = share(VOCAB);
    if (big == null) return null;
    const small = share(40);
    return {
      value: big * 100,
      detail: tr(
        `64 rows of K = 151,936` + (small != null ? `; at K = 40 the sort's share is ${fmt(small * 100)} %` : '; no K = 40 row'),
        `64 hàng với K = 151,936` + (small != null ? `; ở K = 40 tỉ phần của phép sắp xếp là ${fmt(small * 100)} %` : '; không có hàng K = 40'),
      ),
    };
  });

/** Lab X1.3: GPU time per step over one CPU thread's, 1 × 151,936 at llama.cpp's defaults. */
const gpuOverCpu = (reports: Reports) =>
  measure(reports, l3, (s: BenchSet) => {
    const g = secs(benchRow(s, 'sample_step', { rows: 1, vocab: VOCAB }));
    const c = secs(benchRow(s, 'cpu_sample_step', { rows: 1, vocab: VOCAB }));
    if (!g || !c) return null;
    return {
      value: g / c,
      detail: tr(
        `GPU ${ms(g)} per step against ${ms(c)} on one CPU thread, 1 × 151,936, top-k 40`,
        `GPU ${ms(g)} mỗi bước so với ${ms(c)} trên một luồng CPU, 1 × 151,936, top-k 40`,
      ),
    };
  });

/** Lab X1.3: a step at batch 64 over a step at batch 1, both 151,936. */
const batchGrowth = (reports: Reports) =>
  measure(reports, l3, (s: BenchSet) => {
    const one = secs(benchRow(s, 'sample_step', { rows: 1, vocab: VOCAB }));
    const many = secs(benchRow(s, 'sample_step', { rows: 64, vocab: VOCAB }));
    if (!one || !many) return null;
    return {
      value: many / one,
      detail: tr(
        `${ms(many)} per step at batch 64 against ${ms(one)} at batch 1, vocabulary 151,936`,
        `${ms(many)} mỗi bước ở batch 64 so với ${ms(one)} ở batch 1, từ vựng 151,936`,
      ),
    };
  });

const missingBench = (labNum: string) => (
  <Translated
    vi={
      <>
        Phần này cần các hàng benchmark của lab {labNum} từ một thiết bị thật. Hãy chạy{' '}
        <code>python3 labs/pmpp.py run {labNum} --bench</code> khi các kernel của bạn đã qua kiểm thử (hoặc thêm{' '}
        <code>--solution</code> để đo thời gian lời giải tham chiếu). Trình giả lập không bao giờ được đo hiệu năng.
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
const WHAT: Record<string, string> = {
  get topk_select() {
    return tr('your select (both kernels)', 'phép chọn của bạn (cả hai kernel)');
  },
  get full_sort_then_first_k() {
    return tr('lab 14.3 sort, first k', 'phép sắp xếp của lab 14.3, lấy k phần tử đầu');
  },
  get cpu_partial_sort() {
    return tr('std::partial_sort, one CPU thread', 'std::partial_sort, một luồng CPU');
  },
  get sort_rows() {
    return tr('lab 14.3 sort alone', 'riêng phép sắp xếp của lab 14.3');
  },
  get chain_composed() {
    return tr('launchFilterRows, sort included', 'launchFilterRows, tính cả phép sắp xếp');
  },
  get filters_unfused() {
    return tr('composed chain after the sort', 'chuỗi ghép sau phép sắp xếp');
  },
  get filters_fused() {
    return tr('fused kernel after the sort', 'kernel gộp sau phép sắp xếp');
  },
  sample_step: 'launchSampleRows', // i18n-ignore: function name
  get cpu_sample_step() {
    return tr('CPU reference chain, one thread', 'chuỗi tham chiếu trên CPU, một luồng');
  },
  get sample_step_whole_row() {
    return tr('launchSampleRows, top-k 0', 'launchSampleRows, top-k 0');
  },
  get cpu_sample_step_whole_row() {
    return tr('CPU chain, top-k 0, one thread', 'chuỗi trên CPU, top-k 0, một luồng');
  },
};
const what = {
  get head() {
    return tr('What ran', 'Phần được chạy');
  },
  cell: (r: BenchRow) => WHAT[r.name] ?? r.name,
};
const colsCol = (head: string, key: string) => ({ head, num: true, cell: (r: BenchRow) => big(r[key]) });
const pctCol = (head: string, key: string) => ({
  head,
  num: true,
  cell: (r: BenchRow) => (num(r[key]) != null ? fmt(num(r[key])! * 100) : ''),
});

/** Lab X1.3: each stage timed alone, from the stage_0_s .. stage_6_s fields of the step rows. */
const STAGES = () => [
  tr('penalties', 'các penalty'),
  tr('top-k select', 'phép chọn top-k'),
  tr('sort of k', 'sắp xếp k phần tử'),
  tr('softmax', 'softmax'),
  tr('top-p', 'top-p'),
  tr('min-p + combine', 'min-p + kết hợp'),
  tr('draw', 'lần rút'),
];
function StageTable() {
  const { reports, loading } = useLabs();
  const sets = benchSets(reports, l3);
  const rows = sets.flatMap((s) =>
    s.rows.filter((r) => (r.name === 'sample_step' || r.name === 'sample_step_whole_row') && num(r.stage_0_s) != null).map((r) => ({ s, r })),
  );
  if (loading)
    return (
      <p className="muted">
        <Translated vi="Đang đọc báo cáo.">Reading reports.</Translated>
      </p>
    );
  if (!rows.length)
    return (
      <p className="muted" data-stages-empty="x1/lab3">
        <Translated
          vi={
            <>
              Máy này chưa có số đo thời gian từng giai đoạn: chúng có sau khi chạy{' '}
              <code>python3 labs/pmpp.py run x1/lab3 --bench</code>.
            </>
          }
        >
          <>
            No stage timings on this machine yet: they come with <code>python3 labs/pmpp.py run x1/lab3 --bench</code>.
          </>
        </Translated>
      </p>
    );
  return (
    <div className="wide" style={{ maxWidth: '62rem', overflowX: 'auto' }}>
      <table className="data" data-table="stages">
        <thead>
          <tr>
            <th>
              <Translated vi="Backend">Backend</Translated>
            </th>
            <th>
              <Translated vi="Kernel">Kernel</Translated>
            </th>
            <th className="num">
              <Translated vi="Số hàng">Rows</Translated>
            </th>
            <th className="num">
              <Translated vi="Từ vựng">Vocab</Translated>
            </th>
            <th className="num">Top-k</th>
            {STAGES().map((h) => (
              <th key={h} className="num">
                {h}, ms{/* i18n-ignore: unit */}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ s, r }, i) => (
            <tr key={i}>
              <td>{s.backend}</td>
              <td>{s.variant === 'work' ? tr('yours', 'của bạn') : tr('reference solution', 'lời giải tham chiếu')}</td>
              <td className="num">{String(r.rows)}</td>
              <td className="num">{big(r.vocab)}</td>
              <td className="num">{String(r.top_k)}</td>
              {STAGES().map((_, k) => {
                const v = num(r[`stage_${k}_s`]); // i18n-ignore: report field name
                return (
                  <td key={k} className="num">
                    {v == null ? '' : fmt(v * 1000)}
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

/** Lab X1.3: the tokens digest of the decisive draws, per backend and variant. */
// Devices write the full digest (tokens_digest, over decisive_draws of draws) and the core's
// (core_digest, over core_draws); the emulators write the core's only. A run from before the
// core existed wrote tokens_digest alone, over exactly the core's draws: it counts as a core.
type DigestTest = {
  case: string;
  pass: boolean;
  tokens_digest?: string;
  decisive_draws?: number;
  draws?: number;
  core_digest?: string;
  core_draws?: number;
};
type DigestRow = { b: string; variant: 'solution' | 'work'; full?: string; fullDraws?: number; core: string; coreDraws: number };
function digestRows(reports: Reports): DigestRow[] {
  return Object.keys(reports).flatMap((b) =>
    (['solution', 'work'] as const).flatMap((variant): DigestRow[] => {
      const t = (reports[b].results.filter((x) => x.lab === l3.id && x.variant === variant) as RunResult[])
        .flatMap((x) => x.tests as DigestTest[])
        .find((x) => typeof x.core_digest === 'string' || typeof x.tokens_digest === 'string');
      // Only a digest that means something: its case passed, and for your work, the work is not
      // the untouched starter.
      if (!t || !t.pass || (variant === 'work' && reports[b].work_state?.[l3.id] === 'untouched')) return [];
      const legacy = typeof t.core_digest !== 'string';
      const core = legacy ? t.tokens_digest! : t.core_digest!;
      const coreDraws = legacy ? t.decisive_draws ?? 0 : t.core_draws ?? 0;
      const full = !legacy && typeof t.tokens_digest === 'string' && (t.decisive_draws ?? 0) > 0 ? t.tokens_digest : undefined;
      if (coreDraws <= 0 && !full) return [];
      return [{ b, variant, full, fullDraws: full ? t.decisive_draws : undefined, core, coreDraws }];
    }),
  );
}
function DigestTable() {
  const { reports, loading } = useLabs();
  if (loading)
    return (
      <p className="muted">
        <Translated vi="Đang đọc báo cáo.">Reading reports.</Translated>
      </p>
    );
  const found = digestRows(reports);
  const fulls = found.filter((f) => f.full);
  const distinctFull = new Set(fulls.map((f) => f.full)).size;
  const distinctCore = new Set(found.map((f) => f.core)).size;
  const same = distinctFull <= 1 && distinctCore === 1;
  const names = (rows: DigestRow[]) => [...new Set(rows.map((f) => f.b))].join(', ');
  return (
    <>
      {found.length ? (
        <table className="data" data-table="digest">
          <thead>
            <tr>
              <th>
                <Translated vi="Backend">Backend</Translated>
              </th>
              <th>
                <Translated vi="Kernel">Kernel</Translated>
              </th>
              <th>
                <Translated vi="Digest đầy đủ">Full digest</Translated>
              </th>
              <th className="num">
                <Translated vi="Lần rút dứt khoát">Decisive draws</Translated>
              </th>
              <th>
                <Translated vi="Digest phần lõi">Core digest</Translated>
              </th>
              <th className="num">
                <Translated vi="Lần rút dứt khoát">Decisive draws</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            {found.map((f) => (
              <tr key={f.b + f.variant} data-digest-backend={f.b} data-digest-variant={f.variant} data-full={f.full ?? ''} data-core={f.core}>
                <td>{f.b}</td>
                <td>{f.variant === 'work' ? tr('yours', 'của bạn') : tr('reference solution', 'lời giải tham chiếu')}</td>
                <td>
                  {f.full ? (
                    <code>{f.full}</code>
                  ) : (
                    <span className="muted">
                      <Translated vi="chỉ phần lõi">core only</Translated>
                    </span>
                  )}
                </td>
                <td className="num">{f.fullDraws ?? ''}</td>
                <td>
                  <code>{f.core}</code>
                </td>
                <td className="num">{f.coreDraws}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      <p data-digest={found.length ? (same ? 'same' : 'differ') : 'none'}>
        {!found.length
          ? tr(
              'No run of lab X1.3 on this machine has written a tokens digest yet.',
              'Chưa có lần chạy lab X1.3 nào trên máy này ghi ra digest của các token.',
            )
          : same
            ? found.length === 1
              ? tr(
                  'One run has written a digest so far. Run another backend to have something to compare it with.',
                  'Đến giờ mới có một lần chạy ghi ra digest. Hãy chạy thêm một backend khác để có cái mà so sánh.',
                )
              : tr(
                  `Every run listed wrote the same digests: the same decisive tokens in the core on ${names(found)}${fulls.length > 1 ? `, and in the full case on ${names(fulls)}` : ''}.`,
                  `Mọi lần chạy trong bảng đều ghi ra cùng các digest: cùng các token dứt khoát ở phần lõi trên ${names(found)}${fulls.length > 1 ? `, và ở trường hợp đầy đủ trên ${names(fulls)}` : ''}.`,
                )
            : tr(
                `The runs wrote ${distinctFull > 1 ? `${distinctFull} different full digests` : ''}${distinctFull > 1 && distinctCore > 1 ? ' and ' : ''}${distinctCore > 1 ? `${distinctCore} different core digests` : ''}. Some backend drew a different token on a draw the CPU reference calls decisive: compare that backend's failures first.`,
                `Các lần chạy ghi ra ${distinctFull > 1 ? `${distinctFull} digest đầy đủ khác nhau` : ''}${distinctFull > 1 && distinctCore > 1 ? ' và ' : ''}${distinctCore > 1 ? `${distinctCore} digest phần lõi khác nhau` : ''}. Có backend đã rút ra một token khác ở một lần rút mà bản tham chiếu trên CPU coi là dứt khoát: hãy xem các lỗi của backend đó trước.`,
              )}
      </p>
    </>
  );
}

/** Lab X1.4's prediction: how many of the three defects the emulator reports. Not measured from the reports. */
function HuntReveal({ committed }: { committed: number }): ReactNode {
  // The count and the reason come at once; which defect is which waits until the reader says
  // the hunt is over, so the reveal does not hand over the answers before the first run.
  const [fixed, setFixed] = useState(false);
  return (
    <div data-reveal="p-lab4-emu">
      <Translated
        vi={
          <p>
            Khi lab này được xây dựng, <code>emu</code> báo <b>2</b> trong ba lỗi, ở cả ba thứ tự luồng của nó.{' '}
            {committed === 2 ? 'Đúng bằng con số của bạn.' : `Bạn đã dự đoán ${committed}.`}
          </p>
        }
      >
        <p>
          When this lab was built, <code>emu</code> reported <b>2</b> of the three, in all three of its thread orders.{' '}
          {committed === 2 ? 'That is your number.' : `You predicted ${committed}.`}
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Lỗi nó bỏ sót là một race giữa các warp: một lệnh ghi mà trên GPU có thể xảy ra trước khi một warp khác kịp đọc
            giá trị cũ. Trình giả lập chạy mọi luồng tới điểm hẹn kế tiếp rồi mới cho bất kỳ luồng nào đi qua điểm đó, nên
            không warp nào chạy trước warp khác được một quãng. Khi đó, một lệnh đọc nằm trước điểm hẹn đầu tiên và một lệnh
            ghi nằm sau vài điểm hẹn luôn ở đúng thứ tự an toàn, trong mọi thứ tự luồng mà nó thử. Backend race báo một cặp
            dòng như thế; một GPU có thể cho qua nó ở mọi lần chạy.
          </p>
        }
      >
        <p>
          The one it missed is a race between warps: a write that, on a GPU, may land before another warp has read the old
          value. The emulator runs every thread up to its next rendezvous before it lets any thread past one, so no warp is
          ever a stretch ahead of another. A read that comes before the first rendezvous and a write that comes several
          rendezvous later are then always in the safe order, in every thread order it tries. The race backend reports such a
          pair of lines; a GPU may pass it every run.
        </p>
      </Translated>
      {fixed ? (
        <Translated
          vi={
            <p data-reveal-detail="p-lab4-emu">
              Từng lỗi một: <code>emu</code> báo lỗi 1 (khối lượng xác suất tính bao gồm) ở mọi K từ 31 trở lên, và lỗi 2
              (phần mang được cộng sau khi đã quyết định) ở mọi K trên 1,024. Lỗi 3, rào chắn bị thiếu, qua mọi trường hợp
              của <code>emu</code>: mọi luồng đọc tổng của khối trước điểm hẹn cấp warp đầu tiên, còn lệnh ghi tổng của warp
              trong phép scan nằm sau đó vài điểm hẹn.
            </p>
          }
        >
          <p data-reveal-detail="p-lab4-emu">
            Defect by defect: <code>emu</code> reported defect 1 (the inclusive mass) at every K from 31 up, and defect 2 (the
            carry added after the decision) at every K above 1,024. Defect 3, the missing barrier, passed every{' '}
            <code>emu</code> case: every thread reads the block sum before its first warp-level rendezvous, and the scan's
            warp-total write comes several rendezvous later.
          </p>
        </Translated>
      ) : (
        <p>
          <button type="button" onClick={() => setFixed(true)}>
            <Translated
              vi={
                <>
                  Tôi đã sửa cả ba: cho xem <code>emu</code> đã thấy lỗi nào
                </>
              }
            >
              <>
                I have fixed all three: show which <code>emu</code> saw
              </>
            </Translated>
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
              Bốn lab, mỗi lab trong một thư mục riêng dưới <code>labs/x1/</code>. Lab thứ nhất thay phép sắp xếp toàn bộ
              của lab 14.3 bằng một phép chọn cho top-k. Lab thứ hai xây top-p và min-p hai lần, lần đầu ghép từ các lab
              trước, rồi gộp thành một kernel. Lab thứ ba là mốc dự án M5: toàn bộ sampler, nhận logits và số đếm, trả
              ra một token mỗi hàng. Lab thứ tư săn ba lỗi trong kernel gộp của lab thứ hai.
            </>
          }
        >
          <>
            Four labs, each in its own directory under <code>labs/x1/</code>. The first replaces lab 14.3's full sort with a
            selection for top-k. The second builds top-p and min-p twice, composed from earlier labs and then as one kernel.
            The third is milestone M5: the whole sampler, logits and counts in, one token per row out. The fourth hunts three
            defects in the second lab's fused kernel.
          </>
        </Translated>
      </LabsIntro>
      <Translated
        vi={
          <p>
            Học phần này không phải là một chương của sách, nên <code>pmpp.py</code> không biết các lab của nó qua một con
            số như 5.2. Hãy gọi chúng theo đường dẫn: <code>x1/lab1</code> là lab X1.1, và cứ thế đến <code>x1/lab4</code>.
          </p>
        }
      >
        <p>
          This module is not a chapter of the book, so <code>pmpp.py</code> does not know its labs by a number such as 5.2.
          Name them by path: <code>x1/lab1</code> is lab X1.1, and so on to <code>x1/lab4</code>.
        </p>
      </Translated>
      <Beyond title={tr('All of this is ours', 'Tất cả phần này là của khóa học này')}>
        <Translated
          vi={
            <p>
              Sách không có phần lấy mẫu. Mọi hợp đồng, quy tắc phá hòa, thiết kế và kiểm thử trên trang này là của khóa học
              này; các thành phần chúng được xây từ đó là của sách, và mỗi lab nêu tên các mục sách liên quan.
            </p>
          }
        >
          <p>
            The book has no sampling. Every contract, tie rule, design and test on this page is ours; the parts they are
            built from are the book's, and each lab names its sections.
          </p>
        </Translated>
      </Beyond>
      <RunningLabs
        example="x1/lab1"
        alsoShownIn={
          <Translated
            vi={
              <>
                các bảng số đo của <a href={href('x1', 'X1.4')}>X1.4</a> và <a href={href('x1', 'X1.5')}>X1.5</a>
              </>
            }
          >
            <>
              the measured panels of <a href={href('x1', 'X1.4')}>X1.4</a> and <a href={href('x1', 'X1.5')}>X1.5</a>
            </>
          </Translated>
        }
      />
      <BackendsAndEmulator />
      <Translated
        vi={
          <p>
            Mọi lab của học phần này cũng được build cho biến thể race của trình giả lập, vì lab nào cũng có một cập nhật
            chung cần làm đúng: một histogram và số đếm theo từng warp ở lab X1.1, một tổng của khối và một phép scan dùng
            chung một mảng ở lab X1.2 và X1.4, một tổng và các tổng theo đoạn ở lab X1.3.{' '}
            <code>python3 labs/pmpp.py run x1/lab1 --backend race</code> chạy đồng thời các luồng của một khối dưới
            ThreadSanitizer, công cụ báo hai truy cập vào cùng một vị trí mà không có gì sắp thứ tự chúng, kèm cả hai dòng
            mã nguồn. Một lần chạy sạch là bằng chứng, không phải chứng minh: nó chỉ thấy những lịch chạy đã thực sự xảy ra.
          </p>
        }
      >
        <p>
          Every lab of this module is also built for the emulator's race flavour, because each has a shared update to get
          right: a histogram and per-warp counts in lab X1.1, a block sum and a scan sharing one array in labs X1.2 and
          X1.4, a total and chunk totals in lab X1.3. <code>python3 labs/pmpp.py run x1/lab1 --backend race</code> runs a
          block's threads at once under ThreadSanitizer, which reports two accesses to one location with nothing ordering
          them, with both source lines. A clean run is evidence, not proof: it sees only the schedules that happened.
        </p>
      </Translated>
      <VerificationStatus />
      <Translated
        vi={
          <p>
            Ba lab dùng lại lời giải tham chiếu của các lab trước, những file bạn không sửa. Trên CUDA, HIP và các trình giả
            lập, chúng là các bản sao mang tên file riêng, trong <code>labs/x1/shared-g11/</code> (lab X1.1 và X1.3) và{' '}
            <code>labs/x1/shared-g12/</code> (lab X1.2). Bước build đặt tên mỗi file sinh ra theo tên file nguồn, và file
            của lab nào cũng tên là <code>kernel.cu</code>, nên các bản gốc không thể nằm cạnh nhau; thêm nữa, một số hàm phụ
            trợ của chúng không phải <code>static</code>, nên hai bản trong cùng một chương trình giả lập sẽ định nghĩa cùng
            một tên hai lần. Mỗi bản sao chỉ khác bản gốc ở chỗ các hàm phụ trợ đó là <code>static</code>, và một unit test,{' '}
            <code>app/src/chapters/x1/copies-g11.test.ts</code> hoặc <code>copies-g12.test.ts</code>, sẽ thất bại nếu có
            bất cứ chỗ nào khác. Trên Metal, các chương trình điều khiển (driver) nạp chính các file{' '}
            <code>solution/kernel.metal</code> của các lab trước và liên kết <code>launch.mm</code> của chúng. Mỗi lab kiểm
            thử riêng phần mã được cung cấp trước, nên một file được cung cấp mà bị hỏng sẽ lộ lỗi ngay tại chính nó.
          </p>
        }
      >
        <p>
          Three labs reuse earlier labs' reference solutions, which you do not edit. On CUDA, HIP and the emulators they are
          copies under file names of their own, in <code>labs/x1/shared-g11/</code> (labs X1.1 and X1.3) and{' '}
          <code>labs/x1/shared-g12/</code> (lab X1.2). The build names each generated file after its source's name, and
          every lab's file is called <code>kernel.cu</code>, so the originals cannot sit side by side; and some of their
          helper functions are not <code>static</code>, so two of them in one emulator program would define the same name
          twice. Each copy differs from its original only in those helpers being <code>static</code>, and a unit test,{' '}
          <code>app/src/chapters/x1/copies-g11.test.ts</code> or <code>copies-g12.test.ts</code>, fails if anything else
          differs. On Metal the drivers load the earlier labs' own <code>solution/kernel.metal</code> files and link their{' '}
          <code>launch.mm</code>. Each lab tests the provided code on its own first, so a broken provided file shows up as
          itself.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Các bản sao, và phần đếm min-p được cung cấp của lab X1.3, là của khóa học này.</p>}>
          <p>The copies, and the provided min-p count of lab X1.3, are ours.</p>
        </Translated>
      </Beyond>

      {/* ------------------------------------------------------------ X1.1 */}
      <h2>
        <Translated vi="Lab X1.1: top-k bằng radix select, đo so với một phép sắp xếp toàn bộ">
          Lab X1.1: top-k by radix select, measured against a full sort
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một hàm host và hai kernel:{' '}
            <code>launchTopKRows(logits, outIdx, outVal, scratch, rows, cols, k)</code>. Với mỗi hàng của một tensor float32{' '}
            <code>[rows × cols]</code>, theo thứ tự hàng (row-major) với chiều batch đứng đầu, hãy ghi ra min(k, cols) logit
            lớn nhất cùng các cột của chúng, theo thứ tự cột tăng dần, khi hòa ở giá trị thứ k thì các cột thấp
            nhất được chọn, mà không sắp xếp hàng. Lab X1.3 gọi một bản sao lời giải tham chiếu của lab này làm giai đoạn top-k.
          </p>
        }
      >
        <p>
          One host function and two kernels:{' '}
          <code>launchTopKRows(logits, outIdx, outVal, scratch, rows, cols, k)</code>. For every row of a{' '}
          <code>[rows × cols]</code> float32 tensor, row-major with the batch first, write the min(k, cols) largest logits
          and their columns, in ascending column order, every tie at the k-th value going to the lowest columns, without
          sorting the row. Lab X1.3 calls a copy of this lab's reference solution as its top-k stage.
        </p>
      </Translated>
      <Source section="9.4, 12.3, 12.5, 14.9" pages="211–214, 292–293, 297–298, 346" figs="Figs. 9.10 and 12.3" />
      <Translated
        vi={
          <p>
            <code>topkThreshold</code> là quá trình thu hẹp dần của mục <a href={href('x1', 'X1.4')}>X1.4</a>: tối đa bốn
            lượt 8-bit trong một lần khởi chạy, mỗi lượt đếm byte hiện tại của các khóa khớp với tiền tố tới lúc đó vào một
            histogram 256 bin trong bộ nhớ chia sẻ (shared memory) bằng phép toán nguyên tử (atomic) (chương 9), rồi đi qua
            các bin từ trên xuống cho tới khi số đếm đạt số khóa còn cần. Nó để lại khóa thứ k chính xác K* và số giá trị hòa
            ở K* còn cần trong <code>scratch</code>, và dừng sớm khi mọi khóa của bin chứa điểm cắt đều cần lấy.{' '}
            <code>topkGather</code> giữ mọi khóa lớn hơn K* và các giá trị hòa đầu tiên trong phạm vi ngân sách đó, xếp hạng
            bằng một ballot và một popcount trong mỗi warp cùng các số đếm cộng dồn qua các vòng 256 cột: bộ lọc ổn định của
            lab 12.4 kèm một ngân sách cho giá trị hòa. Các kernel không bao giờ làm phép tính số thực; hàm khởi chạy đưa
            logits cho chúng dưới dạng số nguyên không dấu, như lab 14.3 làm.
          </p>
        }
      >
        <p>
          <code>topkThreshold</code> is section <a href={href('x1', 'X1.4')}>X1.4</a>'s narrowing in: up to four 8-bit
          passes in one launch, each counting the current byte of the keys that match the prefix so far in a 256-bin
          shared-memory histogram with atomics (chapter 9), then walking the bins from the top until the count reaches the
          keys still needed. It leaves the exact k-th key K* and the number of ties at K* still needed in{' '}
          <code>scratch</code>, and stops early when every key of the crossing bin is needed. <code>topkGather</code> keeps
          every key above K* and the first ties up to that budget, ranked by a ballot and a popcount within each warp and
          running counts across rounds of 256 columns: lab 12.4's stable filter with a tie budget. The kernels never do
          float arithmetic; the launcher hands them the logits as unsigned integers, as lab 14.3 does.
        </p>
      </Translated>
      <LabFiles
        lab={l1}
        files={[
          ['kernel.cu', <Translated vi={<><code>topkThreshold</code>, <code>topkGather</code> và <code>launchTopKRows</code>; các hàm phụ trợ warp của lab 12.4 cùng lane mask có kiểu của chúng, và khóa của lab 14.3, được cung cấp sẵn trong file</>}><><code>topkThreshold</code>, <code>topkGather</code> and <code>launchTopKRows</code>; lab 12.4's warp helpers with their typed lane mask, and lab 14.3's key, are provided in the file</></Translated>],
          ['kernel.metal', <Translated vi={<><code>topk_threshold</code> và <code>topk_gather</code>, histogram là một mảng <code>threadgroup atomic_uint</code></>}><><code>topk_threshold</code> and <code>topk_gather</code>, the histogram a <code>threadgroup atomic_uint</code> array</></Translated>],
          ['launch.mm', <Translated vi={<><code>encodeTopKRows</code>, cả hai kernel trong cùng một compute encoder được truyền vào</>}><><code>encodeTopKRows</code>, both kernels in the one compute encoder it is given</></Translated>],
        ]}
      />
      <Today checked="2026-11-20 (notes/today/infra-facts.md B3; ch12-facts.md B1, C2; ch04-facts.md C2)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              Phép vote của HIP là 64 bit trên mọi target AMD và các hàm <code>_sync</code> của nó nhận một mask 64 bit.{' '}
              <code>simd_ballot</code> của Metal để các bit phía trên độ rộng SIMD-group ở trạng thái không xác định, và
              cách một threadgroup được chia thành các SIMD-group là do bản cài đặt quyết định.
            </p>
          }
        >
          <p>
            HIP's vote is 64 bits on every AMD target and its <code>_sync</code> functions take a 64-bit mask. Metal's{' '}
            <code>simd_ballot</code> leaves the bits above the SIMD-group width undefined, and how a threadgroup splits into
            SIMD-groups is the implementation's.
          </p>
        </Translated>
      </Today>

      <h3>
        <Translated vi="Các kiểm thử kiểm tra gì">What the tests check</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Hãy chạy <code>python3 labs/pmpp.py run x1/lab1 --backend emu</code> trước (vài phút). Trên một thiết bị: 1, 4 và
            7 hàng nhân với 1, 2, 31, 255, 256, 257, 1,000, 4,096, 32,000, 50,257 và 151,936 cột, với k = 1, 2, 40, 256,
            1,000, độ dài hàng và độ dài hàng cộng 5. Dữ liệu được trộn trong mỗi batch để một độ lệch hàng sai sẽ thất bại:
            logits đều, các hàng có đỉnh nhọn, mọi logit bằng nhau, các giá trị hòa chính xác ở giá trị thứ k được cài ngang
            qua ranh giới warp và ranh giới vòng, −0.0 và +0.0 xen kẽ, các khóa chung 8, 16 hoặc 24 bit đầu (buộc phải chạy 2,
            3 và 4 lượt), bin chứa điểm cắt ở hai đầu của lượt đi, và các hàng chỉ có số âm. Phép so sánh là chính xác: các cột
            bằng của bản tham chiếu, các giá trị là bit của logits, các ô sau min(k, cols) giữ giá trị canh của chương trình
            điều khiển, và 64 ô canh sau mỗi đầu ra không bị động tới. Các trình giả lập chạy kích thước nhỏ hơn; README liệt
            kê chúng. Phép sắp xếp của lab 14.3, đường cơ sở, cũng được kiểm thử.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run x1/lab1 --backend emu</code> first (a few minutes). On a device: rows 1, 4 and 7
          by columns 1, 2, 31, 255, 256, 257, 1,000, 4,096, 32,000, 50,257 and 151,936, with k = 1, 2, 40, 256, 1,000, the
          row length and the row length plus 5. The data are mixed within each batch so that a wrong row offset fails:
          uniform logits, peaked rows, every logit equal, exact ties at the k-th value planted across warp and round seams,
          interleaved −0.0 and +0.0, keys sharing their top 8, 16 or 24 bits (which force 2, 3 and 4 passes), crossing bins
          at either end of the walk, and negative-only rows. The comparison is exact: the columns equal the reference's, the
          values are the logits' bits, slots past min(k, cols) keep the driver's sentinel, and 64 guard entries past every
          output are untouched. The emulators run smaller shapes; the README lists them. Lab 14.3's sort, the baseline, is
          tested too.
        </p>
      </Translated>
      <RaceStatus lab={l1} />

      <h3>
        <Translated vi="Trước khi đo hiệu năng">Before you benchmark</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Mục X1.4 đếm số byte mỗi thiết kế phải di chuyển cho mỗi khóa: 80 cho bốn lượt của phép sắp xếp, 20 cho phép
            chọn cộng 8 cho mỗi cặp được giữ, tức lưu lượng ít hơn khoảng bốn lần. Hãy quyết định tỉ lệ thời gian sẽ cao hơn
            hay thấp hơn con số đó, và vì sao. Cân nhắc ba điều: phép phân tán của phép sắp xếp đi tới 256 thùng, nên các
            lệnh ghi của nó được hợp nhất kém; ở các lượt sau, các phép atomic của phép chọn đều rơi vào vài bin gần điểm
            cắt; và cả hai đều chạy mỗi hàng một khối.
          </p>
        }
      >
        <p>
          Section X1.4 counts the bytes each design must move per key: 80 for the sort's four passes, 20 for the select plus
          8 for each kept pair, so about four times less traffic. Decide whether the time ratio will be above or below that,
          and why. Weigh three things: the sort's scatter goes to 256 buckets, so its writes are poorly coalesced; in the
          later passes the select's atomics all land on the few bins near the crossing; and both run one block per row.
        </p>
      </Translated>
      <LabPrediction
        id="x1/labs/p-lab1-sort"
        prompt={
          <Translated
            vi={
              <>
                Ở 64 hàng gồm 151,936 logit và k = 40, phép chọn của bạn sẽ nhanh hơn phép sắp xếp toàn bộ của lab 14.3 bao
                nhiêu lần?
              </>
            }
          >
            <>At 64 rows of 151,936 logits and k = 40, how many times faster than lab 14.3's full sort will your select be?</>
          </Translated>
        }
        unit={tr("× the sort's speed", '× tốc độ của phép sắp xếp')}
        measure={sortOverSelect}
        missing={missingBench('x1/lab1')}
        reading={
          <Translated
            vi={
              <>
                Cả hai thời gian đều là của thiết bị, mỗi lần gọi. Số byte cho biết mỗi thiết kế phải đòi hỏi gì ở bộ nhớ;
                tỉ lệ là những gì thiết bị làm được với điều đó. Bảng dưới đây có mọi kích thước, và tỉ lệ ở batch 1 trong
                phần chi tiết là trường hợp một khối làm việc còn phần còn lại của thiết bị ngồi chờ.
              </>
            }
          >
            <>
              Both times are the device's, per call. The byte count says what each design must ask of memory; the ratio is
              what the device made of it. The table below has every shape, and the batch-1 ratio in the detail is the case
              where one block works and the rest of the device waits.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l1}
        cols={[
          what,
          rawCol(tr('Rows', 'Số hàng'), 'rows'),
          colsCol(tr('Cols', 'Số cột'), 'cols'),
          rawCol('k', 'k'),
          msCol,
          numCol(tr('Rows/s', 'Hàng/s'), 'rows_per_s'),
          numCol(tr('Bytes per key', 'Byte mỗi khóa'), 'bytes_per_key'),
          numCol(tr('Sort / select', 'Sắp xếp / chọn'), 'sort_over_select'),
          numCol(tr('CPU / select', 'CPU / chọn'), 'cpu_over_select'),
        ]}
      />
      <Translated
        vi={
          <p>
            Cách benchmark hoạt động: 1, 8 và 64 hàng nhân với 32,000 và 151,936 cột, k = 1, 40 và 1,024. Phép chọn của bạn
            và phép sắp xếp của lab 14.3 được đo thời gian trên thiết bị, lấy kết quả tốt nhất trong 5 lần chạy, mỗi lần lặp
            lại lời gọi cho tới khi kéo dài ít nhất 2 ms (20 ms với phép sắp xếp). Một luồng CPU chạy{' '}
            <code>std::partial_sort</code> cho mỗi hàng, đó là top-k phía host của llama.cpp với k tới 128, đo bằng một đồng
            hồ đơn điệu. Mọi kết quả được đo đều được kiểm tra trước khi báo cáo.
          </p>
        }
      >
        <p>
          How the benchmark works: rows 1, 8 and 64 by 32,000 and 151,936 columns, k = 1, 40 and 1,024. Your select and lab
          14.3's sort are timed on the device, best of 5 runs, each repeating the call until it lasts at least 2 ms (20 ms
          for the sort). One CPU thread runs <code>std::partial_sort</code> per row, which is llama.cpp's host top-k for k up
          to 128, timed with a steady clock. Every timed result is checked before it is reported.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md D3)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              Top-k phía host của llama.cpp là một <code>std::partial_sort</code> với k tới 128.
            </p>
          }
        >
          <p>llama.cpp's host top-k is a <code>std::partial_sort</code> for k up to 128.</p>
        </Translated>
      </Today>
      <ReferenceSolution lab={l1} tiers={TIERS_1} />

      {/* ------------------------------------------------------------ X1.2 */}
      <h2>
        <Translated vi="Lab X1.2: top-p và min-p, ghép từ các lab trước, rồi gộp thành một kernel">
          Lab X1.2: top-p and min-p, composed from earlier labs, then fused
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            Phần A ghép. <code>launchFilterRows</code> sắp xếp mỗi hàng gồm K logit ứng viên từ lớn đến nhỏ và ổn định bằng
            lab 14.3, tính softmax ở nhiệt độ 1 bằng lab 10.3, đếm số phần tử còn lại sau top-p bằng quy tắc giữ của lab 11.6
            (giữ token i khi i là 0, hoặc p ít nhất là 1, hoặc khối lượng xác suất xếp trước i nhỏ hơn p), và đếm số phần tử
            còn lại sau min-p bằng <code>minPCount</code> của bạn: các logit đã sắp xếp lớn hơn hoặc bằng logit đầu tiên cộng
            ln(min-p), logarit được tính một lần trên host. Sau đó n = max(1, min(số đếm của top-p, số đếm của min-p)). Đó là
            thứ tự của llama.cpp, top-p rồi min-p, và lấy giá trị nhỏ nhất là đúng vì cả hai đều giữ một tiền tố của hàng đã
            sắp xếp (mục <a href={href('x1', 'X1.6')}>X1.6</a>).
          </p>
        }
      >
        <p>
          Part A composes. <code>launchFilterRows</code> sorts each row of K candidate logits largest first and stably with
          lab 14.3, takes the softmax at temperature 1 with lab 10.3, counts the top-p survivors with lab 11.6's keep rule
          (keep token i when i is 0, or p is at least 1, or the mass ranked before i is below p), and counts the min-p
          survivors with your <code>minPCount</code>: sorted logits at least the first plus ln(min-p), the logarithm computed
          once on the host. Then n = max(1, min(top-p's count, min-p's count)). That is llama.cpp's order, top-p then min-p,
          and taking the minimum works because both keep a prefix of the sorted row (section{' '}
          <a href={href('x1', 'X1.6')}>X1.6</a>).
        </p>
      </Translated>
      <Source section="10.7–10.8, 11.1, 11.4" pages="239–244, 252–254, 261–265" figs="Figs. 10.13, 10.16, 11.8 and 11.9" />
      <Translated
        vi={
          <p>
            Phần B gộp. <code>launchFilterSortedFused</code> bắt đầu từ các hàng đã được sắp xếp, nên giá trị lớn nhất là phần
            tử 0. Mỗi hàng một khối 256 luồng cộng các exp(l − l₀) bằng phép rút gọn (reduction) theo khối của chương 10, rồi
            đi qua hàng theo từng đoạn 1,024 phần tử bằng phép scan theo khối của chương 11, với khối lượng xác suất của các
            đoạn trước được mang theo trong một thanh ghi (register). Nó tính từng xác suất, quyết định top-p và min-p khi khối
            lượng xác suất và logit còn nằm trong thanh ghi, đếm số phần tử còn lại, rồi ghi ra các xác suất và n. Nó đọc
            logits hai lần và ghi các xác suất một lần.
          </p>
        }
      >
        <p>
          Part B fuses. <code>launchFilterSortedFused</code> starts from rows that are already sorted, so the maximum is
          element 0. One block of 256 threads per row adds up exp(l − l₀) with chapter 10's block reduction, then walks the
          row in chunks of 1,024 with chapter 11's block scan and the mass of earlier chunks carried in a register. It
          computes each probability, decides top-p and min-p while the mass and the logit are in registers, counts the
          survivors, and writes the probabilities and n. It reads the logits twice and writes the probabilities once.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Việc gộp và lập luận về tiền tố là của khóa học này. Thứ tự lọc và cả hai quy tắc là của llama.cpp, đã được
              kiểm tra trong lượt rà soát dữ kiện (notes/today/x1-facts.md D1; ch11-facts.md D2; ch12-facts.md D3).
            </p>
          }
        >
          <p>
            The fusion and the prefix argument are ours. The filter order and both rules are llama.cpp's, checked in the
            facts pass (notes/today/x1-facts.md D1; ch11-facts.md D2; ch12-facts.md D3).
          </p>
        </Translated>
      </Beyond>
      <LabFiles
        lab={l2}
        files={[
          ['kernel.cu', <Translated vi={<><code>minPCount</code>, <code>launchMinPCount</code>, <code>launchFilterRows</code>, <code>filterSortedFused</code> và <code>launchFilterSortedFused</code>; phép cộng warp, phép scan warp và phép cộng khối được cung cấp sẵn</>}><><code>minPCount</code>, <code>launchMinPCount</code>, <code>launchFilterRows</code>, <code>filterSortedFused</code> and <code>launchFilterSortedFused</code>; the warp sum, warp scan and block sum are provided</></Translated>],
          ['kernel.metal', <Translated vi={<><code>min_p_count</code> và <code>filter_sorted_fused</code></>}><><code>min_p_count</code> and <code>filter_sorted_fused</code></></Translated>],
          ['launch.mm', <Translated vi={<><code>encodeMinPCount</code>, <code>encodeFilterRows</code> và <code>encodeFilterSortedFused</code></>}><><code>encodeMinPCount</code>, <code>encodeFilterRows</code> and <code>encodeFilterSortedFused</code></></Translated>],
        ]}
      />

      <h3>
        <Translated vi="Các kiểm thử kiểm tra gì">What the tests check</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Cả hai phần chạy trên cùng đầu vào. Các hàng chính xác chứa các nhóm 2ʲ logit bằng 0, phần còn lại ở −1000, nên
            mọi xác suất là một lũy thừa của hai hoặc 0, và mọi tổng riêng phần đều chính xác với bất kỳ thứ tự cộng nào. p nằm
            đúng trên tổng bao gồm tại một token: ở token 0, bên trong một dãy bốn phần tử của một lane, ở token cuối cùng của
            đoạn 1,024 đầu tiên, và sau đó. Token đó phải được giữ và token kế tiếp bị loại. Một logit được cài đúng bằng 0 +
            ln(min-p) theo cách host tính phải được giữ. Trên các hàng có dạng softmax, các xác suất phải nằm trong sai số
            tương đối 1e-4 so với một bản tham chiếu double, và n phải bằng của bản tham chiếu ở bất cứ đâu không có khối
            lượng xác suất hay logit nào nằm trong dải quanh một ranh giới; bên trong dải, giá trị lân cận nào cũng được chấp
            nhận. README suy ra độ rộng của dải.
          </p>
        }
      >
        <p>
          Both parts run on the same inputs. The exact rows hold groups of 2ʲ logits of 0 with the rest at −1000, so every
          probability is a power of two or 0 and every partial sum is exact in any order of addition. p sits exactly on the
          inclusive sum at one token: at token 0, inside a lane's run of four, at the last token of the first chunk of 1,024,
          and past it. That token must be kept and the next one dropped. A logit planted exactly at 0 + ln(min-p) as the
          host computes it must be kept. On softmax-shaped rows the probabilities must be within 1e-4 relative of a double
          reference, and n must equal the reference's wherever no mass and no logit lies within the band of a boundary;
          inside the band either neighbour is accepted. The README derives the band.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Các hàng chính xác dựa vào việc exp(0) đúng bằng 1 và exp(−1000) đúng bằng 0. Giới hạn độ chính xác vài ulp của
            các nhà cung cấp không hứa hẹn điều nào trong hai điều đó. Cả hai đều đúng trên Metal và trên các trình giả lập khi
            lab này được xây dựng; trên CUDA và HIP thì chúng unverified.
          </p>
        }
      >
        <p>
          The exact rows rely on exp(0) being exactly 1 and exp(−1000) exactly 0. Vendors' accuracy bounds of a few ulp do not
          promise either. Both held on Metal and on the emulators when this lab was built; on CUDA and HIP they are
          unverified.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch10-facts.md A11, B4, C4)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              <code>expf</code> của CUDA sai lệch trong phạm vi 2 ulp; <code>exp</code> nhanh của Metal trong phạm vi 3 +
              floor(|2x|) ulp; HIP công bố các sai khác đo được, không phải giới hạn.
            </p>
          }
        >
          <p>
            CUDA's <code>expf</code> is within 2 ulp; Metal's fast <code>exp</code> within 3 + floor(|2x|) ulp; HIP publishes
            measured differences, not bounds.
          </p>
        </Translated>
      </Today>
      <RaceStatus lab={l2} />

      <h3>
        <Translated vi="Trước khi đo hiệu năng">Before you benchmark</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Hãy đếm số byte mỗi token trước. Sau phép sắp xếp, softmax của chuỗi ghép đọc logits ba lần và ghi các xác suất,
            lab 11.6 đọc các xác suất và ghi khối lượng xác suất cùng một byte giữ/bỏ, còn min-p đọc logits thêm một lần nữa:
            29 byte. Kernel gộp đọc logits hai lần và ghi các xác suất một lần: 12. Sau đó hãy quyết định xem cả hai có chạy ở
            cùng băng thông hay không. Với dự đoán thứ hai, hãy tự hỏi lab 14.3 làm bao nhiêu việc cho mỗi hàng bất kể K là
            bao nhiêu, và các bộ lọc làm bao nhiêu; trình khám phá của mục <a href={href('x1', 'X1.5')}>X1.5</a> là chỗ để
            suy luận về điều đó.
          </p>
        }
      >
        <p>
          Count the bytes per token first. After the sort, the composed chain's softmax reads the logits three times and
          writes the probabilities, lab 11.6 reads the probabilities and writes the mass and a keep byte, and min-p reads the
          logits again: 29 bytes. The fused kernel reads the logits twice and writes the probabilities once: 12. Then decide
          whether both run at the same bandwidth. For the second prediction, ask how much work lab 14.3 does per row whatever
          K is, and how much the filters do; section <a href={href('x1', 'X1.5')}>X1.5</a>'s explorer is the place to reason
          about it.
        </p>
      </Translated>
      <LabPrediction
        id="x1/labs/p-lab2-fused"
        prompt={
          <Translated
            vi={<>Ở 64 hàng gồm 151,936 phần tử, kernel gộp sẽ nhanh hơn chuỗi ghép sau phép sắp xếp bao nhiêu lần?</>}
          >
            <>At 64 rows of 151,936, how many times faster than the composed chain after the sort will the fused kernel be?</>
          </Translated>
        }
        unit={tr("× the composed chain's speed", '× tốc độ của chuỗi ghép')}
        measure={fusedSpeedup}
        missing={missingBench('x1/lab2')}
        reading={
          <Translated
            vi={
              <>
                GB/s ở đây là tính theo số byte, không phải đo được: số byte tối thiểu mỗi thiết kế phải di chuyển, chia cho
                thời gian của nó. Nếu cả hai chạy ở cùng băng thông tính theo số byte, mức tăng tốc sẽ bằng tỉ lệ của các số
                byte.
              </>
            }
          >
            <>
              The GB/s are counted, not measured: the minimum bytes each design must move, divided by its time. If both ran at
              the same counted bandwidth, the speed-up would be the ratio of the byte counts.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="x1/labs/p-lab2-sort"
        prompt={
          <Translated
            vi={
              <>
                Ở 64 hàng với K = 151,936, phép sắp xếp sẽ chiếm bao nhiêu phần trăm thời gian của chuỗi ghép, tính cả phép
                sắp xếp?
              </>
            }
          >
            <>At 64 rows of K = 151,936, what percentage of the composed chain's time, sort included, will the sort take?</>
          </Translated>
        }
        unit={tr("% of the composed chain's time", '% thời gian của chuỗi ghép')}
        measure={sortShare}
        missing={missingBench('x1/lab2')}
        reading={
          <Translated
            vi={
              <>
                Phần chi tiết cho cùng tỉ phần đó ở K = 40, top-k mặc định của llama.cpp. Việc tỉ phần có giảm khi K nhỏ đi
                hay không cho biết bao nhiêu phần chi phí của phép sắp xếp phụ thuộc vào K và bao nhiêu là cố định cho mỗi
                hàng.
              </>
            }
          >
            <>
              The detail gives the same share at K = 40, llama.cpp's default top-k. Whether the share falls as K shrinks says
              how much of the sort's cost depends on K and how much is fixed per row.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l2}
        cols={[
          what,
          rawCol(tr('Rows', 'Số hàng'), 'rows'),
          colsCol('K', 'cols'),
          msCol,
          numCol(tr('Counted GB/s', 'GB/s tính theo byte'), 'gb_per_s'),
          rawCol(tr('Bytes per token', 'Byte mỗi token'), 'bytes_per_token'),
          pctCol(tr('Sort share, %', 'Tỉ phần sắp xếp, %'), 'sort_share'),
          numCol(tr('Fused speed-up', 'Tăng tốc khi gộp'), 'speedup_vs_unfused'),
        ]}
      />
      <Translated
        vi={
          <p>
            Cách benchmark hoạt động: 1 và 64 hàng nhân với K = 40, 1,024, 32,000 và 151,936, trên các hàng có dạng softmax ở
            top-p 0.95 và min-p 0.05 mặc định của llama.cpp. Nó đo thời gian riêng phép sắp xếp của lab 14.3, toàn bộ{' '}
            <code>launchFilterRows</code> của bạn, chuỗi ghép sau phép sắp xếp, và kernel gộp trên cùng các logit đã sắp
            xếp. Lấy kết quả tốt nhất trong 5 lần đo, mỗi lần lặp lại lời gọi cho tới khi kéo dài ít nhất 2 ms; mọi kết quả
            được đo đều được kiểm tra trước. Một kích thước 1 hàng chỉ tối đa 600 KB mỗi buffer, nên hãy đọc băng thông ở các
            kích thước 64 hàng.
          </p>
        }
      >
        <p>
          How the benchmark works: rows 1 and 64 by K = 40, 1,024, 32,000 and 151,936, on softmax-shaped rows at llama.cpp's
          default top-p 0.95 and min-p 0.05. It times lab 14.3's sort alone, your whole <code>launchFilterRows</code>, the
          composed chain after the sort, and the fused kernel on the same sorted logits. Best of 5 timed runs, each repeating
          the call until it lasts at least 2 ms; every timed result is checked first. A 1-row shape is at most 600 KB per
          buffer, so read the 64-row shapes for bandwidth.
        </p>
      </Translated>
      <ReferenceSolution lab={l2} tiers={TIERS_2} />

      {/* ------------------------------------------------------------ X1.3 */}
      <h2>
        <Translated vi="Lab X1.3: mốc dự án M5, một sampler GPU theo batch">
          Lab X1.3: milestone M5, a batched GPU sampler
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            <code>launchSampleRows</code> nhận logits <code>[rows × vocab]</code> và số đếm token, một seed, một stream và
            một bước cho mỗi hàng, cùng các thiết lập (top-k, top-p, min-p, nhiệt độ, và các penalty repetition, frequency và
            presence), rồi ghi ra một token id cho mỗi hàng. Mọi giai đoạn trừ ba đều là kernel của một lab trước. Bạn viết
            ba giai đoạn mà chưa lab nào có, cùng phần ghép nối.
          </p>
        }
      >
        <p>
          <code>launchSampleRows</code> takes <code>[rows × vocab]</code> logits and token counts, a seed, a stream and a
          step per row, and the settings (top-k, top-p, min-p, temperature, and the repetition, frequency and presence
          penalties), and writes one token id per row. Every stage but three is an earlier lab's kernel. You write the three
          that none of them has, and the composition.
        </p>
      </Translated>
      <ul>
        <Translated
          vi={
            <li>
              <b>Philox4x32-10</b>, mười vòng của Random123, với tích 32 × 32 ra 64 bit là một phép nhân{' '}
              <code>unsigned long long</code> (<code>ulong</code> trong Metal), và một kernel ghi ra bốn word cho các bộ đếm và
              khóa cho trước, để chương trình điều khiển kiểm tra các vector đã công bố trên thiết bị (mục{' '}
              <a href={href('x1', 'X1.7')}>X1.7</a>).
            </li>
          }
        >
          <li>
            <b>Philox4x32-10</b>, Random123's ten rounds, with the 32 × 32 to 64-bit product as an{' '}
            <code>unsigned long long</code> multiplication (<code>ulong</code> in Metal), and a kernel that writes the four
            words for given counters and keys, so the driver checks the published vectors on the device (section{' '}
            <a href={href('x1', 'X1.7')}>X1.7</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>u</b> = (w &gt;&gt; 8) × 2⁻²⁴, là cùng một số float trên mọi backend.
            </li>
          }
        >
          <li>
            <b>u</b> = (w &gt;&gt; 8) × 2⁻²⁴, which is the same float on every backend.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Repetition penalty</b>, quy tắc của llama.cpp: với một token đã xuất hiện, nhân logit nhỏ hơn hoặc bằng 0 với
              hệ số phạt và chia logit dương cho nó (mục <a href={href('x1', 'X1.3')}>X1.3</a>).
            </li>
          }
        >
          <li>
            <b>The repetition penalty</b>, llama.cpp's rule: for a token already seen, multiply a logit at or below 0 by the
            penalty and divide a positive one (section <a href={href('x1', 'X1.3')}>X1.3</a>).
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Lần rút</b>, mỗi hàng một khối trên n phần tử còn lại: trọng số exp((lᵢ − l₀) / T), tổng của chúng bằng một
              phép rút gọn theo khối, đích u × total, rồi đi theo từng đoạn với một tổng cộng dồn; vị trí đầu tiên có tổng
              bao gồm lớn hơn hẳn đích sẽ thắng, và là trọng số dương cuối cùng nếu sai số làm tròn không để lại vị trí nào
              (mục <a href={href('x1', 'X1.8')}>X1.8</a>).
            </li>
          }
        >
          <li>
            <b>The draw</b>, one block per row over the n survivors: weights exp((lᵢ − l₀) / T), their total by a block
            reduction, target u × total, then chunks with a running sum; the first position whose inclusive sum is strictly
            greater than the target wins, and the last positive weight if rounding leaves none (section{' '}
            <a href={href('x1', 'X1.8')}>X1.8</a>).
          </li>
        </Translated>
      </ul>
      <Today checked="2026-11-20 (notes/today/x1-facts.md A4, A5, B3, D1, D2, E2)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              Ba vector đáp án đã biết cho 10 vòng của Random123; <code>__umulhi</code> có trên CUDA và HIP, và lab dùng tích
              64 bit, thứ không cần intrinsic nào trên backend nào. Chuỗi mặc định và quy tắc repetition của llama.cpp. Một phép
              chuyển đổi 24 bit không bao giờ trả về 1.0.
            </p>
          }
        >
          <p>
            Random123's three 10-round known-answer vectors; <code>__umulhi</code> exists on CUDA and HIP, and the lab uses
            the 64-bit product, which needs no intrinsic on any backend. llama.cpp's default chain and repetition rule. A
            24-bit conversion never returns 1.0.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Trình giả lập và stub front-end của khóa học này cũng khai báo <code>__umulhi</code> (xem{' '}
            <code>labs/common/include/pmpp/emu/emu_device.h</code> và <code>pmpp/check/cuda_stub.h</code>), nên dạng nào cũng
            build được ở bất cứ đâu một file <code>.cu</code> build được.
          </p>
        }
      >
        <p>
          This course's emulator and front-end stub declare <code>__umulhi</code> too (see{' '}
          <code>labs/common/include/pmpp/emu/emu_device.h</code> and <code>pmpp/check/cuda_stub.h</code>), so either form
          builds wherever a <code>.cu</code> file does.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Chuỗi này là chuỗi mặc định của llama.cpp, giới hạn ở các giai đoạn mà M5 xây: chép logits vào vùng làm việc,
            repetition penalty, rồi lượt frequency và presence của lab 9.3; nếu nhiệt độ bằng 0 hoặc thấp hơn thì dùng argmax
            của lab 10.3 và dừng; ngược lại là phép chọn của lab X1.1 (cả hàng khi top-k bằng 0 hoặc ít nhất bằng kích thước
            từ vựng), phép sắp xếp các ứng viên của lab 14.3, softmax của lab 10.3, phép đếm top-p của lab 11.6, phép đếm và
            kết hợp min-p được cung cấp sẵn, và lần rút. Ở mọi giai đoạn, khi hòa thì token id thấp nhất được chọn. Mục{' '}
            <a href={href('x1', 'X1.1')}>X1.1</a> nêu thứ tự này và lý do của nó.
          </p>
        }
      >
        <p>
          The chain is llama.cpp's default restricted to the stages M5 builds: copy the logits into the workspace,
          repetition, then lab 9.3's frequency and presence pass; if the temperature is 0 or below, lab 10.3's argmax and
          stop; otherwise lab X1.1's select (the whole row when top-k is 0 or at least the vocabulary), lab 14.3's sort of the
          candidates, lab 10.3's softmax, lab 11.6's top-p count, the provided min-p count and combine, and the draw. Every
          tie at every stage goes to the lowest token id. Section <a href={href('x1', 'X1.1')}>X1.1</a> gives the order and
          its reasons.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Thiết kế của chúng tôi: bộ (seed, stream, step) cho mỗi hàng, cách ánh xạ bộ đếm và khóa, phép chuyển đổi u,
              quy tắc rút và bố cục vùng làm việc. Theo cách chúng tôi đọc header của cuRAND, cách ánh xạ này bằng lời gọi{' '}
              <code>curand()</code> đầu tiên sau <code>curand_init(seed, stream, 4 × step)</code>; điều đó unverified trên
              thiết bị.
            </p>
          }
        >
          <p>
            Our design: the per-row (seed, stream, step), the counter and key mapping, the u conversion, the draw rule and
            the workspace layout. By our reading of cuRAND's header the mapping equals the first <code>curand()</code> after{' '}
            <code>curand_init(seed, stream, 4 × step)</code>; that is unverified on a device.
          </p>
        </Translated>
      </Beyond>
      <LabFiles
        lab={l3}
        files={[
          ['kernel.cu', <Translated vi={<><code>philox4x32_10</code>, <code>philoxKatKernel</code>, <code>uniformFromBits</code>, <code>applyRepetition</code>, <code>drawRows</code>, <code>sampleWorkspaceBytes</code> và <code>launchSampleRows</code>; các hàm khởi chạy khác và phép sao chép vào vùng làm việc được cung cấp sẵn</>}><><code>philox4x32_10</code>, <code>philoxKatKernel</code>, <code>uniformFromBits</code>, <code>applyRepetition</code>, <code>drawRows</code>, <code>sampleWorkspaceBytes</code> and <code>launchSampleRows</code>; the other launchers and the workspace copy are provided</></Translated>],
          ['kernel.metal', <Translated vi={<><code>philox4x32_10</code>, <code>philox_kat</code>, <code>uniform_from_bits</code>, <code>apply_repetition</code> và <code>draw_rows</code></>}><><code>philox4x32_10</code>, <code>philox_kat</code>, <code>uniform_from_bits</code>, <code>apply_repetition</code> and <code>draw_rows</code></></Translated>],
          ['launch.mm', <Translated vi={<><code>encodePhiloxKat</code>, <code>encodeApplyRepetition</code>, <code>encodeDrawRows</code> và <code>encodeSampleRows</code>; trên Metal, vùng làm việc là một buffer cho mỗi giá trị trung gian, khai báo trong <code>sampler_metal.h</code></>}><><code>encodePhiloxKat</code>, <code>encodeApplyRepetition</code>, <code>encodeDrawRows</code> and <code>encodeSampleRows</code>; on Metal the workspace is one buffer per intermediate, declared in <code>sampler_metal.h</code></></Translated>],
        ]}
      />

      <h3>
        <Translated vi="Các kiểm thử kiểm tra gì">What the tests check</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Trước hết là các đáp án đã biết của Philox trên thiết bị, rồi u khớp từng bit cho 4,096 bộ (seed, stream, step),
            trong đó bộ đầu tiên, toàn số 0, phải cho ra word <code>0x6627e8d5</code>. Tiếp theo là riêng lần rút với u được
            cài sẵn: trọng số bằng nhau, khi u = m × 2⁻²⁴ phải cho đúng vị trí floor(u × n), trọng số bằng 0 không bao giờ
            được thắng, u gần 1, chỉ một phần tử còn lại, ranh giới giữa các đoạn, và phương án dự phòng. Sau đó là quy tắc
            repetition trên các logit cả hai dấu và các số 0 chính xác. Rồi toàn bộ sampler so với bản tham chiếu trên CPU, ở
            các giá trị mặc định của llama.cpp (top-k 40, top-p 0.95, min-p 0.05, nhiệt độ 0.8) có penalty, và ở top-k 1,
            top-k 0, top-p 1, min-p 0, nhiệt độ 0, nhiệt độ 2 và không penalty, 64 seed nhân 4 bước cho mỗi kích thước.
            Cuối cùng là một vòng decode đơn giản có số đếm được giữ trên thiết bị, và một digest của các token.
          </p>
        }
      >
        <p>
          Philox's known answers on the device first, then u bit for bit for 4,096 (seed, stream, step) triples, the first of
          which, all zero, must give the word <code>0x6627e8d5</code>. Then the draw alone with planted u: equal weights,
          where u = m × 2⁻²⁴ must give position floor(u × n) exactly, zero weights that must never win, u near 1, a single
          survivor, chunk seams, and the fallback. Then the repetition rule on logits of both signs and exact zeros. Then the
          whole sampler against the CPU reference, at llama.cpp's defaults (top-k 40, top-p 0.95, min-p 0.05, temperature 0.8)
          with penalties, and at top-k 1, top-k 0, top-p 1, min-p 0, temperature 0, temperature 2 and no penalties, 64 seeds by
          4 steps per shape. Last, a toy decode loop whose counts are kept on the device, and a digest of the tokens.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Token chỉ được so sánh ở những nơi lần rút là dứt khoát. Bản tham chiếu trên CPU ghi lại mỗi quyết định còn cách
            bao xa mới bị lật: một logit đã bị phạt, điểm cắt top-k, số đếm của top-p và min-p, thứ tự của các phần tử lân
            cận, và khoảng cách từ u × total tới một ranh giới của tổng cộng dồn. Một lần rút nằm trong dải quanh một ranh giới
            được đếm và in ra, không bao giờ bị tính là thất bại, vì softmax và scan bằng float của một thiết bị có thể rơi vào
            phía bên kia của nó. Các lần rút dứt khoát phải khớp chính xác, và hơn một nửa số lần rút của mỗi trường hợp phải
            là dứt khoát.
          </p>
        }
      >
        <p>
          Tokens are compared only where the draw is decisive. The CPU reference records how far every decision is from
          flipping: a penalised logit, the top-k cut, top-p's and min-p's counts, the order of neighbours, and the distance
          of u × total from a boundary of the running sum. A draw within a band of a boundary is counted and printed, never
          failed, because a device's float softmax and scan may land on the other side of it. Decisive draws must match
          exactly, and more than half of every case's draws must be decisive.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x1-facts.md C5; ch05-facts.md C7)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              MSL: "Either round ties to even or round toward zero rounding mode may be supported" (có thể hỗ trợ chế độ làm
              tròn về số chẵn gần nhất hoặc làm tròn về 0), và fast math theo mặc định.
            </p>
          }
        >
          <p>
            MSL: "Either round ties to even or round toward zero rounding mode may be supported", and fast math by default.
          </p>
        </Translated>
      </Today>
      <RaceStatus lab={l3} />
      <h3>
        <Translated vi="Cùng seed, cùng token, trên mọi backend">Same seed, same tokens, across backends</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Một digest là giá trị băm của các lần rút dứt khoát trong một trường hợp cố định ở các giá trị mặc định của
            llama.cpp, hàng dài 1,000. Trên GPU, trường hợp đầy đủ là 8 hàng nhân 16 seed nhân 8 bước, 1,024 lần rút, gần như
            tất cả đều dứt khoát: một backend rút ra token khác chỉ một lần trong vài trăm lần rút cũng sẽ làm nó thay đổi.
            Các trình giả lập chạy từng luồng một, quá chậm cho 1,024 lần rút, nên chúng chỉ chạy phần lõi, 2 hàng đầu nhân 2
            seed nhân 2 bước, 8 lần rút; GPU cũng băm riêng phần lõi đó, để mọi backend có thể được so sánh trên nó. Digest bằng
            nhau nghĩa là token bằng nhau ở những lần rút đó, và bảng cho số lần rút bên cạnh mỗi digest. Bảng đọc chúng từ
            các lần chạy trên máy này (<code>reports/runs/</code>); trên một máy CUDA hoặc ROCm, cùng lần chạy đó in ra{' '}
            <code>tokens_digest</code> và <code>core_digest</code> của nó, để bạn so với các giá trị này bằng tay.
          </p>
        }
      >
        <p>
          A digest is a hash of the decisive draws of one fixed case at llama.cpp's defaults, rows of 1,000. On a GPU the
          full case is 8 rows by 16 seeds by 8 steps, 1,024 draws, nearly all of them decisive: a backend that drew a
          different token once in a few hundred draws would change it. The emulators run one thread at a time, far too slowly
          for 1,024 draws, so they run only the core, the first 2 rows by 2 seeds by 2 steps, 8 draws; GPUs hash that core separately
          too, so every backend can be compared on it. Equal digests mean equal tokens on those draws, and the table gives the
          count beside each. It reads them from this machine's runs (<code>reports/runs/</code>); on a CUDA or ROCm machine
          the same run prints its <code>tokens_digest</code> and <code>core_digest</code>, to compare with these by hand.
        </p>
      </Translated>
      <DigestTable />

      <h3>
        <Translated vi="Trước khi đo hiệu năng">Before you benchmark</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Hãy suy luận từ hai sự thật. Mọi kernel chạy mỗi hàng một khối, nên ở batch 1 chỉ một multiprocessor làm việc còn
            phần còn lại ngồi chờ (<a href={href('ch10', '10.10')}>mục 10.10</a>). Và chỉ phép sao chép, các penalty và phép
            chọn đọc toàn bộ từ vựng; sau top-k, mọi giai đoạn chỉ chạm tới 40 giá trị mỗi hàng. Theo mặc định llama.cpp lấy
            mẫu trên một luồng host, và đó là thứ cột CPU dùng để so sánh.
          </p>
        }
      >
        <p>
          Reason from two facts. Every kernel runs one block per row, so at batch 1 one multiprocessor works and the rest
          wait (<a href={href('ch10', '10.10')}>section 10.10</a>). And only the copy, the penalties and the select read the
          whole vocabulary; after top-k every stage touches 40 values per row. llama.cpp samples on one host thread by
          default, which is what the CPU column measures against.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x1-facts.md D1)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              llama.cpp: <code>bool backend_sampling = false;</code> khi sinh văn bản. Đường lấy mẫu trên GPU của nó còn ở
              dạng thử nghiệm.
            </p>
          }
        >
          <p>
            llama.cpp: <code>bool backend_sampling = false;</code> for generation. Its GPU sampling path is experimental.
          </p>
        </Translated>
      </Today>
      <LabPrediction
        id="x1/labs/p-lab3-cpu"
        prompt={
          <Translated
            vi={
              <>
                Ở batch 1 và 151,936 token với các giá trị mặc định của llama.cpp, thời gian mỗi bước lấy mẫu của GPU chia
                cho thời gian của một luồng CPU là bao nhiêu? (Dưới 1 nghĩa là GPU thắng.)
              </>
            }
          >
            <>
              At batch 1 and 151,936 tokens with llama.cpp's defaults, what is the GPU's time per sampling step divided by one
              CPU thread's? (Below 1 means the GPU wins.)
            </>
          </Translated>
        }
        unit={tr("× one CPU thread's time", '× thời gian của một luồng CPU')}
        measure={gpuOverCpu}
        missing={missingBench('x1/lab3')}
        reading={
          <Translated
            vi={
              <>
                Thời gian của Metal là thời gian GPU của command buffer; thời gian của CUDA và HIP là đồng hồ thực phía host
                bao quanh các lần khởi chạy và một lần đồng bộ hóa; thời gian của CPU là một đồng hồ đơn điệu trên host. Bảng
                theo giai đoạn cho thấy thời gian của GPU đã đi đâu.
              </>
            }
          >
            <>
              Metal's time is the command buffer's GPU time; CUDA and HIP times are host wall clock around the launches and a
              synchronise; the CPU's is a steady clock on the host. The stage table shows where the GPU's time went.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="x1/labs/p-lab3-batch"
        prompt={
          <Translated vi={<>Một bước lấy mẫu ở batch 64 sẽ lâu hơn ở batch 1 bao nhiêu lần, với từ vựng 151,936?</>}>
            <>How many times longer will a sampling step take at batch 64 than at batch 1, vocabulary 151,936?</>
          </Translated>
        }
        unit={tr('× the batch-1 step', '× bước ở batch 1')}
        measure={batchGrowth}
        missing={missingBench('x1/lab3')}
        reading={
          <Translated
            vi={
              <>
                Con số 64 sẽ có nghĩa là mỗi hàng tốn bằng hàng đầu tiên, như thể các hàng chạy lần lượt nối tiếp nhau. Một con số gần
                1 nghĩa là các hàng thêm vào đã chạy trên những multiprocessor mà batch 1 để trống.
              </>
            }
          >
            <>
              64 would mean every row costs as much as the first, as if rows ran one after another. A number near 1 means the
              extra rows ran on multiprocessors that batch 1 left idle.
            </>
          </Translated>
        }
      />
      <LabResults
        lab={l3}
        cols={[
          what,
          rawCol(tr('Rows', 'Số hàng'), 'rows'),
          colsCol(tr('Vocab', 'Từ vựng'), 'vocab'),
          rawCol('Top-k', 'top_k'),
          msCol,
          numCol(tr('Tokens/s', 'Token/s'), 'tokens_per_s'),
          numCol(tr('GPU / CPU', 'GPU / CPU'), 'gpu_over_cpu'),
        ]}
      />
      <Translated vi={<p>Mỗi giai đoạn được đo riêng, trên các buffer được chuẩn bị bằng cách chạy các giai đoạn một lần theo thứ tự:</p>}>
        <p>Each stage timed alone, on buffers prepared by running the stages once in order:</p>
      </Translated>
      <StageTable />
      <Translated
        vi={
          <p>
            Cách benchmark hoạt động: 1, 8 và 64 hàng nhân với từ vựng 32,000 và 151,936 ở các giá trị mặc định của
            llama.cpp, có bật penalty, và đường cả hàng (top-k 0) ở 1 và 8 hàng nhân 151,936. Một luồng CPU chạy cùng chuỗi
            đó bằng float, top-k bằng <code>std::partial_sort</code>. Lấy kết quả tốt nhất trong 5 lần đo, mỗi lần lặp lại lời
            gọi cho tới khi kéo dài ít nhất 2 ms; mọi kết quả đều được kiểm tra với bản tham chiếu trên CPU trước.
          </p>
        }
      >
        <p>
          How the benchmark works: rows 1, 8 and 64 by vocabularies of 32,000 and 151,936 at llama.cpp's defaults with the
          penalties on, and the whole-row path (top-k 0) at rows 1 and 8 by 151,936. One CPU thread runs the same chain in
          float, top-k by <code>std::partial_sort</code>. Best of 5 timed runs, each repeating the call until it lasts at
          least 2 ms; every result is checked against the CPU reference first.
        </p>
      </Translated>

      <h3>
        <Translated vi="Mốc dự án">The project milestone</Translated>
      </h3>
      <Translated
        vi={
          <p>
            <code>launchSampleRows</code> hoàn thành mốc dự án M5: logits <code>[rows × vocab]</code> và số đếm token vào, một
            token id mỗi hàng ra, cùng một token cho cùng (seed, stream, step) trên mọi backend ở mọi lần rút dứt khoát. Nó
            theo các quy ước của dự án để các mốc sau có thể gọi nó: float32 theo thứ tự hàng với chiều batch đứng đầu, con
            trỏ device thô và kích thước tường minh, một hàm host <code>launch</code> bên cạnh các kernel, bản tham chiếu
            trên CPU trong <code>main.cpp</code>. Mốc M10 (lab X3.2) gọi nó trên GPU, sau bước decode của M7, khi có lời
            giải của lab này.
          </p>
        }
      >
        <p>
          <code>launchSampleRows</code> completes milestone M5: <code>[rows × vocab]</code> logits and token counts in, one
          token id per row out, the same token for the same (seed, stream, step) on every backend for every decisive draw.
          It follows the project's conventions so later milestones can call it: row-major float32 with the batch first, raw
          device pointers and explicit sizes, a <code>launch</code> host function beside the kernels, the CPU reference in{' '}
          <code>main.cpp</code>. Milestone M10 (lab X3.2) calls it on a GPU, after M7's decode step, when this lab's solution is
          present.
        </p>
      </Translated>
      <ReferenceSolution lab={l3} tiers={TIERS_3} />

      {/* ------------------------------------------------------------ X1.4 */}
      <h2>
        <Translated vi="Lab X1.4: săn lỗi bộ lọc, ba lỗi trong một kernel gộp top-p và min-p">
          Lab X1.4: filter hunt, three defects in a fused top-p and min-p kernel
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn được giao kernel gộp của lab X1.2, bằng CUDA và bằng Metal, với cùng hợp đồng, tên và hàm khởi chạy, và ba lỗi
            độc lập được cài sẵn trong đó. Nó biên dịch được. Hãy tìm và sửa cả ba. Đây là kernel bạn vừa viết, nên các lỗi
            là những lỗi nó dễ mắc: một ở quy tắc ranh giới, một ở phần mang giữa các đoạn, một ở đồng bộ hóa.
          </p>
        }
      >
        <p>
          You are handed lab X1.2's fused kernel, in CUDA and in Metal, with the same contract, names and launchers, and three
          independent defects planted in it. It compiles. Find and fix all three. This is the kernel you have just written,
          so the defects are the ones it invites: one in a boundary rule, one in the carry between chunks, one in
          synchronisation.
        </p>
      </Translated>
      <Source section="10.7–10.8, 11.4" pages="239–244, 261–265" figs="Figs. 10.13, 10.16, 11.8 and 11.9" />
      <LabFiles
        lab={l4}
        files={[
          ['kernel.cu', <Translated vi={<><code>filterSortedFused</code>, với ba lỗi; <code>launchFilterSortedFused</code> thì đúng</>}><><code>filterSortedFused</code>, with the three defects; <code>launchFilterSortedFused</code> is correct</></Translated>],
          ['kernel.metal', <Translated vi={<><code>filter_sorted_fused</code>, với cùng ba lỗi đó</>}><><code>filter_sorted_fused</code>, with the same three</></Translated>],
          ['launch.mm', <Translated vi={<>được cung cấp sẵn và đúng; không cần sửa gì</>}><>provided and correct; nothing to change</></Translated>],
        ]}
      />
      <Translated
        vi={
          <p>
            README liệt kê những gì không phải lỗi: lane mask có kiểu và <code>warpSize</code> ở chỗ sách viết{' '}
            <code>0xffffffff</code> và 32, ngưỡng <code>-FLT_MAX</code> giữ mọi token khi tắt min-p, ln(min-p) được tính trên
            host, và không có lượt tìm max trên một hàng đã sắp xếp.
          </p>
        }
      >
        <p>
          The README lists what is not a defect: the typed lane mask and <code>warpSize</code> where the book writes{' '}
          <code>0xffffffff</code> and 32, the threshold <code>-FLT_MAX</code> that keeps every token when min-p is off,
          ln(min-p) computed on the host, and no max pass on a sorted row.
        </p>
      </Translated>
      <Translated vi={<p>Phương pháp quan trọng hơn các bản sửa:</p>}>
        <p>The method matters more than the fixes:</p>
      </Translated>
      <ol>
        <Translated
          vi={
            <li>
              Trước khi chạy bất cứ thứ gì, hãy đọc <code>work/kernel.cu</code> và ghi ra các lỗi bạn thấy được, và với mỗi
              lỗi, backend mà bạn nghĩ sẽ báo nó: <code>emu</code>, <code>race</code>, GPU của bạn, hoặc không backend nào.
            </li>
          }
        >
          <li>
            Before running anything, read <code>work/kernel.cu</code> and write down the defects you can see, and for each the
            backend you expect to report it: <code>emu</code>, <code>race</code>, your GPU, or none.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Chạy trình giả lập: <code>python3 labs/pmpp.py run x1/lab4 --backend emu</code>. Các thứ tự luồng của nó là cố
              định, nên cái gì nó báo thì lần nào nó cũng báo. Hãy dùng điểm chung của các trường hợp thất bại (K nào, loại
              hàng nào, p nào) và các gợi ý của chương trình điều khiển.
            </li>
          }
        >
          <li>
            Run the emulator: <code>python3 labs/pmpp.py run x1/lab4 --backend emu</code>. Its thread orders are fixed, so
            what it reports, it reports every time. Use what the failing cases have in common (which K, which kind of row,
            which p) and the driver's hints.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Sửa từng lỗi một và chạy lại. Sau đó chạy <code>python3 labs/pmpp.py run x1/lab4 --backend race</code>, backend
              báo hai luồng chạm vào cùng vùng nhớ mà không có gì sắp thứ tự chúng, kèm cả hai dòng mã nguồn, dù có giá trị
              nào bị sai hay không.
            </li>
          }
        >
          <li>
            Fix one defect at a time and run again. Then run <code>python3 labs/pmpp.py run x1/lab4 --backend race</code>,
            which reports two threads touching the same memory with nothing ordering them, with both source lines, whether
            or not a value came out wrong.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Sửa y như vậy trong <code>kernel.metal</code> và chạy trên thiết bị:{' '}
              <code>python3 labs/pmpp.py run x1/lab4</code>. Ghi lại những lỗi nó cho bạn thấy và những lỗi nó che đi.
            </li>
          }
        >
          <li>
            Make the same fixes in <code>kernel.metal</code> and run on the device: <code>python3 labs/pmpp.py run x1/lab4</code>.
            Note which defects it showed you and which it hid.
          </li>
        </Translated>
      </ol>
      <Predict
        id="x1/labs/p-lab4-emu"
        unit={tr('defects', 'lỗi')}
        prompt={
          <Translated
            vi={
              <>
                Sau khi đọc kernel và trước khi chạy nó: trong ba lỗi, <code>emu</code> sẽ báo bao nhiêu lỗi?
              </>
            }
          >
            <>After reading the kernel and before running it: of the three defects, how many will <code>emu</code> report?</>
          </Translated>
        }
      >
        {(c) => <HuntReveal committed={c} />}
      </Predict>
      <Translated
        vi={
          <p>
            Dù GPU của bạn cho thấy gì, hãy đối chiếu nó với những gì các nhà cung cấp ghi trong tài liệu. Các warp khác nhau
            tiến triển độc lập giữa các rào chắn (barrier), và đọc lẫn ghi cùng một vị trí mà không đồng bộ hóa là hành vi
            không xác định. Một trường hợp qua mười lần trên GPU của bạn chỉ cho bạn thấy mười lịch chạy của một thiết bị,
            không phải một kernel đúng.
          </p>
        }
      >
        <p>
          Whatever your GPU showed, read it against what the vendors document. Different warps make independent progress
          between barriers, and reading and writing the same location without synchronisation is undefined behaviour. A case
          that passed ten times on your GPU showed you ten schedules of one device, not a correct kernel.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A9; infra-facts.md C5; ch10-facts.md A1)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              NVIDIA: các warp chạy độc lập. Apple: các SIMD-group "make independent forward progress ... in the absence of
              threadgroup barrier operations" (tiến triển độc lập khi không có thao tác rào chắn threadgroup). CUDA: "Reading
              from or writing to the same memory location without memory fences or synchronization results in undefined
              behavior" (đọc hoặc ghi cùng một vị trí bộ nhớ mà không có memory fence hay đồng bộ hóa dẫn tới hành vi không
              xác định).
            </p>
          }
        >
          <p>
            NVIDIA: warps run independently. Apple: SIMD-groups "make independent forward progress ... in the absence of
            threadgroup barrier operations". CUDA: "Reading from or writing to the same memory location without memory fences
            or synchronization results in undefined behavior".
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Các kiểm thử là phần B của lab X1.2 ở kích thước nhỏ hơn, với các hàng dài hơn một đoạn 1,024 trên mọi backend một
            cách có chủ ý; trên GPU, mỗi batch chạy mười lần trên các đầu ra mới. Lab này không có benchmark.
          </p>
        }
      >
        <p>
          The tests are lab X1.2's part B at smaller sizes, with rows longer than one chunk of 1,024 on every backend on
          purpose; on a GPU every batch runs ten times on fresh outputs. There is no benchmark.
        </p>
      </Translated>
      <RaceStatus lab={l4} />
      <LabResults
        lab={l4}
        noBench={tr(
          'This lab has no benchmark. The result is the line above: every test passing on the emulator, the race backend and a device. As shipped, the kernel is meant to fail.',
          'Lab này không có benchmark. Kết quả là dòng ở trên: mọi kiểm thử đều qua trên trình giả lập, backend race và một thiết bị. Ở dạng ban đầu được giao, kernel này cố ý thất bại.',
        )}
      />
      <ReferenceSolution lab={l4} tiers={TIERS_2} />
    </LabsPage>
  );
}
