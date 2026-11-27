import type { Question } from '../../components/Ask';
import { Ask } from '../../components/Ask';
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
import { Translated, tr } from '../../lib/i18n';
import { benchRows, fmt, labStatus, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 20's own content.
// Predictions are stored, not graded; the one question is about lab 20.3's first defect.
export const questionsLabs: Record<string, Question> = {
  'ch20/labs/q-detector': {
    kind: 'choice',
    id: 'ch20/labs/q-detector',
    prompt: (
      <Translated
        vi={
          <>
            Lab 20.3, lỗi 1: lệnh <code>__syncwarp()</code> sau lần ghi P trong <code>compute_P_and_update_D</code> đã
            bị bỏ (trên Metal là <code>simdgroup_barrier</code>). Trước khi chạy bất cứ thứ gì: bạn nghĩ backend nào sẽ
            báo ra lỗi này?
          </>
        }
      >
        <>
          Lab 20.3, defect 1: the <code>__syncwarp()</code> after the P write in <code>compute_P_and_update_D</code> is
          gone (on Metal, the <code>simdgroup_barrier</code>). Before you run anything: which backend do you expect to
          report it?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="emu: các thứ tự luồng cố định của nó làm mọi rào chắn bị thiếu đều thất bại.">
            emu: its fixed thread orders make every missing barrier fail.
          </Translated>
        ),
        why: (
          <Translated vi="emu chỉ chuyển luồng ở các điểm đồng bộ hóa, và một phép shuffle là điểm mà cả warp gặp nhau. Mọi lane đã ghi P của nó trước khi bất kỳ lane nào rời butterfly tính tổng, nên các lần đọc trong compute_O thấy được các lần ghi.">
            emu switches threads only at synchronisation points, and a shuffle is one where the whole warp meets. Every
            lane has written its P before any lane leaves the sum butterfly, so the reads in compute_O see the writes.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="race: ThreadSanitizer chỉ ra lần ghi P và lần đọc.">
            race: ThreadSanitizer names the P write and the read.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="GPU của bạn: các lane chạy lệch nhau đọc giá trị P cũ từ bộ nhớ chia sẻ (shared memory).">
            Your GPU: lanes that drift apart read stale P values from shared memory.
          </Translated>
        ),
        why: (
          <Translated vi="Không có gì hứa rằng các lane luôn đi cùng nhau, nhưng cũng không có gì buộc chúng chạy tách ra. Khi lab được viết, Metal qua mọi bài kiểm thử khi chỉ có riêng lỗi này.">
            Nothing promises the lanes stay together, but nothing makes them run apart either. When the lab was
            written, Metal passed every test with this defect alone.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Cả ba: thiếu một rào chắn là sai trên mọi backend.">
            All three: a missing barrier is wrong on every backend.
          </Translated>
        ),
        why: (
          <Translated vi="Sai thì đúng là sai; nhưng không lộ ra. Chỉ một công cụ coi shuffle là không sắp thứ tự bộ nhớ, đúng như tài liệu CUDA nói, mới thấy nó mọi lần.">
            Wrong, yes; visible, no. Only a tool that treats shuffles as not ordering memory, as the CUDA documentation
            says they do not, sees it every time.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Shuffle trao đổi thanh ghi; chúng không sắp thứ tự bộ nhớ (ch20-facts A3). Trên <code>race</code> các luồng
            chạy cùng lúc và chỉ <code>__syncwarp()</code> sắp thứ tự bộ nhớ của chúng, nên ThreadSanitizer báo ra cặp
            truy cập đó. Trên <code>emu</code> điểm hẹn của shuffle tình cờ xếp các lane thẳng hàng, còn trên GPU phần
            cứng có thể giữ các lane của một warp đi cùng nhau dù mã có yêu cầu hay không. Đó là lý do đoạn mã của sách,
            vốn không có rào chắn này, có thể qua mọi bài kiểm thử bạn đưa ra. Trình chạy từng bước của kernel ở mục{' '}
            <a href={href('ch20', '20.5')}>20.5</a> cho thấy lịch chạy làm hỏng nó.
          </p>
        }
      >
        <p>
          Shuffles exchange registers; they do not order memory (ch20-facts A3). On <code>race</code> the threads run at
          once and only <code>__syncwarp()</code> orders their memory, so ThreadSanitizer reports the pair. On{' '}
          <code>emu</code> the shuffle rendezvous happens to line the lanes up, and on a GPU the hardware may keep a
          warp&rsquo;s lanes together whether or not the code asks. That is why the book&rsquo;s listing, which has no such
          barrier, can pass every test you give it. Section <a href={href('ch20', '20.5')}>20.5</a>&rsquo;s kernel stepper
          shows the schedule that breaks it.
        </p>
      </Translated>
    ),
    source: 'Figs. 20.13 and 20.15, pp. 501–503; notes/today/ch20-facts.md A3; labs/ch20/lab3_flash_hunt/README.md',
    vi: { source: 'Figs. 20.13 và 20.15, pp. 501–503; notes/today/ch20-facts.md A3; labs/ch20/lab3_flash_hunt/README.md' },
  },
};

export const LABS = defineLabs('ch20', [
  { num: '20.1', dir: 'lab1_causal_softmax', name: 'causal softmax (M6, part one)', vi: { name: 'softmax nhân quả (M6, phần một)' } },
  { num: '20.2', dir: 'lab2_flash', name: 'flash attention forward (M6, part two)', vi: { name: 'flash attention lượt forward (M6, phần hai)' } },
  { num: '20.3', dir: 'lab3_flash_hunt', name: 'flash race hunt', vi: { name: 'săn race trong flash attention' } },
  { num: '20.4', dir: 'lab4_decode', name: 'KV append and decode (M7)', vi: { name: 'nối thêm KV và decode (M7)' } },
  { num: '20.5', dir: 'lab5_gqa_decode', name: 'grouped-query decode', vi: { name: 'decode grouped-query' } },
]);
const [l1, l2, l3, l4, l5] = LABS;

// Identifier colours, by the memory each lives in.
const TIERS_SOFTMAX: Record<string, Tier> = { S: 'global', P: 'global', L: 'global', scratch: 'shared', m: 'reg', s: 'reg' };
const TIERS_FLASH: Record<string, Tier> = {
  Q: 'global', K: 'global', V: 'global', O: 'global', L: 'global',
  KT_j: 'shared', S_i: 'shared', V_j: 'shared',
  Q_i: 'reg', O_i: 'reg', m_i: 'reg', D_i: 'reg', last_m: 'reg', S_ij: 'reg', O_ij: 'reg', P_ij: 'reg',
};
const TIERS_DECODE: Record<string, Tier> = {
  q: 'global', Kc: 'global', Vc: 'global', out: 'global', k_new: 'global', v_new: 'global',
  m_s: 'shared', D_s: 'shared', O_s: 'shared', qr: 'reg',
};

// ---------------------------------------------------------------- reading the benchmark rows
// Lab 20.1 records one row per N, 'unfused_attention', with s_seconds, softmax_seconds,
// o_seconds, total_seconds, softmax_share, softmax_gb_per_s and score_bytes (S and P, all heads).
// Lab 20.2 records 'attention' per n and b_c: flash_seconds, unfused_seconds, flash_over_unfused,
// flash_gflops, unfused_gflops, unfused_score_bytes. Lab 20.4 records 'decode' (batch, len, ms,
// gb_per_s, cache_bytes) and 'append' (batch, us, bytes_moved). Lab 20.5 records 'perhead' and
// 'grouped' per (batch, h_kv) with g_q, blocks, ms, gb_per_s and, on 'grouped', speedup_vs_perhead.
const mib = (v?: number) => (v == null ? '' : `${fmt(v / 2 ** 20)} MiB`); // i18n-ignore: number and unit
const sec = (key: string, head: string): Col => ({
  head,
  num: true,
  cell: (r) => {
    const v = num(r[key]);
    return v == null ? '' : fmt(v * 1000);
  },
});

/** Lab 5.1's measured bandwidth on one backend, from the reference solution or the reader's own run. */
function bandwidth(reports: Reports, backend: string): number | undefined {
  const run = reports[backend];
  const rows = [...benchRows(run, 'ch05/lab1_roofline', 'work'), ...benchRows(run, 'ch05/lab1_roofline', 'solution')];
  return num(rows.find((r) => r.name === 'roofline')?.bandwidth_gb_per_s);
}

/** Lab 20.2's row at n, the book's B_c = 32 where a build has it, otherwise whatever B_c it ran. */
const flashRow = (s: BenchSet, n: number) => benchRow(s, 'attention', { n, b_c: 32 }) ?? benchRow(s, 'attention', { n });

const shareAt4096 = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const r = benchRow(s, 'unfused_attention', { n: 4096 });
    const share = num(r?.softmax_share);
    const sm = num(r?.softmax_seconds);
    const tot = num(r?.total_seconds);
    if (share == null || sm == null || tot == null) return null;
    return {
      value: 100 * share,
      detail: tr(
        `softmax ${fmt(sm * 1000)} ms of ${fmt(tot * 1000)} ms in all, N = 4096, 8 heads`,
        `softmax ${fmt(sm * 1000)} ms trên tổng ${fmt(tot * 1000)} ms, N = 4096, 8 head`,
      ),
    };
  });

const flashRatio4096 = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const r = flashRow(s, 4096);
    const f = num(r?.flash_seconds);
    const u = num(r?.unfused_seconds);
    if (!r || f == null || u == null) return null;
    return {
      value: f / u,
      detail: tr(
        `flash ${fmt(f * 1000)} ms against unfused ${fmt(u * 1000)} ms, n = 4096, B_c = ${String(r.b_c)}`,
        `flash ${fmt(f * 1000)} ms so với bản không gộp ${fmt(u * 1000)} ms, n = 4096, B_c = ${String(r.b_c)}`,
      ),
    };
  });

const decodeMs = (reports: Reports) =>
  measure(reports, l4, (s) => {
    const r = benchRow(s, 'decode', { batch: 1, len: 8192 });
    const ms = num(r?.ms);
    const gb = num(r?.gb_per_s);
    if (ms == null || gb == null) return null;
    const bw = bandwidth(reports, s.backend);
    return {
      value: ms,
      detail: tr(
        `${fmt(gb)} GB/s on the cache bytes${bw ? `, against lab 5.1's ${fmt(bw)} GB/s, ${fmt((100 * gb) / bw)}% of it` : '; no lab 5.1 bandwidth on this backend yet'}`,
        `${fmt(gb)} GB/s trên số byte của cache${bw ? `, so với ${fmt(bw)} GB/s của lab 5.1, bằng ${fmt((100 * gb) / bw)}% mức đó` : '; chưa có băng thông lab 5.1 trên backend này'}`,
      ),
    };
  });

const decodeBatch8 = (reports: Reports) =>
  measure(reports, l4, (s) => {
    const one = num(benchRow(s, 'decode', { batch: 1, len: 8192 })?.ms);
    const eight = num(benchRow(s, 'decode', { batch: 8, len: 8192 })?.ms);
    if (one == null || eight == null) return null;
    return {
      value: eight / one,
      detail: tr(
        `${fmt(eight)} ms at batch 8 against ${fmt(one)} ms at batch 1, len 8192`,
        `${fmt(eight)} ms ở batch 8 so với ${fmt(one)} ms ở batch 1, len 8192`,
      ),
    };
  });

const groupedSpeedup = (reports: Reports) =>
  measure(reports, l5, (s) => {
    const g = benchRow(s, 'grouped', { batch: 1, h_kv: 8 });
    const p = benchRow(s, 'perhead', { batch: 1, h_kv: 8 });
    const gm = num(g?.ms);
    const pm = num(p?.ms);
    if (gm == null || pm == null) return null;
    return {
      value: pm / gm,
      detail: tr(
        `per-head ${fmt(pm)} ms over ${String(p?.blocks)} blocks, grouped ${fmt(gm)} ms over ${String(g?.blocks)} blocks, len 8192`,
        `theo từng head ${fmt(pm)} ms trên ${String(p?.blocks)} khối, theo nhóm ${fmt(gm)} ms trên ${String(g?.blocks)} khối, len 8192`,
      ),
    };
  });

// ---------------------------------------------------------------- results columns
// Built when the table renders, so the headings follow the selected language.
const softmaxCols: () => Col[] = () => [
  rawCol('N', 'n'),
  sec('s_seconds', 'S = Q Kᵀ, ms'), // i18n-ignore: equation and unit
  sec('softmax_seconds', tr('Softmax, ms', 'Softmax, ms')),
  sec('o_seconds', 'O = P V, ms'), // i18n-ignore: equation and unit
  sec('total_seconds', tr('Total, ms', 'Tổng, ms')),
  { head: tr('Softmax share', 'Tỉ phần softmax'), num: true, cell: (r) => (num(r.softmax_share) == null ? '' : `${fmt(100 * num(r.softmax_share)!)}%`) },
  numCol(tr('Softmax GB/s', 'Softmax GB/s'), 'softmax_gb_per_s'),
  { head: tr('S and P', 'S và P'), num: true, cell: (r) => mib(num(r.score_bytes)) },
];
const flashCols: () => Col[] = () => [
  rawCol('n', 'n'),
  rawCol('B_c', 'b_c'),
  sec('flash_seconds', tr('Flash, ms', 'Flash, ms')),
  sec('unfused_seconds', tr('Unfused, ms', 'Không gộp, ms')),
  numCol(tr('Flash / unfused', 'Flash / không gộp'), 'flash_over_unfused'),
  numCol(tr('Flash GFLOP/s', 'Flash GFLOP/s'), 'flash_gflops'),
  numCol(tr('Unfused GFLOP/s', 'Không gộp GFLOP/s'), 'unfused_gflops'),
  { head: tr('Unfused S and P', 'S và P của bản không gộp'), num: true, cell: (r) => mib(num(r.unfused_score_bytes)) },
];
const decodeCols: () => Col[] = () => [
  {
    head: tr('Measured', 'Đã đo'),
    cell: (r) => (r.name === 'append' ? tr('append alone', 'chỉ riêng append') : tr('decode attention', 'decode attention')),
  },
  rawCol(tr('Batch', 'Batch'), 'batch'),
  rawCol('len', 'len'),
  msCol,
  numCol('GB/s', 'gb_per_s'),
  { head: tr('Bytes', 'Byte'), num: true, cell: (r) => mib(num(r.cache_bytes) ?? num(r.bytes_moved)) },
];
const groupedCols: () => Col[] = () => [
  {
    head: tr('Kernel measured', 'Kernel được đo'),
    cell: (r) =>
      r.name === 'perhead'
        ? tr('per-head (lab 20.4)', 'theo từng head (lab 20.4)')
        : r.name === 'grouped'
          ? tr('grouped (this lab)', 'theo nhóm (lab này)')
          : r.name,
  },
  rawCol(tr('Batch', 'Batch'), 'batch'),
  rawCol('h_kv', 'h_kv'),
  rawCol('g_q', 'g_q'),
  rawCol(tr('Blocks', 'Khối'), 'blocks'),
  msCol,
  numCol(tr('GB/s, unique bytes', 'GB/s, byte không trùng lặp'), 'gb_per_s'),
  numCol(tr('× per-head', '× theo từng head'), 'speedup_vs_perhead'),
];

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
            `${s === 'pass' ? 'đạt' : 'chưa đạt'}: ${tests.filter((t) => t.pass).length} trên ${tests.length} bài kiểm thử, ${races ? `${races} báo cáo ThreadSanitizer` : 'không có race nào được báo'}`,
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
      <p data-race="none">
        <Translated vi="Chưa có lần chạy nào trên backend race ở máy này.">
          No run on the race backend on this machine yet.
        </Translated>
      </p>
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
                  <Translated
                    vi={
                      <>
                        Lab {l.num}, {l.vi?.name ?? l.name}
                      </>
                    }
                  >
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

// ---------------------------------------------------------------- lab 20.5 beside Eq. 20.13
/** Measured grouped/per-head ratios per g_q and batch, beside the intensity ratio Eq. 20.13 gives. */
function GroupedVsIntensity() {
  const { reports, loading } = useLabs();
  const sets = preferredBench(reports, l5);
  const rows = sets.flatMap((s) =>
    s.rows
      .filter((r): r is BenchRow => r.name === 'grouped' && num(r.speedup_vs_perhead) != null)
      .map((r) => ({ s, r })),
  );
  if (loading)
    return (
      <p className="muted">
        <Translated vi="Đang đọc báo cáo.">Reading reports.</Translated>
      </p>
    );
  if (!rows.length)
    return (
      <p className="muted" data-gqa="none">
        <Translated
          vi={
            <>
              Chưa có benchmark nào của lab 20.5 trên máy này. Chạy <code>python3 labs/pmpp.py run 20.5 --bench</code>{' '}
              (thêm <code>--solution</code> để có bản tham chiếu), rồi tải lại.
            </>
          }
        >
          No lab 20.5 benchmark on this machine yet. Run <code>python3 labs/pmpp.py run 20.5 --bench</code> (add{' '}
          <code>--solution</code> for the reference), then reload.
        </Translated>
      </p>
    );
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
      <table className="data" data-table="gqa-intensity">
        <thead>
          <tr>
            <th>
              <Translated vi="Backend">Backend</Translated>
            </th>
            <th className="num">
              <Translated vi="Batch">Batch</Translated>
            </th>
            <th className="num">g_q</th>
            <th className="num">
              <Translated vi="Số khối, theo từng head : theo nhóm">Blocks, per-head : grouped</Translated>
            </th>
            <th className="num">
              <Translated vi="Số byte yêu cầu, theo từng head : theo nhóm">Bytes requested, per-head : grouped</Translated>
            </th>
            <th className="num">
              <Translated vi="Mức tăng tốc đo được, theo nhóm so với theo từng head">
                Measured speed-up, grouped over per-head
              </Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ s, r }, i) => {
            const gq = num(r.g_q) ?? 1;
            const ph = benchRow(s, 'perhead', { batch: num(r.batch)!, h_kv: num(r.h_kv)! });
            return (
              <tr key={s.backend + i} data-gqa-row={`${String(r.batch)}-${gq}`}>
                <td>
                  {s.backend}{' '}
                  <span className="muted">
                    ({s.device},{' '}
                    {s.variant === 'work' ? tr('yours', 'của bạn') : tr('reference solution', 'lời giải tham chiếu')})
                  </span>
                </td>
                <td className="num">{String(r.batch)}</td>
                <td className="num">{gq}</td>
                <td className="num">
                  {String(ph?.blocks ?? '?')} : {String(r.blocks)}
                </td>
                <td className="num">{gq} : 1</td>
                <td className="num">{fmt(num(r.speedup_vs_perhead)!)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------- milestones M6 and M7
function MilestoneStatus() {
  const { reports, loading } = useLabs();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const name = (b: string) => (b === 'emu' ? tr('the emulator', 'trình giả lập') : b === 'metal' ? 'Metal' : b.toUpperCase());
  const passes = (l: Lab) => backends.filter((b) => labStatus(reports[b], l.id, 'work') === 'pass');
  const m6 = passes(l1).filter((b) => passes(l2).includes(b));
  const m7 = passes(l4);
  const gqa = passes(l5);
  const say = (bs: string[]) => bs.map(name).join(tr(' and ', ' và '));
  return (
    <p data-milestone={loading ? 'loading' : `m6:${m6.join(',') || 'none'};m7:${m7.join(',') || 'none'}`}>
      {loading ? (
        tr('Reading reports.', 'Đang đọc báo cáo.')
      ) : !backends.length ? (
        tr('No lab results are recorded on this machine yet.', 'Chưa có kết quả lab nào được ghi trên máy này.')
      ) : (
        <>
          {m6.length
            ? tr(
                `M6: your causal softmax and your flash kernel both pass on ${say(m6)} on this machine.`,
                `M6: softmax nhân quả và kernel flash của bạn đều đạt trên ${say(m6)} ở máy này.`,
              )
            : tr(
                'M6: your labs 20.1 and 20.2 do not both pass on any backend of this machine yet.',
                'M6: lab 20.1 và 20.2 của bạn chưa cùng đạt trên backend nào của máy này.',
              )}{' '}
          {m7.length
            ? tr(
                `M7: your append and decode pass on ${say(m7)}${gqa.length ? `, and your grouped kernel on ${say(gqa)}` : ''}.`,
                `M7: append và decode của bạn đạt trên ${say(m7)}${gqa.length ? `, và kernel theo nhóm của bạn đạt trên ${say(gqa)}` : ''}.`,
              )
            : tr(
                'M7: your lab 20.4 does not pass on any backend of this machine yet.',
                'M7: lab 20.4 của bạn chưa đạt trên backend nào của máy này.',
              )}
        </>
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
              Năm lab, mỗi lab trong một thư mục riêng dưới <code>labs/ch20/</code>. Lab 20.1 và 20.2 là mốc dự án M6,
              attention nhân quả cho một lượt prefill: đầu tiên là chặng giữa của pipeline không gộp, rồi toàn bộ
              trong một kernel. Lab 20.3 giấu ba lỗi trong kernel đó, một trong số đó là rào chắn mà đoạn mã của sách bỏ
              sót. Lab 20.4 và 20.5 là mốc dự án M7, một bước decode với KV cache: nối thêm và tính attention, rồi làm
              điều tương tự cho các head grouped-query. Mã bạn không sửa nằm trong <code>labs/ch20/shared-g11/</code>{' '}
              (các bài kiểm thử flash và pipeline không gộp), <code>labs/ch20/shared-g12/</code> (các bài kiểm thử
              decode) và thư mục <code>provided/</code> của lab 20.5.
            </>
          }
        >
          Five labs, each in its own directory under <code>labs/ch20/</code>. Labs 20.1 and 20.2 are milestone M6, causal
          attention for a prefill: first the unfused pipeline&rsquo;s middle stage, then the whole thing as one kernel. Lab
          20.3 hides three defects in that kernel, one of them the barrier the book&rsquo;s listing leaves out. Labs 20.4
          and 20.5 are milestone M7, one decode step with a KV cache: append and attend, then the same for grouped-query
          heads. Code you do not edit sits in <code>labs/ch20/shared-g11/</code> (the flash tests and the unfused
          pipeline), <code>labs/ch20/shared-g12/</code> (the decode tests) and lab 20.5&rsquo;s <code>provided/</code>{' '}
          folder.
        </Translated>
      </LabsIntro>
      <RunningLabs
        example="20.4"
        alsoShownIn={
          <Translated
            vi={
              <>
                các mục <a href={href('ch20', '20.3')}>20.3</a>, <a href={href('ch20', '20.5')}>20.5</a> và{' '}
                <a href={href('ch20', '20.6')}>20.6</a>
              </>
            }
          >
            <>
              sections <a href={href('ch20', '20.3')}>20.3</a>, <a href={href('ch20', '20.5')}>20.5</a> and{' '}
              <a href={href('ch20', '20.6')}>20.6</a>
            </>
          </Translated>
        }
      />
      <BackendsAndEmulator />
      <Translated
        vi={
          <p>
            Mọi lab ở đây cũng được build cho biến thể thứ hai của trình giả lập, <code>race</code>, vốn chạy các luồng
            của một khối cùng lúc dưới ThreadSanitizer và chỉ ra hai dòng mã nguồn của mọi cặp truy cập không được đồng
            bộ hóa. Yêu cầu nó bằng <code>--backend race</code>. Lab nào cũng có lý do để dùng nó: hai phép rút gọn dùng
            chung một ô scratch trong 20.1, các rào chắn warp và ô của kernel flash trong 20.2 và 20.3, và việc các warp
            trộn kết quả qua bộ nhớ chia sẻ trong 20.4 và 20.5. Kích thước bài toán của nó nhỏ, vì một khối 512 luồng dưới
            ThreadSanitizer chạy chậm. Bảng trạng thái bên dưới cho thấy biến thể thứ nhất; bảng này cho thấy biến thể
            thứ hai.
          </p>
        }
      >
        <p>
          Every lab here is also built for the emulator&rsquo;s second flavour, <code>race</code>, which runs one
          block&rsquo;s threads at the same time under ThreadSanitizer and names the two source lines of any
          unsynchronised pair of accesses. Ask for it with <code>--backend race</code>. Each lab has a reason to: two
          reductions sharing a scratch slot in 20.1, the warp and tile barriers of the flash kernel in 20.2 and 20.3, and
          the warps&rsquo; merge through shared memory in 20.4 and 20.5. Its sizes are small, because a 512-thread block
          under ThreadSanitizer is slow. The status table below shows the first flavour; this one shows the second.
        </p>
      </Translated>
      <RaceTable />
      <Translated
        vi={
          <p>
            Có một khoảng trống quan trọng ở chương này hơn hầu hết các chương khác: không trình giả lập nào mô hình hóa
            một warp chạy đồng bộ từng bước (lockstep), và <code>emu</code> coi một phép shuffle là điểm mà cả warp gặp
            nhau. Vì vậy một rào chắn warp bị thiếu vẫn qua trên <code>emu</code> và chỉ <code>race</code> thấy được. Lab
            20.3 được xây dựng trên chính khác biệt đó.
          </p>
        }
      >
        <p>
          One gap matters in this chapter more than in most: neither emulator models a warp running in lockstep, and{' '}
          <code>emu</code> treats a shuffle as a point where the whole warp meets. A missing warp barrier therefore passes
          on <code>emu</code> and is seen only by <code>race</code>. Lab 20.3 is built on that difference.
        </p>
      </Translated>
      <VerificationStatus />

      {/* ------------------------------------------------------------ 20.1 */}
      <h2>
        <Translated vi="Lab 20.1: mặt nạ nhân quả và softmax (mốc dự án M6, phần một)">
          Lab 20.1: causal mask and softmax (milestone M6, part one)
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn viết chặng giữa của attention không gộp:{' '}
            <code>launchCausalSoftmax(S, P, L, batch, nq, nk, scale)</code> biến các điểm số <code>[batch, nq, nk]</code>{' '}
            thành <code>P = softmax(scale · S + M)</code> và ghi logsumexp <code>L = m + log(sum)</code> của mọi hàng vào{' '}
            <code>[batch, nq]</code>. Thiết kế là của Fig. 20.4: một khối 256 luồng cho mỗi hàng, các cột theo bước nhảy,
            mặt nạ làm cận của vòng lặp để một điểm số bị che không bao giờ được đọc, một phép rút gọn max, một rào chắn,
            một phép rút gọn tổng, rồi ghi P. Hai phép rút gọn khối là của lab 10.3, mốc dự án M4, được cung cấp và ghi
            nguồn trong tệp; bạn viết các lượt có mặt nạ, rào chắn giữa hai phép rút gọn, L và lần khởi chạy.
          </p>
        }
      >
        <p>
          You write the middle stage of unfused attention: <code>launchCausalSoftmax(S, P, L, batch, nq, nk, scale)</code>{' '}
          turns scores <code>[batch, nq, nk]</code> into <code>P = softmax(scale · S + M)</code> and writes the logsumexp{' '}
          <code>L = m + log(sum)</code> of every row into <code>[batch, nq]</code>. The design is Fig. 20.4&rsquo;s: one
          block of 256 threads per row, strided columns, the mask as the loop bound so a masked score is never read, a max
          reduction, a barrier, a sum reduction, then the write of P. The two block reductions are lab 10.3&rsquo;s,
          milestone M4, provided and credited in the file; you write the masked passes, the barrier between the
          reductions, L and the launch.
        </p>
      </Translated>
      <Source section="20.3" pages="486–488" figs={tr('Fig. 20.4', 'Hình 20.4')} />
      <Translated
        vi={
          <p>
            Lab khác trang sách ở đâu: <code>gridDim.y</code> là 1, không phải 0, và ở đây nó đi qua batch; giá trị max
            bắt đầu từ <code>-FLT_MAX</code>, vì không header CUDA nào định nghĩa <code>NEG_INFINITY</code>; các phép
            rút gọn của M4 thay cho CUB, vốn đã bỏ <code>cub::Max()</code> và <code>cub::Sum()</code>; và L thay cho mẫu
            số D, vốn không thể dựng lại P nếu thiếu max của hàng. Hai lựa chọn là của khóa học này: kernel tự áp dụng hệ
            số scale, và mặt nạ được căn theo góc dưới bên phải, nên hàng r thấy các cột tới r + (nk − nq). Với nq = nk đó
            là mặt nạ của sách; với nq = 1 đó là một bước decode, thấy mọi key.
          </p>
        }
      >
        <p>
          Where the lab differs from the page: <code>gridDim.y</code> is 1, not 0, and here it walks the batch; the max
          starts at <code>-FLT_MAX</code>, since no CUDA header defines <code>NEG_INFINITY</code>; M4&rsquo;s reductions
          replace CUB, whose <code>cub::Max()</code> and <code>cub::Sum()</code> are gone; and L replaces the
          denominator D, which cannot rebuild P without the row max. Two choices are ours: the kernel applies the scale
          itself, and the mask is aligned to the lower right, so row r sees columns up to r + (nk − nq). With nq = nk that
          is the book&rsquo;s mask; with nq = 1 it is a decode step, which sees every key.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md A1, A5, A7, A8, D4, D5; infra-facts A11)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              FlashAttention từ bản 2.1 và MLX căn mặt nạ của một truy vấn ngắn theo góc dưới bên phải;{' '}
              <code>is_causal</code> của PyTorch căn nó theo góc trên bên trái.
            </p>
          }
        >
          <p>
            FlashAttention since 2.1 and MLX align a short query&rsquo;s mask to the lower right; PyTorch&rsquo;s{' '}
            <code>is_causal</code> aligns it to the upper left.
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
                  <code>causalSoftmaxKernel</code> và <code>launchCausalSoftmax</code>; các phép rút gọn M4 ở đầu tệp đã
                  được cho sẵn
                </>
              }
            >
              <><code>causalSoftmaxKernel</code> and <code>launchCausalSoftmax</code>; the M4 reductions at the top are given</>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated
              vi={
                <>
                  <code>causal_softmax</code>, với <code>simd_max</code>, <code>simd_sum</code> và bộ nhớ threadgroup
                </>
              }
            >
              <><code>causal_softmax</code>, with <code>simd_max</code>, <code>simd_sum</code> and threadgroup memory</>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated
              vi={
                <>
                  lần dispatch trong <code>encodeCausalSoftmax</code>; các phép kiểm tra và binding đã được cho sẵn
                </>
              }
            >
              <>the dispatch in <code>encodeCausalSoftmax</code>; the checks and bindings are given</>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 20.1 --backend emu</code>, rồi <code>--backend race</code>, rồi GPU. Các
            hình dạng đi từ một cột tới 2,048 vuông và 4,097 cột trong một hàng, với nq &lt; nk trong một trường hợp, ở
            batch 1 và 3. Mỗi hàng nhận một công thức dữ liệu, trộn lẫn trong một batch: điểm số đồng đều, cũng vậy nhưng
            dịch đi +90 và −90 (exp(90) không vừa trong một float, nên kernel nào bỏ bước trừ max sẽ thất bại), điểm số
            lớn nhất ở cột nhìn thấy cuối cùng, 10<sup>30</sup> ở mọi cột bị che (kernel nào đọc một cột như vậy sẽ ra số 0
            hoặc NaN), và mọi điểm số bằng nhau. P phải có tổng bằng 1 trên mỗi hàng trong một dung sai mà README suy ra,
            P bị che phải đúng bằng 0, L phải khớp một bản tham chiếu double, và các canary sau P và L phải còn nguyên.
            README cũng nói <code>race</code> ở đây để làm gì: nếu thiếu rào chắn giữa hai phép rút gọn, warp 0 có thể
            ghi đè <code>scratch[0]</code> trước khi một warp chậm kịp đọc giá trị max.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 20.1 --backend emu</code>, then <code>--backend race</code>, then the GPU.
          The shapes run from one column to 2,048 square and 4,097 columns in one row, with nq &lt; nk in one case, at batch
          1 and 3. Each row gets one recipe, mixed within a batch: uniform scores, the same offset by +90 and by −90 (exp(90)
          does not fit in a float, so a kernel that skips the max subtraction fails), the largest score at the last
          visible column, 10<sup>30</sup> in every masked column (a kernel that reads one gets zeros or NaN), and all
          scores equal. P must sum to 1 per row within a tolerance the README derives, masked P must be exactly 0, L must
          match a double reference, and canaries after P and L must survive. The README also says what <code>race</code>{' '}
          is here for: without the barrier between the reductions, warp 0 can overwrite <code>scratch[0]</code> before a
          slow warp has read the max.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Với <code>--bench</code>, chương trình điều khiển (driver) chạy toàn bộ pipeline không gộp ở 8 head, d =
            128 và N = 1024, 2048 và 4096: S = Q Kᵀ bằng M3 của dự án của bạn (bản tham chiếu của lab 15.6), softmax của
            bạn, rồi O = P V bằng M2 (bản tham chiếu của lab 5.3). Nó đo thời gian từng chặng và tính GB/s của softmax
            trên lưu lượng tối thiểu của nó.
          </p>
        }
      >
        <p>
          With <code>--bench</code> the driver runs the whole unfused pipeline at 8 heads, d = 128 and N = 1024, 2048 and
          4096: S = Q Kᵀ with your project&rsquo;s M3 (lab 15.6&rsquo;s reference), your softmax, then O = P V with M2 (lab
          5.3&rsquo;s reference). It times each stage and counts the softmax&rsquo;s GB/s on its minimum traffic.
        </p>
      </Translated>
      <LabPrediction
        id="ch20/labs/p-lab1-share"
        prompt={
          <Translated
            vi={
              <>
                Ở N = 4096, d = 128, kernel mặt nạ và softmax của bạn sẽ chiếm bao nhiêu phần trăm thời gian của pipeline
                ba chặng? Phép nhân S làm 2N²d FLOP, tính cả nửa bị che. Softmax di chuyển khoảng 10N² byte: nửa nhìn thấy
                của S được đọc ba lần, P được ghi một lần. Chia mỗi con số cho tốc độ giới hạn nó, tức trần tính toán và
                băng thông của bạn từ <a href={href('ch05', 'labs')}>lab 5.1</a>.
              </>
            }
          >
            <>
              At N = 4096, d = 128, what percentage of the three-stage pipeline&rsquo;s time will your mask-and-softmax
              kernel take? The S product does 2N²d FLOPs, masked half included. The softmax moves about 10N² bytes: the
              visible half of S read three times, P written once. Divide each by the rate that limits it, your compute
              ceiling and your bandwidth from <a href={href('ch05', 'labs')}>lab 5.1</a>.
            </>
          </Translated>
        }
        unit="%"
        measure={shareAt4096}
        reading={
          <Translated
            vi={
              <>
                Nếu cả hai kernel chạy ở trần của chúng, một thiết bị có điểm gãy (ridge point) R FLOP/B sẽ mất thời gian
                cho phép nhân gấp 2d / (10R) lần so với cho softmax. Rồi hãy hỏi M3 và M2 cách trần tính toán bao xa, và
                softmax của bạn cách trần băng thông bao xa. Mục <a href={href('ch20', '20.3')}>20.3</a> đếm lưu lượng;
                mục <a href={href('ch20', '20.5')}>20.5</a> loại bỏ nó.
              </>
            }
          >
            <>
              If both kernels ran at their ceilings, a device whose ridge point is R FLOP/B would spend 2d / (10R) times as
              long in the product as in the softmax. Then ask how far M3 and M2 are from the compute ceiling, and your
              softmax from the bandwidth one. Section <a href={href('ch20', '20.3')}>20.3</a> counts the traffic; section{' '}
              <a href={href('ch20', '20.5')}>20.5</a> removes it.
            </>
          </Translated>
        }
      />
      <LabResults lab={l1} filter={(r) => r.name === 'unfused_attention'} cols={softmaxCols()} />
      <ReferenceSolution lab={l1} tiers={TIERS_SOFTMAX} />

      {/* ------------------------------------------------------------ 20.2 */}
      <h2>
        <Translated vi="Lab 20.2: flash attention lượt forward (mốc dự án M6, phần hai)">
          Lab 20.2: flash attention forward (milestone M6, part two)
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn viết attention nhân quả thành một kernel, để n × n điểm số của lab 20.1 không bao giờ tồn tại trong bộ nhớ
            device: <code>launchFlashAttention(Q, K, V, O, L, batch, heads, n, scale)</code> trên{' '}
            <code>[batch, heads, n, 128]</code>, n bất kỳ, ghi cùng L như lab 20.1. Thiết kế là của sách: một khối 512
            luồng, 16 warp mỗi warp hai hàng, sở hữu một dải 32 hàng của Q; K, V và ô S/P đi qua bộ nhớ chia sẻ mỗi lần
            một ô B_c key; mỗi luồng giữ phần Q và O của nó cùng max chạy và mẫu số trong thanh ghi. Trục y của lưới đi qua
            các cặp (chuỗi, head).
          </p>
        }
      >
        <p>
          You write causal attention as one kernel, so that the n × n scores of lab 20.1 never exist in device memory:{' '}
          <code>launchFlashAttention(Q, K, V, O, L, batch, heads, n, scale)</code> on <code>[batch, heads, n, 128]</code>,
          any n, writing the same L as lab 20.1. The design is the book&rsquo;s: a block of 512 threads, 16 warps of two
          rows each, owns a panel of 32 rows of Q; K, V and the S/P tile pass through shared memory one tile of B_c keys
          at a time; each thread keeps its slice of Q and O and the running max and denominator in registers. Grid y walks
          the (sequence, head) pairs.
        </p>
      </Translated>
      <Source
        section="20.5"
        pages="492–504"
        figs={tr('Figs. 20.8 to 20.16; Exercises 1 and 3, p. 511', 'Hình 20.8 đến 20.16; Bài tập 1 và 3, p. 511')}
      />
      <Translated
        vi={
          <p>
            Được cung cấp và đã sửa sẵn: các hằng số, kiểu lane mask và warp logic 32 lane, <code>load_Q</code>,{' '}
            <code>load_KT_and_V</code> và <code>store_O</code>. Phần của bạn: <code>initialize</code> (Bài tập 3),{' '}
            <code>compute_S_and_max</code>, <code>update_m_and_D</code>, <code>compute_P_and_update_D</code> với rào chắn
            warp mà trang sách bỏ sót, <code>compute_O</code>, hai vòng lặp của kernel với cả hai rào chắn khối, và lần
            khởi chạy (Bài tập 1). Mục <a href={href('ch20', '20.5')}>20.5</a> liệt kê mọi chỗ lab khác trang sách và lý
            do.
          </p>
        }
      >
        <p>
          Provided and already corrected: the constants, the lane-mask type and the 32-lane logical warp,{' '}
          <code>load_Q</code>, <code>load_KT_and_V</code> and <code>store_O</code>. Yours: <code>initialize</code>{' '}
          (Exercise 3), <code>compute_S_and_max</code>, <code>update_m_and_D</code>, <code>compute_P_and_update_D</code>{' '}
          with the warp barrier the page leaves out, <code>compute_O</code>, the kernel&rsquo;s two loops with both block
          barriers, and the launch (Exercise 1). Section <a href={href('ch20', '20.5')}>20.5</a> lists every place the
          lab departs from the page and why.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Trên Metal, <code>launch.mm</code> cũng là phần của bạn. Các ô của sách ở B_c = 32 chiếm 37,376 byte, nhiều
            hơn 32 KB bộ nhớ threadgroup mà Apple công bố, nên việc tạo pipeline thất bại. Hãy chọn B_c lớn nhất trong 32,
            16, 8 mà các ô vừa <code>maxThreadgroupMemoryLength</code>, biên dịch kernel với giá trị đó qua{' '}
            <code>preprocessorMacros</code>, và kiểm tra rằng pipeline chạy SIMD group 32 và 512 luồng mỗi threadgroup
            trước khi dispatch.
          </p>
        }
      >
        <p>
          On Metal, <code>launch.mm</code> is yours too. The book&rsquo;s tiles at B_c = 32 take 37,376 bytes, more than the
          32 KB of threadgroup memory Apple documents, so pipeline creation fails. Take the largest B_c in 32, 16, 8 whose
          tiles fit <code>maxThreadgroupMemoryLength</code>, compile the kernel with it through{' '}
          <code>preprocessorMacros</code>, and check that the pipeline runs SIMD-groups of 32 and 512 threads per
          threadgroup before dispatching.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md C1, C6; ch05-facts C3; ch03-facts C9)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              Lỗi 32 KB chỉ được quan sát trên một máy Mac. Apple không công bố độ rộng SIMD group cố định: hãy đọc{' '}
              <code>threadExecutionWidth</code> từ pipeline, mà Apple nói trả về kích thước SIMD group.
            </p>
          }
        >
          <p>
            The 32 KB failure was observed on one Mac only. Apple documents no fixed SIMD-group width: read{' '}
            <code>threadExecutionWidth</code> from the pipeline, which Apple says returns the SIMD-group size.
          </p>
        </Translated>
      </Today>
      <LabFiles
        lab={l2}
        files={[
          [
            'kernel.cu',
            <Translated
              vi={
                <>
                  năm hàm device, các vòng lặp và rào chắn của <code>flashAttentionKernel</code>, và{' '}
                  <code>launchFlashAttention</code>
                </>
              }
            >
              <>the five device functions, <code>flashAttentionKernel</code>&rsquo;s loops and barriers, and <code>launchFlashAttention</code></>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated
              vi={
                <>
                  như trên với <code>simd_broadcast</code>, <code>simd_max</code>, <code>simd_sum</code> và{' '}
                  <code>simdgroup_barrier</code>
                </>
              }
            >
              <>the same with <code>simd_broadcast</code>, <code>simd_max</code>, <code>simd_sum</code> and <code>simdgroup_barrier</code></>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated vi={<>việc chọn B_c, biên dịch với nó, các phép kiểm tra pipeline và lần dispatch</>}>
              <>the choice of B_c, the compile with it, the pipeline checks and the dispatch</>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Chạy <code>python3 labs/pmpp.py run 20.2 --backend emu</code> trước; nó chậm, bốn trường hợp dưới ba thứ tự
            luồng. Rồi <code>--backend race</code>, rồi GPU. n chạy qua 1, 2, 31, 32, 33, 63, 64, 65, 100, 257, 1000 và
            2048 với (batch, heads) = (1, 1) và (2, 3). Mỗi head nhận một công thức dữ liệu: dữ liệu đồng đều; max đến
            muộn, nơi điểm số tăng dọc mọi hàng nên max chạy tăng ở mọi ô và mọi lần chỉnh tỉ lệ đều có ý nghĩa; điểm số
            dịch đi khoảng +90; và V có một mức riêng cho mỗi cột. O và L được kiểm tra từng hàng với một bản tham chiếu CPU
            double, với dung sai mà README suy ra từ các cận của exp và phép cộng dồn, các canary sau O và L phải còn
            nguyên, và chương trình điều khiển kiểm tra rằng không có buffer n × n nào được cấp phát.
          </p>
        }
      >
        <p>
          Run <code>python3 labs/pmpp.py run 20.2 --backend emu</code> first; it is slow, four cases under three thread
          orders. Then <code>--backend race</code>, then the GPU. n runs over 1, 2, 31, 32, 33, 63, 64, 65, 100, 257, 1000
          and 2048 with (batch, heads) = (1, 1) and (2, 3). Each head gets one recipe: uniform data; a late max, where the
          score grows along every row so the running max rises in every tile and every rescale matters; scores offset by
          about +90; and V with a distinct level per column. O and L are checked row by row against a double CPU reference
          with a tolerance the README derives from the exp and summation bounds, canaries after O and L must survive, and
          the driver checks that no n × n buffer was allocated.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Với <code>--bench</code>, kernel của bạn chạy đối chọi với pipeline không gộp của lab 20.1 (M3, softmax
            nhân quả tham chiếu, M2) ở 8 head, d = 128, n = 512 đến 4096. GFLOP/s đếm 2n²d mỗi head, một nửa của hai phép
            nhân đầy đủ, vì một nửa S bị che; đường không gộp vẫn tính cả nửa bị che. Trên CUDA và HIP, một bản build
            B_c = 16 chạy bên cạnh bản mặc định 32.
          </p>
        }
      >
        <p>
          With <code>--bench</code>, your kernel runs against lab 20.1&rsquo;s unfused pipeline (M3, the reference causal
          softmax, M2) at 8 heads, d = 128, n = 512 to 4096. GFLOP/s counts 2n²d per head, half of the two full products,
          because half of S is masked; the unfused path computes the masked half anyway. On CUDA and HIP a B_c = 16 build
          runs beside the default 32.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              FlashAttention-2 chỉ chạy ở fp16 và bf16, SDPA của cuDNN không liệt kê fp32 chút nào, và kernel Metal của
              llama.cpp làm hai phép nhân bằng các lệnh ma trận SIMD group (facts A6, D2). Lab này giữ các vòng lặp warp
              fp32 của sách để bạn thấy được mọi lần trao đổi.
            </p>
          }
        >
          <p>
            FlashAttention-2 runs in fp16 and bf16 only, cuDNN&rsquo;s SDPA lists no fp32 at all, and llama.cpp&rsquo;s Metal
            kernel does its two products with SIMD-group matrix instructions (facts A6, D2). This lab keeps the
            book&rsquo;s fp32 warp loops so that you can see every exchange.
          </p>
        </Translated>
      </Beyond>
      <LabPrediction
        id="ch20/labs/p-lab2-ratio"
        prompt={
          <Translated
            vi={
              <>
                Ở n = 4096, thời gian flash của bạn chia cho thời gian của pipeline không gộp. Dưới 1 nghĩa là flash
                nhanh hơn. Hãy cân hai lực. Flash không bao giờ ghi S và P: mỗi head, đường không gộp ghi và đọc lại
                ít nhất 2n² × 4 byte, 128 MiB ở n = 4096. Nhưng hai phép nhân của flash là các vòng lặp warp đơn giản của
                sách, một shuffle cho mỗi phép nhân-cộng khi tính điểm số, trong khi đường không gộp chạy M3 và M2 đã
                tinh chỉnh của bạn.
              </>
            }
          >
            <>
              At n = 4096, your flash time divided by the unfused pipeline&rsquo;s time. Below 1 means flash is faster.
              Weigh two forces. Flash never writes S and P: per head the unfused path writes and rereads at least 2n² × 4
              bytes, 128 MiB at n = 4096. But flash&rsquo;s two products are the book&rsquo;s simple warp loops, one shuffle
              per multiply-add for the scores, while the unfused path runs your tuned M3 and M2.
            </>
          </Translated>
        }
        unit={tr('× the unfused time', '× thời gian bản không gộp')}
        measure={flashRatio4096}
        reading={
          <Translated
            vi={
              <>
                Sách không đưa ra số đo thời gian nào, và không có gì ở đây nói lực nào thắng trên thiết bị của bạn. Chú
                thích 8 (pp. 500–501) nói hai phép nhân có thể dùng các tối ưu của chương 15; các kernel production chạy
                chúng trên phần cứng ma trận với kiểu 16-bit hoặc 8-bit, điều lab này không làm.
              </>
            }
          >
            <>
              The book gives no timing, and nothing here says which force wins on your device. Footnote 8 (pp. 500–501)
              says the two products could use chapter 15&rsquo;s optimisations; production kernels run them on matrix
              hardware in 16-bit or 8-bit types, which this lab does not.
            </>
          </Translated>
        }
      />
      <LabResults lab={l2} filter={(r) => r.name === 'attention'} cols={flashCols()} />
      <ReferenceSolution lab={l2} tiers={TIERS_FLASH} />

      {/* ------------------------------------------------------------ 20.3 */}
      <h2>
        <Translated vi="Lab 20.3: săn race trong flash attention">Lab 20.3: flash attention race hunt</Translated>
      </h2>
      <Translated
        vi={
          <p>
            <code>work/kernel.cu</code> và <code>work/kernel.metal</code> chứa kernel tham chiếu của lab 20.2 với ba lỗi
            độc lập, cùng ba lỗi đó trong cả hai tệp. Nó biên dịch được. Bạn được biết mỗi lỗi thuộc loại gì, không phải nó
            nằm ở đâu: một rào chắn bị thiếu giữa các lane của một warp, một rào chắn bị thiếu trên cả khối, và một giá trị
            sai trong phần sổ sách của online softmax. Hãy tìm và sửa cả ba, và ghi lại backend nào cho bạn thấy lỗi nào.
          </p>
        }
      >
        <p>
          <code>work/kernel.cu</code> and <code>work/kernel.metal</code> hold lab 20.2&rsquo;s reference kernel with three
          independent defects, the same three in both files. It compiles. You are told what kind each one is, not where it
          is: a missing barrier between the lanes of one warp, a missing barrier across the block, and a wrong value in the
          online softmax&rsquo;s bookkeeping. Find and fix all three, and record which backend showed you which.
        </p>
      </Translated>
      <Source section="20.5" pages="492–504" figs={tr('Figs. 20.9, 20.13 to 20.15', 'Hình 20.9, 20.13 đến 20.15')} />
      <LabFiles
        lab={l3}
        files={[
          [
            'kernel.cu',
            <Translated vi={<>kernel của lab 20.2 với ba lỗi</>}>
              <>lab 20.2&rsquo;s kernel with the three defects</>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated vi={<>cùng ba lỗi đó, bằng Metal</>}>
              <>the same three, in Metal</>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated vi={<>phía host tham chiếu của lab 20.2; nó không có lỗi</>}>
              <>lab 20.2&rsquo;s reference host side; it has no defect</>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={<p>Trước khi chạy bất cứ thứ gì, hãy dự đoán. README có một bảng cho cả ba lỗi; đây là lỗi đầu tiên.</p>}
      >
        <p>Before you run anything, predict. The README has a table for all three defects; this is the first.</p>
      </Translated>
      <Ask q={questionsLabs['ch20/labs/q-detector']} label={tr('Predict, then check', 'Dự đoán, rồi kiểm tra')} />
      <Translated vi={<p>Phương pháp là của lab 5.4:</p>}>
        <p>The method is lab 5.4&rsquo;s:</p>
      </Translated>
      <ol>
        <li>
          <Translated
            vi={
              <>
                <code>python3 labs/pmpp.py run 20.3 --backend emu</code>. Các thứ tự luồng của nó cố định, nên một lỗi nó
                thấy được sẽ thất bại mọi lần. Nó chạy chậm với kernel này: hãy khởi động nó và đọc mã trong lúc nó chạy.
              </>
            }
          >
            <code>python3 labs/pmpp.py run 20.3 --backend emu</code>. Its thread orders are fixed, so a defect it can see
            fails every time. It is slow on this kernel: start it and read the code while it runs.
          </Translated>
        </li>
        <li>
          <Translated vi="Sửa một chỗ rồi chạy lại; các lỗi thất bại thay đổi trong lúc bạn làm. Dữ liệu max đến muộn cho thấy vấn đề chỉnh tỉ lệ trước tiên.">
            Fix one thing and run again; the failures change as you go. The late-max data shows a rescale problem first.
          </Translated>
        </li>
        <li>
          <Translated
            vi={
              <>
                <code>python3 labs/pmpp.py run 20.3 --backend race</code>. ThreadSanitizer chỉ ra hai dòng va chạm nhau.
                Một hàm được inline có thể hiện ra là <code>flashAttentionKernel</code> không có số dòng; phía còn lại vẫn
                chỉ ra dòng của nó. Mỗi cặp được báo một lần mỗi lần chạy, nên một báo cáo sạch không phải là bằng chứng.
              </>
            }
          >
            <code>python3 labs/pmpp.py run 20.3 --backend race</code>. ThreadSanitizer names the two lines that collide. An
            inlined function may show as <code>flashAttentionKernel</code> with no line number; the other side still names
            its line. Each pair is reported once per run, so a clean report is not a proof.
          </Translated>
        </li>
        <li>
          <Translated vi="Rồi đến GPU của bạn. Ghi lại những lỗi nào nó che giấu.">
            Then your GPU. Note which defects it hid.
          </Translated>
        </li>
      </ol>
      <Translated
        vi={
          <p>
            Các bài kiểm thử là của lab 20.2. Khi lab được viết, mỗi lỗi được kiểm tra riêng: rào chắn warp chỉ thất bại
            trên <code>race</code>; rào chắn khối thất bại trên <code>emu</code> ở mọi thứ tự luồng, được{' '}
            <code>race</code> báo ra và thất bại trên Metal; lỗi sổ sách thất bại ở mọi nơi.
          </p>
        }
      >
        <p>
          The tests are lab 20.2&rsquo;s. When the lab was written each defect was checked alone: the warp barrier failed
          only on <code>race</code>; the block barrier failed on <code>emu</code> in every thread order, was reported by{' '}
          <code>race</code> and failed on Metal; the bookkeeping error failed everywhere.
        </p>
      </Translated>
      <LabResults
        lab={l3}
        noBench={tr(
          'This lab has no benchmark: lab 20.2 times the same kernel. The result is the line above, every test passing on the emulators and on a device. As shipped, the kernel is meant to fail.',
          'Lab này không có benchmark: lab 20.2 đo thời gian chính kernel này. Kết quả là dòng ở trên, mọi bài kiểm thử đều đạt trên các trình giả lập và trên một thiết bị. Khi phát hành, kernel được cố ý để thất bại.',
        )}
      />
      <ReferenceSolution lab={l3} tiers={TIERS_FLASH} />

      <h3>
        <Translated vi="Dự án: mốc dự án M6">The project: milestone M6</Translated>
      </h3>
      <Translated
        vi={
          <p>
            M6 là attention nhân quả cho một lượt prefill theo các quy ước của dự án: tensor <code>float</code> theo thứ
            tự hàng (row-major), batch đứng đầu, con trỏ device thô và kích thước tường minh, và một hàm{' '}
            <code>launch</code> bên cạnh mỗi kernel. Hai dạng, cả hai cùng một L: pipeline không gộp (M3,{' '}
            <code>launchCausalSoftmax</code>, M2) và <code>launchFlashAttention</code>. Gọi dạng nào ở một n cho trước là
            câu trả lời của benchmark của bạn.
          </p>
        }
      >
        <p>
          M6 is causal attention for a prefill in the project&rsquo;s conventions: row-major <code>float</code> tensors,
          batch first, raw device pointers and explicit sizes, and a <code>launch</code> function beside each kernel. Two
          forms, both with the same L: the unfused pipeline (M3, <code>launchCausalSoftmax</code>, M2) and{' '}
          <code>launchFlashAttention</code>. Which one to call at a given n is your benchmark&rsquo;s answer.
        </p>
      </Translated>

      {/* ------------------------------------------------------------ 20.4 */}
      <h2>
        <Translated vi="Lab 20.4: nối thêm vào KV cache và decode một token (mốc dự án M7)">
          Lab 20.4: KV cache append and single-token decode (milestone M7)
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một bước decode của một lớp attention, cho một batch các chuỗi có độ dài khác nhau. Viết hai hàm host và các
            kernel của chúng: <code>launchKvAppend</code> ghi key và value mới của mỗi chuỗi vào hàng{' '}
            <code>pos[b]</code> của các cache có bố cục <code>[batch, max_seq, h_kv, 128]</code> và không chạm vào gì
            khác, và <code>launchDecodeAttention</code> cho mỗi head truy vấn tính attention trên <code>len[b]</code>{' '}
            hàng đầu tiên trong cache của head K/V của nó, <code>h / (h_q / h_kv)</code>, để một kernel phục vụ cả MHA,
            GQA và MQA. Một truy vấn mới duy nhất không cần mặt nạ.
          </p>
        }
      >
        <p>
          One decode step of one attention layer, for a batch of sequences with different lengths. Write two host
          functions and their kernels: <code>launchKvAppend</code> writes each sequence&rsquo;s new key and value at row{' '}
          <code>pos[b]</code> of caches laid out <code>[batch, max_seq, h_kv, 128]</code> and touches nothing else, and{' '}
          <code>launchDecodeAttention</code> lets each query head attend over the first <code>len[b]</code> cached rows of
          its K/V head, <code>h / (h_q / h_kv)</code>, so one kernel serves MHA, GQA and MQA. A single new query needs no
          mask.
        </p>
      </Translated>
      <Source
        section={tr('20.4 to 20.6', '20.4 đến 20.6')}
        pages="488–492, 493–504, 506–507"
        figs={tr('Figs. 20.5 to 20.7; Eqs. 20.3 to 20.6 and 20.9', 'Hình 20.5 đến 20.7; Pt. 20.3 đến 20.6 và 20.9')}
      />
      <Translated
        vi={
          <p>
            Sách không đưa mã nào cho một cache hay một kernel decode (p. 492); thiết kế là của khóa học này. Một khối 256
            luồng cho mỗi (head, chuỗi). Mỗi warp trong 8 warp nhận cứ mỗi key thứ tám; 32 lane chia chiều head, mỗi lane 4
            giá trị, và một phép cộng butterfly cho mọi lane điểm số. Mỗi warp giữ max chạy, mẫu số và đầu ra chưa chuẩn
            hóa của riêng nó, cập nhật từng key một như kernel ở mục 20.5 cập nhật từng ô một. Cuối cùng 8 warp trộn kết
            quả qua bộ nhớ chia sẻ bằng Pt. 20.4 và 20.6, phép trộn split-K mà FlashAttention-1 dùng giữa các warp và
            Flash-Decoding dùng giữa các khối.
          </p>
        }
      >
        <p>
          The book shows no code for a cache or a decode kernel (p. 492); the design is ours. One block of 256 threads per
          (head, sequence). Each of the 8 warps takes every eighth key; the 32 lanes split the head dimension, 4 values
          each, and a butterfly sum gives every lane the score. Each warp keeps its own running max, denominator and
          unnormalised output, updated one key at a time as section 20.5&rsquo;s kernel does one tile at a time. At the
          end the 8 warps merge through shared memory with Eqs. 20.4 and 20.6, the split-K merge that FlashAttention-1
          used across warps and Flash-Decoding uses across blocks.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md A7, A8, D3, D4)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              Bố cục cache là của mã Llama tham chiếu của Meta; phép append là một phép scatter theo chỉ số, như{' '}
              <code>ggml_set_rows</code> của llama.cpp. Kernel decode Metal của MLX, <code>sdpa_vector</code>, có cùng
              hình dạng với thiết kế này.
            </p>
          }
        >
          <p>
            The cache layout is Meta&rsquo;s reference Llama code&rsquo;s; the append is a scatter by index, as
            llama.cpp&rsquo;s <code>ggml_set_rows</code>. MLX&rsquo;s Metal decode kernel, <code>sdpa_vector</code>, has the
            same shape as this design.
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
                  <code>warpSumAll</code>, <code>kvAppendKernel</code>, <code>decodeAttentionKernel</code> và hai hàm
                  khởi chạy
                </>
              }
            >
              <><code>warpSumAll</code>, <code>kvAppendKernel</code>, <code>decodeAttentionKernel</code> and the two launchers</>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated vi={<><code>kv_append</code> và <code>decode_attention</code></>}>
              <><code>kv_append</code> and <code>decode_attention</code></>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated vi={<>hai lần dispatch; các binding đã được cho sẵn</>}>
              <>the two dispatches; the bindings are given</>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Phép append được kiểm tra từng bit: các hàng mới nằm ở <code>pos[b]</code>, mọi phần tử khác của cả hai cache
            giữ nguyên canary. Decode chạy (h_q, h_kv) = (8, 8), (8, 2), (8, 1) và (6, 2), một đến bốn chuỗi mỗi lời gọi
            với độ dài tới 4,097 trên GPU (nhỏ hơn trên các trình giả lập), trên key ngẫu nhiên và trên một max đến muộn mà
            chỉ một warp thấy, nên phép trộn phải chỉnh tỉ lệ các warp khác. Các hàng từ <code>len[b]</code> trở đi chứa
            NaN. Rồi đến vòng lặp của mốc dự án: 16 bước tự hồi quy, trong đó đầu ra của mỗi bước phải bằng hàng cuối của
            một attention nhân quả đầy đủ tính lại từ đầu trên mọi token cho đến lúc đó. Đó là khẳng định của mục{' '}
            <a href={href('ch20', '20.4')}>20.4</a>, được kiểm thử.
          </p>
        }
      >
        <p>
          The append is checked bit for bit: the new rows land at <code>pos[b]</code>, every other element of both caches
          keeps its canary. Decode runs (h_q, h_kv) = (8, 8), (8, 2), (8, 1) and (6, 2), one to four sequences per call with
          lengths up to 4,097 on a GPU (smaller on the emulators), on random keys and on a late max that only one warp
          sees, so the merge must rescale the others. Rows at and past <code>len[b]</code> hold NaN. Then the milestone
          loop: 16 autoregressive steps where each step&rsquo;s output must equal the last row of a full causal attention
          recomputed from scratch over every token so far. That is section <a href={href('ch20', '20.4')}>20.4</a>&rsquo;s
          claim, tested.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Với <code>--bench</code>, chỉ riêng decode attention ở h_q = h_kv = 32, chiều head 128 (một lớp MHA cỡ 7B, lựa
            chọn của khóa học này), len 512 đến 32,768, batch 1 và 8. GB/s đếm số byte cache mà một lời gọi phải đọc, 2 ×
            len × h_kv × 128 × 4 mỗi chuỗi. Các kích thước không vừa trong một nửa bộ nhớ thiết bị bị bỏ qua và được liệt
            kê. Phép append được đo thời gian riêng.
          </p>
        }
      >
        <p>
          With <code>--bench</code>, decode attention alone at h_q = h_kv = 32, head dimension 128 (a 7B-class MHA layer,
          our choice), len 512 to 32,768, batch 1 and 8. GB/s counts the cache bytes one call must read, 2 × len × h_kv ×
          128 × 4 per sequence. Sizes that do not fit in half the device memory are skipped and listed. The append is timed
          separately.
        </p>
      </Translated>
      <LabPrediction
        id="ch20/labs/p-lab4-ms"
        prompt={
          <Translated
            vi={
              <>
                Dùng băng thông bạn đã đo ở <a href={href('ch05', 'labs')}>lab 5.1</a>, thời gian tính bằng mili giây của
                một lời gọi decode ở len 8192, batch 1, 32 head kích thước 128 ở fp32. Mỗi phần tử 4 byte của K và V được
                dùng trong một phép nhân-cộng: khoảng 0.5 FLOP/B, thấp xa dưới mọi điểm gãy, với 256 MiB cache phải đọc.
              </>
            }
          >
            <>
              Using the bandwidth you measured in <a href={href('ch05', 'labs')}>lab 5.1</a>, the time in milliseconds of
              one decode call at len 8192, batch 1, 32 heads of 128 in fp32. Each 4-byte element of K and V is used in one
              multiply-add: about 0.5 FLOP/B, far below any ridge point, with 256 MiB of cache to read.
            </>
          </Translated>
        }
        unit="ms"
        measure={decodeMs}
        reading={
          <Translated
            vi={
              <>
                Pt. 20.9 có được &ldquo;khoảng 1&rdquo; bằng cách đếm một phép toán cho mỗi phần tử một byte; lập luận
                vẫn vậy. GB/s đo được tiến gần mức của lab 5.1 đến đâu chính là tỉ lệ băng thông mà kernel này đạt được.
                Mục <a href={href('ch20', '20.6')}>20.6</a> đặt cùng các hàng này cạnh mô hình.
              </>
            }
          >
            <>
              Eq. 20.9 gets &ldquo;about 1&rdquo; by counting one operation per one-byte element; the argument is the same.
              How close the measured GB/s comes to lab 5.1&rsquo;s is the fraction of your bandwidth this kernel reaches.
              Section <a href={href('ch20', '20.6')}>20.6</a> sets the same rows beside the model.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch20/labs/p-lab4-batch8"
        prompt={
          <Translated
            vi={
              <>
                Ở cùng độ dài, thời gian của batch 8 bằng bao nhiêu lần của batch 1. Mỗi chuỗi có cache riêng, nên batch 8
                đọc gấp 8 lần số byte; nhưng batch 1 khởi chạy 32 khối còn batch 8 khởi chạy 256.
              </>
            }
          >
            <>
              At the same length, batch 8 as a multiple of batch 1&rsquo;s time. Each sequence has its own cache, so batch
              8 reads 8 times the bytes; but batch 1 launches 32 blocks and batch 8 launches 256.
            </>
          </Translated>
        }
        unit={tr('× batch 1', '× batch 1')}
        measure={decodeBatch8}
        reading={
          <Translated
            vi={
              <>
                Fig. 20.17 nói attention không được lợi từ batching: số byte trên mỗi FLOP không đổi. Một tỉ số thấp hơn
                hẳn 8 cho thấy batch 1 đã không giữ đủ yêu cầu đang bay để đạt băng thông của bạn, đúng điểm mà
                Flash-Decoding nêu về các batch nhỏ (facts A8).
              </>
            }
          >
            <>
              Fig. 20.17 says attention does not benefit from batching: the bytes per FLOP do not change. A ratio well below
              8 says batch 1 was not keeping enough requests in flight to reach your bandwidth, the point Flash-Decoding
              makes about small batches (facts A8).
            </>
          </Translated>
        }
      />
      <LabResults lab={l4} filter={(r) => r.name === 'decode' || r.name === 'append'} cols={decodeCols()} />
      <ReferenceSolution lab={l4} tiers={TIERS_DECODE} />

      {/* ------------------------------------------------------------ 20.5 */}
      <h2>
        <Translated vi="Lab 20.5: decode grouped-query, nạp mỗi hàng K/V một lần">
          Lab 20.5: grouped-query decode, loading each K/V row once
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            <b>Hãy làm xong lab 20.4 trước.</b> Mã khởi đầu (starter) của lab này bắt đầu từ kernel theo từng head tham
            chiếu của lab 20.4, nên mở nó ra là lộ đáp án của lab đó.
          </p>
        }
      >
        <p>
          <b>Finish lab 20.4 first.</b> This lab&rsquo;s starter begins from lab 20.4&rsquo;s reference per-head kernel, so
          opening it gives away that lab&rsquo;s answer.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Trong kernel của lab 20.4, g_q head truy vấn dùng chung một head K/V mỗi head có một khối riêng, và mỗi khối
            đọc cùng các hàng K và V: kernel yêu cầu bộ nhớ g_q lần số byte mà cache chứa. Viết{' '}
            <code>launchDecodeAttentionGrouped</code>, với các đối số và hợp đồng của lab 20.4, sao cho một khối cho mỗi
            (head K/V, chuỗi) nạp mỗi hàng một lần và dùng nó cho cả g_q truy vấn, như đường <code>gqa_ratio</code> của
            llama.cpp. Mỗi warp giữ g_q trạng thái chạy; phép trộn chạy một lần cho mỗi head truy vấn qua cùng buffer
            chia sẻ. Kích thước nhóm là một hằng số lúc biên dịch: một template trên <code>GQ</code> trong CUDA, một macro
            tiền xử lý mà chương trình điều khiển đặt khi biên dịch thư viện Metal.
          </p>
        }
      >
        <p>
          In lab 20.4&rsquo;s kernel the g_q query heads that share a K/V head each get their own block, and each block reads
          the same K and V rows: the kernel asks memory for g_q times the bytes the cache holds. Write{' '}
          <code>launchDecodeAttentionGrouped</code>, with lab 20.4&rsquo;s arguments and contract, so that one block per
          (K/V head, sequence) loads each row once and uses it for all g_q queries, as llama.cpp&rsquo;s{' '}
          <code>gqa_ratio</code> path does. Each warp keeps g_q running states; the merge runs once per query head through
          the same shared buffer. The group size is a compile-time constant: a template on <code>GQ</code> in CUDA, a
          preprocessor macro the driver sets when it compiles the Metal library.
        </p>
      </Translated>
      <Source section="20.7" pages="508–509" figs={tr('Eqs. 20.12 and 20.13', 'Pt. 20.12 và 20.13')} />
      <Translated
        vi={
          <p>
            Ở đây g_q là số head truy vấn trên mỗi head K/V, nghĩa mà Pt. 20.12 và 20.13 cần; văn bản của sách gọi nó là
            số nhóm. Mục <a href={href('ch20', '20.7')}>20.7</a> làm rõ điều này. &ldquo;≈ g_q&rdquo; của Pt. 20.13 đếm
            các phần tử một byte; ở fp32 cường độ tính toán của kernel khoảng g_q / 2 FLOP/B.
          </p>
        }
      >
        <p>
          Here g_q is the number of query heads per K/V head, the meaning Eqs. 20.12 and 20.13 need; the book&rsquo;s prose
          calls it the number of groups. Section <a href={href('ch20', '20.7')}>20.7</a> straightens this out. Eq.
          20.13&rsquo;s &ldquo;≈ g_q&rdquo; counts one-byte elements; in fp32 the kernel&rsquo;s intensity is about g_q / 2
          FLOP/B.
        </p>
      </Translated>
      <LabFiles
        lab={l5}
        files={[
          [
            'kernel.cu',
            <Translated
              vi={
                <>
                  <code>decodeAttentionGroupedKernel&lt;GQ&gt;</code>, bắt đầu từ kernel theo từng head của lab 20.4, và{' '}
                  <code>launchDecodeAttentionGrouped</code>
                </>
              }
            >
              <><code>decodeAttentionGroupedKernel&lt;GQ&gt;</code>, starting from lab 20.4&rsquo;s per-head kernel, and <code>launchDecodeAttentionGrouped</code></>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated vi={<><code>decode_attention_grouped</code>, cùng điểm xuất phát</>}>
              <><code>decode_attention_grouped</code>, the same starting point</>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated vi={<>phép kiểm tra kích thước nhóm và lần dispatch</>}>
              <>the group-size check and the dispatch</>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Các bài kiểm thử là của lab 20.4 với g_q = 1, 2, 3, 4 và 8, rồi vòng lặp 16 bước với phép append của lab 20.4
            và kernel của bạn. Chúng kiểm tra đáp án, không kiểm tra việc gom nhóm: một kernel theo từng head khởi chạy
            trên (h_q, batch) cũng qua được. Chỉ benchmark mới cho thấy mỗi hàng có được đọc một lần hay không. Nó đo thời
            gian kernel tham chiếu của lab 20.4 và kernel của bạn ở h_q = 32, len 8192, batch 1 và 4, và h_kv = 32, 8 và 4,
            với GB/s tính trên số byte cache không trùng lặp cho cả hai.
          </p>
        }
      >
        <p>
          The tests are lab 20.4&rsquo;s with g_q = 1, 2, 3, 4 and 8, then the 16-step loop with lab 20.4&rsquo;s append and
          your kernel. They check the answers, not the grouping: a per-head kernel launched over (h_q, batch) passes them.
          Only the benchmark shows whether each row is read once. It times lab 20.4&rsquo;s reference kernel and yours at
          h_q = 32, len 8192, batch 1 and 4, and h_kv = 32, 8 and 4, with GB/s on the unique cache bytes for both.
        </p>
      </Translated>
      <LabPrediction
        id="ch20/labs/p-lab5-speedup"
        prompt={
          <Translated
            vi={
              <>
                Ở h_kv = 8 (g_q = 4), batch 1: kernel theo nhóm của bạn sẽ nhanh hơn kernel theo từng head bao nhiêu lần?
                Kernel theo từng head yêu cầu gấp 4 lần số byte, nhưng một bộ nhớ đệm (cache) có thể phục vụ các lần lặp
                lại nếu 4 khối của một nhóm chạy gần như cùng lúc; và kernel của bạn khởi chạy 8 khối trong khi kernel
                theo từng head khởi chạy 32.
              </>
            }
          >
            <>
              At h_kv = 8 (g_q = 4), batch 1: how many times faster will your grouped kernel be than the per-head kernel?
              The per-head kernel requests 4 times the bytes, but a cache may serve the repeats if the 4 blocks of a group
              run at about the same time; and your kernel launches 8 blocks where the per-head one launches 32.
            </>
          </Translated>
        }
        unit={tr('× the per-head kernel', '× kernel theo từng head')}
        measure={groupedSpeedup}
        reading={
          <Translated
            vi={
              <>
                Dưới 1 nghĩa là kernel theo nhóm chậm hơn. Cả CUDA lẫn Apple đều không hứa một thứ tự cho các khối, nên
                việc các khối của một nhóm có chồng lên nhau hay không là điều cần đo. Ít khối là lý do Flash-Decoding
                cũng chia các key của một head ra nhiều khối (facts A8).
              </>
            }
          >
            <>
              Below 1 means the grouped kernel is slower. Neither CUDA nor Apple promises an order for blocks, so whether a
              group&rsquo;s blocks overlap is something to measure. Few blocks is the reason Flash-Decoding also splits a
              head&rsquo;s keys across blocks (facts A8).
            </>
          </Translated>
        }
      />
      <Translated
        vi={<p>Mọi tỉ số đo được, bên cạnh hệ số mà Pt. 20.13 nói số byte được yêu cầu giảm đi:</p>}
      >
        <p>Every measured ratio, beside the factor by which Eq. 20.13 says the bytes requested shrink:</p>
      </Translated>
      <GroupedVsIntensity />
      <Translated
        vi={
          <p>
            Ở chỗ việc gom nhóm làm mất khối, cách chữa là chia cả các key, để một head K/V lại có nhiều khối.
          </p>
        }
      >
        <p>
          Where grouping costs blocks, the remedy is to split the keys as well, so that one K/V head gets several blocks
          again.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Bước tiếp theo, không được xây ở đây: cho mỗi (head K/V, chuỗi) nhiều khối trên các lát của key và trộn
              chúng trong một kernel thứ hai bằng cùng Pt. 20.4 và 20.6. Flash-Decoding làm điều này (
              <code>num_splits</code>), llama.cpp trộn bằng <code>flash_attn_combine_results</code>, và MLX chuyển sang
              một kernel hai lượt cho các cache GQA dài (facts A8, D1, D4).
            </p>
          }
        >
          <p>
            The next step, not built here: give each (K/V head, sequence) several blocks over slices of the keys and merge
            them in a second kernel with the same Eqs. 20.4 and 20.6. Flash-Decoding does this (<code>num_splits</code>),
            llama.cpp merges with <code>flash_attn_combine_results</code>, and MLX switches to a two-pass kernel for long
            GQA caches (facts A8, D1, D4).
          </p>
        </Translated>
      </Beyond>
      <LabResults lab={l5} filter={(r) => r.name === 'perhead' || r.name === 'grouped'} cols={groupedCols()} />
      <ReferenceSolution lab={l5} tiers={TIERS_DECODE} />

      <h3>
        <Translated vi="Dự án: mốc dự án M7">The project: milestone M7</Translated>
      </h3>
      <Translated
        vi={
          <p>
            M7 là một bước decode của attention: <code>launchKvAppend</code> và <code>launchDecodeAttention</code> trên
            một cache có bố cục <code>[batch, max_seq, h_kv, 128]</code>, với độ dài theo từng chuỗi, và{' '}
            <code>launchDecodeAttentionGrouped</code> là dạng grouped-query của nó. Nghiên cứu batching của chương 22,
            mốc dự án M8, hỏi việc batching một bước decode như vậy mang lại gì về thông lượng và tốn gì về độ trễ.
          </p>
        }
      >
        <p>
          M7 is one decode step of attention: <code>launchKvAppend</code> and <code>launchDecodeAttention</code> over a
          cache laid out <code>[batch, max_seq, h_kv, 128]</code>, with per-sequence lengths, and{' '}
          <code>launchDecodeAttentionGrouped</code> as its grouped-query form. Chapter 22&rsquo;s batching study, milestone
          M8, asks what batching such a decode step buys in throughput and costs in latency.
        </p>
      </Translated>
      <MilestoneStatus />
    </LabsPage>
  );
}
