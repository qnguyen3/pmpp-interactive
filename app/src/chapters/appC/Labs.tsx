import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Source, Today, type Tier } from '../../components/notes';
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
  preferredBench,
  rawCol,
  useLabs,
  whose,
  type BenchSet,
  type Col,
} from '../../components/LabKit';
import { Translated, tr } from '../../lib/i18n';
import { fmt, num, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { paradigm, parseParadigmRow } from '../../sim/umparadigm';
import { ratioFromProbe } from '../../sim/zerocopy';

// The page is built from components/LabKit.tsx; what is here is appendix C's own content.
// The lab runner resolves "N.M" to chNN/labM_ only, so the lab's number on this page is
// its directory prefix, appC/lab1, which `pmpp.py run` finds by substring.
const LABS = defineLabs('appC', [
  { num: 'appC/lab1', dir: 'lab1_feeding', name: 'three ways to feed a kernel', vi: { name: 'ba cách cấp dữ liệu cho một kernel' } },
]);
const [l1] = LABS;
const RUN = 'python3 labs/pmpp.py run appC/lab1';

// Identifier colours for the reference solution: the three arrays live wherever the path
// puts them, and the kernel's own indices are registers.
const TIERS: Record<string, Tier> = {
  table: 'global',
  ids: 'global',
  out: 'global',
  table_d: 'global',
  ids_d: 'global',
  out_d: 'global',
  i: 'reg',
  r: 'reg',
  c: 'reg',
};

export const questionsLabs: Record<string, Question> = {
  'appC/labs/q-managed-like': {
    kind: 'choice',
    id: 'appC/labs/q-managed-like',
    prompt: (
      <Translated
        vi={
          <>
            Hai máy chạy workload <code>sparse1</code> của lab, trong đó kernel đọc một hàng trong 64 hàng của bảng, một lần.
            Máy A: Windows, một GPU NVIDIA, <code>ConcurrentManagedAccess</code> = 0. Máy B: Linux, một GPU AMD CDNA,{' '}
            <code>HSA_XNACK</code> chưa được đặt. Trên mỗi máy, <code>feedManaged</code> hoạt động giống đường nào?
          </>
        }
      >
        <>
          Two machines run the lab's <code>sparse1</code> workload, in which the kernel reads one row in 64 of the table,
          once. Machine A: Windows, an NVIDIA GPU, <code>ConcurrentManagedAccess</code> = 0. Machine B: Linux, an AMD
          CDNA GPU, <code>HSA_XNACK</code> not set. Which path does <code>feedManaged</code> behave like on each?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="A giống explicit: toàn bộ bộ nhớ managed chuyển sang GPU lúc khởi chạy. B giống mapped: kernel đọc bộ nhớ host được ghim.">
            {'A like explicit: all managed memory moves to the GPU at the launch. B like mapped: the kernel reads pinned host memory.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="A giống mapped, B giống explicit: mô hình hạn chế đọc tại chỗ, và AMD không có XNACK sao chép lúc khởi chạy.">
            {'A like mapped, B like explicit: the limited paradigm reads in place, and AMD without XNACK copies at the launch.'}
          </Translated>
        ),
        why: (
          <Translated vi="Bị đảo ngược. Mô hình hạn chế chuyển bộ nhớ managed hàng loạt lúc khởi chạy, còn AMD không có XNACK thì hoàn toàn không di chuyển.">
            {'Swapped. The limited paradigm moves managed memory in bulk at a launch, and AMD without XNACK does not migrate at all.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Cả hai giống explicit: bộ nhớ managed nằm trong bộ nhớ device trước khi bất kỳ kernel nào đọc nó.">
            Both like explicit: managed memory ends up in device memory before any kernel reads it.
          </Translated>
        ),
        why: (
          <Translated vi="Không đúng với B. Không có XNACK, bộ nhớ managed của AMD không di chuyển và hoạt động như bộ nhớ host được ghim: kernel đọc nó trong bộ nhớ host.">
            {"Not on B. Without XNACK, AMD's managed memory does not migrate and behaves like pinned host memory: the kernel reads it in host memory."}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Cả hai giống mapped về số byte được chuyển: chỉ 1 trong 64 hàng mà kernel chạm tới đi qua đường truyền, trên cả hai máy.">
            {'Both like mapped in bytes moved: only the 1 in 64 rows the kernel touches cross the link, on either machine.'}
          </Translated>
        ),
        why: (
          <Translated vi="Chỉ chuyển những gì được chạm tới, từng trang khi có lỗi trang, là mô hình đầy đủ. Máy A có mô hình hạn chế, chuyển tất cả lúc khởi chạy.">
            {'Moving only what is touched, page by page on a fault, is the full paradigm. Machine A has the limited one, which moves all of it at the launch.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trên A, mô hình hạn chế (mục <a href={href('appC', 'C.7')}>C.7</a>) là hành vi CUDA 6 của sách trên Kepler và
            Maxwell: mỗi khi một kernel bắt đầu, toàn bộ bộ nhớ managed nói chung chuyển sang GPU, nên cả bảng đi qua một lần,
            như trong <code>feedExplicit</code>. Trên B, bộ nhớ managed không có XNACK hoạt động như bộ nhớ host được ghim, và
            mỗi hàng kernel đọc đều đi qua đường truyền, như trong <code>feedMapped</code>. Chỉ trên mô hình đầy đủ, các trang
            mới di chuyển khi, và chỉ khi, GPU chạm tới chúng. Benchmark in ra mô hình của máy bạn trước mọi phép đo thời gian.
          </p>
        }
      >
        <p>
          On A, the limited paradigm (section <a href={href('appC', 'C.7')}>C.7</a>) is the book's CUDA 6 behaviour on
          Kepler and Maxwell: whenever a kernel starts, all managed memory generally moves to the GPU, so the whole table
          crosses once, as in <code>feedExplicit</code>. On B, managed memory without XNACK behaves like pinned host
          memory, and every row the kernel reads crosses the link, as in <code>feedMapped</code>. Only on a full paradigm do pages move
          when, and only if, the GPU touches them. The benchmark prints which paradigm your machine has before any timing.
        </p>
      </Translated>
    ),
    source: 'vendor documentation, not in the book: appC-facts A10 (limited paradigm), B5 and ch02-facts B7 (AMD without XNACK); the book\'s CUDA 6 limits are p. 627',
    vi: {
      source:
        'tài liệu của nhà cung cấp, không có trong sách: appC-facts A10 (mô hình hạn chế), B5 và ch02-facts B7 (AMD không có XNACK); các giới hạn CUDA 6 của sách ở p. 627',
    },
  },
};

// ---------------------------------------------------------------- reading the benchmark rows
const ms = (s: number) => `${fmt(s * 1000)} ms`; // i18n-ignore: number and unit
const mib = (bytes: number) => `${fmt(bytes / 2 ** 20)} MiB`; // i18n-ignore: number and unit
const feedRow = (s: BenchSet, workload: string, path: string): BenchRow | undefined =>
  s.rows.find((r) => r.name === 'feed' && r.workload === workload && r.path === path);

/** time(a) / time(b) for one workload, from the rows of one backend. */
function ratio(s: BenchSet, workload: string, a: string, b: string, extra?: (s: BenchSet) => string) {
  const ra = feedRow(s, workload, a);
  const rb = feedRow(s, workload, b);
  const ta = num(ra?.seconds);
  const tb = num(rb?.seconds);
  if (!ra || !ta || !tb) return null;
  const table = num(ra.table_bytes);
  return {
    value: ta / tb,
    detail: tr(
      `${ms(ta)} for ${a} and ${ms(tb)} for ${b}, ${workload}` +
        (table ? `, a table of ${mib(table)}` : '') +
        (extra ? extra(s) : ''),
      `${ms(ta)} cho ${a} và ${ms(tb)} cho ${b}, ${workload}` +
        (table ? `, một bảng ${mib(table)}` : '') +
        (extra ? extra(s) : ''),
    ),
  };
}

const sparseRatio = (reports: Reports) => measure(reports, l1, (s) => ratio(s, 'sparse1', 'explicit', 'mapped'));
const repeatRatio = (reports: Reports) => measure(reports, l1, (s) => ratio(s, 'dense8', 'mapped', 'explicit'));
const metalRatio = (reports: Reports) =>
  measure(reports, l1, (s) =>
    ratio(s, 'dense8', 'private', 'shared', (set) => {
      const nc = num(feedRow(set, 'dense8', 'nocopy')?.seconds);
      const sh = num(feedRow(set, 'dense8', 'shared')?.seconds);
      return nc && sh ? tr(`; no-copy took ${ms(nc)}, ${fmt(nc / sh)} times shared`, `; no-copy mất ${ms(nc)}, gấp ${fmt(nc / sh)} lần shared`) : '';
    }),
  );

// ---------------------------------------------------------------- what the benchmark printed first
function Instruments() {
  const { reports, loading, reload } = useLabs();
  const sets = preferredBench(reports, l1);
  const lines = sets.flatMap((s) => {
    const out: { key: string; backend: string; what: string; says: string }[] = [];
    const where = `${s.backend} (${s.device}), ${s.date}`;
    const par = s.rows.find((r) => r.name === 'paradigm');
    const attrs = par ? parseParadigmRow(par as Record<string, string | number>) : null;
    if (par)
      out.push({
        key: s.backend + 'paradigm',
        backend: where,
        what: tr('Unified Memory paradigm', 'Mô hình Unified Memory'),
        says: attrs
          ? `${paradigm(attrs).row.name} (ConcurrentManagedAccess ${attrs.concurrentManagedAccess ? 1 : 0}, PageableMemoryAccess ${attrs.pageableMemoryAccess ? 1 : 0}, UsesHostPageTables ${attrs.usesHostPageTables ? 1 : 0})` // i18n-ignore: paradigm name and attribute identifiers
          : tr(`reported as ${String(par.paradigm ?? 'unknown')}`, `được báo là ${String(par.paradigm ?? 'không rõ')}`),
      });
    const probe = s.rows.find((r) => r.name === 'probe');
    const rho = ratioFromProbe(probe);
    if (probe && rho != null)
      out.push({
        key: s.backend + 'probe',
        backend: where,
        what: tr('Link against device memory', 'Đường truyền so với bộ nhớ device'),
        says: tr(
          `${fmt(Number(probe.link_gb_per_s))} GB/s pinned host to device, ${fmt(Number(probe.device_gb_per_s))} GB/s device to device: the link is ${fmt(100 * rho, 2)}% of device bandwidth here`,
          `${fmt(Number(probe.link_gb_per_s))} GB/s từ host được ghim sang device, ${fmt(Number(probe.device_gb_per_s))} GB/s từ device sang device: ở đây đường truyền bằng ${fmt(100 * rho, 2)}% băng thông device`,
        ),
      });
    const addr = s.rows.find((r) => r.name === 'addresses');
    if (addr)
      out.push({
        key: s.backend + 'addresses',
        backend: where,
        what: tr('contents against gpuAddress (observed, not documented)', 'contents so với gpuAddress (quan sát được, không có trong tài liệu)'),
        says: tr(
          `shared buffer: ${addr.shared_same ? 'the same value' : 'different values'}; no-copy buffer: ${addr.nocopy_same ? 'the same value' : 'different values'}`,
          `buffer shared: ${addr.shared_same ? 'cùng một giá trị' : 'hai giá trị khác nhau'}; buffer no-copy: ${addr.nocopy_same ? 'cùng một giá trị' : 'hai giá trị khác nhau'}`,
        ),
      });
    return out;
  });
  return (
    <Bench
      title={tr('What the benchmark reports before it times anything', 'Những gì benchmark báo cáo trước khi đo thời gian bất cứ thứ gì')}
      task={
        <button className="quiet" onClick={reload}>
          <Translated vi="Tải lại kết quả">Reload results</Translated>
        </button>
      }
      foot={tr(
        "From this machine's recorded lab runs. The paradigm and the probe are CUDA and HIP rows; the addresses row is Metal's.",
        'Lấy từ các lần chạy lab đã ghi lại trên máy này. Hàng paradigm và hàng probe là của CUDA và HIP; hàng addresses là của Metal.',
      )}
    >
      {lines.length ? (
        <div className="panel" style={{ overflowX: 'auto' }}>
          <table className="data" style={{ margin: 0 }} data-table="instruments">
            <thead>
              <tr>
                <th>
                  <Translated vi="Backend">Backend</Translated>
                </th>
                <th>
                  <Translated vi="Hàng">Row</Translated>
                </th>
                <th>
                  <Translated vi="Nội dung">What it says</Translated>
                </th>
              </tr>
            </thead>
            <tbody>
              {lines.map((l) => (
                <tr key={l.key} data-row={l.key}>
                  <td>{l.backend}</td>
                  <td>{l.what}</td>
                  <td>{l.says}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="muted" style={{ margin: 0 }} data-instruments-empty>
          {loading ? (
            tr('Reading reports.', 'Đang đọc báo cáo.')
          ) : (
            <Translated
              vi={
                <>
                  Chưa có benchmark nào của lab này trên máy này. Hãy chạy <code>{RUN} --bench</code> trên một máy có GPU, rồi
                  tải lại. Trình giả lập không bao giờ được đo hiệu năng.
                </>
              }
            >
              <>
                No benchmark of this lab on this machine yet. Run <code>{RUN} --bench</code> on a machine with a GPU, then
                reload. The emulator is never benchmarked.
              </>
            </Translated>
          )}
        </p>
      )}
      {sets.length ? (
        <p className="muted" style={{ margin: '0.6rem 0 0' }}>
          <Translated
            vi={
              <>
                Đọc từ {sets.map((s) => `${s.backend}: ${whose(s)}`).join('; ')}. Các hàng này đến từ driver và{' '}
                <code>shared/probe.cu</code>, không phải từ các hàm feed của bạn.
              </>
            }
          >
            <>
              Read from {sets.map((s) => `${s.backend}: ${whose(s)}`).join('; ')}. These rows come from the driver and{' '}
              <code>shared/probe.cu</code>, not from your feed functions.
            </>
          </Translated>
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the results table
const PATH_NAME: Record<string, string> = {
  get explicit() {
    return tr('explicit (copies)', 'explicit (sao chép)');
  },
  get mapped() {
    return tr('mapped host memory', 'bộ nhớ host ánh xạ');
  },
  get managed() {
    return tr('managed', 'managed');
  },
  get private() {
    return tr('private, after a blit', 'private, sau một lần blit');
  },
  get nocopy() {
    return tr('no-copy wrap', 'bọc no-copy');
  },
  get shared() {
    return tr('shared buffer', 'buffer shared');
  },
};
const msOf = (head: string, key: string): Col => ({
  head,
  num: true,
  cell: (r) => {
    const v = num(r[key]);
    return v == null ? '' : fmt(v * 1000);
  },
});

function FeedResults() {
  const { reports } = useLabs();
  const rows = Object.keys(reports).flatMap((b) =>
    reports[b].results.filter((x) => x.lab === l1.id).flatMap((x) => x.bench.filter((r) => r.name === 'feed')),
  );
  // The fastest path for each backend, kernel and workload, for the last column.
  const best = new Map<string, number>();
  for (const r of rows) {
    const k = `${String(r.backend)}/${String(r.variant)}/${String(r.workload)}`;
    const t = num(r.seconds);
    if (t && (!best.has(k) || t < best.get(k)!)) best.set(k, t);
  }
  const metal = rows.some((r) => r.gpu_kernel_s !== undefined);
  const cols: Col[] = [
    { head: tr('Path', 'Đường'), cell: (r) => PATH_NAME[String(r.path)] ?? String(r.path) },
    rawCol(tr('Workload', 'Workload'), 'workload'),
    rawCol(tr('Passes', 'Số lượt'), 'passes'),
    {
      head: tr('Table, MiB', 'Bảng, MiB'),
      num: true,
      cell: (r) => {
        const v = num(r.table_bytes);
        return v == null ? '' : fmt(v / 2 ** 20);
      },
    },
    {
      head: tr('Read, tables', 'Đã đọc, số bảng'),
      num: true,
      cell: (r) => {
        const t = num(r.table_bytes);
        const u = num(r.touched_bytes);
        return t && u != null ? fmt(u / t) : '';
      },
    },
    { ...msCol, head: tr('Whole call, ms', 'Cả lời gọi, ms') },
    ...(metal ? [msOf(tr('GPU copy, ms', 'Sao chép GPU, ms'), 'gpu_copy_s'), msOf(tr('GPU kernels, ms', 'Kernel GPU, ms'), 'gpu_kernel_s')] : []),
    {
      head: tr('× fastest path', '× đường nhanh nhất'),
      num: true,
      cell: (r) => {
        const t = num(r.seconds);
        const b = best.get(`${String(r.backend)}/${String(r.variant)}/${String(r.workload)}`);
        return t && b ? fmt(t / b) : '';
      },
    },
  ];
  return <LabResults lab={l1} filter={(r) => r.name === 'feed'} cols={cols} />;
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS} tiers={TIERS}>
      <LabsIntro>
        <Translated
          vi={
            <>
              Một lab, trong <code>labs/appC/lab1_feeding/</code>. Nó cho một kernel ba chỗ ở trên mỗi backend và đo thời gian
              toàn bộ hành trình mà Fig. C.1 thực hiện: CPU tạo ra một bảng, kernel đọc nó, CPU dùng kết quả. Kernel được cung
              cấp sẵn; bạn viết phía host, nơi phụ lục này thực sự nói tới.
            </>
          }
        >
          <>
            One lab, in <code>labs/appC/lab1_feeding/</code>. It gives one kernel three homes on each backend and times the
            whole trip that Fig. C.1 makes: the CPU produces a table, the kernel reads it, the CPU consumes the result. The
            kernel is provided; you write the host side, which is where this appendix lives.
          </>
        </Translated>
      </LabsIntro>
      <Translated
        vi={
          <p>
            Trình chạy lab chấp nhận số của phụ lục, nên <code>python3 labs/pmpp.py run C.1</code> cũng chạy được. Trang này
            gọi lab bằng phần đầu thư mục của nó, <code>appC/lab1</code>, trong các lệnh và các bảng.
          </p>
        }
      >
        <p>
          The lab runner accepts the appendix number, so <code>python3 labs/pmpp.py run C.1</code> works too. This page
          names the lab by the start of its directory, <code>appC/lab1</code>, in its commands and its tables.
        </p>
      </Translated>
      <RunningLabs
        example="appC/lab1"
        alsoShownIn={
          <Translated
            vi={
              <>
                trình khám phá đường truyền của <a href={href('appC', 'C.2')}>mục C.2</a>, các địa chỉ của{' '}
                <a href={href('appC', 'C.3')}>mục C.3</a> và mô hình của <a href={href('appC', 'C.7')}>mục C.7</a>
              </>
            }
          >
            <>
              the link explorer of <a href={href('appC', 'C.2')}>section C.2</a>, the addresses of{' '}
              <a href={href('appC', 'C.3')}>section C.3</a> and the paradigm of <a href={href('appC', 'C.7')}>section C.7</a>
            </>
          </Translated>
        }
      />
      <BackendsAndEmulator />
      <Translated
        vi={
          <p>
            Lab này không được build cho <code>race</code>, biến thể thứ hai của trình giả lập, chạy các luồng đồng thời dưới
            ThreadSanitizer. Không luồng nào của kernel chạm tới một phần tử mà luồng khác ghi.
          </p>
        }
      >
        <p>
          This lab is not built for <code>race</code>, the emulator's second flavour, which runs threads concurrently
          under ThreadSanitizer. No thread of the kernel touches an element that another thread writes.
        </p>
      </Translated>
      <VerificationStatus />

      {/* ------------------------------------------------------------ C.1 */}
      <h2>
        <Translated vi="Lab C.1: ba cách cấp dữ liệu cho một kernel">Lab C.1: three ways to feed a kernel</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Kernel là <code>gatherAdd</code>: <code>out[r*cols + c] += table[ids[r]*cols + c]</code>. Hàng r của kết quả là
            hàng <code>ids[r]</code> của bảng, tức là một lần tra token embedding: bảng là một bộ từ vựng gồm các hàng
            embedding và ids là các token id. Nó chạy <code>passes</code> lần trên cùng dữ liệu, đại diện cho một kernel đọc
            đầu vào nhiều hơn một lần. Thứ thay đổi giữa các đường chỉ là nơi bảng, ids và kết quả nằm, và do đó kernel nhận
            con trỏ nào và cái gì đi qua giữa các bộ xử lý.
          </p>
        }
      >
        <p>
          The kernel is <code>gatherAdd</code>: <code>out[r*cols + c] += table[ids[r]*cols + c]</code>. Row r of the
          result is row <code>ids[r]</code> of the table, which is a token-embedding lookup: the table is a vocabulary of
          embedding rows and the ids are token ids. It runs <code>passes</code> times over the same data, standing in for
          a kernel that reads its input more than once. What changes from path to path is only where the table, the ids
          and the result live, and so which pointer the kernel gets and what crosses between the processors.
        </p>
      </Translated>
      <Source
        section="C.2, C.3, C.6"
        pages="624–627"
        figs={tr('Fig. C.1, p. 627; the embedding lookup is ch. 20, p. 480', 'Fig. C.1, p. 627; phép tra embedding ở ch. 20, p. 480')}
      />
      <Translated
        vi={
          <p>
            Mỗi đường nhận một <code>FeedJob</code> từ <code>feeding.h</code>: số hàng và số cột của bảng, ids trong bộ nhớ
            host do chương trình điều khiển (driver) sở hữu, số lượt, và hai callback. <code>produce</code> ghi bảng, như{' '}
            <code>fread</code> làm trong Fig. C.1; <code>consume</code> đọc kết quả, như <code>use_data</code> làm. Mỗi đường
            cấp phát ba mảng trong loại bộ nhớ riêng của nó, gọi <code>produce</code> một lần, khởi tạo kết quả bằng 0, khởi
            chạy các lượt, bảo đảm lượt cuối đã xong, gọi <code>consume</code> một lần, giải phóng những gì nó đã cấp phát, và
            trả về 0, hoặc 1 sau khi in ra điều gì đã hỏng.
          </p>
        }
      >
        <p>
          Every path receives a <code>FeedJob</code> from <code>feeding.h</code>: the table's rows and columns, the ids
          in host memory the driver owns, the number of passes, and two callbacks. <code>produce</code> writes the table,
          as <code>fread</code> does in Fig. C.1; <code>consume</code> reads the result, as <code>use_data</code> does.
          Each path allocates the three arrays in its own kind of memory, calls <code>produce</code> once, starts the
          result at zero, launches the passes, makes sure the last one has finished, calls <code>consume</code> once,
          frees what it allocated, and returns 0, or 1 after printing what failed.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Các lượt của kernel, ba đường Metal, các phép dò và mọi phép đo là của chúng tôi. Phụ lục không đưa mã nào cho
              C.2 và không đưa số đo thời gian nào.
            </p>
          }
        >
          <p>
            The kernel's passes, the three Metal paths, the probes and every measurement are ours. The appendix shows no
            code for C.2 and gives no timings.
          </p>
        </Translated>
      </Beyond>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <Translated
              vi={
                <>
                  <code>feedExplicit</code>, <code>feedMapped</code> và <code>feedManaged</code>, bên dưới dòng{' '}
                  <code>your code below</code>. Kernel và <code>launchGatherAdd</code> phía trên đã được cung cấp
                </>
              }
            >
              <>
                <code>feedExplicit</code>, <code>feedMapped</code> and <code>feedManaged</code>, below the line{' '}
                <code>your code below</code>. The kernel and <code>launchGatherAdd</code> above it are provided
              </>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated
              vi={
                <>
                  <code>feedPrivateCopy</code>, <code>feedNoCopy</code> và <code>feedShared</code>.{' '}
                  <code>encodeGatherAdd</code> và <code>waitFor</code> đã được cung cấp. <code>kernel.metal</code> đã hoàn
                  chỉnh và không cần thay đổi
                </>
              }
            >
              <>
                <code>feedPrivateCopy</code>, <code>feedNoCopy</code> and <code>feedShared</code>.{' '}
                <code>encodeGatherAdd</code> and <code>waitFor</code> are provided. <code>kernel.metal</code> is complete
                and needs no change
              </>
            </Translated>,
          ],
        ]}
      />
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-table="paths" style={{ minWidth: '44rem' }}>
          <thead>
            <tr>
              <th>
                <Translated vi="Hàm">Function</Translated>
              </th>
              <th>
                <Translated vi="Backend">Backends</Translated>
              </th>
              <th>
                <Translated vi="Bảng nằm ở đâu">Where the table lives</Translated>
              </th>
              <th>
                <Translated vi="Kernel đọc gì">What the kernel reads</Translated>
              </th>
              <th>
                <Translated vi="Nguồn">From</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><code style={{ whiteSpace: 'nowrap' }}>feedExplicit</code></td>
              <td>CUDA, HIP, emu{/* i18n-ignore: backend names */}</td>
              <td>
                <Translated vi={<>bộ nhớ <code>malloc</code>, được sao chép vào bộ nhớ <code>cudaMalloc</code></>}>
                  <><code>malloc</code> memory, copied into <code>cudaMalloc</code> memory</>
                </Translated>
              </td>
              <td>
                <Translated vi="bộ nhớ device, sau một lần sao chép cả bảng">device memory, after one copy of the whole table</Translated>
              </td>
              <td>
                <Translated vi="mô hình đơn giản, p. 623">the simple model, p. 623</Translated>
              </td>
            </tr>
            <tr>
              <td><code style={{ whiteSpace: 'nowrap' }}>feedMapped</code></td>
              <td>CUDA, HIP, emu{/* i18n-ignore: backend names */}</td>
              <td>
                <Translated vi={<><code>cudaHostAlloc</code> với <code>cudaHostAllocMapped</code></>}>
                  <><code>cudaHostAlloc</code> with <code>cudaHostAllocMapped</code></>
                </Translated>
              </td>
              <td>
                <Translated vi="bộ nhớ host tại chỗ, qua kênh kết nối, ở mọi lần truy cập">
                  host memory in place, over the interconnect, on every access
                </Translated>
              </td>
              <td>
                <Translated vi="C.2, pp. 624–625">C.2, pp. 624–625</Translated>
              </td>
            </tr>
            <tr>
              <td><code style={{ whiteSpace: 'nowrap' }}>feedManaged</code></td>
              <td>CUDA, HIP, emu{/* i18n-ignore: backend names */}</td>
              <td><code>cudaMallocManaged</code></td>
              <td>
                <Translated vi="bất cứ nơi nào runtime đã đặt mỗi trang">wherever the runtime has put each page</Translated>
              </td>
              <td>
                <Translated vi="C.6 và Fig. C.1, pp. 626–627">C.6 and Fig. C.1, pp. 626–627</Translated>
              </td>
            </tr>
            <tr>
              <td><code style={{ whiteSpace: 'nowrap' }}>feedPrivateCopy</code></td>
              <td>Metal{/* i18n-ignore: backend name */}</td>
              <td>
                <Translated vi="một buffer trung gian shared, rồi một lần blit vào buffer private">
                  a shared staging buffer, then a blit into a private buffer
                </Translated>
              </td>
              <td>
                <Translated vi="một buffer chỉ GPU chạm tới được, sau một lần sao chép trên GPU">
                  a buffer only the GPU can touch, after a GPU copy
                </Translated>
              </td>
              <td>
                <Translated vi="của chúng tôi: mô hình explicit trên một bộ nhớ">ours: the explicit model on one memory</Translated>
              </td>
            </tr>
            <tr>
              <td><code style={{ whiteSpace: 'nowrap' }}>feedNoCopy</code></td>
              <td>Metal{/* i18n-ignore: backend name */}</td>
              <td>
                <Translated vi={<>vùng nhớ căn theo trang của chính bạn, được bọc bằng <code>newBufferWithBytesNoCopy</code></>}>
                  <>your own page-aligned memory, wrapped with <code>newBufferWithBytesNoCopy</code></>
                </Translated>
              </td>
              <td>
                <Translated vi={<>các byte mà <code>produce</code> đã ghi, tại chỗ</>}>
                  <>the bytes <code>produce</code> wrote, in place</>
                </Translated>
              </td>
              <td>
                <Translated vi="của chúng tôi: cách llama.cpp trao trọng số được ánh xạ bộ nhớ cho Metal">
                  ours: how llama.cpp hands memory-mapped weights to Metal
                </Translated>
              </td>
            </tr>
            <tr>
              <td><code style={{ whiteSpace: 'nowrap' }}>feedShared</code></td>
              <td>Metal{/* i18n-ignore: backend name */}</td>
              <td>
                <Translated vi={<>một buffer có lưu trữ shared, từ <code>newBufferWithLength</code></>}>
                  <>a buffer with shared storage, from <code>newBufferWithLength</code></>
                </Translated>
              </td>
              <td>
                <Translated vi={<>các byte mà <code>produce</code> đã ghi, tại chỗ</>}>
                  <>the bytes <code>produce</code> wrote, in place</>
                </Translated>
              </td>
              <td>
                <Translated vi="của chúng tôi: lời khuyên của Apple cho dữ liệu CPU điền và GPU đọc">
                  ours: Apple's advice for data the CPU fills and the GPU reads
                </Translated>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <Translated
        vi={
          <p>
            Quy trình của sách cho bộ nhớ ánh xạ, lấy một con trỏ thứ hai từ <code>cudaHostGetDevicePointer()</code> và đưa nó
            cho kernel, là quy trình năm 2009. Hãy đưa cho kernel con trỏ mà <code>cudaHostAlloc</code> trả về: các bài kiểm
            thử kiểm tra rằng kernel nhận đúng con trỏ mà <code>produce</code> đã ghi qua. Trình giả lập hoàn toàn không cung
            cấp <code>cudaHostGetDevicePointer()</code>, nên mã gọi nó không build được ở đó. Dù vậy vẫn hãy giữ cờ mapped, vì
            trên HIP chính nó làm cho bộ nhớ là fine-grained.
          </p>
        }
      >
        <p>
          The book's procedure for mapped memory, which takes a second pointer from <code>cudaHostGetDevicePointer()</code>{' '}
          and gives that to the kernel, is the 2009 one. Give the kernel the pointer <code>cudaHostAlloc</code> returned:
          the tests check that the kernel received the very pointer <code>produce</code> wrote through. The emulator does
          not provide <code>cudaHostGetDevicePointer()</code> at all, so code that calls it does not build there. Keep the
          mapped flag anyway, because on HIP it is what makes the memory fine-grained.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/appC-facts.md A3, B2)" /* i18n-ignore: date and citation only */>
        <Translated
          vi={
            <p>
              NVIDIA: với unified addressing, bộ nhớ <code>cudaHostAlloc</code> truy cập được qua chính con trỏ đó, và không cần{' '}
              <code>cudaHostGetDevicePointer()</code> cho nó.
            </p>
          }
        >
          <p>
            NVIDIA: under unified addressing, <code>cudaHostAlloc</code> memory is reachable through the same pointer, and{' '}
            <code>cudaHostGetDevicePointer()</code> is not needed for it.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Trên Metal không có kênh kết nối nào giữa bộ nhớ của CPU và của GPU, nên ba đường ở đó không phải ba đường của
            sách. Chúng khác nhau ở việc bảng có được sao chép không và ai sở hữu bộ nhớ. Một buffer no-copy cần một con trỏ
            căn theo trang và một độ dài làm tròn lên số trang nguyên, và một lần blit trên macOS cần offset và kích thước là
            bội của 4 byte. Thời gian GPU được báo theo từng command buffer, nên hãy đặt các lần sao chép và các kernel vào các
            command buffer riêng nếu bạn muốn thấy chúng tách biệt.
          </p>
        }
      >
        <p>
          On Metal there is no interconnect between the CPU's memory and the GPU's, so the three paths there are not the
          book's three. They differ in whether the table is copied and in who owns the memory. A no-copy buffer needs a
          page-aligned pointer and a length rounded up to whole pages, and a blit on macOS needs offsets and sizes that are
          multiples of 4 bytes. GPU time is reported per command buffer, so put the copies and the kernels in separate
          command buffers if you want to see them apart.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md C2, C3; ch05-facts C6, C9; appC-facts C2, D4)" /* i18n-ignore: date and citation only */>
        <Translated
          vi={
            <p>
              Apple: buffer private không có địa chỉ CPU. Metal "có thể áp dụng thêm các tối ưu hóa cho tài nguyên private", và
              không nói là tối ưu hóa nào.
            </p>
          }
        >
          <p>
            Apple: private buffers have no CPU address. Metal "may apply additional optimizations to private resources",
            and does not say which.
          </p>
        </Translated>
      </Today>

      <h3>
        <Translated vi="Kiểm thử">Testing</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Hãy chạy <code>{RUN} --backend emu</code> trước, rồi <code>{RUN}</code>. Các kích thước là hàng × cột của{' '}
            <code>float</code>: 1 × 1, 7 × 3, 64 × 33 và 130 × 17, và trên device thêm 4096 × 256. Các ids có dạng dày (mỗi hàng
            một lần, theo thứ tự) hoặc thưa (một hàng trong tám, làm tròn lên, có lặp lại, luôn gồm hàng cuối để lỗi lệch một
            lộ ra), với một lượt và với ba lượt. Với mỗi đường và mỗi trường hợp, chương trình điều khiển kiểm tra bốn điều:
          </p>
        }
      >
        <p>
          Run <code>{RUN} --backend emu</code> first, then <code>{RUN}</code>. The shapes are rows × columns of{' '}
          <code>float</code>: 1 × 1, 7 × 3, 64 × 33 and 130 × 17, and on a device also 4096 × 256. The ids come dense
          (every row once, in order) or sparse (a row in eight, rounded up, with repeats, always including the last row
          so that an off-by-one shows), with one pass and with three. For each path and case the driver checks four things:
        </p>
      </Translated>
      <Translated
        vi={
          <ol>
            <li>
              Kết quả <code>consume</code> thấy bằng <code>passes</code> lần các hàng được gom của tham chiếu CPU trong{' '}
              <code>main.cpp</code>, trong một dung sai. Mọi giá trị đúng cũng là số nguyên, và điều đó cũng được kiểm tra.
            </li>
            <li>
              <code>produce</code> và <code>consume</code> mỗi hàm được gọi đúng một lần, với các con trỏ khác null.
            </li>
            <li>
              Cấu trúc. Với các đường mapped, managed, no-copy và shared, bảng mà launcher được cung cấp ghi lại chính là vùng
              nhớ <code>produce</code> đã ghi: một con trỏ phục vụ cả hai phía. Với các đường explicit và private thì không, nên
              đã có một lần sao chép, và trên Metal buffer kernel đọc có lưu trữ private.
            </li>
            <li>
              <code>feedNoCopy</code>: con trỏ mà <code>produce</code> ghi qua là bội của kích thước trang.
            </li>
          </ol>
        }
      >
        <ol>
          <li>
            The result <code>consume</code> saw is <code>passes</code> times the gathered rows of the CPU reference in{' '}
            <code>main.cpp</code>, within a tolerance. Every correct value is also a whole number, and that is checked too.
          </li>
          <li>
            <code>produce</code> and <code>consume</code> were each called exactly once, with pointers that are not null.
          </li>
          <li>
            Structure. For the mapped, managed, no-copy and shared paths, the table the provided launcher recorded is the
            memory <code>produce</code> wrote: one pointer serves both sides. For the explicit and private paths it is
            not, so a copy happened, and on Metal the buffer the kernel read has private storage.
          </li>
          <li>
            <code>feedNoCopy</code>: the pointer <code>produce</code> wrote through is a multiple of the page size.
          </li>
        </ol>
      </Translated>
      <Translated
        vi={
          <p>
            Như khi phát hành, mọi hàm đều trả về -1 và nói rằng nó chưa được viết, nên mọi trường hợp đều thất bại. Những gì
            các bài kiểm thử không thấy được cũng quan trọng không kém:
          </p>
        }
      >
        <p>
          As shipped, every function returns -1 and says it is not written yet, so every case fails. What the tests
          cannot see matters as much:
        </p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>
              <b>Thiếu một lần đồng bộ trước <code>consume</code>.</b> Trên trình giả lập, mọi lời gọi đều đồng bộ và{' '}
              <code>cudaHostAlloc</code> cùng <code>cudaMallocManaged</code> là các cấp phát host thông thường, nên lỗi này
              qua được ở đó. Trên device nó là một race condition, và với bộ nhớ managed ở nơi{' '}
              <code>ConcurrentManagedAccess</code> bằng 0, bao gồm mọi hệ thống Windows và WSL, là một segmentation fault.
            </li>
            <li>
              <b>Kết quả không bắt đầu từ 0.</b> Các vùng cấp phát mới của trình giả lập đọc ra như một số float khoảng −3 ×
              10<sup>−16</sup>, biến mất khi cộng vào. Kiểm tra số nguyên bắt được nó ở bất cứ đâu một hàng được gom chứa số 0,
              điều mà các trường hợp 64 × 33 và 130 × 17 có.
            </li>
            <li>
              <b>Rò rỉ bộ nhớ</b>, và việc mỗi lần giải phóng có khớp với vùng cấp phát của nó không: không được kiểm tra.
            </li>
            <li>
              <b>Một đường có nhanh hay không.</b> Đó là việc của benchmark.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            <b>A missing synchronise before <code>consume</code>.</b> On the emulator every call is synchronous and{' '}
            <code>cudaHostAlloc</code> and <code>cudaMallocManaged</code> are plain host allocations, so the mistake
            passes there. On a device it is a race, and for managed memory where <code>ConcurrentManagedAccess</code> is 0,
            which includes every Windows and WSL system, a segmentation fault.
          </li>
          <li>
            <b>A result that does not start at zero.</b> The emulator's fresh allocations read as a float of about −3 ×
            10<sup>−16</sup>, which vanishes when added. The whole-number check catches it wherever a gathered row holds a
            0, which the 64 × 33 and 130 × 17 cases contain.
          </li>
          <li>
            <b>Leaks</b>, and whether each free matches its allocation: not checked.
          </li>
          <li>
            <b>Whether a path is fast.</b> That is the benchmark's job.
          </li>
        </ul>
      </Translated>
      <Today checked="2026-11-20 (notes/today/appC-facts.md A10)" /* i18n-ignore: date and citation only */>
        <Translated
          vi={
            <p>
              Ở nơi <code>ConcurrentManagedAccess</code> bằng 0, "các truy cập đồng thời của CPU/GPU, kể cả vào các vùng cấp
              phát bộ nhớ managed khác nhau, sẽ gây ra segmentation fault".
            </p>
          }
        >
          <p>
            Where <code>ConcurrentManagedAccess</code> is 0, "Concurrent CPU/GPU accesses, even to different managed
            memory allocations, will cause a segmentation fault".
          </p>
        </Translated>
      </Today>

      <h3>
        <Translated vi="Trước khi đo hiệu năng, hãy dự đoán">Before you benchmark, predict</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Một công cụ để dự đoán. Trên GPU rời, đường explicit chuyển cả bảng qua kênh kết nối một lần rồi đọc bộ nhớ device.
            Đường mapped chỉ chuyển những gì kernel chạm tới, nhưng đi qua kênh kết nối ở mỗi lần chạm. Bộ nhớ device nhanh hơn
            đường truyền nhiều, nên mapped thắng, đại khái, chỉ khi kernel đọc ít hơn khoảng một bảng dữ liệu trong suốt vòng
            đời của nó. Bảng kết quả ghi lượng đó là "Đã đọc, số bảng": 1 cho <code>dense1</code>, 8 cho <code>dense8</code>,
            1/64 cho <code>sparse1</code>. <a href={href('appC', 'C.2')}>Mục C.2</a> áp dụng cùng quy tắc với một tỷ lệ đường
            truyền do bạn chọn.
          </p>
        }
      >
        <p>
          A tool to predict with. On a discrete GPU the explicit path moves the whole table across the interconnect once
          and then reads device memory. The mapped path moves only what the kernel touches, but crosses the interconnect
          on every touch. Device memory is much faster than the link, so mapped wins, roughly, only when the kernel reads
          less than about one table's worth over its whole life. The results table gives that amount as "Read, tables": 1
          for <code>dense1</code>, 8 for <code>dense8</code>, 1/64 for <code>sparse1</code>.{' '}
          <a href={href('appC', 'C.2')}>Section C.2</a> works the same rule with a link ratio you choose.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Quy tắc hòa vốn này là của chúng tôi, suy ra từ lời khuyên của sách chỉ dùng bộ nhớ zero-copy cho dữ liệu mà
              kernel đọc "thỉnh thoảng và thưa thớt" (p. 625).
            </p>
          }
        >
          <p>
            This break-even rule is ours, derived from the book's advice to use zero-copy memory only for data a kernel
            reads "occasionally and sparsely" (p. 625).
          </p>
        </Translated>
      </Beyond>
      <LabPrediction
        id="appC/labs/p-sparse"
        prompt={
          <Translated
            vi={
              <>
                CUDA hoặc HIP, <code>sparse1</code>: thời gian của đường explicit chia cho thời gian của đường mapped, cho cả lời
                gọi.
              </>
            }
          >
            <>
              CUDA or HIP, <code>sparse1</code>: the explicit path's time divided by the mapped path's time, for the
              whole call.
            </>
          </Translated>
        }
        unit={tr("× the mapped path's time", '× thời gian của đường mapped')}
        measure={sparseRatio}
        missing={
          <Translated
            vi={
              <>
                Tỷ lệ này chỉ tồn tại trên CUDA và HIP: Metal không có kênh kết nối giữa các bộ nhớ của nó, nên không có đường
                mapped hay explicit. Hãy chạy <code>{RUN} --bench</code> trên một máy có CUDA hoặc ROCm.
              </>
            }
          >
            <>
              This ratio exists only on CUDA and HIP: Metal has no interconnect between its memories, so it has no mapped
              or explicit path. Run <code>{RUN} --bench</code> on a machine with CUDA or ROCm.
            </>
          </Translated>
        }
        reading={
          <Translated
            vi={
              <>
                Trên 1, đọc tại chỗ đã thắng. Đường explicit trả giá cho cả bảng đi qua; đường mapped trả giá cho một hàng trong
                64, ở tốc độ đường truyền. Hàng probe ở trên cho biết đường truyền của bạn nhanh thế nào so với bộ nhớ device.
              </>
            }
          >
            <>
              Above 1, reading in place won. The explicit path paid for the whole table to cross; the mapped path paid for
              one row in 64, at link speed. The probe row above says how fast your link is against your device memory.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="appC/labs/p-repeat"
        prompt={
          <Translated
            vi={
              <>
                CUDA hoặc HIP, <code>dense8</code>: thời gian của đường mapped chia cho thời gian của đường explicit. Kernel đọc
                cả bảng tám lần.
              </>
            }
          >
            <>
              CUDA or HIP, <code>dense8</code>: the mapped path's time divided by the explicit path's time. The kernel reads
              the whole table eight times.
            </>
          </Translated>
        }
        unit={tr("× the explicit path's time", '× thời gian của đường explicit')}
        measure={repeatRatio}
        missing={
          <Translated
            vi={
              <>
                Tỷ lệ này chỉ tồn tại trên CUDA và HIP. Hãy chạy <code>{RUN} --bench</code> trên một máy có CUDA hoặc ROCm.
              </>
            }
          >
            <>
              This ratio exists only on CUDA and HIP. Run <code>{RUN} --bench</code> on a machine with CUDA or ROCm.
            </>
          </Translated>
        }
        reading={
          <Translated
            vi={
              <>
                Trên 1, sao chép trước đã thắng. Đường mapped đi qua đường truyền tám lần cho mỗi byte; đường explicit một lần.
                Lời khuyên của sách ở p. 625 chính là phép đo này gói trong một câu.
              </>
            }
          >
            <>
              Above 1, copying first won. The mapped path crossed the link eight times for every byte; the explicit path
              once. The book's advice on p. 625 is this measurement in one sentence.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="appC/labs/p-metal"
        prompt={
          <Translated
            vi={
              <>
                Metal, <code>dense8</code>: thời gian của đường private-copy chia cho thời gian của đường shared. Hãy quyết định
                luôn xem no-copy có khác shared hay không.
              </>
            }
          >
            <>
              Metal, <code>dense8</code>: the private-copy path's time divided by the shared path's time. Decide also
              whether no-copy will differ from shared at all.
            </>
          </Translated>
        }
        unit={tr("× the shared path's time", '× thời gian của đường shared')}
        measure={metalRatio}
        missing={
          <Translated vi={<>Tỷ lệ này chỉ tồn tại trên Metal. Hãy chạy <code>{RUN} --bench</code> trên một máy Mac.</>}>
            <>
              This ratio exists only on Metal. Run <code>{RUN} --bench</code> on a Mac.
            </>
          </Translated>
        }
        reading={
          <Translated
            vi={
              <>
                Đường private sao chép bảng trên GPU rồi đọc một buffer mà CPU không chạm tới được; Apple không nói việc đọc nó có
                nhanh hơn hay không. Thời gian cả lời gọi gồm cả cấp phát, nên hãy xem các cột sao chép GPU và kernel GPU trước khi
                kết luận một khác biệt đến từ đâu.
              </>
            }
          >
            <>
              The private path copies the table on the GPU and then reads a buffer the CPU cannot touch; Apple does not say
              whether that is faster to read. The whole-call time includes allocation, so look at the GPU copy and GPU
              kernel columns before deciding where a difference comes from.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Thêm một câu nữa, không có con số. Hãy tra mô hình managed của bạn ở mục <a href={href('appC', 'C.7')}>C.7</a>, hoặc
            trong bảng đầu tiên bên dưới khi bạn đã chạy benchmark, và nói <code>managed</code> sẽ giống đường nào.
          </p>
        }
      >
        <p>
          One more, without a number. Look up your managed paradigm in section <a href={href('appC', 'C.7')}>C.7</a>, or
          in the first table below once you have run the benchmark, and say which path <code>managed</code> will resemble.
        </p>
      </Translated>
      <Ask q={questionsLabs['appC/labs/q-managed-like']} />

      <h3>
        <Translated vi="Máy của bạn đã đo được gì">What your machine measured</Translated>
      </h3>
      <Instruments />
      <Translated
        vi={
          <p>
            Trên CUDA và HIP, phép dò đo thời gian một lần sao chép từ host được ghim sang device và một lần sao chép từ device
            sang device với kích thước bằng bảng, bằng event. Băng thông device tính cả byte đọc lẫn byte ghi. Tỷ lệ của chúng
            là câu trả lời của máy bạn cho con số "dưới 10%" của sách. <code>feedExplicit</code> sao chép từ bộ nhớ{' '}
            <code>malloc</code>, không phải bộ nhớ ghim, nên lần sao chép của nó có thể chạy dưới con số của phép dò.
          </p>
        }
      >
        <p>
          On CUDA and HIP the probe times a pinned host-to-device copy and a device-to-device copy of the table's size,
          with events. Device bandwidth counts bytes read plus bytes written. Their ratio is your machine's answer to the
          book's "less than 10%". <code>feedExplicit</code> copies from <code>malloc</code> memory, not pinned memory, so
          its copy may run below the probe's figure.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/appC-facts.md A4, C3)" /* i18n-ignore: date and citation only */>
        <Translated
          vi={
            <p>
              NVIDIA không còn nêu một tỷ lệ nào, và các hệ thống NVLink-C2C thu hẹp khoảng cách. Apple không ghi nhận gì về
              quan hệ giữa <code>gpuAddress</code> của một buffer và <code>contents</code> của nó; hàng Metal là một quan sát.
            </p>
          }
        >
          <p>
            NVIDIA no longer states a ratio, and NVLink-C2C systems narrow the gap. Apple documents nothing about how a
            buffer's <code>gpuAddress</code> relates to its <code>contents</code>; the Metal row is an observation.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Các số đo thời gian sau đây là cả lời gọi feed, lấy lần tốt nhất trong vài lần sau một lần khởi động không đo: cấp
            phát, <code>produce</code>, các lần sao chép, các lượt, đồng bộ, <code>consume</code> và giải phóng. Cấp phát được
            tính vào có chủ ý, vì hàm của Fig. C.1 cũng cấp phát. Trên Metal, "Sao chép GPU" và "Kernel GPU" cộng dồn thời gian
            GPU của các command buffer mà mỗi đường đã chờ. Kích thước bảng được chọn từ bộ nhớ mà device báo, tối đa 256 MiB để
            một lần chạy ngắn gọn; mức trần này là của lab, không phải một sự thật về device nào.
          </p>
        }
      >
        <p>
          The timings that follow are the whole feed call, the best of several after one untimed warm-up: allocation,{' '}
          <code>produce</code>, copies, passes, synchronise, <code>consume</code> and free. Allocation is inside on
          purpose, because Fig. C.1's function allocates too. On Metal, "GPU copy" and "GPU kernels" add up the GPU time
          of the command buffers each path waited on. The table size is chosen from the memory the device reports, up to a
          cap of 256 MiB that keeps a run short; the cap is the lab's, not a fact about any device.
        </p>
      </Translated>
      <FeedResults />
      <ReferenceSolution lab={l1} />

      <h2>
        <Translated vi="Những gì lab này để lại cho sau">What this lab leaves for later</Translated>
      </h2>
      <Translated
        vi={
          <ul>
            <li>
              <code>feedExplicit</code> dùng bộ nhớ <code>malloc</code> làm vùng trung gian. Hãy dùng <code>cudaMallocHost</code>{' '}
              thay thế và so sánh lần sao chép của nó với con số của phép dò. Khi vùng trung gian đã được ghim, nó có còn đáng
              giữ không, hay <code>produce</code> nên ghi thẳng vào buffer ghim và lần sao chép đi từ đó?
            </li>
            <li>
              Prefetch bảng sang device trong <code>feedManaged</code> trước lượt đầu tiên. Lời gọi này đã đổi trong CUDA 13
              sang nhận một <code>cudaMemLocation</code>, còn header mặc định của HIP vẫn nhận một device kiểu <code>int</code>,
              nên dòng này cần một <code>#if</code>. Cả hai trình giả lập đều không có lời gọi này: phần này chỉ dành cho device
              thật.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            <code>feedExplicit</code> stages through <code>malloc</code> memory. Stage through <code>cudaMallocHost</code>{' '}
            instead and compare its copy with the probe's figure. Once the staging buffer is pinned, is it still worth
            having, or should <code>produce</code> write the pinned buffer and the copy go from there?
          </li>
          <li>
            Prefetch the table to the device in <code>feedManaged</code> before the first pass. The call changed in CUDA
            13 to take a <code>cudaMemLocation</code>, and HIP's default header still takes an <code>int</code> device, so
            the line needs an <code>#if</code>. Neither emulator has the call: this one is for real devices only.
          </li>
        </ul>
      </Translated>
      <Today checked="2026-11-20 (notes/today/appC-facts.md A14, B6)" /* i18n-ignore: date and citation only */>
        <Translated
          vi={
            <p>
              HIPIFY đánh dấu các dạng CUDA 13 của <code>cudaMemPrefetchAsync</code> và <code>cudaMemAdvise</code> là không được
              hỗ trợ; các hàm tương ứng của HIP là các hàm <code>_v2</code>.
            </p>
          }
        >
          <p>
            HIPIFY marks the CUDA 13 forms of <code>cudaMemPrefetchAsync</code> and <code>cudaMemAdvise</code>{' '}
            unsupported; their HIP counterparts are the <code>_v2</code> functions.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Lab này không phải một mốc dự án được đánh số của dự án xuyên suốt. <code>gatherAddKernel</code> và{' '}
            <code>launchGatherAdd</code> của nó theo các quy ước của dự án (<code>float</code> thứ tự hàng, con trỏ thô và kích
            thước tường minh, tham chiếu CPU trong <code>main.cpp</code>), nên chúng có thể làm phép tra token embedding cho{' '}
            <code>nanoinfer</code> khi mô hình cần. Trang <a href={href('appC', 'wrap')}>tổng kết</a> có bài kiểm tra tổng hợp
            của phụ lục và danh sách những chỗ trang in và tài liệu ngày nay khác nhau.
          </p>
        }
      >
        <p>
          The lab is not a numbered milestone of the course project. Its <code>gatherAddKernel</code> and{' '}
          <code>launchGatherAdd</code> follow the project's conventions (row-major <code>float</code>, raw pointers and
          explicit sizes, the CPU reference in <code>main.cpp</code>), so they can serve as <code>nanoinfer</code>'s
          token-embedding lookup when the model needs one. The <a href={href('appC', 'wrap')}>summary</a> has the
          appendix's mixed check and the list of places where the printed page and today's documentation differ.
        </p>
      </Translated>
    </LabsPage>
  );
}
