import type { Question } from '../../components/Ask';
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
  numCol,
  preferredBench,
  rawCol,
  useLabs,
  whose,
  type BenchSet,
  type Lab,
} from '../../components/LabKit';
import { fmt, labStatus, num, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';

// The page is built from components/LabKit.tsx; what is here is appendix A's own content.
// The lab runner resolves "A.1" to appA/lab1_ and also finds a directory prefix by
// substring; the lab's number on this page is its directory prefix, appA/lab1, the start
// of the id the run reports record (appA/lab1_sumerror). Lessons call it lab A.1.
// No question on this page: the lab is the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('appA', [{ num: 'appA/lab1', dir: 'lab1_sumerror', name: 'summation error', vi: { name: 'sai số của phép cộng' } }]);
const [l1] = LABS;
const RUN = 'python3 labs/pmpp.py run appA/lab1';
const FACTS = 'notes/today/appA-facts.md';

// Identifier colours for the reference solution: the input and the results are global,
// the tree's array is shared, a thread's running sum and its compensation are registers.
const TIERS: Record<string, Tier> = {
  x: 'global',
  out: 'global',
  partials: 'global',
  partial_s: 'shared',
  sum: 'reg',
  c: 'reg',
  y: 'reg',
  t: 'reg',
};

// ---------------------------------------------------------------- what each prediction is measured against
const accRow = (s: BenchSet, method: string, acc: string, data: string) =>
  s.rows.find((r) => r.name === 'accuracy' && r.method === method && r.acc === acc && r.data === data);

/** p-lab1-half-loop: what the half serial kernel returned for 65,536 uniform values. */
const halfLoop = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const r = accRow(s, 'serial', 'half', 'uniform');
    const v = num(r?.result);
    const ref = num(r?.reference);
    if (!r || v == null || ref == null) return null;
    return {
      value: v,
      detail: tr(
        `n = ${fmt(num(r.n) ?? NaN)}, against a true sum of ${fmt(ref, 6)}; ${fmt(num(r.ulps) ?? NaN)} ULPs of half at the reference`,
        `n = ${fmt(num(r.n) ?? NaN)}, so với tổng đúng ${fmt(ref, 6)}; lệch ${fmt(num(r.ulps) ?? NaN)} ULP của half tại giá trị tham chiếu`,
      ),
    };
  });

/** p-lab1-kahan-cost: Kahan's time over the plain tree's, float accumulator, the 'time' table. */
const kahanCost = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const tree = s.rows.find((r) => r.name === 'time' && r.method === 'tree' && r.acc === 'float');
    const kahan = s.rows.find((r) => r.name === 'time' && r.method === 'kahan' && r.acc === 'float');
    const a = num(kahan?.seconds);
    const b = num(tree?.seconds);
    if (!tree || !kahan || !a || !b) return null;
    return {
      value: a / b,
      detail: tr(
        `${fmt(a * 1000)} ms against ${fmt(b * 1000)} ms on ${fmt(num(tree.n) ?? NaN)} floats at coarse ${String(tree.coarse)}`,
        `${fmt(a * 1000)} ms so với ${fmt(b * 1000)} ms trên ${fmt(num(tree.n) ?? NaN)} số float với coarse ${String(tree.coarse)}`,
      ),
    };
  });

/** p-lab1-coarse: the plain float tree's error in ULPs at coarse 4096, with Kahan's beside it. */
const coarseUlps = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const tree = s.rows.find((r) => r.name === 'coarse' && r.method === 'tree' && num(r.coarse) === 4096);
    const kahan = s.rows.find((r) => r.name === 'coarse' && r.method === 'kahan' && num(r.coarse) === 4096);
    const v = num(tree?.ulps);
    if (!tree || v == null) return null;
    const k = num(kahan?.ulps);
    return {
      value: v,
      detail: tr(
        `${fmt(num(tree.n) ?? NaN)} uniform floats; Kahan at the same factor: ${k == null ? 'not reported' : `${fmt(k)} ULPs`}`,
        `${fmt(num(tree.n) ?? NaN)} số float phân bố đều; Kahan ở cùng hệ số: ${k == null ? 'chưa có báo cáo' : `${fmt(k)} ULP`}`,
      ),
    };
  });

// ---------------------------------------------------------------- what the probe found
const yes = (v: unknown) => v === true || v === 1 || v === 'true';

function Probe() {
  const { reports, loading } = useLabs();
  const sets = preferredBench(reports, l1);
  const found = sets.flatMap((s) => {
    const p = s.rows.find((r) => r.name === 'probe');
    return p ? [{ s, p }] : [];
  });
  return found.length ? (
    <div className="panel" style={{ overflowX: 'auto' }}>
      <table className="data" style={{ margin: 0 }} data-table="probe">
        <thead>
          <tr>
            <th>
              <Translated vi="Backend">Backend</Translated>
            </th>
            <th>1 + 0.75 ULP</th>{/* i18n-ignore: a numeric expression, same in both languages */}
            <th>
              <Translated vi="Denormal FP32">FP32 denormals</Translated>
            </th>
            <th>a·b + c</th>
          </tr>
        </thead>
        <tbody>
          {found.map(({ s, p }) => (
            <tr key={s.backend} data-backend={s.backend}>
              <td>
                {s.backend} <span className="muted">({s.device})</span>
              </td>
              <td>{p.rounding === 'nearest' ? tr('rounds to nearest', 'làm tròn về gần nhất') : p.rounding === 'toward-zero' ? tr('rounds toward zero', 'làm tròn về phía 0') : String(p.rounding ?? tr('not reported', 'chưa có báo cáo'))}</td>
              <td>{p.keeps_denormals === undefined ? tr('not reported', 'chưa có báo cáo') : yes(p.keeps_denormals) ? tr('kept', 'giữ lại') : tr('flushed to zero', 'bị ép về 0')}</td>
              <td>{p.fuses_mul_add === undefined ? tr('not reported', 'chưa có báo cáo') : yes(p.fuses_mul_add) ? tr('fused, one rounding', 'gộp, một lần làm tròn') : tr('two roundings', 'hai lần làm tròn')}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  ) : (
    <p className="muted" style={{ margin: 0 }} data-probe-empty>
      {loading ? tr('Reading reports.', 'Đang đọc báo cáo.') : (
        <Translated
          vi={
            <>
              Máy này chưa có dòng probe nào. Hãy chạy <code>{RUN} --bench</code> trên một máy có GPU, rồi tải lại.
            </>
          }
        >
          <>
            No probe row on this machine yet. Run <code>{RUN} --bench</code> on a machine with a GPU, then reload.
          </>
        </Translated>
      )}
    </p>
  );
}

// ---------------------------------------------------------------- the accuracy table, one per backend
const DATA = ['ones', 'uniform', 'mixed', 'mixed-asc', 'mixed-desc', 'cancel', 'softmax', 'big-small'];
const METHODS = ['serial', 'tree', 'kahan'];
const ACCS = ['float', 'half', 'bfloat'];

function cellOf(r: BenchRow | undefined) {
  if (!r) return { text: '', cls: '', title: tr('not in this run', 'không có trong lần chạy này') };
  const res = num(r.result);
  if (res == null || !isFinite(res)) return { text: tr('overflow', 'tràn số'), cls: 'fail', title: tr('the sum overflowed: stored as null', 'tổng bị tràn số: lưu thành null') };
  const u = num(r.ulps);
  const mm = r.matches_model as unknown; // a boolean in the reports, though BenchRow types it as string | number
  const same = mm === undefined ? null : mm === true || mm === 1 || mm === 'true';
  const model = same == null ? tr('model not compared', 'chưa so với mô hình') : same ? tr('the same bits as the CPU model of this order', 'trùng từng bit với mô hình CPU của thứ tự này') : tr('not the same bits as the CPU model', 'không trùng từng bit với mô hình CPU');
  return { text: u == null ? '' : fmt(u), cls: same === false ? 'differs' : '', title: tr(`result ${fmt(res, 9)}, reference ${fmt(num(r.reference) ?? NaN, 9)}; ${model}`, `kết quả ${fmt(res, 9)}, tham chiếu ${fmt(num(r.reference) ?? NaN, 9)}; ${model}`) };
}

function Accuracy() {
  const { reports, loading, reload } = useLabs();
  const sets = preferredBench(reports, l1).filter((s) => s.rows.some((r) => r.name === 'accuracy'));
  return (
    <Bench
      title={tr('Error by order, accumulator and data', 'Sai số theo thứ tự, bộ tích lũy và dữ liệu')}
      task={
        <button className="quiet" onClick={reload}>
          <Translated vi="Tải lại kết quả">Reload results</Translated>
        </button>
      }
      foot={tr(
        "ULPs of the accumulator's format at the reference value. Hover a cell for the result and the reference. A cell marked * did not have the same bits as the CPU model of that order. Measurements, not expected values.",
        'Tính bằng ULP của định dạng bộ tích lũy tại giá trị tham chiếu. Rê chuột lên một ô để xem kết quả và giá trị tham chiếu. Ô có dấu * không trùng từng bit với mô hình CPU của thứ tự đó. Đây là số đo, không phải giá trị kỳ vọng.',
      )}
    >
      {sets.length ? (
        sets.map((s) => {
          const one = s.rows.find((r) => r.name === 'accuracy' && r.method === 'tree');
          return (
            <div key={s.backend} style={{ marginBottom: '0.8rem' }}>
              <Translated
                vi={
                  <p className="muted" style={{ margin: '0 0 0.4rem' }}>
                    {s.backend} ({s.device}), {s.date}, {whose(s)}. n = {String(one?.n ?? '')}; tree và Kahan với coarse{' '}
                    {String(one?.coarse ?? '')}, khối {String(one?.block ?? '')}.
                  </p>
                }
              >
                <p className="muted" style={{ margin: '0 0 0.4rem' }}>
                  {s.backend} ({s.device}), {s.date}, {whose(s)}. n = {String(one?.n ?? '')}; tree and Kahan at coarse{' '}
                  {String(one?.coarse ?? '')}, block {String(one?.block ?? '')}.
                </p>
              </Translated>
              <div className="panel" style={{ overflowX: 'auto' }}>
                <table className="data" style={{ margin: 0 }} data-table="accuracy" data-backend={s.backend}>
                  <thead>
                    <tr>
                      <th>
                        <Translated vi="Tổng">Sum</Translated>
                      </th>
                      {DATA.map((d) => (
                        <th key={d} className="num">
                          {d}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {ACCS.flatMap((a) =>
                      METHODS.map((m) => (
                        <tr key={m + a} data-row={`${m}/${a}`}>
                          <td style={{ whiteSpace: 'nowrap' }}>
                            {m}, {a}
                          </td>
                          {DATA.map((d) => {
                            const c = cellOf(accRow(s, m, a, d));
                            return (
                              <td key={d} className="num" title={c.title} data-cell={d} style={c.cls === 'fail' ? { color: 'var(--fail)' } : undefined}>
                                {c.text}
                                {c.cls === 'differs' ? ' *' : ''}
                              </td>
                            );
                          })}
                        </tr>
                      )),
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          );
        })
      ) : (
        <p className="muted" style={{ margin: 0 }} data-accuracy-empty>
          {loading ? tr('Reading reports.', 'Đang đọc báo cáo.') : (
            <Translated
              vi={
                <>
                  Máy này chưa có dòng accuracy nào. Hãy chạy <code>{RUN} --bench</code> trên một máy có GPU, rồi tải
                  lại. Trình giả lập không bao giờ được đo hiệu năng.
                </>
              }
            >
              <>
                No accuracy rows on this machine yet. Run <code>{RUN} --bench</code> on a machine with a GPU, then reload.
                The emulator is never benchmarked.
              </>
            </Translated>
          )}
        </p>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- the race flavour, which the status table does not show
function RaceLine({ lab: l }: { lab: Lab }) {
  const { reports, loading } = useLabs();
  const run = reports.race;
  const rs = run?.results.filter((x) => x.lab === l.id) ?? [];
  const s = labStatus(run, l.id, 'solution');
  const n = rs.filter((x) => x.variant === 'solution').flatMap((x) => x.tests).length;
  return (
    <p data-race-status={loading ? 'loading' : s}>
      {loading ? (
        tr('Reading reports.', 'Đang đọc báo cáo.')
      ) : !run || s === 'none' ? (
        <Translated
          vi={
            <>
              Máy này chưa chạy lab này trên backend race: <code>{RUN} --backend race</code>.
            </>
          }
        >
          <>
            No run of this lab on the race backend on this machine yet: <code>{RUN} --backend race</code>.
          </>
        </Translated>
      ) : s === 'pass' ? (
        <Translated vi={<>Trên backend race của máy này, lời giải tham chiếu đã qua cả {n} bài kiểm thử và không có race nào được báo.</>}>
          <>On the race backend of this machine the reference solution passed all {n} tests with no race reported.</>
        </Translated>
      ) : (
        <Translated vi={<>Trên backend race của máy này, lời giải tham chiếu đã thất bại: ở đó nó là unverified.</>}>
          <>On the race backend of this machine the reference solution failed: unverified there.</>
        </Translated>
      )}
    </p>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS} tiers={TIERS}>
      <LabsIntro>
        <Translated
          vi={
            <>
              Một lab, trong <code>labs/appA/lab1_sumerror/</code>. Lab cộng một mảng float32 theo ba cách: một vòng lặp,
              một cây và một cây có bù, với ba kiểu bộ tích lũy, rồi đo xem thứ tự và độ rộng làm gì với kết quả và mỗi
              cách tốn bao nhiêu thời gian. Đây là mục A.6 được đo trên phần cứng của bạn, và là bằng chứng cho bộ tích lũy
              mà module X2 chọn cho các phép GEMV lượng tử hóa của nó.
            </>
          }
        >
          One lab, in <code>labs/appA/lab1_sumerror/</code>. It sums one float32 array three ways, a loop, a tree and a
          compensated tree, with three accumulator types, and measures what the order and the width do to the result and
          what each costs in time. It is section A.6 measured on your hardware, and the evidence for the accumulator
          module X2 chooses for its quantised GEMVs.
        </Translated>
      </LabsIntro>
      <Translated
        vi={
          <p>
            Trình chạy lab nhận số của lab, <code>pmpp.py run A.1</code>, và cả phần đầu thư mục của nó,{' '}
            <code>appA/lab1</code>. Mọi lệnh trên trang này dùng cách thứ hai, tức phần đầu của id mà các báo cáo chạy ghi
            lại (<code>appA/lab1_sumerror</code>), và các bảng dưới đây cũng gọi nó như vậy. Các bài học gọi nó là lab A.1.
          </p>
        }
      >
        <p>
          The lab runner accepts the lab's number, <code>pmpp.py run A.1</code>, and the start of its directory,{' '}
          <code>appA/lab1</code>. Every command on this page uses the second, the start of the id the run reports record
          (<code>appA/lab1_sumerror</code>), and the tables below call it that too. The lessons call it lab A.1.
        </p>
      </Translated>
      <RunningLabs
        example="appA/lab1"
        alsoShownIn={
          <Translated
            vi={
              <>
                bảng "Thiết bị của bạn" ở <a href={href('appA', 'A.4')}>mục A.4</a>, nơi đọc kết quả probe của lab này
              </>
            }
          >
            <>
              the "Your device" panel of <a href={href('appA', 'A.4')}>section A.4</a>, which reads this lab's probe
            </>
          </Translated>
        }
      />
      <BackendsAndEmulator />
      <Translated
        vi={
          <p>
            Lab này cũng được build cho <code>race</code>, biến thể thứ hai của trình giả lập (emulator), chạy đồng thời
            các luồng của một khối dưới ThreadSanitizer. Cây là một lần cập nhật bộ nhớ chia sẻ (shared memory) giữa các
            rào chắn (barrier), và một rào chắn đặt sai chỗ sẽ được báo ở đó kèm dòng mã nguồn. Chạy nó bằng{' '}
            <code>{RUN} --backend race</code>.
          </p>
        }
      >
        <p>
          This lab is also built for <code>race</code>, the emulator's second flavour, which runs a block's threads at once
          under ThreadSanitizer. The tree is a shared-memory update between barriers, and a barrier in the wrong place is
          reported there with its source line. Run it with <code>{RUN} --backend race</code>.
        </p>
      </Translated>
      <RaceLine lab={l1} />
      <VerificationStatus />

      {/* ------------------------------------------------------------ A.1 */}
      <h2>
        <Translated vi="Lab A.1: sai số của phép cộng: thứ tự, độ rộng bộ tích lũy và phép bù">
          Lab A.1: summation error: order, accumulator width and compensation
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mục A.6 đưa ra hai khẳng định mà bạn có thể đo. Thứ tự của một tổng float thay đổi kết quả của nó, nên một cây
            và một vòng lặp cho kết quả khác nhau là chuyện hợp lệ. Và có một thứ chắc chắn hơn việc sắp xếp lại thứ tự:
            phép cộng có bù của Kahan, mà sách có nêu tên nhưng không mô tả. Bạn viết ba phép cộng <code>n</code> giá trị
            float32, mỗi phép là một template theo kiểu bộ tích lũy, để mỗi phép cộng đều được làm tròn về kiểu đó:
          </p>
        }
      >
        <p>
          Section A.6 makes two claims you can measure. The order of a float sum changes its result, so a tree and a loop
          legitimately disagree. And there is something sturdier than reordering: Kahan's compensated summation, which the
          book names and does not describe. You write three sums of <code>n</code> float32 values, each a template on its
          accumulator type, so that every addition is rounded to that type:
        </p>
      </Translated>
      <Source
        section="A.6"
        pages="593–594"
        figs={tr(
          'the tree is ch. 10 Fig. 10.10, p. 237, behind the coarsening loop of Fig. 10.20, p. 247',
          'cây là ch. 10 Hình 10.10, p. 237, đặt sau vòng lặp gộp việc của Hình 10.20, p. 247',
        )}
      />
      <Translated
        vi={
          <ul>
            <li>
              <b>
                <code>sumSerial</code>
              </b>
              : một luồng cộng <code>x[0]</code>, <code>x[1]</code>, ... theo thứ tự chỉ số, đúng thứ tự của vòng lặp CPU.
            </li>
            <li>
              <b>
                <code>sumTree</code>
              </b>
              , tất định và không dùng atomic: luồng t của khối b cộng <code>coarse</code> phần tử của nó{' '}
              <code>x[b*B*coarse + t + j*B]</code> theo thứ tự của j, để các lần đọc của một warp được hợp nhất. Sau đó khối
              kết hợp các tổng của nó bằng cây chia đôi của chương 10, mỗi tầng một rào chắn, và một lần khởi chạy thứ hai
              của cùng kernel đó rút gọn các tổng riêng phần của các khối.
            </li>
            <li>
              <b>
                <code>sumKahan</code>
              </b>
              : như cây, nhưng vòng lặp của mỗi luồng là một tổng có bù. Bản thân cây vẫn là cây thường. Mục{' '}
              <a href={href('appA', 'A.6')}>A.6</a> đi qua từng dòng trong bốn dòng của vòng lặp.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            <b>
              <code>sumSerial</code>
            </b>
            : one thread adds <code>x[0]</code>, <code>x[1]</code>, ... in index order, the CPU loop's order.
          </li>
          <li>
            <b>
              <code>sumTree</code>
            </b>
            , deterministic and without atomics: thread t of block b adds its <code>coarse</code> elements{' '}
            <code>x[b*B*coarse + t + j*B]</code> in order of j, so that a warp's loads are coalesced. The block then
            combines its sums with chapter 10's halving tree, one barrier per level, and a second launch of the same kernel
            reduces the blocks' partial sums.
          </li>
          <li>
            <b>
              <code>sumKahan</code>
            </b>
            : as the tree, but each thread's loop is a compensated sum. The tree itself stays plain. Section{' '}
            <a href={href('appA', 'A.6')}>A.6</a> steps through the four lines of the loop.
          </li>
        </ul>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Phụ lục A không đưa ra mã nào. Các kernel, các bộ tích lũy 16-bit, vòng lặp có bù, phần probe thiết bị, các
              cận sai số và mọi phép đo đều là của khóa học này.
            </p>
          }
        >
          <p>
            Appendix A shows no code. The kernels, the 16-bit accumulators, the compensated loop, the device probe, the
            error bounds and every measurement are ours.
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
                  <code>sumSerial</code>, <code>sumTree</code>, <code>sumKahan</code> và ba hàm khởi chạy của chúng; các bản
                  khởi tạo 16-bit nằm trong các khối <code>#if PMPP_NARROW_ACC</code> được cung cấp sẵn
                </>
              }
            >
              <>
                <code>sumSerial</code>, <code>sumTree</code>, <code>sumKahan</code> and their three launchers; the 16-bit
                instantiations go inside the provided <code>#if PMPP_NARROW_ACC</code> blocks
              </>
            </Translated>,
          ],
          [
            'kernel.metal',
            <Translated
              vi={
                <>
                  <code>sum_serial</code>, <code>sum_tree</code> và <code>sum_kahan</code>; chín bản khởi tạo{' '}
                  <code>[[host_name]]</code> ở cuối tệp đã được cung cấp sẵn
                </>
              }
            >
              <>
                <code>sum_serial</code>, <code>sum_tree</code> and <code>sum_kahan</code>; the nine{' '}
                <code>[[host_name]]</code> instantiations at the bottom are provided
              </>
            </Translated>,
          ],
          [
            'launch.mm',
            <Translated vi={<><code>encodeSumSerial</code>, <code>encodeSumTree</code> và <code>encodeSumKahan</code>, hai lệnh dispatch cho hai lượt của cây</>}>
              <><code>encodeSumSerial</code>, <code>encodeSumTree</code> and <code>encodeSumKahan</code>, two dispatches for the tree's two passes</>
            </Translated>,
          ],
        ]}
      />
      <Translated
        vi={
          <p>
            Mỗi hàm khởi chạy trả về false, không khởi chạy gì, với một kích thước nằm ngoài hợp đồng của nó hoặc một bộ
            tích lũy mà bản build này không có. Những bộ tích lũy nào tồn tại phụ thuộc vào backend:
          </p>
        }
      >
        <p>
          Each launcher returns false, without launching, for a shape outside its contract or an accumulator this build
          does not have. Which accumulators exist depends on the backend:
        </p>
      </Translated>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-table="accumulators">
          <thead>
            <tr>
              <th>
                <Translated vi="Backend">Backend</Translated>
              </th>
              <th>float</th>{/* i18n-ignore: type name */}
              <th>half</th>{/* i18n-ignore: type name */}
              <th>bfloat16</th>{/* i18n-ignore: type name */}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Metal</td>
              <td><code>float</code></td>
              <td><code>half</code></td>
              <td><code>bfloat</code></td>
            </tr>
            <tr>
              <td>CUDA</td>
              <td><code>float</code></td>
              <td><code>__half</code></td>
              <Translated vi={<td><code>__nv_bfloat16</code>, compute capability 8.0 trở lên (driver kiểm tra)</td>}>
                <td><code>__nv_bfloat16</code>, compute capability 8.0 or higher (the driver checks)</td>
              </Translated>
            </tr>
            <tr>
              <td>HIP</td>
              <td><code>float</code></td>
              <td><code>__half</code></td>
              <Translated vi={<td><code>__hip_bfloat16</code>, được gọi là <code>__nv_bfloat16</code> qua một typedef trong <code>sumerror.h</code></td>}>
                <td><code>__hip_bfloat16</code>, named <code>__nv_bfloat16</code> by a typedef in <code>sumerror.h</code></td>
              </Translated>
            </tr>
            <tr>
              <td>emu, race</td>{/* i18n-ignore: backend names */}
              <td><code>float</code></td>
              <td>
                <Translated vi="không có">none</Translated>
              </td>
              <td>
                <Translated vi="không có">none</Translated>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <Translated
        vi={
          <p>
            Các trình giả lập và phần kiểm tra front-end CUDA có mô hình hóa <code>__half</code> nhưng không có{' '}
            <code>__nv_bfloat16</code>, và guard <code>PMPP_NARROW_ACC</code> của lab này loại cả hai nhánh 16-bit của{' '}
            <code>kernel.cu</code> ra khỏi những bản build đó. Vì vậy trên máy này, không trình biên dịch nào biên dịch các
            nhánh đó cả. Chúng là unverified, thậm chí chưa phải front-end-checked, cho đến khi một máy CUDA hoặc ROCm chạy
            chúng.
          </p>
        }
      >
        <p>
          The emulators and the CUDA front-end check model <code>__half</code> but not <code>__nv_bfloat16</code>, and
          this lab's <code>PMPP_NARROW_ACC</code> guard leaves both 16-bit branches of <code>kernel.cu</code> out of
          those builds. On this machine those branches are therefore compiled by no compiler at all. They are unverified,
          and not even front-end-checked, until a CUDA or ROCm machine runs them.
        </p>
      </Translated>
      <Today checked={`2026-11-20 (${FACTS} A16, B1, C1)`}>
        <Translated
          vi={
            <p>
              <code>__nv_bfloat16</code> của CUDA cần compute capability 8.0 trở lên. HIP gọi kiểu BF16 của nó là{' '}
              <code>__hip_bfloat16</code>. Metal có <code>bfloat</code> từ Metal 3.1, và không có <code>double</code>, nên
              phần tham chiếu chạy trên CPU.
            </p>
          }
        >
          <p>
            CUDA's <code>__nv_bfloat16</code> needs compute capability 8.0 or higher. HIP names its BF16 type{' '}
            <code>__hip_bfloat16</code>. Metal has <code>bfloat</code> from Metal 3.1, and no <code>double</code>, so the
            reference runs on the CPU.
          </p>
        </Translated>
      </Today>

      <h3>
        <Translated vi="Kiểm thử">Testing</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Chạy <code>{RUN} --backend emu</code> trước, rồi đến <code>{RUN}</code>. Trình giả lập thực hiện hơn một trăm lần
            khởi chạy cho mỗi bài kiểm thử, mỗi lần với ba thứ tự luồng, nên hãy cho nó vài phút. Mọi kết quả đều được so
            với một tổng có bù tính bằng double trên CPU. Đầu ra bắt đầu là NaN, nên một kết quả chưa từng được ghi sẽ bị
            phát hiện, và mỗi trường hợp chạy hai lần trên cùng các buffer. Các bài kiểm thử kiểm tra bốn điều:
          </p>
        }
      >
        <p>
          Run <code>{RUN} --backend emu</code> first, then <code>{RUN}</code>. The emulator makes over a hundred launches
          per test, each under three thread orders, so give it a few minutes. Every result is compared with a
          compensated sum in double on the CPU. Outputs start as NaN, so a result never written is caught, and every case
          runs twice on the same buffers. The tests check four things:
        </p>
      </Translated>
      <Translated
        vi={
          <ol>
            <li>
              <b>Các trường hợp chính xác.</b> Toàn số một, và các số nguyên từ −3 đến 3, phải ra kết quả chính xác ở mọi
              thứ tự: không tổng riêng phần nào cần làm tròn.
            </li>
            <li>
              <b>Cận sai số.</b> Trên dữ liệu phân bố đều, dữ liệu có độ lớn trộn lẫn và dữ liệu triệt tiêu nhau, mỗi tổng
              phải nằm trong cận chuẩn cho thứ tự của nó: số lần làm tròn mà một phần tử đi qua, nhân hai lần đơn vị làm
              tròn (unit roundoff), nhân tổng các |x|. Hai lần, vì Metal có thể làm tròn về phía 0. README suy ra từng cận.
            </li>
            <li>
              <b>Bài kiểm thử của chính Kahan.</b> Số 1 theo sau bởi rất nhiều bản sao của nửa ULP của 1, với coarse 64. Một
              vòng lặp thường đánh rơi mọi giá trị nhỏ và kết thúc thấp hơn khoảng 32 ULP. Kernel có bù phải nằm trong
              khoảng 2.
            </li>
            <li>
              <b>Các định dạng hẹp</b> (Metal; CUDA và HIP nơi được build). 4,097 số một đi qua vòng lặp và cây half và
              bfloat16 phải cho ra đúng từng bit như một mô hình CPU cùng thứ tự. Một kernel âm thầm tích lũy bằng float sẽ
              trả về 4,097 ở đó.
            </li>
          </ol>
        }
      >
        <ol>
          <li>
            <b>Exact cases.</b> All ones, and whole numbers from −3 to 3, must come out exact in every order: no partial sum
            needs rounding.
          </li>
          <li>
            <b>Error bounds.</b> On uniform, mixed-magnitude and cancelling data, each sum must stay within the standard
            bound for its order: the number of roundings an element passes through, times twice the unit roundoff, times
            the sum of |x|. Twice, because Metal may round toward zero. The README derives each bound.
          </li>
          <li>
            <b>Kahan's own test.</b> 1 followed by many copies of half an ULP of 1, at coarse 64. A plain loop drops every
            small value and ends about 32 ULPs low. The compensated kernel must land within 2.
          </li>
          <li>
            <b>The narrow formats</b> (Metal; CUDA and HIP where built). 4,097 ones through the half and bfloat16 loop and
            tree must give the same bits as a CPU model of the same order. A kernel that quietly accumulates in float
            returns 4,097 there.
          </li>
        </ol>
      </Translated>
      <Translated
        vi={
          <p>
            Ở trạng thái ban đầu, mọi hàm khởi chạy đều trả về false, nên mọi bài kiểm thử phép cộng đều thất bại và chỉ
            probe qua được. Những gì các bài kiểm thử không thấy cũng quan trọng không kém:
          </p>
        }
      >
        <p>
          As shipped, every launcher returns false, so every sum test fails and only the probe passes. What the tests
          cannot see matters as much:
        </p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>
              <b>Trình biên dịch GPU làm gì với vòng lặp của bạn.</b> Trình biên dịch của trình giả lập là trình biên dịch
              của host, vốn mặc định không sắp xếp lại (reassociate) các phép cộng float, nên một vòng lặp có bù chạy qua
              được ở đó không nói lên điều gì về GPU. Nếu <code>kahan, float, big-small</code> thất bại trên Metal mà qua
              được trên trình giả lập, câu hỏi về fast-math ở <a href={href('appA', 'A.6')}>mục A.6</a> có lời giải thích.
              README có một gợi ý, được gập lại.
            </li>
            <li>
              <b>Mọi lỗi về thứ tự.</b> Phép kiểm tra trùng từng bit dùng các số một, nên bắt được một bộ tích lũy float
              nhưng không bắt được một thứ tự sai giữa các giá trị bằng nhau. Benchmark so mọi tập dữ liệu với mô hình CPU
              của thứ tự bạn dùng.
            </li>
            <li>
              <b>Tốc độ.</b> Đó là việc của benchmark.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            <b>What a GPU compiler does to your loop.</b> The emulator's compiler is the host's, which does not reassociate
            float additions by default, so a compensated loop that passes there says nothing about a GPU. If{' '}
            <code>kahan, float, big-small</code> fails on Metal and passes on the emulator, the fast-math question in{' '}
            <a href={href('appA', 'A.6')}>section A.6</a> has the reason. The README has a hint, folded away.
          </li>
          <li>
            <b>Every order mistake.</b> The bit-exact check uses ones, which catches a float accumulator but not a wrong
            order of equal values. The benchmark compares every data set with the CPU model of your order.
          </li>
          <li>
            <b>Speed.</b> That is the benchmark's job.
          </li>
        </ul>
      </Translated>
      <Today checked={`2026-11-20 (${FACTS} A13, B5, C2, C3)`}>
        <Translated
          vi={
            <p>
              NVIDIA không ghi tài liệu về việc nvcc có sắp xếp lại các phép cộng float hay không; lab dùng{' '}
              <code>__fadd_rn</code> và bài kiểm thử chính là phép đo. HIP-Clang mặc định là{' '}
              <code>-fno-associative-math</code>. Chế độ toán mặc định của Metal cho phép sắp xếp lại, và chỉ chế độ an toàn
              (safe) của nó mới cấm.
            </p>
          }
        >
          <p>
            NVIDIA does not document whether nvcc reassociates float additions; the lab uses <code>__fadd_rn</code> and the
            test is the measurement. HIP-Clang defaults to <code>-fno-associative-math</code>. Metal's default math mode
            allows reassociation, and only its safe mode forbids it.
          </p>
        </Translated>
      </Today>

      <h3>
        <Translated vi="Trước khi benchmark, hãy dự đoán">Before you benchmark, predict</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Dòng đầu tiên của benchmark là probe: thiết bị của bạn có làm tròn về gần nhất không, có giữ denormal FP32
            không và có gộp phép nhân-cộng không. Sau đó là ba bảng, được mô tả ở phần số đo bên dưới. Hãy chốt từng
            con số trước. Với con số đầu tiên, trình khám phá khoảng cách ở <a href={href('appA', 'A.2')}>mục A.2</a> cho
            thấy khoảng cách giữa các giá trị half ở bất kỳ độ lớn nào, còn <a href={href('appA', 'A.4')}>mục A.4</a> cho
            thấy bốn chế độ làm tròn.
          </p>
        }
      >
        <p>
          The benchmark's first line is the probe: whether your device rounds to nearest, keeps FP32 denormals and fuses a
          multiply-add. Then come three tables, described under the measurements below. Commit to each number first. For
          the first, <a href={href('appA', 'A.2')}>section A.2</a>'s spacing explorer shows the gap between half values at
          any magnitude, and <a href={href('appA', 'A.4')}>section A.4</a> the four rounding modes.
        </p>
      </Translated>
      <LabPrediction
        id="appA/labs/p-lab1-half-loop"
        prompt={
          <Translated
            vi={
              <>
                Mô hình của mục A.6 làm tròn về gần nhất, và vòng lặp FP16 của nó trên 65,536 giá trị phân bố đều trong
                [0, 1) dừng ở 2048. Kernel serial half của lab chạy tổng đó trên thiết bị của bạn, và Metal được phép làm
                tròn về phía 0 thay vào đó. Kernel trên thiết bị của bạn sẽ trả về bao nhiêu? Hãy tính xem làm tròn về phía
                0 sẽ cho ra gì trước khi chốt.
              </>
            }
          >
            <>
              Section A.6's model rounds to nearest, and its FP16 loop over 65,536 uniform values from [0, 1) stops at 2048.
              The lab's half serial kernel runs that sum on your device, and Metal is allowed to round toward zero instead.
              What will your device's kernel return? Work out what rounding toward zero would give before you commit.
            </>
          </Translated>
        }
        unit={tr('returned by the half loop', 'do vòng lặp half trả về')}
        measure={halfLoop}
        missing={
          <Translated
            vi={
              <>
                Phần này cần một bộ tích lũy half, mà trên máy này chỉ Metal build được. Hãy chạy{' '}
                <code>{RUN} --bench</code> trên một máy Mac, hoặc trên một máy CUDA hay ROCm.
              </>
            }
          >
            <>
              This needs a half accumulator, which only Metal builds on this machine. Run <code>{RUN} --bench</code> on a
              Mac, or on a CUDA or ROCm machine.
            </>
          </Translated>
        }
        reading={
          <Translated
            vi={
              <>
                Làm tròn về gần nhất đánh rơi một số hạng nhỏ hơn nửa khoảng cách quanh tổng đang chạy, và khoảng cách đạt
                2 ở 2048. Làm tròn về phía 0 đánh rơi mọi số hạng nhỏ hơn trọn một khoảng cách, và khoảng cách đạt 1 ở 1024:
                vòng lặp dừng gần 1024, cao hơn một chút vì vài đầu vào được làm tròn thành đúng 1 trong half. Dòng làm tròn
                của probe, ở đầu benchmark, cho biết thiết bị của bạn làm theo cách nào.
              </>
            }
          >
            <>
              Rounding to nearest drops a term below half the gap around the running sum, and the gap reaches 2 at 2048.
              Rounding toward zero drops any term below a whole gap, and the gap reaches 1 at 1024: the loop stops near
              1024, a little above it because a few inputs round to exactly 1 in half. The probe's rounding line, at the top
              of the benchmark, says which your device does.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="appA/labs/p-lab1-kahan-cost"
        prompt={
          <Translated
            vi={
              <>
                Trên 2<sup>24</sup> số float với cùng hệ số gộp việc, kernel Kahan mất thời gian gấp bao nhiêu lần kernel cây
                thường? (1 nghĩa là bằng nhau.)
              </>
            }
          >
            <>
              On 2<sup>24</sup> floats at the same coarsening factor, how many times longer does the Kahan kernel take than
              the plain tree kernel? (1 means the same time.)
            </>
          </Translated>
        }
        unit={tr("× the plain tree's time", '× thời gian của cây thường')}
        measure={kahanCost}
        reading={
          <Translated
            vi={
              <>
                Vòng lặp có bù làm bốn phép cộng cho mỗi phần tử trong khi vòng lặp thường làm một, và cả hai đều đọc 4 byte
                cho mỗi phần tử. Phần số học thêm vào có lộ ra hay không phụ thuộc vào việc phép cộng đang chờ bộ nhớ hay chờ
                tính toán. Hãy so cột GB/s với băng thông của thiết bị bạn từ lab 5.1.
              </>
            }
          >
            <>
              The compensated loop does four additions per element where the plain one does one, and both read 4 bytes per
              element. Whether the extra arithmetic shows depends on whether the sum was waiting for memory or for
              arithmetic. Compare the GB/s column with your device's bandwidth from lab 5.1.
            </>
          </Translated>
        }
      />
      <LabPrediction
        id="appA/labs/p-lab1-coarse"
        prompt={
          <Translated
            vi={
              <>
                Trên 2<sup>22</sup> số float phân bố đều với hệ số gộp việc 4096, cây float thường sẽ lệch bao nhiêu ULP?
                Hãy quyết định luôn liệu Kahan có tốt hơn ở cùng hệ số hay không.
              </>
            }
          >
            <>
              On 2<sup>22</sup> uniform floats at coarsening factor 4096, how many ULPs off will the plain float tree be?
              Decide also whether Kahan will be better at the same factor.
            </>
          </Translated>
        }
        unit={tr('ULPs of float, plain tree', 'ULP của float, cây thường')}
        measure={coarseUlps}
        reading={
          <Translated
            vi={
              <>
                Ở hệ số này, vòng lặp tuần tự của mỗi luồng dài 4,096 phép cộng. Các giá trị phân bố đều đều có cùng một cỡ,
                nên một vòng lặp dài trên chúng có thể mất ít hơn bạn nghĩ; trình khám phá ở mục{' '}
                <a href={href('appA', 'A.6')}>A.6</a> cho thấy loại dữ liệu mà một đoạn dài thực sự gây hại.
              </>
            }
          >
            <>
              Each thread's sequential loop is 4,096 additions long at this factor. Uniform values are all of one size, so
              a long loop of them may lose less than you expect; section <a href={href('appA', 'A.6')}>A.6</a>'s explorer
              shows the data where a long chunk does hurt.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Thêm hai câu nữa, không cần con số. Kernel serial sẽ xử lý thứ tự nào tốt hơn, <code>mixed-asc</code> hay{' '}
            <code>mixed-desc</code>, và vì sao? Và thiết bị của bạn có giữ denormal FP32 không? Bảng đầu tiên dưới đây trả
            lời câu thứ hai.
          </p>
        }
      >
        <p>
          Two more, without a number. Which order will the serial kernel handle better, <code>mixed-asc</code> or{' '}
          <code>mixed-desc</code>, and why? And does your device keep FP32 denormals? The first table below answers the
          second.
        </p>
      </Translated>

      <h3>
        <Translated vi="Máy của bạn đã đo được gì">What your machine measured</Translated>
      </h3>
      <Bench
        title={tr('What the probe found', 'Probe tìm thấy gì')}
        foot={tr(
          "From this machine's recorded lab runs, under each backend's default compiler settings. The probe reads its inputs from a buffer, so nothing is worked out at build time.",
          'Lấy từ các lần chạy lab đã ghi lại trên máy này, với thiết lập trình biên dịch mặc định của từng backend. Probe đọc đầu vào từ một buffer, nên không có gì được tính sẵn lúc build.',
        )}
      >
        <Probe />
      </Bench>
      <Translated
        vi={
          <p>
            Bảng accuracy cộng 65,536 giá trị của mỗi tập dữ liệu bằng mọi phương pháp và bộ tích lũy, và cho sai số tính
            bằng ULP của định dạng bộ tích lũy đó. <code>mixed</code> lấy độ lớn từ 2<sup>−14</sup> đến 2<sup>−2</sup>;{' '}
            <code>mixed-asc</code> và <code>mixed-desc</code> là cùng các giá trị đó đã sắp xếp; <code>cancel</code> gán cho
            chúng dấu ngẫu nhiên; <code>softmax</code> là các số hạng của mẫu số softmax; <code>big-small</code> là bài kiểm
            thử của Kahan ở trên. Tràn số được hiển thị đúng là tràn số: tổng đúng của 65,536 số một lớn hơn giá trị half lớn
            nhất, 65,504.
          </p>
        }
      >
        <p>
          The accuracy table sums 65,536 values of each data set with every method and accumulator, and gives the error in
          ULPs of that accumulator's format. <code>mixed</code> draws magnitudes from 2<sup>−14</sup> to 2<sup>−2</sup>;{' '}
          <code>mixed-asc</code> and <code>mixed-desc</code> are the same values sorted; <code>cancel</code> gives them
          random signs; <code>softmax</code> is the terms of a softmax denominator; <code>big-small</code> is Kahan's
          test above. An overflow is shown as such: the true sum of 65,536 ones is above the largest half, 65,504.
        </p>
      </Translated>
      <Accuracy />
      <Translated
        vi={
          <p>
            Tiếp theo là các bảng đo thời gian. "coarse" chạy tree và Kahan bằng float trên 2<sup>22</sup> số float phân bố
            đều với hệ số gộp việc từ 1 đến 4096; "time" chạy từng phép cộng với từng bộ tích lũy, tree và Kahan trên{' '}
            2<sup>24</sup> số float và vòng lặp một luồng chỉ trên 2<sup>20</sup>, vì một luồng chạy qua 2<sup>24</sup> số
            float có thể chạm giới hạn thời gian lệnh của GPU. GB/s là 4 byte cho mỗi float đầu vào chia cho thời gian: số
            byte mà một phép cộng phải đọc, không phải lượng mà hệ thống bộ nhớ đã chuyển. Mỗi thời gian là lần tốt nhất
            trong vài lần chạy sau một lần chạy không tính giờ.
          </p>
        }
      >
        <p>
          The timed tables follow. "coarse" runs tree and Kahan in float on 2<sup>22</sup> uniform floats at coarsening
          factors from 1 to 4096; "time" runs each sum in each accumulator, tree and Kahan on 2<sup>24</sup> floats and
          the one-thread loop on 2<sup>20</sup> only, since one thread over 2<sup>24</sup> floats can run into a GPU's
          command time limit. GB/s is 4 bytes per input float over the time: the bytes a sum must read, not what the
          memory system moved. Each time is the best of several runs after an untimed one.
        </p>
      </Translated>
      <LabResults
        lab={l1}
        filter={(r) => r.name === 'coarse' || r.name === 'time'}
        cols={[
          rawCol(tr('Table', 'Bảng'), 'name'),
          rawCol(tr('Method', 'Phương pháp'), 'method'),
          rawCol(tr('Accumulator', 'Bộ tích lũy'), 'acc'),
          rawCol('n', 'n'),
          rawCol(tr('Coarse', 'Coarse'), 'coarse'),
          msCol,
          numCol('GB/s', 'gb_per_s'),
          numCol('ULPs', 'ulps'),
        ]}
      />
      <ReferenceSolution lab={l1} />

      <h2>
        <Translated vi="Lab này phục vụ cho điều gì">What this lab feeds</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lab này không phải là một mốc dự án được đánh số của dự án khóa học. Bộ tích lũy dạng template của nó, cùng sai
            số và chi phí đo được của từng lựa chọn, là bằng chứng cho hai quyết định về sau: bộ tích lũy của các phép GEMV
            trọng số int8 và 4-bit trong module X2 và của mốc M9, bước decode lượng tử hóa xây trên chúng, và việc giữ
            softmax của mốc M4, lab 10.3, ở float. Ba hướng để đi xa hơn, đều là của khóa học này:
          </p>
        }
      >
        <p>
          The lab is not a numbered milestone of the course project. Its templated accumulator, and the measured error and
          cost of each choice, are the evidence for two later decisions: the accumulator of module X2's int8- and 4-bit-weight GEMVs and
          of milestone M9, the quantised decode step built on them, and keeping milestone M4's softmax, lab 10.3, in float. Three ways to go further, all ours:
        </p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>
              <b>Một cây có bù.</b> Giữ cặp (tổng, phần bù) qua cả cây, kết hợp hai cặp ở mỗi tầng. Sai số của Kahan ở
              coarse 1, nơi cây làm toàn bộ công việc, có thay đổi không?
            </li>
            <li>
              <b>Bộ cộng cắt bớt của sách.</b> Bài tập 3 hỏi sai số của một bộ cộng làm tròn về phía 0. Trên CUDA,{' '}
              <code>__fadd_rz</code> chính là bộ cộng đó: hãy viết một phép cộng serial với nó và so sai số của nó với sai
              số của <code>__fadd_rn</code>. Trên HIP, các dạng <code>_rz</code> cần{' '}
              <code>OCML_BASIC_ROUNDED_OPERATIONS</code>; Metal không có chế độ làm tròn theo từng phép toán.
            </li>
            <li>
              <b>Cộng theo cặp trong thanh ghi.</b> Làm cho vòng lặp của mỗi luồng cộng theo cặp trên <code>coarse</code>{' '}
              phần tử của nó, theo cách <code>sum</code> của NumPy nhóm theo vị trí, và so nó với Kahan trong bảng
              "coarse".
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            <b>A compensated tree.</b> Keep the pair (sum, compensation) through the tree as well, combining two pairs at
            each level. Does Kahan's error at coarse 1, where the tree does all the work, change?
          </li>
          <li>
            <b>The book's truncating adder.</b> Exercise 3 asks for the error of an adder that rounds toward zero. On CUDA,{' '}
            <code>__fadd_rz</code> is that adder: write a serial sum with it and compare its error with{' '}
            <code>__fadd_rn</code>'s. On HIP the <code>_rz</code> forms need <code>OCML_BASIC_ROUNDED_OPERATIONS</code>;
            Metal has no per-operation rounding modes.
          </li>
          <li>
            <b>Pairwise in registers.</b> Make each thread's loop pairwise over its <code>coarse</code> elements, the way
            NumPy's <code>sum</code> groups by position, and compare it with Kahan in the "coarse" table.
          </li>
        </ul>
      </Translated>
      <Today checked={`2026-11-20 (${FACTS} A9, B4, C2, D4)`}>
        <Translated
          vi={
            <p>
              <code>__fadd_rz</code> có trong tài liệu của NVIDIA; các dạng làm tròn của HIP cần macro ở trên; Metal cho phép
              hoặc làm tròn về số chẵn gần nhất hoặc làm tròn về phía 0, và không cho chọn theo từng phép toán.{' '}
              <code>sum</code> của NumPy dùng phép cộng theo cặp một phần (partial pairwise).
            </p>
          }
        >
          <p>
            <code>__fadd_rz</code> is documented by NVIDIA; HIP's rounded forms need the macro above; Metal allows either round
            to nearest even or round toward zero, and offers no per-operation choice. NumPy's <code>sum</code> uses partial
            pairwise summation.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Phần <a href={href('appA', 'wrap')}>tổng kết</a> có bài kiểm tra tổng hợp của phụ lục, nơi từng bài tập được
            giải, và những chỗ PDF có vẻ sai.
          </p>
        }
      >
        <p>
          The <a href={href('appA', 'wrap')}>summary</a> has the appendix's mixed check, where each exercise is worked,
          and the places where the PDF looks wrong.
        </p>
      </Translated>
    </LabsPage>
  );
}
