import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { functionSource } from '../../sim/toolchain';
import {
  FAILURES,
  FLAG_TEXT,
  FORMS,
  FORM_LABEL,
  IN_BRANCH,
  LINE_PAIRS,
  PRESET_LABEL,
  WIDTHS,
  binary,
  halves,
  hex,
  preset,
  presetsFor,
  vote,
  type Form,
  type Group,
  type VoteResult,
  type Width,
} from '../../sim/warpvote';
import { OneVoteScene } from './OneVoteScene';

// Section 12.3, Coalescing atomic operations with warp-level primitives (pp. 291-295).
// Listings: Figs. 12.3 and 12.4 from labs/ch12/snippets/ (lines pinned and run on the
// emulator by S123.snippets.test.ts), and lab 12.1's ballot form. The visualiser, the
// questions and the scene run on src/sim/warpvote.ts. Vendor facts: notes/today/
// ch12-facts.md (A1 to A7, B1 to B4, C1 to C3, D1) and infra-facts.md (A3, B3, B4).

const FIG3 = 'ch12/snippets/s123_fig12_3.cu';
const FIG4 = 'ch12/snippets/s123_fig12_4_cg.cu';
const LAB = 'ch12/lab1_unstable/solution/';
const STATUS = {
  get fig3() {
    return tr(
      'front-end-checked; emulator-checked; unverified on CUDA hardware; CUDA only',
      'front-end-checked; emulator-checked; unverified trên phần cứng CUDA; chỉ CUDA',
    );
  },
  get fig4() {
    return tr(
      'front-end-checked; emulator-checked; unverified on CUDA hardware; not built for HIP',
      'front-end-checked; emulator-checked; unverified trên phần cứng CUDA; không dựng cho HIP',
    );
  },
};
/** A lab's reference-solution status, in the labs page's words (LabKit's solutionTabStatus), from this machine's runs. */
function useLabStatus(lab: string) {
  const { reports } = useReports();
  return {
    cu: solutionTabStatus(reports, lab, 'kernel.cu'),
    metal: solutionTabStatus(reports, lab, 'kernel.metal'),
    mm: solutionTabStatus(reports, lab, 'launch.mm'),
  };
}
const TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', outputSize: 'global', outputSize_ref: 'global',
  val: 'reg', i: 'reg', j: 'reg', activeThreads: 'reg', leader: 'reg', numActive: 'reg', previousThreads: 'reg',
  previousActiveThreads: 'reg', offset: 'reg', vote: 'reg', keep: 'reg', lane: 'reg', numKept: 'reg', lanesBelow: 'reg',
  flag: 'reg', rank: 'reg', count: 'reg', full: 'reg',
};

const missing = (s: string | null) => !s || s.startsWith('// could not load');
/** A kernel's text with the comment lines above it removed, so line 1 is the book's line 01. */
function kernelOf(src: string | null, name: string): string | null {
  if (missing(src)) return null;
  const f = functionSource(src!, name);
  if (!f) return null;
  const lines = f.text.split('\n');
  let k = 0;
  while (k < lines.length && lines[k].trim().startsWith('//')) k++;
  return lines.slice(k).join('\n');
}
/** The text between "// region: name" and "// endregion" in a lab file, or the named function. */
function labPart(src: string | null, region: string, fn: string): string | null {
  if (missing(src)) return null;
  const m = new RegExp('//\\s*region:\\s*' + region + '\\s*\\n([\\s\\S]*?)\\n\\s*//\\s*endregion').exec(src!);
  if (m) return m[1];
  return functionSource(src!, fn)?.text ?? null;
}

// ---------------------------------------------------------------- listings
function Fig3({ code }: { code: string | null }) {
  if (!code) return null;
  return (
    <Code
      tabs={[{ label: tr('Fig. 12.3', 'Hình 12.3'), code, tiers: TIERS, status: STATUS.fig3 }]}
      highlight={[7, 10, 13, 17, 20, 22, 23, 24]}
      caption={
        <Translated
          vi={
            <>
              Kernel của sách cùng số dòng, từ <code>labs/{FIG3}</code>, nơi có kèm một <code>cond()</code> và{' '}
              <code>laneIdx()</code> của chương 10. Được đánh dấu: bốn bước và hạng. Một thay đổi: dòng 08 gán cho{' '}
              <code>j</code> một giá trị (xem bên dưới).
            </>
          }
        >
          <>
            The book's kernel with its line numbers, from <code>labs/{FIG3}</code>, where a <code>cond()</code> and
            chapter 10's <code>laneIdx()</code> sit beside it. Marked: the four steps and the rank. One change: line 08
            gives <code>j</code> a value (see below).
          </>
        </Translated>
      }
    />
  );
}

function BallotListing() {
  const status = useLabStatus('ch12/lab1_unstable');
  const cu = useLabFile(LAB + 'kernel.cu');
  const metal = useLabFile(LAB + 'kernel.metal');
  const helpers = labPart(cu, 'helpers', 'maskPopc');
  const k = labPart(cu, 'warp', 'filterWarpAggregatedKernel');
  const m = labPart(metal, 'warp', 'filter_warp_aggregated');
  if (!k) return null;
  return (
    <Code
      tabs={[
        { label: tr('CUDA and HIP', 'CUDA và HIP'), code: (helpers ? helpers + '\n\n' : '') + k, tiers: TIERS, status: status.cu },
        ...(m ? [{ label: 'Metal', code: m, tiers: TIERS, status: status.metal }] : []),
      ]}
      caption={
        <Translated
          vi={
            <>
              Kernel gộp theo warp của lab 12.1, từ <code>labs/{LAB}</code>: kiểu mặt nạ và hai hàm đếm bit của nó, rồi đến
              kernel. Tab Metal là cùng kernel đó viết bằng các hàm SIMD group.
            </>
          }
        >
          <>
            Lab 12.1's warp-aggregated kernel, from <code>labs/{LAB}</code>: the mask type and its two bit counts, then
            the kernel. The Metal tab is the same kernel in SIMD-group functions.
          </>
        </Translated>
      }
    />
  );
}

function Fig4({ code }: { code: string | null }) {
  if (!code) return null;
  return (
    <Code
      tabs={[{ label: tr('Fig. 12.4', 'Hình 12.4'), code, tiers: TIERS, status: STATUS.fig4 }]}
      highlight={[7, 10, 12, 19, 21]}
      caption={
        <Translated
          vi={
            <>
              Kernel của sách cùng số dòng, từ <code>labs/{FIG4}</code>, có kèm header và namespace mà trang sách bỏ qua. Các
              dòng 07, 08 và 12 khác trang sách (xem bên dưới).
            </>
          }
        >
          <>
            The book's kernel with its line numbers, from <code>labs/{FIG4}</code>, where the header and namespace the
            page leaves out are included. Lines 07, 08 and 12 differ from the page (see below).
          </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- the visualiser
const bits = (r: VoteResult, g: Group) => (r.input.width === 8 ? 8 : r.input.form === 'metal' ? r.input.width : g.maskBits);
const maskText = (r: VoteResult, g: Group) => binary(g.mask, bits(r, g));

function groupLines(r: VoteResult, g: Group, gi: number): { code: ReactNode; value: ReactNode }[] {
  const { form, width } = r.input;
  const L: { code: ReactNode; value: ReactNode }[] = [];
  const m = <span className="mono" data-mask-bits="">{maskText(r, g)}</span>;
  const atomicLine = g.atomic
    ? (
      <Translated vi={<>lane {g.atomic.lane} cộng {g.atomic.add}, nhận lại <b>{g.atomic.before}</b>; bộ đếm {g.atomic.before} → {g.atomic.after}</>}>
        <>lane {g.atomic.lane} adds {g.atomic.add}, gets back <b>{g.atomic.before}</b>; counter {g.atomic.before} → {g.atomic.after}</>
      </Translated>
    )
    : <span className="t-fail"><Translated vi="không có atomic">no atomic</Translated></span>;
  if (form === 'metal') {
    L.push({ code: <code>count = simd_sum(keep ? 1 : 0)</code>, value: g.count });
    L.push({ code: <code>simd_is_first()</code>, value: <Translated vi={<>true chỉ ở lane 0: mọi lane đều hoạt động</>}><>true in lane 0 only: every lane is active</></Translated> });
    L.push({
      code: <code>{r.input.guard ? 'if (simd_is_first() && count > 0) j = atomic_fetch_add_explicit(...)' : 'if (simd_is_first()) j = atomic_fetch_add_explicit(...)'}</code>,
      value: g.atomic ? atomicLine : <Translated vi={<>bỏ qua: count bằng 0</>}><>skipped: count is 0</></Translated>,
    });
    L.push({ code: <code>j = simd_broadcast_first(j)</code>, value: <Translated vi={<>mọi lane nhận {g.start}</>}><>every lane gets {g.start}</></Translated> });
    L.push({ code: <code>simd_ballot(keep), for reference</code>, value: <>{m} <span className="muted">ctz = {g.ctz}{g.ctz === 64 ? tr(' (no lane)', ' (không có lane nào)') : ''}</span></> });
    return L;
  }
  if (form === 'ballot') {
    L.push({ code: <code>vote = __ballot_sync(full, keep)</code>, value: <>{m} <span className="muted">{hex(g.mask, g.maskBits)}</span></> });
    if (g.skipped) {
      L.push({ code: <code>if (vote == 0) return;</code>, value: <Translated vi={<>rẽ nhánh: cả warp rời đi</>}><>taken: the whole warp leaves</></Translated> });
      return L;
    }
    L.push({ code: <code>leader = maskFfs(vote) - 1</code>, value: <>{g.ffs} - 1 = <b>{g.leader}</b></> });
    L.push({ code: <code>numKept = maskPopc(vote)</code>, value: g.count });
    L.push({ code: <code>fetch_add(numKept)</code>, value: atomicLine });
    L.push({ code: <code>j = __shfl_sync(full, j, leader)</code>, value: <Translated vi={<>mọi lane nhận {g.start}</>}><>every lane gets {g.start}</></Translated> });
    return L;
  }
  const naive = form === 'hip-naive';
  L.push({
    code: <code>{naive ? 'unsigned int activeThreads = __activemask()' : 'activeThreads = __activemask()'}</code>,
    value: (
      <>
        {m} <span className="muted">{hex(g.mask, g.maskBits)}</span>
        {naive && width === 64 ? <span className="muted"><Translated vi=" (mặt nạ 64 bit, bị cắt còn 32 bit thấp)"> (the 64-bit mask, cut to its low 32 bits)</Translated></span> : null}
        {r.groups.length > 1 ? <span className="muted"><Translated vi={<> ảnh chụp {gi + 1}: các lane {g.lanes.join(', ')}</>}> snapshot {gi + 1}: lanes {g.lanes.join(', ')}</Translated></span> : null}
      </>
    ),
  });
  L.push({
    code: <code>{form === 'bug-ffs' ? 'leader = __ffs(activeThreads)' : 'leader = __ffs(activeThreads) - 1'}</code>,
    value: (
      <>
        {form === 'bug-ffs' ? <b>{g.leader}</b> : <>{g.ffs} - 1 = <b>{g.leader}</b></>}
        {g.leader === -1 ? <span className="muted"><Translated vi=" (0xFFFFFFFF khi hiểu là unsigned)"> (0xFFFFFFFF as unsigned)</Translated></span> : null}
        {!g.leaderRuns ? <span className="t-fail"><Translated vi=" không phải lane chạy đoạn mã này"> not a lane that runs this code</Translated></span> : null}
      </>
    ),
  });
  L.push({ code: <code>numActive = __popc(activeThreads)</code>, value: g.count });
  L.push({ code: <code>if (laneIdx() == leader) fetch_add(numActive)</code>, value: atomicLine });
  L.push({
    code: <code>j = __shfl_sync(activeThreads, j, leader)</code>,
    value: g.start != null ? (
      <Translated vi={<>các lane qua kiểm tra nhận {g.start}</>}><>passing lanes get {g.start}</></Translated>
    ) : (
      <span className="t-fail"><Translated vi="không xác định: lane nguồn nằm ngoài mặt nạ">undefined: the source lane is outside the mask</Translated></span>
    ),
  });
  return L;
}

function notesFor(r: VoteResult): string[] {
  const { form, width, pass } = r.input;
  const out: string[] = [];
  if (r.compileError) return [r.compileError];
  const k = r.passing.length;
  const g = r.groups[0];
  if (k === 0) {
    if (form === 'metal')
      out.push(r.atomics
        ? tr(
            'Nobody keeps, yet lane 0 still does the atomic and adds 0: harmless, but wasted. Tick the guard to skip it, as lab 12.1 does.',
            'Không lane nào giữ, vậy mà lane 0 vẫn thực hiện phép atomic và cộng 0: vô hại, nhưng phí. Đánh dấu điều kiện chặn để bỏ qua nó, như lab 12.1 làm.',
          )
        : tr('Nobody keeps: the count is 0 and the guard skips the atomic.', 'Không lane nào giữ: count bằng 0 và điều kiện chặn bỏ qua phép atomic.'));
    else if (IN_BRANCH[form]) out.push(tr('No lane passes, so no lane enters the branch and nothing happens.', 'Không lane nào qua kiểm tra, nên không lane nào vào nhánh và không có gì xảy ra.'));
    else out.push(tr('Nobody keeps: the vote is 0 and the whole warp leaves before the atomic.', 'Không lane nào giữ: phiếu bầu bằng 0 và cả warp rời đi trước phép atomic.'));
    return out;
  }
  if (form === 'metal' && !pass[0])
    out.push(tr(
      'Lane 0 keeps nothing, yet it does the atomic: in uniform control flow every lane is active, so lane 0 is the first active thread. The count and the ranks still include only the lanes that keep.',
      'Lane 0 không giữ gì, vậy mà nó thực hiện phép atomic: trong luồng điều khiển đồng nhất mọi lane đều hoạt động, nên lane 0 là luồng hoạt động đầu tiên. Count và các hạng vẫn chỉ tính các lane có giữ.',
    ));
  if (r.groups.length > 1)
    out.push(tr(
      `The passing lanes reached __activemask() as ${r.groups.length} snapshots: ${r.atomics} atomics, each reserving a range of its own. Every key still gets its own slot.`,
      `Các lane qua kiểm tra tới __activemask() thành ${r.groups.length} ảnh chụp: ${r.atomics} phép atomic, mỗi phép đặt trước một phạm vi riêng. Mỗi khóa vẫn có ô riêng của mình.`,
    ));
  if (form === 'hip-naive' && width === 64 && r.passing.some((l) => l >= 32)) {
    if (g.leader === -1)
      out.push(tr(
        'No passing lane is below 32, so the 32-bit mask is 0, __ffs gives 0 and the leader is -1: nobody does the atomic, and the shuffle has no lane to read.',
        'Không lane qua kiểm tra nào dưới 32, nên mặt nạ 32 bit bằng 0, __ffs cho 0 và leader là -1: không ai thực hiện phép atomic, và phép shuffle không có lane nào để đọc.',
      ));
    else
      out.push(tr(
        `Lanes 32 to 63 vanish from the 32-bit mask: the count is ${g.count} for ${k} passing lanes, and 1 << lane is undefined for the upper lanes, so their ranks are too.`,
        `Các lane 32 đến 63 biến mất khỏi mặt nạ 32 bit: count là ${g.count} cho ${k} lane qua kiểm tra, và 1 << lane không xác định với các lane phía trên, nên hạng của chúng cũng vậy.`,
      ));
  }
  if (form === 'hip-naive' && width === 64 && !r.passing.some((l) => l >= 32))
    out.push(tr(
      'Every passing lane is below 32, so the narrow mask loses nothing: this input passes. Try "only lanes 32 to 63".',
      'Mọi lane qua kiểm tra đều dưới 32, nên mặt nạ hẹp không mất gì: đầu vào này vượt qua. Hãy thử "only lanes 32 to 63".',
    ));
  if (form === 'hip-naive' && width < 64) out.push(tr('A 32-wide wavefront fits in 32 bits: the naive port is right here.', 'Một wavefront rộng 32 vừa trong 32 bit: bản chuyển đổi ngây thơ đúng ở đây.'));
  if (form === 'bug-ffs') {
    const bad = r.groups.find((x) => !x.leaderRuns);
    if (bad)
      out.push(tr(
        `Without the - 1, lane ${bad.leader} is chosen to lead. It does not run this code, so nobody does the atomic and the shuffle reads a lane outside the mask.`,
        `Không có - 1, lane ${bad.leader} được chọn làm leader. Nó không chạy đoạn mã này, nên không ai thực hiện phép atomic và phép shuffle đọc một lane ngoài mặt nạ.`,
      ));
    else
      out.push(tr(
        `Lane ${g.leader} leads instead of lane ${g.leader - 1}. It passes too, so it reserves the same range: right by luck. Make lane ${g.leader} fail.`,
        `Lane ${g.leader} làm leader thay cho lane ${g.leader - 1}. Nó cũng qua kiểm tra, nên nó đặt trước cùng phạm vi: đúng nhờ may mắn. Hãy làm lane ${g.leader} không qua.`,
      ));
  }
  if (form === 'bug-below')
    out.push(tr(
      'Each lane counts itself as well as the lanes below it, so every rank is one too high: the first slot of the range stays empty and the last key lands past it.',
      'Mỗi lane đếm cả chính nó lẫn các lane bên dưới, nên mọi hạng đều thừa một: ô đầu tiên của phạm vi bỏ trống và khóa cuối rơi ra ngoài phạm vi.',
    ));
  if (r.correct && form !== 'bug-ffs')
    out.push(tr(
      `${k} ${k === 1 ? 'key' : 'keys'} in slots ${r.counterBefore} to ${r.counterAfter - 1}, ${r.atomics} ${r.atomics === 1 ? 'atomic' : 'atomics'}.`,
      `${k} khóa ở các ô ${r.counterBefore} đến ${r.counterAfter - 1}, ${r.atomics} phép atomic.`,
    ));
  return out;
}

function WarpVote() {
  const [width, setWidth] = useState<Width>(8);
  const [form, setForm] = useState<Form>('book');
  const [pass, setPass] = useState<boolean[]>(() => preset('book', 8));
  const [counter, setCounter] = useState(40);
  const [split, setSplit] = useState(false);
  const [guard, setGuard] = useState(true);
  const splitApplies = IN_BRANCH[form] && form !== 'hip-naive';
  const r = useMemo(
    () => vote({ width, pass, counter, form, guard, snapshots: split && splitApplies ? halves(pass) : undefined }),
    [width, pass, counter, form, guard, split, splitApplies],
  );
  const resize = (w: Width) => {
    setWidth(w);
    setPass((p) => Array.from({ length: w }, (_, l) => (l < p.length ? p[l] : false)));
  };
  const lanes = Array.from({ length: width }, (_, l) => l);
  const rows = width === 64 ? [lanes.slice(0, 32), lanes.slice(32)] : [lanes];
  const leaders = new Set(r.groups.filter((g) => g.atomic).map((g) => g.atomic!.lane));
  const writers = r.lanes.filter((L) => L.writes);
  // The output strip: the reserved range, plus any slot a lane writes outside it.
  const slots = writers.flatMap((L) => (L.slot == null ? [] : [L.slot]));
  const lo = Math.min(r.counterBefore, ...slots);
  const hi = Math.max(r.counterAfter, ...slots.map((s) => s + 1), r.counterBefore + 1);
  const failures = r.flags.filter((f) => FAILURES.includes(f));
  const showBits = width === 64 ? 'hex' : 'binary';
  return (
    <Bench
      title={tr('Warp vote visualiser', 'Trực quan hóa bỏ phiếu warp')}
      task={tr(
        'Click lanes to make them pass or fail. Change the form, the width, the counter.',
        'Bấm vào các lane để cho chúng qua hoặc không qua kiểm tra. Đổi dạng mã, độ rộng, bộ đếm.',
      )}
      foot={
        <Translated
          vi={
            <>
              Các lane giữ khóa, mặt nạ và hạng của mình trong thanh ghi (register); bộ đếm và đầu ra nằm trong bộ nhớ toàn cục.
              Một lane viền nét đứt nằm ngoài nhánh: nó không qua kiểm tra và không bao giờ chạy mã bỏ phiếu.
            </>
          }
        >
          <>
            Lanes hold their key, mask and rank in registers; the counter and the output are in global memory. A dashed
            lane is outside the branch: it failed the test and never runs the vote code.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span className="field ui" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
          <Translated vi="Số lane">Lanes</Translated>
          <span className="seg" role="group" aria-label={tr('warp width', 'độ rộng warp')}>
            {WIDTHS.map((w) => (
              <button key={w} aria-pressed={width === w} onClick={() => resize(w)}>{w === 8 ? tr('8 (toy)', '8 (đồ chơi)') : w}</button>
            ))}
          </span>
        </span>
        <label className="field">
          <Translated vi="Mã">Code</Translated>
          <select value={form} onChange={(e) => setForm(e.target.value as Form)} aria-label={tr('form', 'dạng mã')}>
            {FORMS.map((f) => <option key={f} value={f}>{FORM_LABEL[f]}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Bộ đếm trước">Counter before</Translated>
          <input type="number" value={counter} min={0} aria-label={tr('counter before', 'bộ đếm trước')} onChange={(e) => setCounter(Math.max(0, Math.floor(+e.target.value || 0)))} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="muted"><Translated vi="Qua kiểm tra:">Pass:</Translated></span>
        {presetsFor(width).map((p) => (
          <button key={p} className="quiet" onClick={() => setPass(preset(p, width))}>{PRESET_LABEL[p]}</button>
        ))}
        {splitApplies ? (
          <label className="check">
            <input type="checkbox" checked={split} onChange={(e) => setSplit(e.target.checked)} /> <Translated vi="tách các lane qua kiểm tra thành hai ảnh chụp">split the passing lanes into two snapshots</Translated>
          </label>
        ) : null}
        {form === 'metal' ? (
          <label className="check">
            <input type="checkbox" checked={guard} onChange={(e) => setGuard(e.target.checked)} /> <Translated vi="bỏ qua phép atomic khi không lane nào giữ (lab 12.1 làm vậy)">skip the atomic when nobody keeps (lab 12.1 does)</Translated>
          </label>
        ) : null}
      </div>

      <div className="mx-label t-reg"><Translated vi="Các lane của warp">Lanes of the warp</Translated></div>
      {rows.map((row, ri) => (
        <div key={ri} style={{ display: 'grid', gridTemplateColumns: `repeat(${row.length}, minmax(1.55rem, ${row.length > 8 ? '1fr' : '3.4rem'}))`, gap: 2, marginBottom: 6, minWidth: row.length > 8 ? 620 : undefined }}>
          {row.map((l) => {
            const L = r.lanes[l];
            const outside = !L.votes;
            const lead = leaders.has(l);
            return (
              <button key={l} data-lane={l} data-pass={L.pass ? 1 : 0} data-rank={L.rank ?? ''}
                data-slot={L.writes ? (L.slot == null ? '?' : L.slot) : ''} aria-pressed={L.pass}
                title={tr(`lane ${l}: ${L.pass ? 'passes' : 'fails'}`, `lane ${l}: ${L.pass ? 'qua' : 'không qua'}`)}
                onClick={() => setPass((p) => p.map((x, i) => (i === l ? !x : x)))}
                style={{
                  padding: '0.15rem 0', minWidth: 0, fontSize: '0.7rem', lineHeight: 1.25, borderRadius: 2,
                  background: L.pass ? 'var(--reg-wash)' : 'var(--cell)', color: 'var(--ink)',
                  border: lead ? '2px solid var(--ink)' : `1px ${outside ? 'dashed' : 'solid'} ${L.pass ? 'var(--reg)' : 'var(--rule)'}`,
                  fontWeight: 450,
                }}>
                <span style={{ display: 'block', color: 'var(--ink-3)' }}>{l}</span>
                <span style={{ display: 'block', fontWeight: 650 }}>{L.pass ? '✓' : '·'}</span>
                <span style={{ display: 'block', color: L.undefinedSlot ? 'var(--fail)' : 'var(--global)' }}>
                  {L.writes ? (L.slot == null ? '?' : L.slot) : ' '}
                </span>
              </button>
            );
          })}
        </div>
      ))}
      <Translated
        vi={
          <p className="status-line" style={{ margin: '0 0 0.8rem' }}>
            Mỗi lane: số của nó, nó có qua kiểm tra không, và ô nó ghi (? = không xác định). Lane được viền đậm thực hiện phép
            atomic.
          </p>
        }
      >
        <p className="status-line" style={{ margin: '0 0 0.8rem' }}>
          Each lane: its number, whether it passes, and the slot it writes (? = undefined). The outlined lane does the
          atomic.
        </p>
      </Translated>

      {r.compileError ? (
        <div className="narrate bad" role="status" data-compile-error="">{r.compileError}</div>
      ) : (
        <>
          <div className="cols" style={{ gridTemplateColumns: `repeat(${Math.max(1, Math.min(r.groups.length, 2))}, minmax(0, 1fr))`, gap: '0.6rem', marginBottom: '0.6rem' }}>
            {r.groups.length === 0 ? <p className="status-line"><Translated vi="Không lane nào chạy mã bỏ phiếu.">No lane runs the vote code.</Translated></p> : null}
            {r.groups.map((g, gi) => (
              <div key={gi} className="panel" data-group={gi} data-mask={hex(g.mask, g.maskBits)}
                data-leader={g.leader} data-count={g.count} data-atomic={g.atomic ? `${g.atomic.before}>${g.atomic.after}` : 'none'}
                data-start={g.start ?? 'undefined'}>
                <div className="mx-label">{r.groups.length > 1 ? tr(`Snapshot ${gi + 1} of ${r.groups.length}`, `Ảnh chụp ${gi + 1} / ${r.groups.length}`) : tr('The warp', 'Warp')}</div>
                {groupLines(r, g, gi).map((x, k) => (
                  <div key={k} className="cols" style={{ gridTemplateColumns: 'minmax(0, 1.15fr) minmax(0, 1fr)', gap: '0.1rem 0.8rem', padding: '0.25rem 0', borderTop: k ? '1px solid var(--rule-soft)' : undefined, fontSize: '0.8rem' }}>
                    <div style={{ overflowWrap: 'anywhere' }}>{x.code}</div>
                    <div>{x.value}</div>
                  </div>
                ))}
              </div>
            ))}
          </div>
          <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)' }}>
          <div>
            {writers.length ? (
              <table className="threads" aria-label={tr('Ranks of the passing lanes', 'Hạng của các lane qua kiểm tra')} style={{ marginBottom: '0.8rem' }}>
                <thead>
                  <tr><th>lane{/* i18n-ignore: lane is kept in English */}</th><th>{form === 'metal' ? tr('lanes below', 'các lane bên dưới') : tr('lanes-below mask', 'mặt nạ các lane bên dưới')}</th><th><Translated vi="&amp; mặt nạ">&amp; mask</Translated></th><th><Translated vi="hạng">rank</Translated></th><th><Translated vi="ô">slot</Translated></th></tr>
                </thead>
                <tbody>
                  {(writers.length > 10 ? [...writers.slice(0, 5), null, ...writers.slice(-3)] : writers).map((L, k) => {
                    if (!L)
                      return (
                        <tr key={'gap' + k}><td colSpan={5} className="muted"><Translated vi={<>… thêm {writers.length - 8} lane qua kiểm tra …</>}>… {writers.length - 8} more passing lanes …</Translated></td></tr>
                      );
                    const g = L.group == null ? null : r.groups[L.group];
                    const b = g ? (width === 8 ? 8 : form === 'metal' ? width : g.maskBits) : 32;
                    const show = (m: bigint | null) => (m == null ? <span className="t-fail"><Translated vi="không xác định">undefined</Translated></span> : showBits === 'hex' ? hex(m, b) : binary(m, b));
                    return (
                      <tr key={L.lane} data-row-lane={L.lane}>
                        <td>{L.lane}</td>
                        <td className="mono">{show(L.below)}</td>
                        <td className="mono">{show(L.and)}</td>
                        <td>{L.rank ?? <span className="t-fail">?</span>}</td>
                        <td className={L.undefinedSlot ? 't-fail' : ''}>{L.slot ?? '?'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            ) : null}
          </div>
          <div>
            <div className="mx-label t-global"><Translated vi="Đầu ra (bộ nhớ toàn cục)">Output (global memory)</Translated></div>
            <div className="mx global" style={{ gridTemplateColumns: `repeat(${Math.min(hi - lo, 16)}, auto)`, marginBottom: '0.5rem' }} data-strip="">
              {Array.from({ length: hi - lo }, (_, k) => lo + k).map((s) => {
                const who = writers.filter((L) => L.slot === s).map((L) => L.lane);
                const reserved = s >= r.counterBefore && s < r.counterAfter;
                const bad = who.length > 1 || (who.length === 1 && !reserved) || (reserved && who.length === 0);
                return (
                  <span key={s} className={bad ? 'bad' : who.length ? 'on' : reserved ? '' : 'void'} data-out-slot={s} data-out-lanes={who.join(',')}
                    title={tr(`slot ${s}`, `ô ${s}`)} style={{ minWidth: '2.7rem', height: '2.3rem', alignItems: 'start', paddingTop: '0.2rem' }}>
                    {who.length ? 'L' + who.join('+') : '·'}
                    <sub>{s}</sub>
                  </span>
                );
              })}
            </div>
            {writers.some((L) => L.undefinedSlot) ? (
              <p className="status-line t-fail" style={{ margin: 0 }}>
                <Translated vi={<>{writers.filter((L) => L.undefinedSlot).length} lane qua kiểm tra ghi vào một địa chỉ không xác định, không hiển thị ở đây.</>}>
                  {writers.filter((L) => L.undefinedSlot).length} passing {writers.filter((L) => L.undefinedSlot).length === 1 ? 'lane writes' : 'lanes write'} at an
                  undefined address, not shown here.
                </Translated>
              </p>
            ) : null}
          </div>
          </div>
        </>
      )}
      <div className={'narrate' + (failures.length ? ' bad' : '')} role="status" style={{ marginTop: '0.4rem' }}
        data-correct={r.correct ? 1 : 0} data-atomics={r.atomics} data-flags={r.flags.join(' ')} data-after={r.counterAfter}>
        {failures.length ? (
          <p style={{ margin: '0 0 0.4rem' }}>
            {failures.map((f) => <span key={f} className="chip fail" style={{ marginRight: 6 }}>{FLAG_TEXT[f]}</span>)}
          </p>
        ) : r.passing.length ? (
          <p style={{ margin: '0 0 0.4rem' }}><span className="chip ok"><Translated vi="mọi lane qua kiểm tra đều có ô riêng trong phạm vi đã đặt trước">every passing lane has its own slot in the reserved range</Translated></span></p>
        ) : null}
        {notesFor(r).map((t, i) => <p key={i} style={{ margin: 0 }}>{t}</p>)}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- Fig. 12.3 against Fig. 12.4
function WhichLine({ fig3, fig4 }: { fig3: string | null; fig4: string | null }) {
  const [sel, setSel] = useState<string | null>(null);
  const [tries, setTries] = useState<Record<string, number>>({});
  const [solved, setSolved] = useState<Record<string, boolean>>({});
  if (!fig3 || !fig4) return null;
  const l3 = fig3.split('\n');
  const l4 = fig4.split('\n');
  const pair = LINE_PAIRS.find((p) => p.id === sel) ?? null;
  const pick = (n: number) => {
    if (!pair) return;
    setTries((t) => ({ ...t, [pair.id]: n }));
    if (n === pair.fig4) setSolved((s) => ({ ...s, [pair.id]: true }));
  };
  const last = pair ? tries[pair.id] : undefined;
  const fmt = (n: number) => String(n).padStart(2, '0');
  const done = LINE_PAIRS.filter((p) => solved[p.id]).length;
  return (
    <Bench
      title={tr('Which line is which?', 'Dòng nào ứng với dòng nào?')}
      task={tr(
        'Pick a step of Fig. 12.3, then the line of Fig. 12.4 that does the same job.',
        'Chọn một bước của Hình 12.3, rồi chọn dòng của Hình 12.4 làm cùng việc đó.',
      )}
      foot={
        <span data-matched={done}>
          <Translated vi={<>Đã ghép {done} / {LINE_PAIRS.length}.</>}>{done} of {LINE_PAIRS.length} matched.</Translated>
        </span>
      }
    >
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
        <div>
          <div className="mx-label"><Translated vi="Hình 12.3">Fig. 12.3</Translated></div>
          <div style={{ display: 'grid', gap: 4 }}>
            {LINE_PAIRS.map((p) => (
              <button key={p.id} data-pair={p.id} data-solved={solved[p.id] ? 1 : 0} aria-pressed={sel === p.id} onClick={() => setSel(p.id)}
                style={{ textAlign: 'left', fontFamily: 'var(--sans)', fontWeight: 450, whiteSpace: 'normal' }}>
                {p.fig3.map((n) => (
                  <span key={n} className="mono" style={{ display: 'block', fontSize: '0.78rem' }}>
                    {fmt(n)} {l3[n - 1]?.trim()}
                  </span>
                ))}
                {solved[p.id] ? <span className="chip ok" style={{ marginTop: 4 }}><Translated vi={<>dòng {fmt(p.fig4)} của Hình 12.4</>}>line {fmt(p.fig4)} of Fig. 12.4</Translated></span> : null}
              </button>
            ))}
          </div>
        </div>
        <div>
          <div className="mx-label"><Translated vi={<>Hình 12.4{pair ? '' : ': hãy chọn một bước ở bên trái trước'}</>}>Fig. 12.4{pair ? '' : ': pick a step on the left first'}</Translated></div>
          <div style={{ display: 'grid', gap: 2 }}>
            {l4.map((text, k) => {
              const n = k + 1;
              const code = text.trim();
              if (!code || code.startsWith('//') || code === '}' || code === '{') return null;
              const isLast = last === n;
              return (
                <button key={n} data-fig4-line={n} disabled={!pair} onClick={() => pick(n)} className="quiet"
                  style={{
                    textAlign: 'left', fontWeight: 450, padding: '0.15rem 0.5rem', whiteSpace: 'pre-wrap', fontSize: '0.76rem',
                    borderColor: isLast ? (n === pair?.fig4 ? 'var(--ok)' : 'var(--fail)') : undefined,
                    background: isLast ? (n === pair?.fig4 ? 'var(--ok-wash)' : 'var(--fail-wash)') : undefined,
                  }}>
                  <span className="mono">{fmt(n)} {code}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>
      {pair && last != null ? (
        <div className={'narrate' + (last === pair.fig4 ? '' : ' bad')} role="status" style={{ marginTop: '0.8rem' }} data-verdict={last === pair.fig4 ? 'right' : 'wrong'}>
          {last === pair.fig4 ? (
            <><b><Translated vi="Đúng.">Yes.</Translated></b> {pair.step}</>
          ) : (
            <>
              <b><Translated vi="Không phải dòng đó.">Not that one.</Translated></b>{' '}
              <Translated vi="Hãy tìm dòng làm việc này:">Look for the line that does this:</Translated> {pair.step}
            </>
          )}
        </div>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS123: Record<string, Question> = {
  'ch12/12.3/q-rank': {
    kind: 'numeric',
    id: 'ch12/12.3/q-rank',
    prompt: (
      <Translated
        vi={
          <>
            Mặt nạ của một warp là <code>1011 0100</code> ở dạng nhị phân (bit 0 là lane 0, ở bên phải: các lane 2, 4, 5 và 7 qua
            kiểm tra). <code>fetch_add</code> của leader trả về 40. Lane 7 ghi vào ô nào?
          </>
        }
      >
        <>
          A warp's mask is <code>1011 0100</code> in binary (bit 0 is lane 0, on the right: lanes 2, 4, 5 and 7 pass).
          The leader's <code>fetch_add</code> returns 40. Which slot does lane 7 write?
        </>
      </Translated>
    ),
    answer: 43,
    unit: 'slot',
    vi: { unit: 'ô' },
    explain: (
      <Translated
        vi={
          <p>
            Các lane qua kiểm tra dưới lane 7 là 2, 4 và 5, nên hạng của nó là 3 và ô của nó là 40 + 3 = 43. Leader là lane 2
            (<code>__ffs</code> cho 3, trừ 1), nó cộng 4, và bộ đếm kết thúc ở 44.
          </p>
        }
      >
        <p>
          The passing lanes below lane 7 are 2, 4 and 5, so its rank is 3 and its slot 40 + 3 = 43. The leader is lane 2
          (<code>__ffs</code> gives 3, minus 1), it adds 4, and the counter ends at 44.
        </p>
      </Translated>
    ),
    source: '§12.3, pp. 292–293',
  },
  'ch12/12.3/q-ffs': {
    kind: 'choice',
    id: 'ch12/12.3/q-ffs',
    prompt: (
      <Translated
        vi={
          <>
            Ai đó bỏ mất <code>- 1</code>: <code>leader = __ffs(activeThreads)</code>. Với những tập lane qua kiểm tra nào thì
            đầu ra vẫn đúng?
          </>
        }
      >
        <>
          Someone drops the <code>- 1</code>: <code>leader = __ffs(activeThreads)</code>. For which sets of passing
          lanes is the output still correct?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không bao giờ.">
            Never.
          </Translated>
        ),
        why: (
          <Translated vi="Nếu lane đó qua kiểm tra, nó thực hiện cùng phép atomic với cùng count và phát cùng điểm bắt đầu. Chỉ leader thay đổi.">
            If that lane passes, it does the same atomic with the same count and broadcasts the same start. Only the leader changes.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Luôn luôn: __ffs chỉ chọn một leader khác, và bất kỳ lane nào của warp cũng có thể thực hiện phép atomic thay các lane khác.">
            Always: __ffs just picks a different leader, and any lane of the warp can do the atomic for the others.
          </Translated>
        ),
        why: (
          <Translated vi="Nếu lane đó không qua kiểm tra, không lane nào thực hiện phép atomic, và phép shuffle đọc một lane ngoài mặt nạ, điều này không xác định. Khi chỉ lane 31 qua kiểm tra, leader sẽ là 32, một lane mà warp không có.">
            If that lane does not pass, no lane does the atomic, and the shuffle reads a lane outside the mask, which is undefined. With only lane 31 passing, the leader would be 32, a lane the warp does not have.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Đúng khi lane ngay trên lane qua kiểm tra thấp nhất cũng qua kiểm tra.">
            Exactly when the lane just above the lowest passing lane also passes.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chỉ khi lane 0 qua kiểm tra.">
            Only when lane 0 passes.
          </Translated>
        ),
        why: (
          <Translated vi="Bản thân lane 0 không quan trọng. Thử các lane 3 và 4: leader thành 4, lane này qua kiểm tra, và đầu ra đúng.">
            Lane 0 does not matter as such. Try lanes 3 and 4: the leader becomes 4, which passes, and the output is right.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một lỗi phụ thuộc dữ liệu: các đầu vào kiểm thử dày đặc, nơi các lane cạnh nhau đều qua kiểm tra, không bao giờ làm
            lộ nó. Hãy kiểm thử cả với các lane qua kiểm tra thưa và rời rạc; lỗi cài sẵn của trình trực quan hóa giúp bạn tìm
            một trường hợp như vậy.
          </p>
        }
      >
        <p>
          A data-dependent bug: dense test inputs, where neighbouring lanes all pass, never show it. Test with sparse and
          isolated passing lanes too; the visualiser's planted bug lets you find one.
        </p>
      </Translated>
    ),
    source: '§12.3, p. 293; notes/today/ch12-facts.md A2, A4',
    vi: { source: '§12.3, p. 293; notes/today/ch12-facts.md, mục A2, A4' },
  },
  'ch12/12.3/q-hip64': {
    kind: 'choice',
    id: 'ch12/12.3/q-hip64',
    prompt: (
      <Translated
        vi={
          <>
            Bạn chuyển Hình 12.3 sang một GPU AMD có wavefront rộng 64. Trình biên dịch từ chối phép shuffle, nên bạn chỉ đổi
            đối số mặt nạ của nó thành <code>(unsigned long long)activeThreads</code>. Nó biên dịch được. Điều gì xảy ra?
          </>
        }
      >
        <>
          You port Fig. 12.3 to an AMD GPU with 64-wide wavefronts. The compiler rejects the shuffle, so you change only
          its mask argument to <code>(unsigned long long)activeThreads</code>. It compiles. What happens?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Đầu ra sai bất cứ khi nào có lane qua kiểm tra nằm trong 32 đến 63: mặt nạ, __popc, __ffs và 1 << lane đều vẫn là 32 bit.">
            {'Wrong output whenever a passing lane is in 32 to 63: the mask, __popc, __ffs and 1 << lane all stay 32-bit.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Đúng: phép shuffle là lời gọi duy nhất phụ thuộc vào độ rộng, và mặt nạ của nó giờ là 64 bit.">
            Correct: the shuffle was the only call that depended on the width, and its mask is now 64 bits.
          </Translated>
        ),
        why: (
          <Translated vi="Chính mặt nạ đã bị cắt còn 32 bit ở dòng 07, trước khi phép shuffle thấy nó.">
            The mask itself was cut to 32 bits on line 07, before the shuffle ever sees it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Vẫn là lỗi biên dịch.">
            Still a compile error.
          </Translated>
        ),
        why: (
          <Translated vi="Một mặt nạ 64 bit thỏa static_assert, và không gì khác được kiểm tra lúc biên dịch.">
            A 64-bit mask satisfies the static_assert, and nothing else is checked at compile time.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ đúng trên thiết bị rộng 64, nơi mặt nạ shuffle 64 bit khớp với wavefront.">
            Correct on 64-wide devices only, where the 64-bit shuffle mask matches the wavefront.
          </Translated>
        ),
        why: (
          <Translated vi="Ngược lại: trên wavefront rộng 32, mặt nạ hẹp không mất gì.">
            The other way round: on 32-wide wavefronts the narrow mask loses nothing.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Cách sửa là một mặt nạ có kiểu 64 bit, giữ nguyên 64 bit suốt đường đi: <code>__popcll</code>, <code>__ffsll</code>{' '}
            và <code>{'1ull << lane'}</code>. Trên GPU AMD rộng 32 (RDNA), bản chuyển đổi ngây thơ tình cờ chạy đúng, đó là cách
            nó lọt qua kiểm thử.
          </p>
        }
      >
        <p>
          The fix is a typed 64-bit mask kept 64-bit all the way: <code>__popcll</code>, <code>__ffsll</code> and{' '}
          <code>{'1ull << lane'}</code>. On 32-wide AMD GPUs (RDNA) the naive port happens to work, which is how it
          survives testing.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch12-facts.md B1; infra-facts.md B3',
    vi: { source: 'notes/today/ch12-facts.md, mục B1; infra-facts.md, mục B3' },
  },
  'ch12/12.3/q-snapshot': {
    kind: 'choice',
    id: 'ch12/12.3/q-snapshot',
    prompt: (
      <Translated
        vi={
          <>
            Trên một GPU NVIDIA, các lane qua kiểm tra của một warp tới <code>__activemask()</code> thành hai nhóm: lane 1 và 2,
            rồi lane 4 và 5. Hình 12.3 làm gì?
          </>
        }
      >
        <>
          On an NVIDIA GPU the passing lanes of one warp reach <code>__activemask()</code> as two groups: lanes 1 and 2,
          then lanes 4 and 5. What does Fig. 12.3 do?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Lane 4 và 5 bị mất: chỉ leader của nhóm đầu thực hiện phép atomic, và phạm vi của warp đã được dùng.">
            Lanes 4 and 5 are lost: only the first group's leader does an atomic, and the warp's range is already used.
          </Translated>
        ),
        why: (
          <Translated vi="Chúng chạy cùng đoạn mã một lát sau với mặt nạ, leader và phép atomic của riêng mình.">
            They run the same code a moment later with their own mask, leader and atomic.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó bị deadlock trong phép shuffle: lane 1 và 2 chờ ở đó lane 4 và 5, vốn tới sau.">
            It deadlocks in the shuffle: lanes 1 and 2 wait there for lanes 4 and 5, which arrive later.
          </Translated>
        ),
        why: (
          <Translated vi="Các lane có thể gọi một intrinsic _sync đồng thời với các mặt nạ khác nhau miễn là các mặt nạ không giao nhau (A4).">
            Lanes may call a _sync intrinsic concurrently with different masks as long as the masks are disjoint (A4).
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó đặt trước bốn ô hai lần.">
            It reserves four slots twice.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi ảnh chụp chỉ đếm các lane của riêng nó: 2 và 2.">
            Each snapshot counts only its own lanes: 2 and 2.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Hai phép atomic, mỗi phép đặt trước một phạm vi liền kề, và cả bốn khóa được lưu đúng.">
            Two atomics, each reserving a contiguous range, and all four keys stored correctly.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Kernel vẫn đúng; điều không được hứa là "một phép atomic mỗi warp" của sách. Một bài thực hành đếm số phép atomic mỗi
            warp trên NVIDIA có thể thấy nhiều hơn một.
          </p>
        }
      >
        <p>
          The kernel stays correct; what is not promised is the book's "one atomic per warp". A lab that counts atomics
          per warp on NVIDIA may see more than one.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch12-facts.md A1, A4',
    vi: { source: 'notes/today/ch12-facts.md, mục A1, A4' },
  },
  'ch12/12.3/q-metal': {
    kind: 'choice',
    id: 'ch12/12.3/q-metal',
    prompt: (
      <Translated
        vi={
          <>
            Kernel Metal của lab 12.1 bỏ phiếu trong luồng điều khiển đồng nhất: mọi lane gọi{' '}
            <code>simd_prefix_exclusive_sum(keep ? 1 : 0)</code> và <code>simd_sum(keep ? 1 : 0)</code>, và lane có{' '}
            <code>simd_is_first()</code> là true thực hiện phép atomic. Lane 0 không qua kiểm tra; lane 3 và 9 qua. Lane nào thực
            hiện phép atomic, và đầu ra có đúng không?
          </>
        }
      >
        <>
          Lab 12.1's Metal kernel votes in uniform control flow: every lane calls{' '}
          <code>simd_prefix_exclusive_sum(keep ? 1 : 0)</code> and <code>simd_sum(keep ? 1 : 0)</code>, and the lane for
          which <code>simd_is_first()</code> is true does the atomic. Lane 0 fails the test; lanes 3 and 9 pass. Which
          lane does the atomic, and is the output right?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Lane 3, lane đầu tiên có giữ.">
            Lane 3, the first lane that keeps.
          </Translated>
        ),
        why: (
          <Translated vi="simd_is_first() nghĩa là luồng hoạt động đầu tiên. Trong luồng điều khiển đồng nhất mọi lane đều hoạt động, nên đó là lane 0.">
            simd_is_first() means the first active thread. In uniform control flow every lane is active, so that is lane 0.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Lane 0, cộng 2, và đầu ra đúng.">
            Lane 0, adding 2, and the output is right.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Lane 0, cộng 3, nên một ô bị phí.">
            Lane 0, adding 3, so one slot is wasted.
          </Translated>
        ),
        why: (
          <Translated vi="Tổng đếm các cờ keep, không phải các lane hoạt động: 0 + 1 + 1 = 2.">
            The sum counts the keep flags, not the active lanes: 0 + 1 + 1 = 2.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không ai: lane 0 không qua, nên phép atomic bị bỏ qua và các khóa bị mất.">
            Nobody: lane 0 failed, so the atomic is skipped and the keys are lost.
          </Translated>
        ),
        why: (
          <Translated vi="Không qua kiểm tra không làm một lane ngừng hoạt động ở đây; nó chỉ làm cờ keep của lane đó bằng 0.">
            Failing the test does not make a lane inactive here; it only makes its keep flag 0.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Luồng hoạt động đầu tiên là lane 0 vì mọi lane đều hoạt động, trong khi count và các hạng chỉ tính các lane có giữ.
            Nếu không lane nào giữ, lane 0 cộng 0 trừ khi mã kiểm tra count, điều mà lab 12.1 làm.
          </p>
        }
      >
        <p>
          The first active thread is lane 0 because every lane is active, while the count and the ranks include only the
          lanes that keep. If no lane keeps, lane 0 adds 0 unless the code checks the count, which lab 12.1 does.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch12-facts.md C1 (MSL spec 6.10.2)',
    vi: { source: 'notes/today/ch12-facts.md, mục C1 (đặc tả MSL 6.10.2)' },
  },
  'ch12/12.3/q-fullmask': {
    kind: 'choice',
    id: 'ch12/12.3/q-fullmask',
    prompt: (
      <Translated
        vi={
          <>
            Bên trong <code>if(cond(val))</code> một đồng nghiệp "đơn giản hóa" dòng 20 thành{' '}
            <code>j = __shfl_sync(0xffffffff, j, leader);</code>. Vì sao điều đó sai?
          </>
        }
      >
        <>
          Inside <code>if(cond(val))</code> a colleague "simplifies" line 20 to{' '}
          <code>j = __shfl_sync(0xffffffff, j, leader);</code>. Why is that wrong?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Nó chỉ chậm hơn: phép shuffle chờ các lane đã bỏ qua nhánh, rồi đưa j của leader cho mọi lane.">
            It is only slower: the shuffle waits for the lanes that skipped the branch, then hands every lane the leader's j.
          </Translated>
        ),
        why: (
          <Translated vi="NVIDIA xếp một mặt nạ chứa lane không gọi vào loại không hợp lệ, không phải chậm.">
            NVIDIA classes a mask that names a lane which does not call as invalid, not slow.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các lane không qua cond có bit được bật nhưng không bao giờ tới phép shuffle; mọi lane chưa thoát trong mặt nạ phải gọi nó với cùng mặt nạ, nên lời gọi không hợp lệ.">
            Lanes that failed cond have their bit set but never reach the shuffle; every non-exited lane in the mask must call it with the same mask, so the call is invalid.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Các lane không qua nhận 0, điều này vô hại vì chúng không bao giờ lưu gì.">
            The lanes that failed receive 0, which is harmless because they never store anything.
          </Translated>
        ),
        why: (
          <Translated vi="Chúng không bao giờ tới lời gọi. Vấn đề là các lane đang chờ chúng.">
            They never reach the call. The problem is the lanes that are waiting for them.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hình 12.3 truyền <code>activeThreads</code> vì đúng những lane đó thực thi lời gọi. Mặt nạ đầy đủ chỉ đúng khi mọi
            lane của warp tới được phép shuffle, như trong dạng ballot của lab 12.1.
          </p>
        }
      >
        <p>
          Fig. 12.3 passes <code>activeThreads</code> because exactly those lanes execute the call. The full mask is
          right only where every lane of the warp reaches the shuffle, as in lab 12.1's ballot form.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch12-facts.md A4 (CUDA guide 5.4.6.6)',
    vi: { source: 'notes/today/ch12-facts.md, mục A4 (hướng dẫn CUDA 5.4.6.6)' },
  },
};

// ---------------------------------------------------------------- the page
export function S123() {
  const src3 = useLabFile(FIG3);
  const src4 = useLabFile(FIG4);
  const fig3 = kernelOf(src3, 'filter_kernel');
  const fig4 = kernelOf(src4, 'filter_kernel');
  return (
    <>
      <Translated
        vi={
          <p>
            Trong Hình 12.2, các lane của một warp có giữ khóa cùng tới phép atomic một lúc, nhắm vào cùng một bộ đếm. Chúng chắc
            chắn va chạm và được phục vụ lần lượt. Nhưng các lane của một warp có thể trao đổi giá trị với giá rẻ, vậy hãy để
            chúng thống nhất tổng trước rồi cho một lane gửi một phép atomic duy nhất thay cho tất cả: một{' '}
            <b>phép toán nguyên tử được hợp nhất</b> (coalesced atomic operation). Sách so sánh nó với cách phần cứng gộp các lần
            load và store của một warp thành ít giao dịch bộ nhớ hơn; văn bản của sách chỉ tới chương 5, còn khóa học này trình
            bày điều đó ở <a href={href('ch06', '6.1')}>mục 6.1</a>.
          </p>
        }
      >
        <p>
          In Fig. 12.2 the lanes of one warp that keep a key reach the atomic together, aimed at the same counter. They
          are certain to collide and be served one after another. But lanes of a warp can exchange values cheaply, so
          let them agree on the total first and have one lane send a single atomic for all of them: a{' '}
          <b>coalesced atomic operation</b>. The book compares it with the way the hardware merges a warp's loads and
          stores into fewer memory transactions; its text points to chapter 5, and this course covers that in{' '}
          <a href={href('ch06', '6.1')}>section 6.1</a>.
        </p>
      </Translated>
      <Source section="12.3" pages="291–292" />
      <Translated
        vi={
          <p>
            Bốn bước làm được việc đó. Chọn một <b>leader</b> trong số các lane đang hoạt động tại phép atomic. Tìm tổng mà chúng
            lẽ ra đã cộng: mỗi lane một, tức là số lane hoạt động. Để leader thực hiện một phép atomic và <b>phát</b> (broadcast)
            điểm bắt đầu của phạm vi nó nhận được. Để mỗi lane tự tính <b>offset</b> của mình trong phạm vi đó và lưu khóa của
            mình vào đó.
          </p>
        }
      >
        <p>
          Four steps do it. Pick a <b>leader</b> among the lanes that are active at the atomic. Find the total they
          would have added: one each, so the number of active lanes. Let the leader perform one atomic and{' '}
          <b>broadcast</b> the start of the range it got. Let each lane work out its own <b>offset</b> in that range and
          store its key there.
        </p>
      </Translated>
      <Source section="12.3" pages="292" />
      <Translated
        vi={
          <p>
            Cảnh này chạy cả hai kernel trên một warp đồ chơi tám lane: bốn phép atomic xếp hàng ở một bộ đếm, hai lần, rồi cùng
            warp đó với một lần bỏ phiếu và một phép atomic.
          </p>
        }
      >
        <p>
          The scene runs both kernels on an eight-lane toy warp: four atomics queueing at one counter, twice, and then
          the same warp with one vote and one atomic.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một warp đồ chơi tám lane, từng bước một. Hai thứ tự mà các phép atomic được phục vụ là các lần rút có seed, không
              phải mô hình của phần cứng nào, và không có gì được đo thời gian.
            </p>
          }
        >
          <p>
            A toy warp of eight lanes, one step at a time. The two orders in which the atomics are served are seeded
            draws, not a model of any hardware, and nothing is timed.
          </p>
        </Translated>
      </Simplification>
      <OneVoteScene />

      <h2><Translated vi="Hình 12.3, từng bước">Fig. 12.3, step by step</Translated></h2>
      <Translated
        vi={
          <p>
            Hình 12.3 viết bốn bước bằng các intrinsic warp. Hãy đọc nó với warp đồ chơi trong đầu; các dòng được đánh dấu là
            các bước.
          </p>
        }
      >
        <p>
          Fig. 12.3 writes the four steps with warp intrinsics. Read it with the toy warp in mind; the marked lines are
          the steps.
        </p>
      </Translated>
      <Fig3 code={fig3} />
      <Translated
        vi={
          <p>
            Dòng 07: <code>__activemask()</code> trả về một số nguyên 32 bit có bit i được bật khi lane i của warp đang hoạt động
            tại điểm đó. Bên trong cả hai câu <code>if</code>, ý định là các lane nằm trong phạm vi và đã qua kiểm tra; với các
            lane 1, 2, 4 và 5 ở đó, ví dụ của sách cho <code>...0011 0110</code>.
          </p>
        }
      >
        <p>
          Line 07: <code>__activemask()</code> returns a 32-bit integer whose bit i is set when lane i of the warp is
          active at that point. Inside both <code>if</code>s that is meant to be the lanes that are in range and passed
          the test; with lanes 1, 2, 4 and 5 there, the book's example gives <code>...0011 0110</code>.
        </p>
      </Translated>
      <Source section="12.3" pages="292" />
      <Translated
        vi={
          <p>
            Dòng 10 và 11: leader là lane hoạt động có chỉ số nhỏ nhất, tức vị trí của bit được bật thấp nhất.{' '}
            <code>__ffs</code> (find first set) trả về vị trí đó đếm từ 1, vì 0 được dành cho "không bit nào được bật"; do đó có{' '}
            <code>- 1</code>. Dòng 13: mỗi lane hoạt động lẽ ra đã cộng 1, nên tổng là số bit được bật, <code>__popc</code>{' '}
            (population count). Dòng 17: <code>fetch_add(numActive)</code> của leader trả về <code>j</code>, điểm bắt đầu của
            phạm vi.
          </p>
        }
      >
        <p>
          Lines 10 and 11: the leader is the active lane with the smallest index, the position of the lowest set bit.{' '}
          <code>__ffs</code> (find first set) returns that position counting from 1, because 0 is reserved for "no bit
          set"; hence the <code>- 1</code>. Line 13: each active lane would have added 1, so the total is the number of
          set bits, <code>__popc</code> (population count). Line 17: the leader's <code>fetch_add(numActive)</code>{' '}
          returns <code>j</code>, the start of the range.
        </p>
      </Translated>
      <Source section="12.3" pages="293" />
      <Translated
        vi={
          <p>
            Dòng 20: một phép shuffle từ leader đưa <code>j</code> cho các lane hoạt động khác. Mặt nạ của nó là{' '}
            <code>activeThreads</code>, không phải <code>0xffffffff</code>, vì chỉ những lane này thực thi lời gọi (các quy tắc
            của chương 10 cho intrinsic <code>_sync</code>, <a href={href('ch10', '10.7')}>mục 10.7</a>).
          </p>
        }
      >
        <p>
          Line 20: a shuffle from the leader gives <code>j</code> to the other active lanes. Its mask is{' '}
          <code>activeThreads</code>, not <code>0xffffffff</code>, because only these lanes execute the call (chapter
          10's rules for <code>_sync</code> intrinsics, <a href={href('ch10', '10.7')}>section 10.7</a>).
        </p>
      </Translated>
      <Source section="12.3" pages="292–293" />
      <Ask q={questionsS123['ch12/12.3/q-fullmask']} />
      <Translated
        vi={
          <p>
            Dòng 22 đến 24: offset của một lane là số lane hoạt động bên dưới nó. Lane i dựng một mặt nạ có các bit 0 đến i − 1
            được bật, <code>{'(1 << i) - 1'}</code>, AND nó với mặt nạ hoạt động, rồi đếm phần còn lại. Đây là một{' '}
            <b>tổng tiền tố nhị phân</b> (binary prefix sum): một scan loại trừ trên các cờ một bit, thực hiện bằng hai phép
            toán bit.
          </p>
        }
      >
        <p>
          Lines 22 to 24: a lane's offset is the number of active lanes below it. Lane i builds a mask with bits 0 to
          i − 1 set, <code>{'(1 << i) - 1'}</code>, ANDs it with the active mask, and counts what is left. This is a{' '}
          <b>binary prefix sum</b>: an exclusive scan of one-bit flags, done with two bit operations.
        </p>
      </Translated>
      <Source section="12.3" pages="293" />
      <Translated
        vi={
          <p>
            Có một dòng khác trang sách. Sách khai báo <code>unsigned int j;</code> ở dòng 08 và chỉ gán giá trị cho nó trong
            leader, vậy mà mọi lane đều truyền <code>j</code> cho phép shuffle ở dòng 20. C++ coi việc đọc một biến chưa khởi
            tạo là không xác định, và khi trình giả lập của khóa học được biên dịch (clang, <code>-O1</code>), trình biên dịch đã
            tận dụng quyền đó: mọi lane qua kiểm tra đều chạy nhánh của leader, bốn phép atomic thay vì một. Đoạn mã viết{' '}
            <code>unsigned int j = 0;</code>, như lab 12.1. Chưa kiểm tra xem nvcc có làm điều gì tương tự không.
          </p>
        }
      >
        <p>
          One line differs from the page. The book declares <code>unsigned int j;</code> on line 08 and gives it a value
          only in the leader, yet every lane passes <code>j</code> to the shuffle on line 20. C++ makes reading an
          uninitialised variable undefined, and when the course's emulator was compiled (clang, <code>-O1</code>) the
          compiler took that licence: every passing lane ran the leader's branch, four atomics instead of one. The
          snippet writes <code>unsigned int j = 0;</code>, as lab 12.1 does. Whether nvcc does anything similar was not
          checked.
        </p>
      </Translated>
      <Beyond title={tr('Ours', 'Của khóa học này')}>
        <Translated
          vi={
            <p>
              Phát hiện khi chạy đoạn mã trên trình giả lập CPU của khóa học; phần đầu của đoạn mã ghi lại điều này. Không phải
              sự kiện từ nhà sản xuất, và chưa thấy trên GPU.
            </p>
          }
        >
          <p>
            Found while running the listing on the course's CPU emulator; the snippet's header records it. Not a vendor
            fact, and not seen on a GPU.
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Thử xem">Try it</Translated></h2>
      <Predict
        id="ch12/12.3/p-rank"
        prompt={
          <Translated vi={<>Các lane 1, 2, 4 và 5 qua kiểm tra (ví dụ của sách). Lane 5 tính được offset nào ở dòng 22 đến 24?</>}>
            <>Lanes 1, 2, 4 and 5 pass (the book's example). Which offset does lane 5 compute on lines 22 to 24?</>
          </Translated>
        }
        unit={tr('offset', 'offset')}
      >
        {(v) => (
          <p>
            <Translated vi={<>Các lane 1, 2 và 4 nằm dưới lane 5 và qua kiểm tra, nên offset của nó là {vote({ width: 8, pass: preset('book', 8), counter: 0, form: 'book' }).lanes[5].rank}.</>}>
              Lanes 1, 2 and 4 are below lane 5 and pass, so its offset is {vote({ width: 8, pass: preset('book', 8), counter: 0, form: 'book' }).lanes[5].rank}.
            </Translated>{' '}
            {v === 3
              ? tr('That is your number.', 'Đó là con số của bạn.')
              : v === 5
                ? tr('Not 5: that is the lane number. The offset counts only the passing lanes below it.', 'Không phải 5: đó là số của lane. Offset chỉ đếm các lane qua kiểm tra bên dưới nó.')
                : v === 4
                  ? tr('Not 4: that counts lane 5 itself.', 'Không phải 4: con số đó tính cả chính lane 5.')
                  : tr('The offset counts passing lanes below the lane, and nothing else.', 'Offset đếm các lane qua kiểm tra bên dưới lane đó, và không gì khác.')}{' '}
            <Translated vi="Trình trực quan hóa mở ra với mặt nạ này.">The visualiser opens on this mask.</Translated>
          </p>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Trình trực quan hóa chạy một warp qua đoạn mã bạn chọn. Bắt đầu với Hình 12.3 trên các lane của sách, rồi: tách các
            lane qua kiểm tra thành hai ảnh chụp; chuyển sang bản chuyển đổi HIP ngây thơ trên 64 lane với chỉ các lane 32 đến 63
            qua kiểm tra; chọn một lỗi cài sẵn và tìm một đầu vào nó xử lý đúng và một đầu vào nó xử lý sai.
          </p>
        }
      >
        <p>
          The visualiser runs one warp through the code you pick. Start with Fig. 12.3 on the book's lanes, then: split
          the passing lanes into two snapshots; switch to the naive HIP port on 64 lanes with only lanes 32 to 63
          passing; pick a planted bug and find an input it gets right and one it gets wrong.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Không có thời gian. Những lane qua kiểm tra nào cùng tới <code>__activemask()</code> là do bạn chọn ở đây (một ảnh
              chụp, hoặc hai), không phải mô hình phần cứng. Giá trị không xác định hiển thị là "?"; một GPU thật sẽ cho ra một
              con số nào đó.
            </p>
          }
        >
          <p>
            No timing. Which passing lanes reach <code>__activemask()</code> together is your choice here (one snapshot,
            or two), not a model of the hardware. Undefined values are shown as "?"; a real GPU would produce some number.
          </p>
        </Translated>
      </Simplification>
      <WarpVote />
      <Ask q={questionsS123['ch12/12.3/q-rank']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS123['ch12/12.3/q-ffs']} />

      <h2><Translated vi="Một ảnh chụp, không phải nhánh">A snapshot, not the branch</Translated></h2>
      <Translated
        vi={
          <p>
            Hình 12.3 dựa vào việc <code>__activemask()</code> trả về đúng các lane đã vào nhánh. Hướng dẫn của NVIDIA giờ nói
            không thể dùng nó cho việc đó: mặt nạ là một ảnh chụp các lane tình cờ đang hội tụ tại lời gọi, và các lane qua kiểm
            tra có thể tới thành nhiều hơn một nhóm.
          </p>
        }
      >
        <p>
          Fig. 12.3 relies on <code>__activemask()</code> returning exactly the lanes that took the branch. NVIDIA's
          guide now says it cannot be used for that: the mask is a snapshot of the lanes that happen to be converged at
          the call, and the passing lanes may arrive in more than one group.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch12-facts.md A1, A4)', '2026-11-20 (notes/today/ch12-facts.md, mục A1, A4)')}>
        <Translated
          vi={
            <p>
              Hướng dẫn CUDA: <code>__activemask()</code> "cannot be used to determine which warp lanes execute a given branch"
              và "only provides an instantaneous snapshot of the active threads within a warp". Các GPU từ compute capability
              7.0 lập lịch các luồng một cách độc lập.
            </p>
          }
        >
          <p>
            The CUDA guide: <code>__activemask()</code> "cannot be used to determine which warp lanes execute a given
            branch" and "only provides an instantaneous snapshot of the active threads within a warp". GPUs from compute
            capability 7.0 schedule threads independently.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Dù vậy Hình 12.3 vẫn đúng. Mọi lane trong một ảnh chụp đọc cùng mặt nạ, bầu cùng leader và đặt trước số đếm của riêng
            mình; phép shuffle <code>_sync</code> sau đó gom đúng những lane đó, và các mặt nạ không giao nhau có thể chạy phép
            shuffle đồng thời. Điều không còn được hứa là "một phép atomic mỗi warp". Đánh dấu ô tách trong trình trực quan hóa:
            hai phép atomic, cùng bốn ô.
          </p>
        }
      >
        <p>
          Fig. 12.3 stays correct anyway. Every lane in one snapshot reads the same mask, elects the same leader and
          reserves its own count; the <code>_sync</code> shuffle then gathers exactly those lanes, and disjoint masks
          may run their shuffles concurrently. What is no longer promised is "one atomic per warp". Tick the split box in
          the visualiser: two atomics, the same four slots.
        </p>
      </Translated>
      <Ask q={questionsS123['ch12/12.3/q-snapshot']} />

      <h2><Translated vi="Dạng nên viết">The form to write</Translated></h2>
      <Translated
        vi={
          <p>
            Cách có tài liệu để biết chính xác lane nào qua kiểm tra là một lần bỏ phiếu mà mọi lane tham gia, trong luồng điều
            khiển đồng nhất, trước mọi nhánh: <code>__ballot_sync(full, keep)</code> bật bit i khi <code>keep</code> của lane i
            là true. Một warp có phiếu bầu bằng 0 thì rời đi (một nhánh mà mọi lane rẽ cùng một hướng), leader là{' '}
            <code>ffs(vote) - 1</code>, count là <code>popc(vote)</code>, và hạng của mỗi lane là popcount của phần phiếu bầu
            bên dưới nó. Đây là những gì lab 12.1 viết.
          </p>
        }
      >
        <p>
          The documented way to learn exactly which lanes pass is a vote that every lane takes, in uniform control flow,
          before any branch: <code>__ballot_sync(full, keep)</code> sets bit i when lane i's <code>keep</code> is true.
          A warp whose vote is 0 leaves (a branch every lane takes the same way), the leader is <code>ffs(vote) - 1</code>,
          the count <code>popc(vote)</code>, and each lane's rank is the popcount of the vote below it. This is what lab
          12.1 writes.
        </p>
      </Translated>
      <Source section="12.3" pages="292–293" />
      <BallotListing />
      <Translated
        vi={
          <p>
            Chỗ nó khác Hình 12.3: phiếu bầu thay cho <code>__activemask()</code>; mọi lane, trong phạm vi hay không, đều thực thi
            phép shuffle, nên mặt nạ của nó là mặt nạ đầy đủ; mặt nạ có kiểu riêng, <code>lane_mask_t</code>, 32 bit trên CUDA và
            64 bit trên HIP, được đếm bằng các hàm hỗ trợ chọn <code>__popc</code> hoặc <code>__popcll</code> theo kiểu đó; chỉ
            số lane lấy từ <code>warpSize</code>; và <code>j</code> bắt đầu từ 0.
          </p>
        }
      >
        <p>
          Where it differs from Fig. 12.3: the vote replaces <code>__activemask()</code>; every lane, in range or not,
          executes the shuffle, so its mask is the full one; the mask has its own type, <code>lane_mask_t</code>, 32
          bits on CUDA and 64 on HIP, counted by helpers that pick <code>__popc</code> or <code>__popcll</code> by that
          type; the lane index comes from <code>warpSize</code>; and <code>j</code> starts at 0.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách dùng <code>__activemask()</code>. Dạng ballot theo hướng dẫn của NVIDIA (ch12-facts A1) và là lựa chọn của
              khóa học. Những kernel top-k thực tế nào dùng nó: xem <a href={href('ch12', 'wrap')}>trang tổng kết</a>.
            </p>
          }
        >
          <p>
            The book uses <code>__activemask()</code>. The ballot form follows NVIDIA's guide (ch12-facts A1) and is the
            course's choice. Which real top-k kernels use it: the <a href={href('ch12', 'wrap')}>summary page</a>.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Trên Metal mỗi bước là một hàm SIMD group, và hai trong số đó là trọn một bước: <code>simd_sum</code> là count và{' '}
            <code>simd_prefix_exclusive_sum</code> là hạng. <code>simd_is_first()</code> chọn leader và{' '}
            <code>simd_broadcast_first</code> phát điểm bắt đầu. Chúng hoạt động trên các luồng đang hoạt động, và Apple không nói
            liệu các luồng hoạt động bên trong một nhánh có đúng là những luồng đã vào nhánh không, nên lab gọi chúng từ mọi
            lane, ngoài mọi nhánh.
          </p>
        }
      >
        <p>
          On Metal every step is a SIMD-group function, and two of them are whole steps: <code>simd_sum</code> is the
          count and <code>simd_prefix_exclusive_sum</code> is the rank. <code>simd_is_first()</code> picks the leader
          and <code>simd_broadcast_first</code> hands out the start. They work over the active threads, and Apple does
          not say whether the active threads inside a branch are exactly the ones that took it, so the lab calls them
          from every lane, outside any branch.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Metal không có trong sách. Các tên đã được biên dịch với bộ công cụ Metal cục bộ; ngữ nghĩa lấy từ đặc tả MSL
              6.10.2 (ch12-facts C1, C3).
            </p>
          }
        >
          <p>
            Metal is not in the book. Names compiled against the local Metal toolchain; semantics from the MSL
            specification 6.10.2 (ch12-facts C1, C3).
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS123['ch12/12.3/q-metal']} />

      <h2><Translated vi="Trên HIP: mặt nạ 64 bit">On HIP: masks of 64 bits</Translated></h2>
      <Translated
        vi={
          <p>
            Hình 12.3 như in trong sách không biên dịch được dưới HIP: mọi intrinsic <code>_sync</code> của HIP đều đòi mặt nạ
            64 bit. Chỉ nới rộng mặt nạ của phép shuffle thì biên dịch được nhưng âm thầm sai trên wavefront rộng 64.{' '}
            <code>activeThreads</code> vẫn là <code>unsigned int</code>, nên các lane 32 đến 63 biến mất khỏi nó;{' '}
            <code>__ffs</code> và <code>__popc</code> là 32 bit; và <code>{'1 << lane'}</code> không xác định với các lane từ 32
            trở lên. Một warp có mọi lane qua kiểm tra nằm trên 31 nhận leader -1. Cách sửa là một mặt nạ có kiểu 64 bit với{' '}
            <code>__popcll</code>, <code>__ffsll</code> và <code>{'1ull << lane'}</code>. Trên AMD, khác với NVIDIA, header HIP
            hứa rằng <code>__activemask()</code> là "the exact active mask".
          </p>
        }
      >
        <p>
          Fig. 12.3 as printed does not compile under HIP: every HIP <code>_sync</code> intrinsic insists on a 64-bit
          mask. Widening only the shuffle's mask makes it compile and leaves it silently wrong on a 64-wide wavefront.{' '}
          <code>activeThreads</code> is still an <code>unsigned int</code>, so lanes 32 to 63 vanish from it;{' '}
          <code>__ffs</code> and <code>__popc</code> are 32-bit; and <code>{'1 << lane'}</code> is undefined for lanes
          32 and up. A warp whose passing lanes all lie above 31 gets the leader -1. The fix is a typed 64-bit mask with{' '}
          <code>__popcll</code>, <code>__ffsll</code> and <code>{'1ull << lane'}</code>. On AMD, unlike NVIDIA, the HIP
          header promises that <code>__activemask()</code> is "the exact active mask".
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              HIP không có trong sách. Lấy từ header và tài liệu HIP ở ROCm 10.2 (ch12-facts B1; infra-facts B3).
            </p>
          }
        >
          <p>
            HIP is not in the book. From the HIP headers and docs at ROCm 10.2 (ch12-facts B1; infra-facts B3).
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS123['ch12/12.3/q-hip64']} />
      <Translated
        vi={
          <p>
            llama.cpp bịt cái bẫy này bằng hệ thống kiểu. <code>ggml_cuda_popc</code> của nó gọi <code>__popc</code> hoặc{' '}
            <code>__popcll</code> tùy theo kích thước của kiểu mặt nạ, rồi xóa template cho mọi kiểu đối số khác, "Makes implicit
            type casts result in an error". <code>maskPopc</code> của lab 12.1 chép ý tưởng đó. Đừng dùng{' '}
            <code>__lanemask_lt()</code> của HIP thay thế: nó có trong header nhưng không có trong tài liệu, và nó chỉ trả về các
            lane hoạt động bên dưới lane gọi, khác với thanh ghi cùng tên của PTX.
          </p>
        }
      >
        <p>
          llama.cpp closes this trap with the type system. Its <code>ggml_cuda_popc</code> calls <code>__popc</code> or{' '}
          <code>__popcll</code> according to the size of its mask type, and then deletes the template for every other
          argument type, "Makes implicit type casts result in an error". Lab 12.1's <code>maskPopc</code> copies the
          idea. Do not reach for HIP's <code>__lanemask_lt()</code> instead: it is in the headers but not the
          documentation, and it returns only the active lanes below the caller, unlike PTX's register of the same name.
        </p>
      </Translated>
      <Beyond title={tr('Not in the book: llama.cpp', 'Không có trong sách: llama.cpp')}>
        <Translated vi={<p>ggml/src/ggml-cuda/common.cuh, đọc ngày 2026-11-20 (ch12-facts D1); header của HIP: B2.</p>}>
          <p>ggml/src/ggml-cuda/common.cuh, read on 2026-11-20 (ch12-facts D1); HIP's header: B2.</p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Trên CUDA, mặt nạ các lane bên dưới cũng có các dạng làm sẵn có tài liệu: thanh ghi đặc biệt của PTX{' '}
            <code>%lanemask_lt</code>, và <code>cuda::device::lane_mask::all_less()</code> từ <code>&lt;cuda/warp&gt;</code>{' '}
            trong CCCL 3.1 trở lên (chưa kiểm tra toolkit nào đầu tiên kèm CCCL đó; libhipcxx của AMD không cung cấp nó). Nếu
            bạn tự viết, hãy viết <code>{'(1u << lane) - 1'}</code>: với lane 31, <code>{'1 << 31'}</code> dịch vào bit dấu của
            một <code>int</code> có dấu.
          </p>
        }
      >
        <p>
          On CUDA the lanes-below mask also has documented ready-made forms: the PTX special register{' '}
          <code>%lanemask_lt</code>, and <code>cuda::device::lane_mask::all_less()</code> from{' '}
          <code>&lt;cuda/warp&gt;</code> in CCCL 3.1 and later (which toolkit first ships that CCCL was not checked;
          AMD's libhipcxx does not provide it). If you write it by hand, write <code>{'(1u << lane) - 1'}</code>: with
          lane 31, <code>{'1 << 31'}</code> shifts into the sign bit of a signed <code>int</code>.
        </p>
      </Translated>
      <Aside>
        <Translated vi={<p>Nguồn: ch12-facts A3 (PTX ISA 9.5 mục 10.20; tài liệu CCCL; cppreference về phép dịch bit).</p>}>
          <p>Sources: ch12-facts A3 (PTX ISA 9.5 section 10.20; CCCL docs; cppreference on shifts).</p>
        </Translated>
      </Aside>

      <h2><Translated vi="Với cooperative groups">With cooperative groups</Translated></h2>
      <Translated
        vi={
          <p>
            Hình 12.4 làm điều tương tự với một <b>coalesced group</b>, một handle cooperative groups cho các lane hoạt động của
            một warp. <code>num_threads()</code> là count, <code>thread_rank()</code> là chỉ số của một lane trong số các lane
            hoạt động, nên rank 0 là leader và rank chính là offset, và <code>shfl(j, 0)</code> đọc từ rank 0, không phải từ
            lane 0.
          </p>
        }
      >
        <p>
          Fig. 12.4 does the same with a <b>coalesced group</b>, a cooperative-groups handle for the active lanes of a
          warp. <code>num_threads()</code> is the count, <code>thread_rank()</code> is a lane's index among the active
          lanes, so rank 0 is the leader and the rank is the offset, and <code>shfl(j, 0)</code> reads from rank 0, not
          from lane 0.
        </p>
      </Translated>
      <Source section="12.3" pages="293–294" figs={tr('Fig. 12.4', 'Hình 12.4')} />
      <Fig4 code={fig4} />
      <Translated
        vi={
          <p>
            Chỗ đoạn mã khác trang sách: trang sách không có header và namespace, nên tệp include{' '}
            <code>&lt;cooperative_groups.h&gt;</code> (trên HIP, <code>&lt;hip/hip_cooperative_groups.h&gt;</code>) và dòng 07
            ghi <code>cg::</code>; dòng 12 gọi <code>num_threads()</code>, mà <code>size()</code> của sách giờ là một bí danh
            kiểu cũ của nó; và <code>j</code> bắt đầu từ 0, như trong Hình 12.3. Metal không có cooperative groups: các hàm SIMD
            group ở trên là thứ thay thế.
          </p>
        }
      >
        <p>
          Where the snippet differs from the page: the page shows no header and no namespace, so the file includes{' '}
          <code>&lt;cooperative_groups.h&gt;</code> (on HIP, <code>&lt;hip/hip_cooperative_groups.h&gt;</code>) and line
          07 says <code>cg::</code>; line 12 calls <code>num_threads()</code>, of which the book's <code>size()</code> is
          now a legacy alias; and <code>j</code> starts at 0, as in Fig. 12.3. Metal has no cooperative groups: its
          SIMD-group functions above are the replacement.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch12-facts.md A5, A6, B3; infra-facts B4)', '2026-11-20 (notes/today/ch12-facts.md, mục A5, A6, B3; infra-facts B4)')}>
        <Translated
          vi={
            <p>
              <code>coalesced_threads()</code> là "opportunistic", giống <code>__activemask()</code>. Các lời gọi tính hạng và
              nối thêm trong một lần ở cuối mục này có trên CUDA; HIP có <code>exclusive_scan</code> nhưng không có{' '}
              <code>exclusive_scan_update</code>. Các trình giả lập không có cả hai.
            </p>
          }
        >
          <p>
            <code>coalesced_threads()</code> is "opportunistic", like <code>__activemask()</code>. The one-call rank and
            append at the end of this section exist on CUDA; HIP has <code>exclusive_scan</code> but not{' '}
            <code>exclusive_scan_update</code>. The emulators have neither.
          </p>
        </Translated>
      </Today>
      <WhichLine fig3={fig3} fig4={fig4} />

      <h2><Translated vi="Trình biên dịch có làm việc này giúp bạn không?">Does the compiler do this for you?</Translated></h2>
      <Translated
        vi={
          <p>
            Sách nói trên thực tế là có: trình biên dịch đã tự hợp nhất các phép atomic như vậy, và không làm khi nó biết chỉ
            có một lane hoạt động, nên mục này có mặt để dạy bỏ phiếu warp, các intrinsic bit và coalesced group. Với NVIDIA,
            khẳng định đó dựa trên một bài blog cập nhật lần cuối năm 2017, không nêu điều kiện nào; chi tiết "chỉ một lane"
            không có trong nguồn nào của NVIDIA tìm được. Trình biên dịch của AMD thì có ghi nhận, trong chú thích của một pass
            LLVM upstream viết lại một phép atomic trên địa chỉ đồng nhất thành đúng bốn bước của sách: một ballot các lane hoạt
            động, đếm các lane hoạt động bên dưới mỗi lane, lane thấp nhất thực hiện phép atomic, và phát kết quả của nó. Nó chạy
            ở <code>-O1</code> trở lên. Apple không ghi nhận gì.
          </p>
        }
      >
        <p>
          The book says that in practice it does: the compiler already coalesces such atomics, and refrains when it can
          tell that only one lane is active, so the section is there to teach warp voting, the bit intrinsics and
          coalesced groups. For NVIDIA the claim rests on a blog post last updated in 2017, which gives no conditions;
          the "only one lane" refinement is in no NVIDIA source found. AMD's compiler does document it, in the comments
          of an upstream LLVM pass that rewrites an atomic on a uniform address into exactly the book's four steps: a
          ballot of the active lanes, a count of the active lanes below each one, the lowest lane doing the atomic, and a
          broadcast of its result. It runs at <code>-O1</code> and above. Apple documents nothing.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch12-facts.md A7, B4)', '2026-11-20 (notes/today/ch12-facts.md, mục A7, B4)')}>
        <Translated
          vi={
            <p>
              Pass của AMD là <code>AMDGPUAtomicOptimizer</code> của LLVM upstream; việc <code>amdclang++</code> của ROCm 10.2
              có cùng mặc định chưa được xác nhận. Đọc mã được sinh ra là cách duy nhất để thấy trình biên dịch của bạn đã làm
              gì; lab 12.1 thì đo thay vào đó.
            </p>
          }
        >
          <p>
            The AMD pass is upstream LLVM's <code>AMDGPUAtomicOptimizer</code>; that ROCm 10.2's{' '}
            <code>amdclang++</code> carries the same default was not confirmed. Reading the generated code is the only
            way to see what your compiler did; lab 12.1 measures instead.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Các phép bỏ phiếu khác">Other votes</Translated></h2>
      <Translated
        vi={
          <p>
            <code>__activemask()</code> là một phép bỏ phiếu warp. Sách nêu thêm ba phép nữa mà không kèm tham số:{' '}
            <code>int __all_sync(unsigned mask, int predicate)</code>, true nếu mọi lane trong mặt nạ có predicate true;{' '}
            <code>int __any_sync(unsigned mask, int predicate)</code>, true nếu có lane nào đó như vậy; và{' '}
            <code>unsigned __ballot_sync(unsigned mask, int predicate)</code>, mặt nạ các lane có predicate true. Trên HIP các
            mặt nạ và ballot là <code>unsigned long long</code>. Trên Metal chúng là <code>simd_all</code>,{' '}
            <code>simd_any</code> và <code>simd_ballot</code>. Ballot của Metal là 64 bit bất kể độ rộng SIMD group, và các bit
            trên cùng không xác định khi ít hơn 64 lane. Một hạng, tức <code>popcount</code> của phần phiếu bầu dưới một lane,
            không bao giờ đọc chúng; một phép đếm trên toàn bộ phiếu bầu thì có, nên hãy lấy count từ <code>simd_sum</code> hoặc
            che bỏ các lane không tồn tại. <code>ctz</code> của phiếu bầu là lane bỏ phiếu thấp nhất, và là 64 khi không có lane
            nào.
          </p>
        }
      >
        <p>
          <code>__activemask()</code> is one warp vote. The book names three more without their parameters:{' '}
          <code>int __all_sync(unsigned mask, int predicate)</code>, true if every lane in the mask has a true predicate;{' '}
          <code>int __any_sync(unsigned mask, int predicate)</code>, true if some lane does; and{' '}
          <code>unsigned __ballot_sync(unsigned mask, int predicate)</code>, the mask of lanes whose predicate is true.
          On HIP the masks and the ballot are <code>unsigned long long</code>. On Metal they are{' '}
          <code>simd_all</code>, <code>simd_any</code> and <code>simd_ballot</code>. Metal's ballot is 64 bits whatever
          the SIMD-group width, and its top bits are undefined below 64 lanes. A rank, the <code>popcount</code> of the
          vote below a lane, never reads them; a count over the whole vote does, so take the count from{' '}
          <code>simd_sum</code> or mask off the lanes that do not exist. <code>ctz</code> of the vote is the lowest
          voting lane, and 64 when there is none.
        </p>
      </Translated>
      <Source section="12.3" pages="295" />
      <Translated
        vi={
          <p>
            Một lợi ích nữa mà sách chỉ ra: các khóa được giữ của một warp giờ rơi vào các ô liền kề, nên các lần store của
            chúng vào bộ nhớ toàn cục dễ được hợp nhất hơn. Chạy trên trình giả lập của khóa học, cả hai đoạn mã trên đều đặt các
            khóa được giữ của mỗi warp cạnh nhau, theo thứ tự lane. Mục <a href={href('ch12', '12.4')}>12.4</a> mở rộng điều này
            ra cả khối.
          </p>
        }
      >
        <p>
          One more benefit the book points out: the kept keys of a warp now land in adjacent slots, so their stores to
          global memory are more likely to be coalesced. Run on the course's emulator, both listings above put every
          warp's kept keys side by side, in lane order. Section <a href={href('ch12', '12.4')}>12.4</a> extends this to
          whole blocks.
        </p>
      </Translated>
      <Source section="12.3" pages="293" />

      <h2><Translated vi="Các backend thực sự khác nhau ở đâu">Where the backends really differ</Translated></h2>
      <Translated
        vi={
          <p>
            Một tệp nguồn phục vụ cả CUDA và HIP cho luồng, khối, rào chắn và atomic, nhưng không cho mã warp của mục này: kiểu
            mặt nạ, các hàm đếm bit và header cooperative groups đều khác. Bảng này gom cả mục vào một chỗ.
          </p>
        }
      >
        <p>
          One source file serves CUDA and HIP for threads, blocks, barriers and atomics, but not for this section's warp
          code: the mask types, the bit counts and the cooperative-groups headers differ. This table is the section in
          one place.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ dùng CUDA. Mọi ô HIP và Metal lấy từ notes/today/ch12-facts.md (bảng tổng hợp); các tên Metal đã được biên
              dịch, chưa chạy, trong lần kiểm tra sự kiện. Không gì ở đây đã chạy trên phần cứng CUDA hay AMD.
            </p>
          }
        >
          <p>
            The book uses CUDA only. Every HIP and Metal cell is from notes/today/ch12-facts.md (summary table); the
            Metal names were compiled, not run, by the fact check. Nothing here was run on CUDA or AMD hardware.
          </p>
        </Translated>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th><Translated vi="Bước">Step</Translated></th><th>CUDA</th><th>HIP (ROCm 10.2)</th><th>Metal (MSL 4.1)</th></tr>{/* i18n-ignore: product and version names */}
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="Mặt nạ lane hoạt động">Active-lane mask</Translated></td>
            <td><Translated vi={<><code>unsigned __activemask()</code>; một ảnh chụp, không phải các lane của nhánh</>}><code>unsigned __activemask()</code>; a snapshot, not the branch's lanes</Translated></td>
            <td><Translated vi={<><code>unsigned long long __activemask()</code>; theo header: "the exact active mask"</>}><code>unsigned long long __activemask()</code>; "the exact active mask"</Translated></td>
            <td><Translated vi={<><code>simd_active_threads_mask()</code>, một phiếu bầu 64 bit; bên trong một nhánh: không được nêu</>}><code>simd_active_threads_mask()</code>, a 64-bit vote; inside a branch: not stated</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Bỏ phiếu trên một predicate">Vote on a predicate</Translated></td>
            <td><Translated vi={<><code>__ballot_sync(mask, p)</code> trả về <code>unsigned</code></>}><code>__ballot_sync(mask, p)</code> returns <code>unsigned</code></Translated></td>
            <td><Translated vi="cùng tên; mặt nạ 64 bit vào và ra">same name; 64-bit mask in and out</Translated></td>
            <td><Translated vi={<><code>simd_ballot(p)</code>; các bit trên cùng không xác định khi ít hơn 64 lane</>}><code>simd_ballot(p)</code>; top bits undefined below 64 lanes</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Leader">Leader</Translated></td>
            <td><code>__ffs(mask) - 1</code></td>
            <td><Translated vi={<><code>__ffsll(mask) - 1</code>; <code>__ffs</code> cắt cụt một mặt nạ 64 bit</>}><code>__ffsll(mask) - 1</code>; <code>__ffs</code> truncates a 64-bit mask</Translated></td>
            <td><Translated vi={<><code>simd_is_first()</code>, hoặc <code>ctz(vote)</code> (64 khi không có)</>}><code>simd_is_first()</code>, or <code>ctz(vote)</code> (64 for none)</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Đếm">Count</Translated></td>
            <td><code>__popc(mask)</code></td>
            <td><Translated vi={<><code>__popcll(mask)</code>; <code>__popc</code> cắt cụt</>}><code>__popcll(mask)</code>; <code>__popc</code> truncates</Translated></td>
            <td><Translated vi={<><code>simd_sum(keep ? 1 : 0)</code>, hoặc <code>popcount</code> của phiếu bầu đã che</>}><code>simd_sum(keep ? 1 : 0)</code>, or <code>popcount</code> of the masked vote</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Mặt nạ các lane bên dưới">Lanes-below mask</Translated></td>
            <td><Translated vi={<><code>{'(1u << lane) - 1'}</code>; PTX <code>%lanemask_lt</code>; <code>lane_mask::all_less()</code> (CCCL 3.1 trở lên)</>}><code>{'(1u << lane) - 1'}</code>; PTX <code>%lanemask_lt</code>; <code>lane_mask::all_less()</code> (CCCL 3.1+)</Translated></td>
            <td><Translated vi={<><code>{'(1ull << lane) - 1'}</code>; <code>__lanemask_lt()</code> chỉ có trong header, chỉ đếm các lane hoạt động</>}><code>{'(1ull << lane) - 1'}</code>; header-only <code>__lanemask_lt()</code> counts active lanes only</Translated></td>
            <td><Translated vi={<><code>{'(1ul << lane) - 1'}</code> trên phiếu bầu <code>ulong</code></>}><code>{'(1ul << lane) - 1'}</code> on the <code>ulong</code> vote</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Hạng trong một lời gọi">Rank in one call</Translated></td>
            <td><code>cg::exclusive_scan(coalesced_threads(), 1)</code></td>
            <td><Translated vi={<><code>exclusive_scan</code> trong <code>&lt;hip/cooperative_groups/hip_scan.h&gt;</code></>}><code>exclusive_scan</code> in <code>&lt;hip/cooperative_groups/hip_scan.h&gt;</code></Translated></td>
            <td><code>simd_prefix_exclusive_sum(keep ? 1 : 0)</code></td>
          </tr>
          <tr>
            <td><Translated vi="Phát điểm bắt đầu">Broadcast the start</Translated></td>
            <td><Translated vi={<><code>__shfl_sync(mask, j, leader)</code> hoặc <code>g.shfl(j, 0)</code></>}><code>__shfl_sync(mask, j, leader)</code> or <code>g.shfl(j, 0)</code></Translated></td>
            <td><Translated vi="cùng tên, mặt nạ 64 bit">same names, 64-bit mask</Translated></td>
            <td><code>simd_broadcast_first(j)</code></td>
          </tr>
          <tr>
            <td><Translated vi="Cả thao tác nối thêm trong một lời gọi">Whole append in one call</Translated></td>
            <td><code>cg::exclusive_scan_update(group, atomic_ref, 1)</code></td>
            <td><Translated vi="không có">none</Translated></td>
            <td><Translated vi="không có">none</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Coalesced group">Coalesced group</Translated></td>
            <td><Translated vi={<><code>&lt;cooperative_groups.h&gt;</code>; <code>size()</code> là bí danh kiểu cũ của <code>num_threads()</code></>}><code>&lt;cooperative_groups.h&gt;</code>; <code>size()</code> is a legacy alias of <code>num_threads()</code></Translated></td>
            <td><Translated vi={<><code>&lt;hip/hip_cooperative_groups.h&gt;</code>; mặt nạ thành viên 64 bit</>}><code>&lt;hip/hip_cooperative_groups.h&gt;</code>; 64-bit member mask</Translated></td>
            <td><Translated vi="không có">none</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Trình biên dịch tự gộp phép atomic">Compiler aggregates the atomic itself</Translated></td>
            <td><Translated vi={<>"in many cases" (blog năm 2017); điều kiện không có tài liệu</>}>"in many cases" (2017 blog); conditions undocumented</Translated></td>
            <td><Translated vi={<>pass LLVM upstream, ở <code>-O1</code> trở lên</>}>upstream LLVM pass, at <code>-O1</code> and above</Translated></td>
            <td><Translated vi="không có tài liệu">not documented</Translated></td>
          </tr>
        </tbody>
      </table>
    </>
  );
}
