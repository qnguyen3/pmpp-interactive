import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { OP_TEXT, PRESETS, atomicOrder, enumerate, nextOp, run, type PresetId, type Run } from '../../sim/histrace';

const LAB = 'ch09/lab1_racehunt';
const KERNEL_TIERS: Record<string, Tier> = {
  image: 'global', bins: 'global', bins_ref: 'global',
  i: 'reg', b: 'reg', tid: 'reg', bid: 'reg', bdim: 'reg',
};
const T = (t: number) => tr(`thread ${t + 1}`, `luồng ${t + 1}`);
/** Code in the backends table: may wrap at its spaces, never inside an identifier. */
const C = ({ children }: { children: ReactNode }) => (
  <code style={{ whiteSpace: 'normal', overflowWrap: 'normal', wordBreak: 'normal' }}>{children}</code>
);
const names = (ts: number[]) =>
  ts.length === 1 ? T(ts[0]) : ts.slice(0, -1).map(T).join(', ') + tr(' and ', ' và ') + T(ts[ts.length - 1]);

/** The lines between `// region: <name>` and `// endregion`, or null if the file or the markers are missing. */
function region(src: string | null, name: string): { text: string; first: number; last: number } | null {
  if (!src || src.startsWith('// could not load')) return null;
  const lines = src.split('\n');
  const a = lines.findIndex((l) => l.trim() === `// region: ${name}`);
  const b = a < 0 ? -1 : lines.findIndex((l, k) => k > a && l.trim() === '// endregion');
  if (a < 0 || b < 0) return null;
  return { text: lines.slice(a + 1, b).join('\n'), first: a + 2, last: b };
}

// ---------------------------------------------------------------- the stepper
/** Whose lost increment was erased by which store, for the closing sentence. */
function lostStory(r: Run): string {
  return r.lost
    .map((j) => {
      const k = [...r.steps].reverse().find((s) => s.erased.includes(j))!;
      return tr(
        `${T(j)}'s increment is missing: ${T(k.thread)}'s store at time ${k.time} wrote ${k.shown} over it`,
        `phép tăng của ${T(j)} bị mất: lần ghi của ${T(k.thread)} ở thời điểm ${k.time} đã ghi ${k.shown} đè lên nó`,
      );
    })
    .join('; ');
}

function RaceStepper() {
  const [n, setN] = useState(2);
  const [atomic, setAtomic] = useState(false);
  const [presetId, setPresetId] = useState<PresetId | null>(null);
  const [schedule, setSchedule] = useState<number[]>([]);
  const [at, setAt] = useState(0);
  const [playing, setPlaying] = useState(false);

  const shown = schedule.slice(0, at);
  const opts = { n, atomic };
  const r = useMemo(() => run(shown, opts), [shown.join(','), n, atomic]); // eslint-disable-line react-hooks/exhaustive-deps
  const cur = r.steps[r.steps.length - 1];

  useEffect(() => {
    if (!playing) return;
    if (at >= schedule.length) return setPlaying(false);
    const id = window.setTimeout(() => setAt((x) => x + 1), 450);
    return () => window.clearTimeout(id);
  }, [playing, at, schedule.length]);

  const load = (id: PresetId | null, nn: number, at0: boolean) => {
    const p = PRESETS.find((x) => x.id === id);
    const base = p ? p.schedule(nn) : [];
    setSchedule(atomic && p ? atomicOrder(base) : base);
    setAt(0);
    setPlaying(!!p && !at0);
  };
  const choosePreset = (id: PresetId) => {
    const p = PRESETS.find((x) => x.id === id)!;
    const nn = p.n ?? n;
    setN(nn);
    setPresetId(id);
    const base = p.schedule(nn);
    setSchedule(atomic ? atomicOrder(base) : base);
    setAt(0);
    setPlaying(true);
  };
  const setThreads = (nn: number) => {
    setN(nn);
    setPresetId(null);
    load(null, nn, true);
  };
  const toggleAtomic = (on: boolean) => {
    setAtomic(on);
    const p = PRESETS.find((x) => x.id === presetId);
    const base = p && (p.n == null || p.n === n) ? p.schedule(n) : [];
    setSchedule(on && p ? atomicOrder(base) : base);
    if (!p) setPresetId(null);
    setAt(0);
    setPlaying(!!p);
  };
  const advance = (t: number) => {
    if (nextOp(shown, t, opts) == null) return;
    setPlaying(false);
    setPresetId(null);
    const s = [...shown, t];
    setSchedule(s);
    setAt(s.length);
  };

  const threads = Array.from({ length: n }, (_, t) => t);
  return (
    <Bench
      title={tr('Race on one bin', 'Tranh chấp trên một bin')}
      task={
        <Translated vi="Làm cho bins[x] kết thúc nhỏ hơn số luồng. Rồi bật atomic và thử làm lại.">
          Make bins[x] end below the number of threads. Then turn on atomic and try to do it again.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Mỗi luồng chạy <code>Old ← bins[x]</code>, <code>New ← Old + 1</code>, <code>bins[x] ← New</code> trên một bin
              bắt đầu từ 0. Bấm nút của một luồng để chạy bước tiếp theo của nó; bảng được vẽ giống của sách, giá trị trong
              ngoặc là giá trị mà bước đó đã nạp, tính hoặc ghi. Bốn preset hai luồng tái hiện các kết quả của Hình 9.4 và
              9.5.
            </>
          }
        >
          <>
            Every thread runs <code>Old ← bins[x]</code>, <code>New ← Old + 1</code>, <code>bins[x] ← New</code> on a bin
            that starts at 0. Click a thread's button to run its next step; the table is drawn like the book's, the value in
            parentheses being what the step loaded, computed or stored. The four two-thread presets reproduce the outcomes
            of Figs. 9.4 and 9.5.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Số luồng">Threads</Translated>
          <span className="seg" role="group" aria-label={tr('Threads', 'Số luồng')}>
            {[2, 3, 4].map((k) => (
              <button key={k} aria-pressed={n === k} onClick={() => setThreads(k)}>{k}</button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi="Preset">Preset</Translated>
          <span className="seg" role="group" aria-label={tr('Preset', 'Preset')} style={{ flexWrap: 'wrap' }}>
            {PRESETS.map((p) => (
              <button key={p.id} aria-pressed={presetId === p.id} onClick={() => choosePreset(p.id)} title={p.book ?? tr('all threads in step', 'mọi luồng chạy đồng bộ')}>
                {p.label}
              </button>
            ))}
          </span>
        </label>
        <label className="check">
          <input type="checkbox" checked={atomic} onChange={(e) => toggleAtomic(e.target.checked)} /> <Translated vi="nguyên tử (atomic)">atomic</Translated>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setAt(0); }} disabled={at === 0}><Translated vi="Làm lại">Restart</Translated></button>
        <button onClick={() => { setPlaying(false); setAt(at - 1); }} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
        <button onClick={() => { setPlaying(false); setAt(at + 1); }} disabled={at >= schedule.length}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= schedule.length}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Chạy')}</button>
        <span className="muted" aria-live="polite">
          <Translated vi="bước">step</Translated> {at} <Translated vi="trên">of</Translated> {schedule.length}
          {presetId ? ` (${PRESETS.find((p) => p.id === presetId)!.label}${atomic ? tr(', atomic', ', nguyên tử') : ''})` : ''}
        </span>
      </div>

      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'stretch' }}>
        {threads.map((t) => {
          const op = nextOp(shown, t, opts);
          const reg = cur ? cur.regs[t] : { old: null, new: null };
          return (
            <button key={t} data-thread={t} onClick={() => advance(t)} disabled={op == null} style={{ textAlign: 'left', minWidth: '10.5rem' }}>
              <span style={{ fontWeight: 650 }}>{T(t)}</span>
              <br />
              <span style={{ fontWeight: 400, fontSize: '0.78rem' }}>
                {op ? tr(`next: ${OP_TEXT[op]}`, `tiếp: ${OP_TEXT[op]}`) : tr('finished', 'đã xong')}
                <br />
                Old = <span className="t-reg">{reg.old ?? '–'}</span>, New = <span className="t-reg">{reg.new ?? '–'}</span>
              </span>
            </button>
          );
        })}
        <div className="panel" style={{ minWidth: '7rem' }}>
          <div className="mx-label t-global">bins[x]</div>
          <div style={{ fontSize: '1.6rem', fontWeight: 650 }} data-memory>{cur ? cur.memory : 0}</div>
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="data" data-part="race-table" style={{ background: 'var(--page)', margin: 0 }}>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}><Translated vi="Thời điểm">Time</Translated></th>
              {threads.map((t) => <th key={t}>{T(t)}</th>)}
              <th>bins[x]</th>
            </tr>
          </thead>
          <tbody>
            {r.steps.map((s) => (
              <tr key={s.time} style={s === cur ? { background: 'var(--rule-soft)' } : undefined}>
                <td style={{ paddingLeft: '0.6rem' }}>{s.time}</td>
                {threads.map((t) => (
                  <td key={t} className="nowrap">
                    {t === s.thread ? <>({s.shown}) {OP_TEXT[s.op]}</> : ''}
                  </td>
                ))}
                <td>
                  <span className="t-global">{s.memory}</span>
                  {s.erased.length ? (
                    <Translated vi={<span className="t-fail"> ghi đè lên cập nhật của {names(s.erased)}</span>}>
                      <span className="t-fail"> overwrites {names(s.erased)}'s update</span>
                    </Translated>
                  ) : null}
                </td>
              </tr>
            ))}
            {r.steps.length === 0 ? (
              <tr>
                <td colSpan={n + 2} className="muted" style={{ paddingLeft: '0.6rem' }}>
                  <Translated vi="Chưa có gì chạy. Bấm một luồng ở trên, hoặc chọn một preset.">
                    Nothing has run yet. Click a thread above, or pick a preset.
                  </Translated>
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
      {r.complete ? (
        <p style={{ margin: '0.9rem 0 0' }} data-final={r.final} data-correct={r.correct ? 'yes' : 'no'}>
          {r.correct ? (
            <Translated vi={<span className="chip ok">bins[x] = {r.final}: đếm đủ cả {n} phép tăng</span>}>
              <span className="chip ok">bins[x] = {r.final}: all {n} increments counted</span>
            </Translated>
          ) : (
            <Translated vi={<span className="chip fail">bins[x] = {r.final} sau {n} phép tăng</span>}>
              <span className="chip fail">bins[x] = {r.final} after {n} increments</span>
            </Translated>
          )}{' '}
          {r.correct
            ? atomic
              ? tr(
                  `The order was ${r.steps.map((s) => s.thread + 1).join(', ')}: atomics decided nothing about who went first, only that no two overlapped.`,
                  `Thứ tự là ${r.steps.map((s) => s.thread + 1).join(', ')}: phép toán nguyên tử không quyết định ai đi trước, chỉ đảm bảo không có hai phép nào chồng lên nhau.`,
                )
              : tr(
                  'No thread loaded while another was between its load and its store.',
                  'Không luồng nào nạp trong lúc một luồng khác đang ở giữa lần nạp và lần ghi của nó.',
                )
            : lostStory(r) + '.'}
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the census
function Census() {
  const rows = useMemo(() => [2, 3, 4].map((k) => ({ plain: enumerate(k), atom: enumerate(k, { atomic: true }) })), []);
  const pct = (a: number, b: number) => {
    const v = (100 * a) / b;
    return (v >= 1 ? v.toFixed(0) : v >= 0.1 ? v.toFixed(2) : v.toFixed(4)) + '%';
  };
  const fmtInt = (x: number) => x.toLocaleString('en-US');
  return (
    <div style={{ overflowX: 'auto' }}>
      <Translated
        vi={
          <p className="muted" style={{ margin: '0.4rem 0 0' }}>
            Thứ tự: các cách xen kẽ giữ nguyên thứ tự các bước của chính mỗi luồng. → <i>v</i>: bao nhiêu cách trong số đó
            kết thúc với bin bằng <i>v</i>. Nguyên tử: cùng phép đếm đó với mỗi phép tăng là không thể chia cắt.
          </p>
        }
      >
        <p className="muted" style={{ margin: '0.4rem 0 0' }}>
          Orders: interleavings that keep each thread's own steps in order. → <i>v</i>: how many of them end with the bin
          at <i>v</i>. Atomic: the same census with each increment indivisible.
        </p>
      </Translated>
      <table className="data" data-part="census" style={{ margin: '0.4rem 0 0.6rem' }}>
        <thead>
          <tr>
            <th><Translated vi="Số luồng">Threads</Translated></th>
            <th className="num"><Translated vi="Thứ tự">Orders</Translated></th>
            <th className="num nowrap">→ 1</th>
            <th className="num nowrap">→ 2</th>
            <th className="num nowrap">→ 3</th>
            <th className="num nowrap">→ 4</th>
            <th className="num"><Translated vi="Đúng">Correct</Translated></th>
            <th className="num"><Translated vi="Nguyên tử">Atomic</Translated></th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ plain: c, atom: a }) => (
            <tr key={c.n} data-row={c.n} data-total={c.interleavings} data-correct={c.correct}>
              <td>{c.n}</td>
              <td className="num">{fmtInt(c.interleavings)}</td>
              {[1, 2, 3, 4].map((v) => (
                <td key={v} className="num">{v <= c.n ? fmtInt(c.byFinal[v]) : ''}</td>
              ))}
              <td className="num">
                <b className="nowrap">{fmtInt(c.correct)} <Translated vi="trên">of</Translated> {fmtInt(c.interleavings)}</b>
                <br />
                <span className="muted">{pct(c.correct, c.interleavings)}</span>
              </td>
              <td className="num">{fmtInt(a.correct)} <Translated vi="trên">of</Translated> {fmtInt(a.interleavings)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------- the listing
function BasicListing() {
  const { reports } = useReports();
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  const k = region(cu, 'basic');
  const m = region(metal, 'basic');
  // The same words as the labs page's status table and reference-solution tabs.
  const cuStatus = solutionTabStatus(reports, LAB, 'kernel.cu');
  const mStatus = solutionTabStatus(reports, LAB, 'kernel.metal');
  if (!k)
    return (
      <Translated
        vi={
          <p className="muted" data-part="listing-missing">
            Kernel được nạp từ <code>labs/{LAB}/solution/kernel.cu</code> (các dòng giữa{' '}
            <code>// region: basic</code> và <code>// endregion</code>) khi lab 9.1 đã sẵn sàng.
          </p>
        }
      >
        <p className="muted" data-part="listing-missing">
          The kernel is loaded from <code>labs/{LAB}/solution/kernel.cu</code> (the lines between{' '}
          <code>// region: basic</code> and <code>// endregion</code>) once lab 9.1 is in place.
        </p>
      </Translated>
    );
  const lines = (text: string, needles: string[]) => text.split('\n').flatMap((l, i) => (needles.some((x) => l.includes(x)) ? [i + 1] : []));
  return (
    <Code
      tabs={[
        { label: tr('CUDA and HIP', 'CUDA và HIP'), code: k.text, tiers: KERNEL_TIERS, status: cuStatus, highlight: lines(k.text, ['atomic_ref', 'fetch_add']) },
        ...(m ? [{ label: 'Metal', code: m.text, tiers: KERNEL_TIERS, status: mStatus, highlight: lines(m.text, ['atomic_fetch_add_explicit', 'atomic_uint']) }] : []),
      ]}
      caption={
        <Translated
          vi={
            <>
              Kernel cơ bản từ lời giải tham chiếu của lab 9.1, các dòng {k.first} đến {k.last} của{' '}
              <code>labs/{LAB}/solution/kernel.cu</code>
              {m ? <> và vùng tương ứng của <code>kernel.metal</code></> : null}. Nó là Hình 9.6 với ba khác biệt:
              tên, <code>histogram_kernel</code> thành <code>histogramBasic</code>; <code>const</code> trên con trỏ
              ảnh; và các bin được host xóa trước khi khởi chạy, điều mà sách không hề nói ai làm. Hãy thử lab
              trước khi bạn nghiên cứu phần này.
            </>
          }
        >
          <>
            The basic kernel from lab 9.1's reference solution, lines {k.first} to {k.last} of{' '}
            <code>labs/{LAB}/solution/kernel.cu</code>
            {m ? <> and the matching region of <code>kernel.metal</code></> : null}. It is Fig. 9.6 with three differences:
            the name, <code>histogram_kernel</code> becoming <code>histogramBasic</code>; <code>const</code> on the image
            pointer; and bins that the host clears before the launch, which the book never says anyone does. Try the lab
            before you study this.
          </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- questions
export const questionsS92: Record<string, Question> = {
  'ch09/9.2/q-range': {
    kind: 'choice',
    id: 'ch09/9.2/q-range',
    prompt: (
      <Translated vi={<>Bốn luồng, mỗi luồng chạy một lệnh <code>++bins[x]</code> thường trên một bin bắt đầu từ 0. Những giá trị cuối nào có thể xảy ra?</>}>
        <>Four threads each run a plain <code>++bins[x]</code> on a bin that starts at 0. Which final values are possible?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Chỉ 4.">Only 4.</Translated>,
        why: (
          <Translated vi="Chỉ khi không phép tăng nào chồng lên phép khác. Không có gì trong một lệnh ++ thường ngăn sự chồng lấn.">
            Only if no increment overlaps another. Nothing in a plain ++ prevents overlap.
          </Translated>
        ),
      },
      {
        label: <Translated vi="1 hoặc 4: hoặc mọi thứ va chạm, hoặc không gì va chạm.">1 or 4: either everything collides or nothing does.</Translated>,
        why: (
          <Translated vi="Va chạm có thể chỉ một phần: hai luồng có thể chồng lấn trong khi hai luồng kia chạy trọn vẹn.">
            Collisions can be partial: two threads may overlap while the other two run cleanly.
          </Translated>
        ),
      },
      { label: <Translated vi="Bất kỳ giá trị nào trong 1, 2, 3 và 4.">Any of 1, 2, 3 and 4.</Translated>, correct: true },
      {
        label: <Translated vi="Bất kỳ giá trị nào từ 0 đến 4.">Any value from 0 to 4.</Translated>,
        why: (
          <Translated vi="Mỗi luồng ghi ít nhất 1, nên bin không thể kết thúc ở 0.">
            Every thread stores at least 1, so the bin cannot end at 0.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            1 khi cả bốn đều nạp trước khi bất kỳ luồng nào ghi, đó là điều một warp chạy đồng bộ (lockstep) làm; 4 chỉ khi bốn
            phép tăng hoàn toàn không chồng lấn. Chồng lấn một phần cho ra 2 và 3. Trình liệt kê ở trên đếm mỗi trường hợp
            xảy ra bao nhiêu lần nếu mọi cách xen kẽ đều có khả năng như nhau; trên phần cứng thật, xác suất phụ thuộc vào
            thời gian, nhưng tập các kết quả có thể xảy ra vẫn như vậy.
          </p>
        }
      >
        <p>
          1 when all four load before any of them stores, which is what a warp in lockstep does; 4 only when the four
          increments do not overlap at all. Partial overlaps give 2 and 3. The enumerator above counts how often each
          happens if every interleaving were equally likely; on real hardware the odds depend on timing, but the set of
          possible results is the same.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.2, pp. 204–206; the counts are from our enumeration',
    vi: { source: 'ch. 9 §9.2, pp. 204–206; các con số đếm là từ phép liệt kê của khóa học này' },
  },
  'ch09/9.2/q-order': {
    kind: 'choice',
    id: 'ch09/9.2/q-order',
    prompt: (
      <Translated
        vi={
          <>
            Luồng A và B mỗi luồng gọi <code>ref.fetch_add(1, cuda::memory_order_relaxed)</code> trên một bộ đếm đang giữ 5,
            và mỗi luồng dùng giá trị trả về làm chỉ số vào một mảng đầu ra. A nhận chỉ số nào?
          </>
        }
      >
        <>
          Threads A and B each call <code>ref.fetch_add(1, cuda::memory_order_relaxed)</code> on a counter that holds 5,
          and each uses the returned value as an index into an output array. Which index does A get?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="5, vì A là luồng có số thứ tự nhỏ hơn.">5, because A is the lower-numbered thread.</Translated>,
        why: (
          <Translated vi="Số thứ tự luồng không quyết định gì cả. Các phép toán nguyên tử có thể xảy ra theo thứ tự bất kỳ.">
            Thread numbers decide nothing. The atomics can happen in either order.
          </Translated>
        ),
      },
      { label: <Translated vi="5 hoặc 6, và B nhận giá trị còn lại.">5 or 6, and B gets the other one.</Translated>, correct: true },
      {
        label: <Translated vi="5 hoặc 6, có thể giống với B.">5 or 6, possibly the same as B.</Translated>,
        why: (
          <Translated vi="Đó sẽ là một lần cập nhật bị mất. Một thao tác đọc-sửa-ghi nguyên tử không thể chồng lên một thao tác khác trên cùng vị trí, nên hai lời gọi thấy hai giá trị cũ khác nhau.">
            That would be a lost update. An atomic read-modify-write cannot overlap another on the same location, so the two calls see different old values.
          </Translated>
        ),
      },
      {
        label: <Translated vi="6: fetch_add trả về giá trị sau khi cộng.">6: fetch_add returns the value after the add.</Translated>,
        why: (
          <Translated vi="Nó trả về giá trị cũ, giá trị trước khi cộng (p. 209).">
            It returns the old value, the one before the add (p. 209).
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một phép toán nguyên tử cấm chồng lấn; nó không chọn ai đi trước. Một lời gọi thấy 5 và để lại 6, lời gọi kia
            thấy 6 và để lại 7. Mỗi ô của mảng đầu ra được dùng đúng một lần, theo một thứ tự không đoán trước được. Chương 12
            xây dựng nén gọn (compaction) dựa đúng trên tính chất này.
          </p>
        }
      >
        <p>
          An atomic operation forbids overlap; it does not choose who goes first. One call sees 5 and leaves 6, the other
          sees 6 and leaves 7. Each slot of the output array is used exactly once, in an unpredictable order. Chapter 12
          builds stream compaction on exactly this property.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.2, pp. 206, 209',
  },
  'ch09/9.2/q-relaxed': {
    kind: 'choice',
    id: 'ch09/9.2/q-relaxed',
    prompt: (
      <Translated vi={<>Vì sao <code>cuda::memory_order_relaxed</code> là đủ trong kernel histogram cơ bản?</>}>
        <>Why is <code>cuda::memory_order_relaxed</code> enough in the basic histogram kernel?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Relaxed là mặc định của fetch_add, nên dù sao mọi phép toán nguyên tử trong kernel cũng dùng nó.">
            Relaxed is the default for fetch_add, so it is what every atomic in the kernel uses anyway.
          </Translated>
        ),
        why: (
          <Translated vi="Mặc định của fetch_add là seq_cst (notes/today/ch09-facts.md A1). Sách truyền relaxed một cách có chủ đích.">
            The default for fetch_add is seq_cst (notes/today/ch09-facts.md A1). The book passes relaxed on purpose.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Relaxed đánh đổi một chút độ chính xác lấy tốc độ, và một histogram chịu được vài lần đếm bị mất.">
            Relaxed trades a little accuracy for speed, and a histogram can tolerate a few lost counts.
          </Translated>
        ),
        why: (
          <Translated vi="Relaxed không làm mất cập nhật nào: mọi fetch_add vẫn là nguyên tử. Thứ tự chỉ nói các truy cập bộ nhớ khác của luồng có thể dịch chuyển quanh phép toán nguyên tử như thế nào.">
            Relaxed loses no update: every fetch_add is still atomic. The order only says how the thread's other memory accesses may move around the atomic.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Truy cập bộ nhớ duy nhất còn lại của nó, lần nạp điểm ảnh, dù sao cũng phải xảy ra trước: địa chỉ của bin phụ thuộc vào nó.">
            Its one other memory access, the pixel load, must come first anyway: the bin's address depends on it.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Thứ tự bộ nhớ nói phép toán nguyên tử có thể được sắp xếp lại thế nào so với các truy cập bộ nhớ khác của cùng
            luồng. Ở đây chỉ có một truy cập như vậy, và sự phụ thuộc qua <code>b</code> đã buộc nó xảy ra trước. Không luồng
            nào khác đọc các bin trong lúc kernel chạy, nên cũng không cần thứ tự nào giữa các luồng.
          </p>
        }
      >
        <p>
          The memory order says how the atomic may be reordered against the same thread's other memory accesses. Here
          there is one, and the dependency through <code>b</code> already forces it to happen first. No other thread
          reads the bins during the kernel, so no ordering between threads is needed either.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.2, pp. 208–209',
  },
  'ch09/9.2/q-scope': {
    kind: 'choice',
    id: 'ch09/9.2/q-scope',
    prompt: (
      <Translated
        vi={
          <>
            Một đồng nghiệp viết <code>cuda::atomic_ref&lt;unsigned int, cuda::thread_scope_block&gt;</code> trong kernel cơ
            bản, lập luận rằng phạm vi hẹp hơn thì nhanh hơn. Sai ở đâu?
          </>
        }
      >
        <>
          A colleague writes <code>cuda::atomic_ref&lt;unsigned int, cuda::thread_scope_block&gt;</code> in the basic
          kernel, reasoning that a narrower scope is faster. What is wrong?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không sai gì: phạm vi khối nhanh hơn, và số đếm vẫn ra như cũ vì mỗi cập nhật vẫn là nguyên tử.">
            Nothing: block scope is faster, and the counts come out the same because each update is still atomic.
          </Translated>
        ),
        why: (
          <Translated vi="Có thể nó nhanh hơn thật. Nhưng nó chỉ nguyên tử đối với các luồng trong khối của chính nó, và không gì đảm bảo số đếm.">
            It may well be faster. But it is atomic only against the threads of its own block, and nothing guarantees the counts.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó không biên dịch được: bộ nhớ toàn cục cần phạm vi device, còn phạm vi khối chỉ dành cho bộ nhớ chia sẻ.">
            It does not compile: global memory needs device scope, and block scope is only for shared memory.
          </Translated>
        ),
        why: (
          <Translated vi="Nó biên dịch được. Trong CUDA, phạm vi độc lập với không gian bộ nhớ; mục 9.4 dùng phạm vi khối trên bộ nhớ toàn cục một cách có chủ đích.">
            It compiles. In CUDA the scope is independent of the memory space; section 9.4 uses block scope on global memory on purpose.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các khối khác cập nhật cùng các bin, nên một phép cộng chỉ nguyên tử trong một khối là một data race với các phép cộng của chúng.">
            Other blocks update the same bins, so an add atomic only within one block is a data race with theirs.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Phạm vi chỉ ra những luồng đồng bộ hóa qua phép toán nguyên tử. Các luồng của kernel cơ bản đến từ mọi khối,
            nên phạm vi device là lựa chọn đúng hẹp nhất (p. 208). Mô hình bộ nhớ của CUDA coi một data race như vậy là hành
            vi không xác định. Với phạm vi khối, kết quả có thể trông đúng trên một
            GPU và sai trên GPU khác. Cả hai trình giả lập của khóa học đều không cho phạm vi tác dụng gì (labs/README.md), nên
            không bài kiểm tra nào ở đây bắt được lỗi này: bạn phải đọc mã để thấy nó.
          </p>
        }
      >
        <p>
          Scope names the threads that synchronise through the atomic. The basic kernel's threads come from every block,
          so device scope is the narrowest correct choice (p. 208). CUDA's memory model makes such a data race undefined
          behaviour. With block scope the result might look right on one
          GPU and not on another. Neither of the course's emulators gives scopes any effect (labs/README.md), so no test
          here would catch the mistake: you have to read for it.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.2, p. 208; data race: notes/today/ch09-facts.md A2', // i18n-ignore: citation only
  },
  'ch09/9.2/q-emu-blind': {
    kind: 'choice',
    id: 'ch09/9.2/q-emu-blind',
    prompt: (
      <Translated
        vi={
          <>
            Bạn thay phép toán nguyên tử trong kernel cơ bản bằng một lệnh <code>++bins[b]</code> thường và thử nó trên một
            ảnh hằng, nơi mọi luồng đều trúng một bin. Lần chạy nào chắc chắn qua?
          </>
        }
      >
        <>
          You replace the atomic in the basic kernel with a plain <code>++bins[b]</code> and test it on a constant image,
          where every thread hits one bin. Which run is certain to pass?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="race, vốn chạy các luồng đồng thời dưới ThreadSanitizer.">
            race, which runs the threads concurrently under ThreadSanitizer.
          </Translated>
        ),
        why: (
          <Translated vi="race báo cáo lần đọc và lần ghi không có thứ tự trên bin, kể cả trong một lần chạy mà số đếm ra đúng, và một báo cáo làm bài kiểm tra thất bại.">
            race reports the unordered read and write of the bin, even in a run whose counts came out right, and a report fails the test.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Một lần chạy trên GPU, dù bạn lặp lại bao nhiêu lần.">A GPU run, however many times you repeat it.</Translated>,
        why: (
          <Translated vi="Việc một lần đếm có bị mất hay không phụ thuộc vào thời gian. Trên ảnh hằng thì thường là có; không gì làm cho kết quả nào chắc chắn.">
            Whether a count is lost depends on timing. On a constant image it usually is; nothing makes either outcome certain.
          </Translated>
        ),
      },
      { label: <Translated vi="emu, dưới cả ba thứ tự luồng của nó.">emu, under each of its three thread orders.</Translated>, correct: true },
      {
        label: (
          <Translated vi="Không cái nào: race có lộ ra hay không phụ thuộc vào thời gian, trên mọi backend.">
            None of them: whether a race shows depends on timing, on every backend.
          </Translated>
        ),
        why: (
          <Translated vi="Không đúng với emu: ở đó không có thời gian. Một luồng không bao giờ bị ngắt giữa lần nạp và lần ghi của nó, nên mọi thứ tự đều là tuần tự.">
            Not on emu: there is no timing there. A thread is never interrupted between its load and its store, so every order is a serial one.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>emu</code> chạy từng luồng một và chỉ chuyển luồng tại các điểm đồng bộ hóa. Kernel cơ bản không có điểm
            nào, nên mỗi phép tăng chạy từ lần nạp đến lần ghi trước khi bất kỳ luồng nào khác di chuyển: trường hợp tuần tự
            trong trình chạy từng bước, luôn đúng, dù bao nhiêu luồng dùng chung bin. Việc qua ở đó không nói gì về tính
            nguyên tử. Backend{' '}
            <code>race</code> chạy các luồng đồng thời dưới ThreadSanitizer chính vì lý do này.
          </p>
        }
      >
        <p>
          <code>emu</code> runs one thread at a time and switches only at synchronisation points. The basic kernel has
          none, so every increment runs from load to store before any other thread moves: the serial case in the stepper,
          always correct, however many threads share the bin. A pass there says nothing about atomicity. The{' '}
          <code>race</code> backend runs the threads concurrently under ThreadSanitizer for exactly this reason.
        </p>
      </Translated>
    ),
    source: 'labs/README.md, "The race-detection flavour"',
    vi: { source: 'labs/README.md, mục "The race-detection flavour"' },
  },
};

export function S92() {
  return (
    <>
      <Translated
        vi={
          <p>
            Cách trực tiếp nhất để song song hóa vòng lặp của mục 9.1 là khởi chạy một luồng cho mỗi điểm ảnh. Mỗi luồng
            tính chỉ số toàn cục của nó, <code>i = blockIdx.x*blockDim.x + threadIdx.x</code>, kiểm tra rằng <code>i</code>{' '}
            nằm trong ảnh, nạp điểm ảnh của nó và cộng một vào bin của điểm ảnh đó. Không gì ngăn nhiều luồng giữ các
            điểm ảnh có cùng giá trị, và khi đó tất cả chúng nhắm vào cùng một bin.
          </p>
        }
      >
        <p>
          The most direct way to parallelise the loop of section 9.1 is to launch one thread per pixel. Each thread works
          out its global index, <code>i = blockIdx.x*blockDim.x + threadIdx.x</code>, checks that <code>i</code> is
          inside the image, loads its pixel and adds one to that pixel's bin. Nothing stops several threads from holding
          pixels of the same value, and then they all target the same bin.
        </p>
      </Translated>
      <Source section="9.2" pages="203–204" figs={tr('Fig. 9.3', 'Hình 9.3')} />
      <Translated
        vi={
          <p>
            Vấn đề là một phép tăng không phải là một hành động. <code>++bins[b]</code> đọc bin, cộng một, rồi ghi kết quả
            trở lại: một thao tác <b>đọc-sửa-ghi</b> (read-modify-write). Nếu một luồng khác đọc cùng bin đó giữa lần đọc và lần
            ghi của luồng này, cả hai ghi lại cùng một giá trị và một phép tăng bị mất. Điều đó có xảy ra hay không phụ thuộc
            vào thời gian tương đối giữa các luồng, và chính điều này làm nó thành một race condition (tranh chấp dữ liệu).
          </p>
        }
      >
        <p>
          The trouble is that an increment is not one action. <code>++bins[b]</code> reads the bin, adds one, and writes
          the result back: a <b>read-modify-write</b>. If another thread reads the same bin between this thread's read and
          its write, both write back the same value and one increment is lost. Whether that happens depends on the
          relative timing of the threads, which is what makes it a race.
        </p>
      </Translated>
      <Source section="9.2" pages="204–205" />

      <h2><Translated vi="Tranh chấp trên một bin">Race on one bin</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn quyết định thời gian. Bắt đầu với hai luồng và các preset: bốn preset đầu tái hiện các kết quả trong hai bảng
            của sách, một đúng và một sai cho mỗi trường hợp luồng nào đi trước. Rồi tự dựng thứ tự của bạn bằng cách
            bấm, và thử ba hoặc bốn luồng.
          </p>
        }
      >
        <p>
          You decide the timing. Start with two threads and the presets: the first four reproduce the outcomes of the
          book's two tables, one correct and one wrong for each thread going first. Then build your own orders by
          clicking, and try three or four threads.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi lần một bước ở mức mã nguồn, không có thời gian và không có bộ nhớ đệm. Về hình thức, hai luồng chạy một lệnh{' '}
              <code>++</code> thường trên cùng vị trí là một data race và do đó là hành vi không xác định (notes/today/ch09-facts.md
              A2, A3). Cập nhật bị mất là kết quả điển hình, không phải một lời hứa.
            </p>
          }
        >
          <p>
            One source-level step at a time, no timing and no caches. Formally, two threads running a plain{' '}
            <code>++</code> on the same location is a data race and therefore undefined behaviour (notes/today/ch09-facts.md
            A2, A3). The lost update is the typical outcome, not a promise.
          </p>
        </Translated>
      </Simplification>
      <RaceStepper />
      <Translated
        vi={
          <p>
            Preset lockstep là trường hợp NVIDIA có ghi trong tài liệu: khi nhiều luồng của một warp ghi cùng một địa chỉ bằng
            một lệnh ghi thường, "luồng nào thực hiện lần ghi cuối cùng là không xác định". Tất cả chúng đều đã nạp giá trị cũ, nên
            dù lần ghi nào còn lại, bin cũng chỉ tăng thêm một.
          </p>
        }
      >
        <p>
          The lockstep preset is the case NVIDIA documents: when several threads of one warp write the same address with
          a plain store, "which thread performs the final write is undefined". All of them loaded the old value, so
          whichever write survives, the bin goes up by one.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch09-facts.md, A3, B1)', '2026-11-20 (notes/today/ch09-facts.md, mục A3, B1)')}>
        <Translated
          vi={
            <p>
              Câu trích dẫn là từ CUDA Programming Guide, mục 3.2.2.1.1. AMD ghi trong tài liệu rằng các lane của một warp
              thực thi đồng bộ (lockstep), đó là giả định của preset này.
            </p>
          }
        >
          <p>
            The quotation is from the CUDA Programming Guide, section 3.2.2.1.1. AMD documents that the lanes of a warp
            execute in lockstep, which is this preset's assumption.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Có bao nhiêu thứ tự là đúng?">How many orders are correct?</Translated></h2>
      <Translated
        vi={
          <p>
            Với hai luồng, mỗi luồng giữ nguyên thứ tự ba bước của chính nó, có 20 cách xen kẽ sáu bước đó.
          </p>
        }
      >
        <p>
          With two threads, each keeping its own three steps in order, there are 20 ways to interleave the six steps.
        </p>
      </Translated>
      <Predict
        id="ch09/9.2/p-correct-orders"
        prompt={
          <Translated vi={<>Hai luồng, mỗi luồng nạp, cộng một và ghi, trên một bin bắt đầu từ 0. Trong 20 cách xen kẽ, bao nhiêu cách kết thúc với bin bằng 2?</>}>
            <>Two threads each load, add one and store, on a bin that starts at 0. Of the 20 interleavings, how many end with the bin at 2?</>
          </Translated>
        }
        unit={tr('of 20', 'trên 20')}
      >
        {(v) => (
          <>
            <Translated
              vi={
                <p>
                  <b>2</b>: chỉ hai thứ tự tuần tự, một luồng chạy hết hoàn toàn trước luồng kia.{' '}
                  {v === 2
                    ? 'Đó là đáp án của bạn.'
                    : 'Trong mọi thứ tự khác, hai khoảng giữa lần nạp và lần ghi của mỗi luồng chồng lên nhau, nên luồng sau nạp trước khi luồng trước ghi, và cả hai đều ghi 1.'}{' '}
                  Đây là mọi cách xen kẽ cho hai, ba và bốn luồng, chạy qua cùng mô hình với trình chạy từng bước:
                </p>
              }
            >
              <p>
                <b>2</b>: only the two serial orders, one thread entirely before the other.{' '}
                {v === 2
                  ? 'That is what you had.'
                  : 'In every other order the two intervals between a thread\'s load and its store overlap, so the later thread loads before the earlier one stores, and both store 1.'}{' '}
                Here is every interleaving for two, three and four threads, run through the same model as the stepper:
              </p>
            </Translated>
            <Census />
            <Translated
              vi={
                <p>
                  Các lần chạy đúng chính là các lần chạy tuần tự, có <i>n</i>! cách như vậy, và tỉ lệ của chúng sụt mạnh khi
                  thêm luồng: một phần mười với hai luồng, 24 trên 369,600 với bốn luồng. Với phép toán nguyên tử, mọi thứ tự đều đúng.
                </p>
              }
            >
              <p>
                The correct runs are exactly the serial ones, <i>n</i>! of them, and their share collapses as threads are
                added: one in ten for two threads, 24 in 369,600 for four. With atomics every order is correct.
              </p>
            </Translated>
          </>
        )}
      </Predict>
      <Ask q={questionsS92['ch09/9.2/q-range']} />

      <h2><Translated vi="Phép toán nguyên tử">Atomic operations</Translated></h2>
      <Translated
        vi={
          <p>
            Một <b>phép toán nguyên tử</b> (atomic) trên một vị trí bộ nhớ là một thao tác đọc-sửa-ghi mà không thao tác
            đọc-sửa-ghi nào khác trên cùng vị trí có thể chồng lên: ba phần của nó tạo thành một đơn vị không thể chia cắt.
            Phần cứng hiện thực điều này bằng cách giữ các thao tác khác ngoài vị trí đó cho đến khi thao tác hiện tại xong.
          </p>
        }
      >
        <p>
          An <b>atomic operation</b> on a memory location is a read-modify-write that no other read-modify-write on the
          same location can overlap: its three parts form one indivisible unit. Hardware implements it by keeping other
          operations out of that location until the current one has finished.
        </p>
      </Translated>
      <Source section="9.2" pages="206" />
      <Translated
        vi={
          <p>
            Điều một phép toán nguyên tử không làm là chọn thứ tự. Luồng nào cũng có thể đi trước; lời hứa duy nhất là luồng
            thứ hai không thể bắt đầu cho đến khi luồng thứ nhất xong. Vì vậy trên một vị trí, các phép toán nguyên tử được
            thực hiện lần lượt từng cái một, điều mà mục 9.3 biến thành một giới hạn tốc độ. Đánh dấu <i>atomic</i> trong
            trình chạy từng bước: mỗi luồng giờ chỉ có một bước, mọi thứ tự bạn bấm được vẫn được phép, và không thứ tự nào
            làm mất cập nhật.
          </p>
        }
      >
        <p>
          What an atomic does not do is choose an order. Either thread may go first; the only promise is that the second
          cannot start until the first has finished. On one location, atomics are therefore carried out one at a time,
          which section 9.3 turns into a speed limit. Tick <i>atomic</i> in the stepper: each thread now has one step,
          every order you can click is still allowed, and none loses an update.
        </p>
      </Translated>
      <Source section="9.2" pages="206" />
      <Ask q={questionsS92['ch09/9.2/q-order']} />

      <h2><Translated vi="Phép toán nguyên tử trong CUDA C++">Atomics in CUDA C++</Translated></h2>
      <Translated
        vi={
          <p>
            Ấn bản thứ 5 của sách dùng thư viện atomic của CUDA C++, được thêm vào bằng <code>#include &lt;cuda/atomic&gt;</code>.
            Bạn không biến các bin thành đối tượng nguyên tử. Bạn dựng một <i>tham chiếu</i> nguyên tử tới đúng bin bạn muốn cập nhật,
            và gọi một hàm thành viên trên nó:
          </p>
        }
      >
        <p>
          The book's 5th edition uses the CUDA C++ atomics library, included with <code>#include &lt;cuda/atomic&gt;</code>.
          You do not make the bins atomic objects. You build an atomic <i>reference</i> to the one bin you want to update,
          and call a member function on it:
        </p>
      </Translated>
      <Source section="9.2" pages="206–208" figs={tr('Fig. 9.6', 'Hình 9.6')} />
      <pre className="mono" style={{ margin: '0 0 1.2rem', padding: '0.6rem 0.8rem', background: 'var(--cell)', border: '1px solid var(--rule)', borderRadius: 4, overflowX: 'auto' }}>
        {`cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
bins_ref.fetch_add(1, cuda::memory_order_relaxed);`}
      </pre>
      <Translated
        vi={
          <p>
            Hai đối số trong các dòng đó là những lựa chọn bạn sẽ phải đưa ra trong mọi phép toán nguyên tử bạn viết: phạm vi
            (scope) và thứ tự bộ nhớ. Bạn đã biết thứ tự bộ nhớ từ C++; phạm vi là phần CUDA thêm vào. Hãy quyết định cả hai
            trước khi đọc tiếp.
          </p>
        }
      >
        <p>
          Two arguments in those lines are choices you will make in every atomic you write: the scope and the memory
          order. You know memory orders from C++; a scope is CUDA's addition. Decide both before reading on.
        </p>
      </Translated>
      <Ask q={questionsS92['ch09/9.2/q-scope']} label={tr('Predict', 'Dự đoán')} />
      <Ask q={questionsS92['ch09/9.2/q-relaxed']} label={tr('Predict', 'Dự đoán')} />
      <Translated
        vi={
          <p>
            Bốn lựa chọn ẩn trong hai dòng đó. <b>Một tham chiếu, không phải một đối tượng nguyên tử</b>: thư viện cũng có{' '}
            <code>cuda::atomic</code>, thứ làm mọi truy cập tới một đối tượng thành nguyên tử ngay từ khi tạo ra nó; sách dùng
            tham chiếu vì muốn đọc và ghi các bin theo cách không nguyên tử ở chỗ khác. Việc dựng tham chiếu tốn rất
            ít lúc chạy; chủ yếu nó cho trình biên dịch biết cần sinh mã gì cho lời gọi theo sau.{' '}
            <b>Phạm vi</b> nói những luồng nào đồng bộ hóa qua phép toán nguyên tử: trong một khối, trong thiết bị, hay giữa
            các thiết bị. Các luồng tăng một bin có thể nằm ở các khối khác nhau, nên kernel cơ bản cần phạm vi device.
          </p>
        }
      >
        <p>
          Four choices hide in those two lines. <b>A reference, not an atomic object</b>: the library also has{' '}
          <code>cuda::atomic</code>, which makes every access to an object atomic from its creation; the book uses a
          reference because it wants to read and write the bins non-atomically elsewhere. Building the reference costs
          little at run time; it mostly tells the compiler what code to generate for the call that follows.{' '}
          <b>The scope</b> says which threads synchronise through the atomic: within a block, within the device, or across
          devices. The threads that increment a bin can be in different blocks, so the basic kernel needs device scope.
        </p>
      </Translated>
      <Source section="9.2" pages="207–208" />
      <Translated
        vi={
          <p>
            <b>Giá trị trả về</b>: <code>fetch_add</code> trả về giá trị của bin trước khi cộng. Kernel này bỏ qua
            nó; chương 12 và 18 dựa vào nó. <b>Thứ tự bộ nhớ</b> giới hạn cách trình biên dịch và phần cứng có thể sắp xếp lại
            phép toán nguyên tử so với các truy cập bộ nhớ khác của cùng luồng. Ở đây truy cập duy nhất còn lại là lần nạp
            điểm ảnh, và địa chỉ của bin phụ thuộc vào nó, nên lần nạp đó xảy ra trước dù thứ tự nói gì. Relaxed, vốn
            không thêm ràng buộc nào, là đủ. Chương 11 có một trường hợp cần thứ tự chặt hơn.
          </p>
        }
      >
        <p>
          <b>The return value</b>: <code>fetch_add</code> returns the bin's value from before the add. This kernel ignores
          it; chapters 12 and 18 depend on it. <b>The memory order</b> limits how the compiler and the hardware may reorder
          the atomic against the same thread's other memory accesses. Here the only other access is the load of the
          pixel, and the bin's address depends on it, so that load happens first whatever the order says. Relaxed, which
          adds no constraint, is enough. Chapter 11 has a case that needs a stricter order.
        </p>
      </Translated>
      <Source section="9.2" pages="208–209" />

      <h3><Translated vi="Những gì các khai báo được in bỏ sót">What the printed declarations leave out</Translated></h3>
      <Translated
        vi={
          <p>
            Sách in template của <code>atomic_ref</code> và chữ ký của <code>fetch_add</code> mà không có giá trị
            mặc định, và các giá trị mặc định lại là những giá trị chặt. Bỏ phạm vi đi thì bạn nhận{' '}
            <code>cuda::thread_scope_system</code>, bao gồm cả CPU và các GPU khác. Bỏ thứ tự đi thì{' '}
            <code>fetch_add</code> dùng <code>seq_cst</code>. <code>++bins_ref</code> cũng là nguyên tử, nhưng với thứ tự
            mặc định đó: chỉ <code>fetch_add(1, cuda::memory_order_relaxed)</code> khớp với sách. Hàm khởi tạo là{' '}
            <code>explicit</code> và nhận một tham chiếu, nên <code>bins_ref(bins[b])</code> là đúng; ví dụ trong hướng dẫn
            của chính NVIDIA ở mục 2.3.5.1 truyền một con trỏ, nên đừng chép nó.
          </p>
        }
      >
        <p>
          The book prints <code>atomic_ref</code>'s template and <code>fetch_add</code>'s signature without their
          defaults, and the defaults are the strict ones. Leave out the scope and you get{' '}
          <code>cuda::thread_scope_system</code>, which includes the CPU and other GPUs. Leave out the order and{' '}
          <code>fetch_add</code> uses <code>seq_cst</code>. <code>++bins_ref</code> is atomic too, but with that default
          order: only <code>fetch_add(1, cuda::memory_order_relaxed)</code> matches the book. The constructor is{' '}
          <code>explicit</code> and takes a reference, so <code>bins_ref(bins[b])</code> is right; NVIDIA's own guide
          example in section 2.3.5.1 passes a pointer, so do not copy it.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch09-facts.md, A1; infra-facts.md, A1)', '2026-11-20 (notes/today/ch09-facts.md, mục A1; infra-facts.md, A1)')}>
        <Translated
          vi={
            <p>
              NVIDIA giờ gọi <code>atomicAdd</code> và các hàm họ hàng là "legacy" và khuyến nghị các atomic của libcu++ mà sách
              dùng. <code>atomicAdd</code> legacy được ghi trong tài liệu là relaxed ở phạm vi device; <code>atomicAdd_block</code> là
              biến thể phạm vi khối của nó.
            </p>
          }
        >
          <p>
            NVIDIA now calls <code>atomicAdd</code> and its relatives "legacy" and recommends the libcu++ atomics the book
            uses. Legacy <code>atomicAdd</code> is documented as relaxed at device scope; <code>atomicAdd_block</code> is
            its block-scope variant.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Kernel cơ bản">The basic kernel</Translated></h2>
      <Translated
        vi={
          <p>
            Ghép lại, kernel là vòng lặp tuần tự với hai thay đổi: vòng lặp trở thành một chỉ số luồng và một lần kiểm tra
            biên, và phép tăng trở thành tham chiếu nguyên tử cùng <code>fetch_add</code> của nó.
          </p>
        }
      >
        <p>
          Put together, the kernel is the sequential loop with two changes: the loop becomes a thread index and a bounds
          check, and the increment becomes the atomic reference and its <code>fetch_add</code>.
        </p>
      </Translated>
      <Source section="9.2" pages="207" figs={tr('Fig. 9.6', 'Hình 9.6')} />
      <BasicListing />

      <h2><Translated vi="Cùng kernel trên các backend khác">The same kernel on other backends</Translated></h2>
      <Translated
        vi={
          <p>
            HIP giữ cách viết của sách. Metal hoàn toàn không có góc nhìn nguyên tử lên một mảng thường: kernel khai báo
            buffer bins là một mảng số nguyên nguyên tử, trong khi host vẫn cấp phát các bộ đếm 32-bit thường. Điều đó
            thay đổi hình dạng của mã nhiều hơn là thay đổi tên.
          </p>
        }
      >
        <p>
          HIP keeps the book's spelling. Metal does not have an atomic view of a plain array at all: the kernel declares
          the bins buffer as an array of atomic integers, while the host still allocates plain 32-bit counters. That
          changes the shape of the code more than the names.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Sách chỉ dùng CUDA. Các cột HIP và Metal lấy từ tài liệu và header của nhà cung cấp, được kiểm tra vào 2026-11-20 (notes/today/ch09-facts.md, "Where the backends really differ").</p>}>
          <p>The book uses CUDA only. The HIP and Metal columns come from vendor documentation and headers checked on 2026-11-20 (notes/today/ch09-facts.md, "Where the backends really differ").</p>
        </Translated>
      </Beyond>
      {/* Code here breaks only at spaces, never inside an identifier; below about 44rem the
          table keeps its width and scrolls inside its wrapper instead of squeezing columns. */}
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" style={{ minWidth: '44rem', margin: '0.4rem 0 1.4rem' }} data-part="backends">
          <colgroup>
            <col style={{ width: '8.5rem' }} />
            <col />
            <col style={{ width: '10rem' }} />
            <col />
          </colgroup>
          <thead>
            <tr><th></th><th>CUDA</th><th>HIP (ROCm 10.2)</th><th>Metal (MSL 4.1)</th></tr>{/* i18n-ignore: product names and versions */}
          </thead>
          <tbody>
            <tr>
              <td><Translated vi="Góc nhìn nguyên tử lên một mảng thường">Atomic view of a plain array</Translated></td>
              <Translated vi={<td><C>cuda::atomic_ref&lt;T, Scope&gt; r(x[i]);</C> từ <C>&lt;cuda/atomic&gt;</C></td>}>
                <td><C>cuda::atomic_ref&lt;T, Scope&gt; r(x[i]);</C> from <C>&lt;cuda/atomic&gt;</C></td>
              </Translated>
              <td><Translated vi="như trên, qua libhipcxx (ROCm 10.1 trở lên)">same, through libhipcxx (ROCm 10.1 and later)</Translated></td>
              <Translated vi={<td>không có: khai báo tham số là <C>device atomic_uint*</C></td>}>
                <td>none: declare the parameter <C>device atomic_uint*</C></td>
              </Translated>
            </tr>
            <tr>
              <td><Translated vi="Phép cộng">The add</Translated></td>
              <Translated vi={<td><C>r.fetch_add(1, cuda::memory_order_relaxed)</C>; <C>atomicAdd</C> legacy</td>}>
                <td><C>r.fetch_add(1, cuda::memory_order_relaxed)</C>; legacy <C>atomicAdd</C></td>
              </Translated>
              <Translated vi={<td>như trên; <C>atomicAdd</C> legacy</td>}>
                <td>same; legacy <C>atomicAdd</C></td>
              </Translated>
              <td><C>atomic_fetch_add_explicit(&amp;bins[b], 1, memory_order_relaxed)</C></td>
            </tr>
            <tr>
              <td><Translated vi="Thứ tự nếu bạn bỏ nó đi">Order if you leave it out</Translated></td>
              <Translated vi={<td><C>seq_cst</C> cho <C>fetch_add</C> và <C>++r</C>; các hàm legacy là relaxed</td>}>
                <td><C>seq_cst</C> for <C>fetch_add</C> and <C>++r</C>; legacy functions are relaxed</td>
              </Translated>
              <td><Translated vi="như trên">same</Translated></td>
              <td><Translated vi="không thể bỏ đi; với fetch-and-modify, spec chỉ cho phép relaxed">cannot be left out; for fetch-and-modify the spec allows relaxed only</Translated></td>
            </tr>
            <tr>
              <td><Translated vi="Phạm vi trên phép cập nhật">Scope on the update</Translated></td>
              <Translated vi={<td>block, cluster, device, system; mặc định system; các biến thể legacy <C>_block</C> và <C>_system</C></td>}>
                <td>block, cluster, device, system; default system; legacy <C>_block</C> and <C>_system</C> variants</td>
              </Translated>
              <Translated vi={<td>cùng các tên nhưng không có cluster; legacy chỉ có <C>_system</C>, không có <C>_block</C></td>}>
                <td>same names without cluster; legacy <C>_system</C> only, no <C>_block</C></td>
              </Translated>
              <td><Translated vi="không có trên đọc-sửa-ghi; phạm vi chỉ tồn tại trên fence">none on read-modify-write; scopes exist only on fences</Translated></td>
            </tr>
          </tbody>
        </table>
      </div>
      <Translated
        vi={
          <p>
            Một cầu nối tới mã production. vLLM giữ, cho mỗi request đang chạy, một số đếm <code>int32</code> cho mỗi mục của
            bộ từ vựng trong bộ nhớ toàn cục, và kernel Triton của nó cộng một cho mỗi token được sinh ra bằng{' '}
            <code>tl.atomic_add</code>, không có bản sao riêng nào. Đó chính là kernel này, với token id thay cho điểm ảnh.
          </p>
        }
      >
        <p>
          One bridge to production code. vLLM keeps, for every running request, an <code>int32</code> count per
          vocabulary entry in global memory, and its Triton kernel adds one per generated token with{' '}
          <code>tl.atomic_add</code>, with no private copies. That is this kernel, with token ids for pixels.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch09-facts.md, D6)', '2026-11-20 (notes/today/ch09-facts.md, mục D6)')}>
        <Translated
          vi={
            <p>
              <code>vllm/v1/worker/gpu/sample/penalties.py</code>: tensor có dạng <code>[max_num_reqs, vocab_size]</code>; chú thích
              của nó cảnh báo nó "can be very large, taking up GBs of GPU memory" (có thể rất lớn, chiếm hàng GB bộ nhớ GPU).
            </p>
          }
        >
          <p>
            <code>vllm/v1/worker/gpu/sample/penalties.py</code>: the tensor is <code>[max_num_reqs, vocab_size]</code>; its
            comment warns it "can be very large, taking up GBs of GPU memory".
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Công cụ nào thấy được lỗi ++ thường?">Which tool sees the plain ++ bug?</Translated></h2>
      <Translated
        vi={
          <p>
            Lab 9.1 đưa bạn kernel có <code>++bins[b]</code> thay cho phép toán nguyên tử. Khóa học cho bạn ba cách
            chạy nó: backend <code>emu</code>, backend <code>race</code> và GPU của bạn.
          </p>
        }
      >
        <p>
          Lab 9.1 hands you the kernel with <code>++bins[b]</code> in place of the atomic. The course gives you three ways
          to run it: the <code>emu</code> backend, the <code>race</code> backend and your GPU.
        </p>
      </Translated>
      <Ask q={questionsS92['ch09/9.2/q-emu-blind']} label={tr('Predict', 'Dự đoán')} />
      <Translated
        vi={
          <p>
            Trên <code>emu</code> nó qua, vì một luồng không bao giờ bị ngắt giữa lần nạp và lần ghi của nó. Trên{' '}
            <code>race</code>, ThreadSanitizer
            báo cáo lần đọc và lần ghi bin là một cặp không được đồng bộ hóa, kèm cả hai dòng mã nguồn. Trên GPU, một ảnh
            lệch là nơi số đếm bị mất. Công cụ của chính các nhà cung cấp GPU không giúp được nhiều ở đây:{' '}
            <code>compute-sanitizer --tool racecheck</code> của NVIDIA chỉ kiểm tra bộ nhớ chia sẻ, nên không thể thấy race
            này trong bộ nhớ toàn cục; AMD không ghi trong tài liệu trình phát hiện race nào; shader validation của Metal không có kiểm tra race.
          </p>
        }
      >
        <p>
          On <code>emu</code> it passes, because a thread is never interrupted between its load and its store. On{' '}
          <code>race</code>, ThreadSanitizer
          reports the read and the write of the bin as an unsynchronised pair, with both source lines. On a GPU a skewed
          image is where counts go missing. The GPU vendors' own tools do not help much here: NVIDIA's{' '}
          <code>compute-sanitizer --tool racecheck</code> checks shared memory only, so it cannot see this race in global
          memory; AMD documents no race detector; Metal's shader validation has no race check.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch09-facts.md, A10, B6, C5, E1)', '2026-11-20 (notes/today/ch09-facts.md, mục A10, B6, C5, E1)')}>
        <Translated
          vi={
            <p>
              Racecheck: "Currently, this tool only supports detecting accesses to on-chip shared memory." (Hiện tại, công cụ
              này chỉ hỗ trợ phát hiện truy cập tới bộ nhớ chia sẻ trên chip.) Một lệnh{' '}
              <code>++</code> thường trên một histogram trong bộ nhớ chia sẻ, ở mục 9.4, nằm trong tầm của nó.
            </p>
          }
        >
          <p>
            Racecheck: "Currently, this tool only supports detecting accesses to on-chip shared memory." A plain{' '}
            <code>++</code> on a shared-memory histogram, section 9.4, is within its reach.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Lab 9.1 nằm ở <a href={href('ch09', 'labs')}>trang lab</a>.
          </p>
        }
      >
        <p>
          Lab 9.1 is on the <a href={href('ch09', 'labs')}>labs page</a>.
        </p>
      </Translated>
    </>
  );
}
