import { useEffect, useMemo, useState } from 'react';
import { Code } from '../../components/Code';
import { Bench, type Tier } from '../../components/notes';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { useReports } from '../../lib/reports';
import {
  CORRECT,
  EX3,
  PRESET_GRAPH,
  SCHEDULES,
  SCHEDULE_LABEL,
  TOO_LARGE,
  UNVISITED,
  UNWRITTEN,
  VARIANT,
  kernelBody,
  residency,
  simulate,
  variantOf,
  type Preset,
  type Rule,
  type Schedule,
  type Shape,
  type Step,
} from '../../sim/gridbfs';

// "Three grid barriers" and "Does it fit?" (section 18.6, book §18.7, Fig. 18.17 lines 65-74,
// pp. 445-448, and Exercise 3). Everything shown is computed by src/sim/gridbfs.ts, whose
// kernels are pinned to labs/ch18/snippets/s186_coop_swap.cu and run on the emulator by
// gridbfs.test.ts.

const TIERS: Record<string, Tier> = {
  prevFrontier: 'global', currFrontier: 'global', numCurrFrontier: 'global', numFrontier: 'global', level: 'global',
  csrGraph: 'global', numCurrFrontier_ref: 'global', numFrontier_ref: 'global',
  numPrevFrontier: 'reg', currLevel: 'reg', i: 'reg', vertex: 'reg', edge: 'reg', neighbor: 'reg', currFrontierIdx: 'reg', tmp: 'reg',
};

const show = (v: number) => (v === UNWRITTEN ? '' : String(v));

function Barriers() {
  const [shape, setShape] = useState<Shape>(CORRECT);
  const [preset, setPreset] = useState<Preset>('grid');
  const [blocks, setBlocks] = useState(2);
  const [schedule, setSchedule] = useState<Schedule>('forward');
  const [rule, setRule] = useState<Rule>('gpu');
  const [n, setN] = useState(0);
  const locale = useLocale();
  const graph = useMemo(() => PRESET_GRAPH[preset](), [preset]);
  const trace = useMemo(() => simulate(shape, schedule, graph, blocks, 2, rule), [shape, schedule, graph, blocks, rule, locale]);
  useEffect(() => setN(0), [trace]);
  const total = trace.steps.length;
  const at = Math.min(n, total);
  const cur: Step | undefined = at > 0 ? trace.steps[at - 1] : undefined;
  const snap = cur ? cur.after : trace.initial;
  const ex3 = shape.kernel === 'ex3';
  const variant = variantOf(shape);
  const events = trace.history.filter((e) => e.step < at).slice(-12);
  const nextBarrier = () => {
    for (let k = at; k < total; k++) if (trace.steps[k].op === 'release' || trace.steps[k].op === 'deadlock') return setN(k + 1);
    setN(total);
  };
  const sync = (k: 'sync1' | 'sync2' | 'sync3', label: string) => (
    <label className="check">
      <input type="checkbox" checked={shape[k]} aria-label={label} onChange={(e) => setShape({ ...shape, [k]: e.target.checked })} /> {label}
    </label>
  );
  const W = graph.kind === 'grid' ? graph.W! : Math.min(graph.numVertices, 11);
  // Frontier buffers: draw the widest level of the search, and any slot written beyond it.
  const widest = Math.max(...trace.reference.reduce<number[]>((c, l) => (l === UNVISITED ? c : ((c[l] = (c[l] ?? 0) + 1), c)), []));
  const lastWritten = (b: number[]) => b.reduce((m, x, i) => (x === UNWRITTEN ? m : i + 1), 0);
  const slots = Math.max(widest, lastWritten(snap.A), lastWritten(snap.B));
  const counterLabel = ex3 ? 'numFrontier[L]' : '*numCurrFrontier';

  return (
    <>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi="Kernel">Kernel</Translated>
          <span className="seg" role="group" aria-label={tr('Kernel', 'Kernel')}>
            <button aria-pressed={!ex3} onClick={() => setShape(CORRECT)}><Translated vi="Phép hoán đổi của Hình 18.17">Fig. 18.17's swap</Translated></button>
            <button aria-pressed={ex3} onClick={() => setShape(EX3)}><Translated vi="Bài tập 3: một bộ đếm cho mỗi mức">Exercise 3: a counter per level</Translated></button>
          </span>
        </label>
        {ex3 ? sync('sync1', tr('the one grid.sync()', 'grid.sync() duy nhất')) : <>{sync('sync1', tr('grid.sync() 1', 'grid.sync() số 1'))}{sync('sync2', tr('grid.sync() 2', 'grid.sync() số 2'))}{sync('sync3', tr('grid.sync() 3', 'grid.sync() số 3'))}</>}
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi="Đồ thị">Graph</Translated>
          <select value={preset} aria-label={tr('Graph', 'Đồ thị')} onChange={(e) => setPreset(e.target.value as Preset)}>
            <option value="grid">{tr('a 5 × 5 grid from a corner', 'lưới 5 × 5, bắt đầu từ một góc')}</option>
            <option value="tree">{tr('a 4-ary tree of depth 2', 'cây 4 nhánh độ sâu 2')}</option>
          </select>
        </label>
        <label className="field">
          <Translated vi="Số khối thường trú, mỗi khối 2 luồng">Resident blocks of 2 threads</Translated>
          <span className="seg" role="group" aria-label={tr('Blocks', 'Khối')}>
            {[2, 3].map((b) => <button key={b} aria-pressed={blocks === b} onClick={() => setBlocks(b)}>{b}</button>)}
          </span>
        </label>
        <label className="field">
          <Translated vi="Thứ tự">Order</Translated>
          <select value={schedule} aria-label={tr('Order', 'Thứ tự')} onChange={(e) => setSchedule(e.target.value as Schedule)}>
            {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Một rào chắn mà vài luồng đã rời khỏi kernel">A barrier some threads have left</Translated>
          <span className="seg" role="group" aria-label={tr('Barrier rule', 'Quy tắc rào chắn')}>
            <button aria-pressed={rule === 'gpu'} onClick={() => setRule('gpu')}><Translated vi="không bao giờ mở (theo hợp đồng)">never opens (the contract)</Translated></button>
            <button aria-pressed={rule === 'emu'} onClick={() => setRule('emu')}><Translated vi="mở ra (trình giả lập của khóa học)">opens (the course's emulator)</Translated></button>
          </span>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setN(0)} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => setN(at - 1)} disabled={at === 0}><Translated vi="Lùi lại">Back</Translated></button>
        <button className="primary" onClick={() => setN(at + 1)} disabled={at >= total}><Translated vi="Bước">Step</Translated></button>
        <button onClick={nextBarrier} disabled={at >= total}><Translated vi="Chạy đến rào chắn kế tiếp">Run to next barrier</Translated></button>
        <button onClick={() => setN(total)} disabled={at >= total}><Translated vi="Chạy đến hết">Run to the end</Translated></button>
        {trace.firstFailure >= 0 ? <button onClick={() => setN(trace.firstFailure + 1)}><Translated vi="Đến lỗi đầu tiên">Go to the first failure</Translated></button> : null}
        <span className="muted" aria-live="polite"><Translated vi={<>bước {at} / {total}</>}>step {at} of {total}</Translated></span>
      </div>
      <div className={'narrate' + (cur?.violation ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }} data-narrate>
        {cur ? (
          <>
            {cur.text}
            {cur.violation ? <><br /><b className="t-fail"><Translated vi={<>Lỗi: {cur.violation.message}.</>}>Failure: {cur.violation.message}.</Translated></b></> : null}
            {cur.note ? <><br /><span className="muted">{cur.note}</span></> : null}
          </>
        ) : (
          <Translated
            vi={
              <>
                Chưa có gì chạy. Một lần khởi chạy gồm {blocks} khối, mỗi khối 2 luồng, khối nào cũng thường trú; đỉnh 0 là gốc,
                và frontier trong buffer A chứa nó. Kernel: {variant ? VARIANT[variant].label : tr('your own combination', 'tổ hợp của riêng bạn')}. Mỗi
                bước chạy một luồng từ rào chắn lưới này đến rào chắn kế tiếp. Bấm Bước.
              </>
            }
          >
            <>
              Nothing has run. One launch of {blocks} blocks of 2 threads, every block resident; vertex 0 is the root, and
              the frontier in buffer A holds it. Kernel: {variant ? VARIANT[variant].label : tr('your own combination', 'tổ hợp của riêng bạn')}. A step
              runs one thread from one grid barrier to the next. Press Step.
            </>
          </Translated>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 9fr) minmax(0, 5fr)' }}>
        <div style={{ minWidth: 0 }}>
          <Code tabs={[{ label: tr('kernel body, Fig. 18.17 without its private queue', 'thân kernel, Hình 18.17 không có hàng đợi riêng'), code: kernelBody(shape, true), tiers: TIERS }]} highlight={cur ? cur.lines : []} />
        </div>
        <div style={{ minWidth: 0 }}>
          <table className="threads" aria-label={tr('Threads', 'Các luồng')}>
            <thead>
              <tr><th><Translated vi="luồng">thread</Translated></th><th><Translated vi="mức">level</Translated></th><th title={tr('its own numPrevFrontier', 'numPrevFrontier của riêng nó')}>numPrev</th><th title={tr('its prevFrontier and currFrontier', 'prevFrontier và currFrontier của nó')}>prev/curr</th><th><Translated vi="hiện tại">now</Translated></th></tr>
            </thead>
            <tbody>
              {snap.threads.map((t) => (
                <tr key={t.rank} className={cur?.rank === t.rank ? 'cur' : ''} data-thread={t.rank}>
                  <td>{t.rank} (b{t.block})</td>
                  <td>{t.currLevel}</td>
                  <td className="t-reg">{t.numPrev}</td>
                  <td>{t.prev}/{t.curr}</td>
                  <td>{t.status === 'waiting' ? (ex3 ? tr('at the sync', 'tại sync') : tr(`at sync ${t.at}`, `tại sync ${t.at}`)) : t.status === 'exited' ? tr('has left', 'đã rời đi') : tr('running', 'đang chạy')}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="mx-label t-global" style={{ marginTop: '0.7rem' }}><Translated vi="level[] (bộ nhớ toàn cục (global memory)); · là UNVISITED">level[] (global memory); · is UNVISITED</Translated></div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${W}, auto)` }} data-levels={snap.level.map((l) => (l === UNVISITED ? '.' : l)).join(',')}>
            {snap.level.map((l, v) => (
              <span key={v} className={l === UNVISITED ? 'void' : l !== trace.reference[v] ? 'bad' : 'on'}>
                {l === UNVISITED ? '·' : l}<sub>{v}</sub>
              </span>
            ))}
          </div>
          {(['A', 'B'] as const).map((buf) => (
            <div key={buf}>
              <div className="mx-label t-global" style={{ marginTop: '0.5rem' }}>
                <Translated vi="buffer">buffer</Translated> {buf}{snap.threads[0].prev === buf ? tr(' (prevFrontier of thread 0)', ' (prevFrontier của luồng 0)') : tr(' (currFrontier of thread 0)', ' (currFrontier của luồng 0)')}
              </div>
              <div className="mx global" style={{ gridTemplateColumns: `repeat(${Math.min(8, slots)}, auto)` }}>
                {snap[buf].slice(0, slots).map((x, i) => (
                  <span key={i} className={x === UNWRITTEN ? 'void' : 'on'}>{show(x)}<sub>{i}</sub></span>
                ))}
              </div>
            </div>
          ))}
          <div className="panel" style={{ marginTop: '0.7rem' }} data-counter={snap.counters.join(',')}>
            <b>{counterLabel}</b>{' '}
            {ex3 ? snap.counters.map((c, l) => (c ? `[${l}] ${c}` : null)).filter(Boolean).join(', ') || tr('all 0', 'tất cả là 0') : snap.counters[0]}
            <div className="muted" style={{ fontSize: '0.8rem', marginTop: '0.3rem' }}><Translated vi="Lịch sử đến giờ (12 mục cuối):">History so far (last 12):</Translated></div>
            <div style={{ fontSize: '0.8rem' }} data-history>
              {events.length ? events.map((e, k) => (
                <span key={k} className={'chip ' + (e.kind === 'reset' ? 'fail' : e.kind === 'read' ? 'reg' : 'global')} style={{ margin: '0 0.25rem 0.25rem 0' }}>
                  {e.kind === 'append' ? `t${e.rank} +1 → ${e.after}` : e.kind === 'read' ? tr(`t${e.rank} reads ${e.value}`, `t${e.rank} đọc ${e.value}`) : tr(`t${e.rank} resets ${e.before} → 0`, `t${e.rank} đặt lại ${e.before} → 0`)}
                </span>
              )) : <span className="muted"><Translated vi="chưa có gì">nothing yet</Translated></span>}
            </div>
          </div>
          {at >= total ? (
            <p style={{ margin: '0.9rem 0 0' }} data-outcome={trace.outcome.label}>
              <span className={'chip ' + (trace.outcome.kind === 'correct' ? 'ok' : 'fail')}>{tr(trace.outcome.label, ({ 'an out-of-range access': 'một truy cập ngoài phạm vi', 'early exit, deadlock': 'thoát sớm, deadlock', 'early exit': 'thoát sớm', 'appends lost to a late reset': 'các lần nối thêm bị mất vì đặt lại muộn', 'a thread read a partial count': 'một luồng đọc một số đếm dở dang', 'levels wrong': 'mức sai', correct: 'đúng' } as Record<string, string>)[trace.outcome.label] ?? trace.outcome.label)}</span>
              {trace.outcome.consequence ? <> {trace.outcome.consequence}.</> : null}
              {trace.outcome.kind === 'correct' ? <Translated vi={<> Mọi đỉnh đều đã có mức BFS của mình.</>}><> Every vertex has its BFS level.</></Translated> : null}
            </p>
          ) : null}
        </div>
      </div>
    </>
  );
}

function Fit() {
  const { reports } = useReports();
  const [sms, setSms] = useState(2);
  const [perSM, setPerSM] = useState(2);
  const [grid, setGrid] = useState(6);
  const r = residency(sms, perSM, grid);
  // A device's SM count, if a CUDA or HIP run reported one; Metal reports none.
  const mine = (['cuda', 'hip'] as const)
    .map((b) => ({ b, v: Number(reports[b]?.device?.multiProcessorCount) }))
    .find((x) => x.v > 0);
  const num = (label: string, value: number, min: number, max: number, set: (v: number) => void, name: string) => (
    <label className="field">
      {label}
      <input type="number" min={min} max={max} value={value} aria-label={label} data-input={name} style={{ width: '5rem' }}
        onChange={(e) => set(Math.max(min, Math.min(max, Math.round(Number(e.target.value)) || min)))} />
    </label>
  );
  const drawSMs = Math.min(sms, 8);
  return (
    <>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        {num(tr('SMs', 'Số SM'), sms, 1, 512, setSms, 'sms')}
        {num(tr('blocks per SM (the occupancy query)', 'số khối mỗi SM (truy vấn occupancy (mức chiếm dụng))'), perSM, 1, 32, setPerSM, 'per-sm')}
        {num(tr('blocks in the grid, G', 'số khối trong lưới, G'), grid, 1, 16384, setGrid, 'grid')}
        <button onClick={() => setGrid(r.slots)}><Translated vi="Đặt kích thước: G = số SM × số khối mỗi SM">Size it: G = SMs × blocks per SM</Translated></button>
        {mine ? <button className="quiet" onClick={() => setSms(mine.v)}><Translated vi={<>Thiết bị {mine.b.toUpperCase()} của bạn: {mine.v} SM</>}>Your {mine.b.toUpperCase()} device: {mine.v} SMs</Translated></button> : null}
      </div>
      <p className="muted" style={{ marginTop: 0 }}>
        <Translated vi="Số chỗ thường trú">Resident slots</Translated> R = {sms} × {perSM} = <b data-slots>{r.slots}</b>. {mine ? '' : tr('No CUDA or HIP run has reported an SM count on this machine; Metal reports none.', 'Chưa có lần chạy CUDA hay HIP nào báo số SM trên máy này; Metal không báo con số này.')}
      </p>
      <div className="row" style={{ gap: '0.5rem', alignItems: 'flex-start', marginBottom: '0.8rem' }} aria-label={tr('Resident slots', 'Số chỗ thường trú')}>
        {Array.from({ length: drawSMs }, (_, s) => (
          <div key={s} className="mx global" style={{ gridTemplateColumns: `repeat(${Math.min(perSM, 4)}, auto)` }} title={tr(`SM ${s}`, `SM ${s}`)}>
            {Array.from({ length: Math.min(perSM, 8) }, (_, k) => {
              const b = s * perSM + k;
              return <span key={k} className={b < r.resident ? 'on' : 'void'}>{b < r.resident ? `b${b}` : ''}</span>;
            })}
          </div>
        ))}
        {sms > drawSMs ? <span className="muted"><Translated vi={<>và thêm {sms - drawSMs} SM nữa</>}>and {sms - drawSMs} more SMs</Translated></span> : null}
      </div>
      {r.pending ? (
        <div style={{ marginBottom: '0.8rem' }}>
          <div className="mx-label"><Translated vi="đang chờ một chỗ mà không khối thường trú nào giải phóng">waiting for a slot that no resident block frees</Translated></div>
          <div className="mx" style={{ gridTemplateColumns: `repeat(${Math.min(r.pending, 8)}, auto)` }} data-pending={r.pending}>
            {Array.from({ length: Math.min(r.pending, 16) }, (_, k) => <span key={k} className="void">b{r.resident + k}</span>)}
          </div>
          {r.pending > 16 ? <span className="muted"><Translated vi={<>và thêm {r.pending - 16} khối nữa</>}>and {r.pending - 16} more</Translated></span> : null}
        </div>
      ) : null}
      <table className="data" style={{ margin: 0 }}>
        <thead><tr><th><Translated vi="Khởi chạy">Launch</Translated></th><th><Translated vi="Điều gì xảy ra">What happens</Translated></th></tr></thead>
        <tbody>
          <tr data-ordinary={r.ordinary}>
            <td><Translated vi="thông thường,">ordinary,</Translated> <code>&lt;&lt;&lt; &gt;&gt;&gt;</code></td>
            <td>
              {r.fits
                ? tr(
                    'Every block could be resident, but grid.sync() is documented only under the cooperative launch, and nothing checks the size for you. grid.is_valid() reports whether the grid can synchronise.',
                    'Mọi khối đều có thể thường trú, nhưng grid.sync() chỉ được tài liệu hóa dưới cooperative launch, và không có gì kiểm tra kích thước giúp bạn. grid.is_valid() cho biết lưới có thể đồng bộ hóa hay không.',
                  )
                : tr(
                    `Never completes, by the book's argument (p. 445): ${r.resident} resident blocks wait at grid.sync() for ${r.pending} that cannot start until a resident block leaves. What grid.sync() actually does under this launch is not documented; grid.is_valid() reports whether the grid can synchronise.`,
                    `Không bao giờ kết thúc, theo lập luận của sách (p. 445): ${r.resident} khối thường trú chờ tại grid.sync() cho ${r.pending} khối không thể bắt đầu cho đến khi một khối thường trú rời đi. grid.sync() thực sự làm gì dưới kiểu khởi chạy này thì không được tài liệu hóa; grid.is_valid() cho biết lưới có thể đồng bộ hóa hay không.`,
                  )}
            </td>
          </tr>
          <tr data-cooperative={r.cooperative}>
            <td><Translated vi="cooperative (hợp tác)">cooperative</Translated></td>
            <td>
              {r.fits ? (
                <span className="chip ok"><Translated vi="khởi chạy được">launches</Translated></span>
              ) : (
                <Translated
                  vi={
                    <>
                      <span className="chip fail">bị từ chối</span> khi khởi chạy. CUDA: <code>{TOO_LARGE.cuda}</code>. HIP:{' '}
                      <code>{TOO_LARGE.hip}</code>. Metal: không có tương ứng, vì Metal không có cooperative launch và không có
                      rào chắn lưới.
                    </>
                  }
                >
                  <>
                    <span className="chip fail">refused</span> at launch. CUDA: <code>{TOO_LARGE.cuda}</code>. HIP:{' '}
                    <code>{TOO_LARGE.hip}</code>. Metal: no counterpart, because it has no cooperative launch and no grid
                    barrier.
                  </>
                </Translated>
              )}
            </td>
          </tr>
        </tbody>
      </table>
    </>
  );
}

export function GridBarrierStepper() {
  const [tab, setTab] = useState<'barriers' | 'fit'>('barriers');
  return (
    <Bench
      title={tr('Three grid barriers', 'Ba rào chắn lưới')}
      task={tab === 'barriers'
        ? tr('Untick one grid.sync() at a time and find the order that breaks it. Then try Exercise 3.', 'Bỏ chọn từng grid.sync() một và tìm thứ tự làm nó hỏng. Sau đó thử Bài tập 3.')
        : tr('Make the grid one block too large, and see where the failure is caught.', 'Cho lưới lớn hơn một khối so với mức vừa, và xem lỗi bị bắt ở đâu.')}
      foot={
        tab === 'barriers' ? (
          <Translated
            vi={
              <>
                Toàn bộ quá trình tìm kiếm trong một lần cooperative launch, mỗi đỉnh mới được nối thẳng vào bộ đếm chung
                (phép nối của Hình 18.12) để một luồng chạy từ rào chắn lưới này đến rào chắn kế tiếp mà không dừng. Các luồng
                lần lượt chạy: trong mỗi lượt, mọi luồng đang chạy đi đến grid.sync() kế tiếp của nó hoặc ra khỏi kernel, theo
                thứ tự đã chọn; rồi rào chắn mở ra. "In order" và "reversed" là hai thứ tự đầu tiên của trình giả lập.
              </>
            }
          >
            <>
              The whole search in one cooperative launch, with each new vertex appended straight to the public counter
              (Fig. 18.12's append) so that a thread runs from one grid barrier to the next without stopping. Threads take
              turns: in each pass every running thread runs to its next grid.sync() or out of the kernel, in the chosen
              order; then the barrier opens. "In order" and "reversed" are the emulator's first two orders.
            </>
          </Translated>
        ) : (
          <Translated
            vi={
              <>
                Chỉ là số đếm. Số chỗ thường trú lấy từ công thức ở pp. 445–446; một kernel thật có đạt được chừng đó hay không
                tùy vào thanh ghi (register) và bộ nhớ chia sẻ (shared memory) của nó, và đó chính là điều truy vấn occupancy tính ra.
              </>
            }
          >
            <>
              Counts only. The resident slots come from the formula of pp. 445–446; whether a real kernel gets that many
              depends on its registers and shared memory, which is what the occupancy query works out.
            </>
          </Translated>
        )
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="tablist" aria-label={tr('View', 'Chế độ xem')}>
          <button role="tab" aria-selected={tab === 'barriers'} aria-pressed={tab === 'barriers'} onClick={() => setTab('barriers')}><Translated vi="Ba rào chắn lưới">Three grid barriers</Translated></button>
          <button role="tab" aria-selected={tab === 'fit'} aria-pressed={tab === 'fit'} onClick={() => setTab('fit')}><Translated vi="Có vừa không?">Does it fit?</Translated></button>
        </span>
      </div>
      {tab === 'barriers' ? <Barriers /> : <Fit />}
    </Bench>
  );
}
