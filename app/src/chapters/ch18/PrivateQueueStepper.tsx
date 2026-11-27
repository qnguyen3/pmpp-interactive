import { useEffect, useMemo, useState } from 'react';
import { Code } from '../../components/Code';
import { Bench, type Tier } from '../../components/notes';
import { Translated, tr, useLocale } from '../../lib/i18n';
import {
  BLOCK_ORDER_LABEL,
  CORRECT,
  GARBAGE,
  MAX_CAPACITY,
  PRESETS,
  PRESET_LABEL,
  SCHEDULES,
  SCHEDULE_LABEL,
  SITE_TEXT,
  UNWRITTEN,
  blockOrderOf,
  kernelBody,
  listingLine,
  races,
  simulate,
  summary,
  variantOf,
  VARIANT,
  type BlockOrder,
  type Preset,
  type Reset,
  type Schedule,
  type Shape,
  type Step,
} from '../../sim/bfspriv';

// "One block's private frontier" (section 18.6, book §18.6, Fig. 18.15, pp. 442-444). Every
// number comes from src/sim/bfspriv.ts, whose kernels are pinned to
// labs/ch18/snippets/s186_privatised_variants.cu and run on the emulator and its race flavour
// by bfspriv.test.ts.

export const PRIV_TIERS: Record<string, Tier> = {
  currFrontier_s: 'shared', numCurrFrontier_s: 'shared', numCurrFrontier_s_ref: 'shared', currFrontierStartIdx: 'shared',
  currFrontier: 'global', numCurrFrontier: 'global', numCurrFrontier_ref: 'global', prevFrontier: 'global', level: 'global',
  csrGraph: 'global',
  i: 'reg', vertex: 'reg', edge: 'reg', neighbor: 'reg', currFrontierIdx_s: 'reg', currFrontierIdx: 'reg', currLevel: 'reg',
};

/** A shared-memory word: CD for the emulator's fill, else its value. */
const cell = (v: number) => (v === GARBAGE ? 'CD' : String(v));

/** The verdict is a sentence, so it wraps inside its panel (a .chip alone is nowrap). */
const WRAP = { whiteSpace: 'normal', maxWidth: '100%', boxSizing: 'border-box' } as const;

export function PrivateQueueStepper() {
  const [capacity, setCapacity] = useState(2);
  const [threads, setThreads] = useState(4);
  const [blocks, setBlocks] = useState(2);
  const [preset, setPreset] = useState<Preset>('fanout');
  const [shape, setShape] = useState<Shape>(CORRECT);
  const [schedule, setSchedule] = useState<Schedule>('forward');
  const [order, setOrder] = useState<BlockOrder>('in-order');
  const [seed, setSeed] = useState(1);
  const [small, setSmall] = useState(false);
  const [n, setN] = useState(0);

  const blockOrder = useMemo(() => blockOrderOf(order, blocks, seed), [order, blocks, seed]);
  const opts = useMemo(
    () => ({ capacity, blocks, threads, preset, blockOrder, leftover: small ? 1 : undefined }),
    [capacity, blocks, threads, preset, blockOrder, small],
  );
  const locale = useLocale();
  const trace = useMemo(() => simulate(shape, schedule, opts), [shape, schedule, opts, locale]);
  const raced = useMemo(() => races(shape, opts), [shape, opts]);
  useEffect(() => setN(0), [trace]);

  const total = trace.steps.length;
  const at = Math.min(n, total);
  const cur: Step | undefined = at > 0 ? trace.steps[at - 1] : undefined;
  const snap = cur ? cur.after : trace.initial;
  const prev = at > 1 ? trace.steps[at - 2].after : trace.initial;
  const block = cur ? cur.block : blockOrder[0];
  const listing = kernelBody(shape, true);
  const done = trace.steps.slice(0, at);
  const newSoFar = done.filter((s) => s.op === 'count').length;
  const reserved = trace.reservations.filter((r) => r.step < at);
  const variant = variantOf(shape);
  const expected = trace.graph.expected.length;
  // Public slots worth drawing: the expected frontier, and anything written past it.
  let shownSlots = Math.max(expected, snap.count);
  snap.how.forEach((h, j) => { if (h) shownSlots = Math.max(shownSlots, j + 1); });
  shownSlots = Math.min(shownSlots, snap.pub.length);

  const nextBarrier = () => {
    for (let k = at; k < total; k++) if (trace.steps[k].op === 'release') return setN(k + 1);
    setN(total);
  };
  const flag = (key: 'barrier1' | 'barrier2' | 'barrier3' | 'overflowCheck', label: string) => (
    <label className="check">
      <input type="checkbox" checked={shape[key]} onChange={(e) => setShape({ ...shape, [key]: e.target.checked })} /> {label}
    </label>
  );
  const range = (label: string, value: number, min: number, max: number, set: (v: number) => void, name: string) => (
    <label className="field">
      {label} <b data-value={name}>{value}</b>
      <input type="range" min={min} max={max} value={value} aria-label={label} onChange={(e) => set(Number(e.target.value))} />
    </label>
  );

  return (
    <Bench
      title={tr("One block's private frontier", 'Frontier riêng của một khối')}
      task={tr(
        'Find when overflow starts, then break the kernel one barrier at a time and find an order that shows it.',
        'Tìm lúc tràn bắt đầu, rồi làm hỏng kernel từng rào chắn một và tìm một thứ tự để lộ ra lỗi đó.',
      )}
      foot={
        <Translated
          vi={
            <>
              Một mức của BFS: luồng i của lần khởi chạy giữ đỉnh i của frontier trước. Các khối chạy lần lượt, theo thứ tự
              đã chọn; trong một khối, một luồng chạy đến khi gặp rào chắn, rồi luồng kế tiếp tiếp quản, như trên trình giả
              lập của khóa học. Bộ nhớ chia sẻ (shared memory) bắt đầu là các byte 0xCD (hiển thị là CD) trừ khi bạn chọn giá
              trị sót lại nhỏ. Phép kiểm tra race chạy ba thứ tự luồng của trình giả lập và báo hai truy cập vào cùng một vị
              trí khi có một truy cập ghi, chúng không cùng là atomic, và không có rào chắn hay thứ tự chương trình nào đặt
              một truy cập trước.
            </>
          }
        >
          <>
            One level of a BFS: thread i of the launch holds previous-frontier vertex i. Blocks run one after another, in
            the order chosen; inside a block a thread runs until it reaches a barrier, then the next takes over, as on the
            course's emulator. Shared memory starts as 0xCD bytes (shown as CD) unless you choose the small leftover. The
            race check runs the emulator's three thread orders and reports two accesses to one location when one writes,
            they are not both atomic, and no barrier or program order puts one first.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        {range('PRIVATE_FRONTIER_CAPACITY', capacity, 1, MAX_CAPACITY, setCapacity, 'capacity')}
        {range(tr('threads per block', 'số luồng mỗi khối'), threads, 2, 8, setThreads, 'threads')}
        {range(tr('blocks', 'số khối'), blocks, 1, 4, setBlocks, 'blocks')}
        <label className="field">
          <Translated vi="Mức">Level</Translated>
          <select value={preset} aria-label={tr('Level preset', 'Mức có sẵn')} onChange={(e) => setPreset(e.target.value as Preset)}>
            {PRESETS.map((p) => <option key={p} value={p}>{PRESET_LABEL[p]}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi="Đặt lại bộ đếm khi tràn">Counter reset on overflow</Translated>
          <span className="seg" role="group" aria-label={tr('Counter reset', 'Đặt lại bộ đếm')}>
            {(['plain', 'atomic', 'clamp'] as Reset[]).map((r) => (
              <button key={r} aria-pressed={shape.reset === r} data-reset={r} onClick={() => setShape({ ...shape, reset: r })}>
                {r === 'plain' ? tr('plain store', 'ghi thường') : r === 'atomic' ? tr('atomic store', 'ghi atomic') : tr('clamp after barrier 2', 'kẹp giá trị sau rào chắn 2')}
              </button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi="Ghi ra (commit)">Commit</Translated>
          <span className="seg" role="group" aria-label={tr('Commit', 'Ghi ra (commit)')}>
            <button aria-pressed={shape.commit === 'loop'} onClick={() => setShape({ ...shape, commit: 'loop' })}><Translated vi="vòng lặp có bước nhảy">strided loop</Translated></button>
            <button aria-pressed={shape.commit === 'if'} onClick={() => setShape({ ...shape, commit: 'if' })}><Translated vi="một lệnh if">an if</Translated></button>
          </span>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        {flag('barrier1', tr('barrier 1', 'rào chắn 1'))}
        {flag('barrier2', tr('barrier 2', 'rào chắn 2'))}
        {flag('barrier3', tr('barrier 3', 'rào chắn 3'))}
        {flag('overflowCheck', tr('overflow check', 'kiểm tra tràn'))}
        <label className="field">
          <Translated vi="Các luồng trong một khối chạy">Threads in a block run</Translated>
          <select value={schedule} aria-label={tr('Thread order', 'Thứ tự luồng')} onChange={(e) => setSchedule(e.target.value as Schedule)}>
            {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Các khối chạy và đặt chỗ">Blocks run and reserve</Translated>
          <select value={order} aria-label={tr('Block order', 'Thứ tự khối')} onChange={(e) => setOrder(e.target.value as BlockOrder)}>
            {(['in-order', 'reversed', 'random'] as BlockOrder[]).map((o) => <option key={o} value={o}>{BLOCK_ORDER_LABEL[o]}</option>)}
          </select>
        </label>
        {order === 'random' ? <button className="quiet" onClick={() => setSeed((s) => s + 1)}><Translated vi="Thứ tự khác">Another order</Translated></button> : null}
        <label className="field">
          <Translated vi="Bộ nhớ chia sẻ bắt đầu là">Shared memory starts as</Translated>
          <select value={small ? 'small' : 'cd'} aria-label={tr('Shared memory starts as', 'Bộ nhớ chia sẻ bắt đầu là')} onChange={(e) => setSmall(e.target.value === 'small')}>
            <option value="cd">{tr('0xCD bytes (the emulator)', 'các byte 0xCD (trình giả lập)')}</option>
            <option value="small">{tr('a leftover 1 (as a GPU may leave)', 'giá trị 1 sót lại (như GPU có thể để lại)')}</option>
          </select>
        </label>
        <button className="quiet" onClick={() => { setShape(CORRECT); setSmall(false); }}><Translated vi="Kernel đã sửa">Corrected kernel</Translated></button>
      </div>

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setN(0)} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => setN(at - 1)} disabled={at === 0}><Translated vi="Lùi lại">Back</Translated></button>
        <button className="primary" onClick={() => setN(at + 1)} disabled={at >= total}><Translated vi="Bước">Step</Translated></button>
        <button onClick={nextBarrier} disabled={at >= total}><Translated vi="Chạy đến rào chắn kế tiếp">Run to next barrier</Translated></button>
        <button onClick={() => setN(total)} disabled={at >= total}><Translated vi="Chạy đến hết">Run to the end</Translated></button>
        {trace.firstFailure >= 0 ? <button onClick={() => setN(trace.firstFailure + 1)}><Translated vi="Đến lỗi đầu tiên">Go to the first failure</Translated></button> : null}
        <span className="muted" aria-live="polite" data-step={at}><Translated vi={<>bước {at} / {total}</>}>step {at} of {total}</Translated></span>
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
                Chưa có gì chạy. {blocks} khối, mỗi khối {threads} luồng, sẽ tìm ra {expected} đỉnh mới.
                Các khối chạy theo thứ tự {blockOrder.join(', ')}. Kernel: {variant ? VARIANT[variant].label : tr('your own combination of changes', 'tổ hợp thay đổi của riêng bạn')}.
                Bấm Bước.
              </>
            }
          >
            <>
              Nothing has run. {blocks} block{blocks === 1 ? '' : 's'} of {threads} threads will find {expected} new vertices.
              Blocks run in the order {blockOrder.join(', ')}. Kernel: {variant ? VARIANT[variant].label : tr('your own combination of changes', 'tổ hợp thay đổi của riêng bạn')}.
              Press Step.
            </>
          </Translated>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 9fr) minmax(0, 5fr)' }}>
        <div style={{ minWidth: 0 }}>
          <Code tabs={[{ label: tr(`kernel body: ${variant ? VARIANT[variant].label : tr('your combination of changes', 'tổ hợp thay đổi của bạn')}`, `thân kernel: ${variant ? VARIANT[variant].label : tr('your combination of changes', 'tổ hợp thay đổi của bạn')}`), code: listing, tiers: PRIV_TIERS }]} highlight={cur ? cur.lines : []} />
          <table className="threads" aria-label={tr('Threads of the current block', 'Các luồng của khối hiện tại')}>
            <thead>
              <tr><th><Translated vi={<>khối {block}</>}>block {block}</Translated></th><th><Translated vi="đỉnh">vertex</Translated></th><th><Translated vi="đang làm">doing</Translated></th></tr>
            </thead>
            <tbody>
              {Array.from({ length: threads }, (_, t) => (
                <tr key={t} className={cur?.block === block && cur?.thread === t ? 'cur' : ''}>
                  <td><Translated vi={<>luồng {t}</>}>thread {t}</Translated></td>
                  <td>{block * threads + t}</td>
                  <td>{snap.blocks[block].status[t]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ minWidth: 0 }}>
          <div className="mx-label t-shared"><Translated vi="bộ nhớ chia sẻ của mỗi khối: currFrontier_s và bộ đếm của nó">each block's shared memory: currFrontier_s and its counter</Translated></div>
          <div className="row" style={{ gap: '0.7rem', alignItems: 'flex-start', marginBottom: '0.9rem' }}>
            {snap.blocks.map((sb, b) => (
              <div key={b} data-block={b} style={{ opacity: cur && cur.block !== b ? 0.6 : 1 }}>
                <div className="muted" style={{ fontSize: '0.72rem', fontWeight: block === b ? 650 : 400 }}><Translated vi={<>khối {b}</>}>block {b}</Translated></div>
                <div className="mx shared" style={{ gridTemplateColumns: `repeat(${Math.min(capacity, 4)}, auto)` }} aria-label={tr(`currFrontier_s of block ${b}`, `currFrontier_s của khối ${b}`)}>
                  {sb.private_.map((v, k) => {
                    const hot = cur && cur.block === b && cur.op === 'store' && cur.text.includes(`currFrontier_s[${k}]`);
                    return (
                      <span key={k} data-slot={k} className={!sb.written.private_[k] ? 'void' : hot ? 'hot' : 'on'}>
                        {cell(v)}<sub>{k}</sub>
                      </span>
                    );
                  })}
                </div>
                <div style={{ fontSize: '0.72rem', marginTop: '0.25rem', fontFamily: 'var(--sans)' }}>
                  <div data-counter={b}>numCurrFrontier_s{/* i18n-ignore: a variable name */} <b className={sb.written.counter ? 't-shared' : 'muted'}>{cell(sb.counter)}</b></div>
                  <div data-start={b}>currFrontierStartIdx <b className={sb.written.start ? 't-shared' : 'muted'}>{cell(sb.start)}</b></div>
                </div>
              </div>
            ))}
          </div>

          <div className="mx-label t-global">
            <Translated vi="frontier chung currFrontier (bộ nhớ toàn cục (global memory)), bộ đếm">public frontier currFrontier (global memory), counter</Translated> <span data-public-count>{snap.count}</span>
          </div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${Math.min(8, Math.max(1, shownSlots))}, auto)` }} aria-label={tr('public frontier', 'frontier chung')}>
            {snap.pub.slice(0, shownSlots).map((v, j) => {
              const h = snap.how[j];
              const fresh = !!cur && (v !== prev.pub[j] || h?.block !== prev.how[j]?.block);
              return (
                <span key={j} data-public={j} data-via={h?.via ?? ''} className={!h ? 'void' : fresh ? 'hot' : j >= snap.count ? 'bad' : 'on'}
                  title={h ? tr(`written by block ${h.block}, ${h.via === 'overflow' ? tr('straight from the overflow path', 'thẳng từ nhánh tràn') : tr('in the commit', 'trong bước commit')}`, `được ghi bởi khối ${h.block}, ${h.via === 'overflow' ? tr('straight from the overflow path', 'thẳng từ nhánh tràn') : tr('in the commit', 'trong bước commit')}`) : tr('not written', 'chưa được ghi')}>
                  {v === UNWRITTEN ? '' : v}
                  {h ? <sub>{h.via === 'overflow' ? `o${h.block}` : `b${h.block}`}</sub> : null}
                </span>
              );
            })}
          </div>
          <p className="muted" style={{ margin: '0.3rem 0 0.8rem', fontSize: '0.8rem' }}>
            <Translated vi={<>Chữ nhỏ b<i>k</i>: được commit bởi khối <i>k</i>. Chữ nhỏ o<i>k</i>: được gửi thẳng ra bởi nhánh tràn của khối <i>k</i>.</>}>
              Small b<i>k</i>: committed by block <i>k</i>. Small o<i>k</i>: sent straight out by block <i>k</i>'s overflow path.
            </Translated>
          </p>

          <div className="panel" style={{ marginBottom: '0.8rem' }} data-reservations={reserved.map((r) => `${r.block}:${r.start}+${r.count}`).join(' ')}>
            <b><Translated vi="Các lần đặt chỗ, theo thứ tự các khối chạm tới bộ đếm chung">Reservations, in the order blocks reached the public counter</Translated></b>
            {reserved.length ? (
              <ol style={{ margin: '0.3rem 0 0', paddingLeft: '1.2rem' }}>
                {reserved.map((r) => (
                  <li key={r.block}><Translated vi={<>khối {r.block}: {r.count === GARBAGE ? 'CD' : r.count} chỗ từ {r.start}</>}>block {r.block}: {r.count === GARBAGE ? 'CD' : r.count} slot{r.count === 1 ? '' : 's'} from {r.start}</Translated></li>
                ))}
              </ol>
            ) : <div className="muted"><Translated vi="chưa có">none yet</Translated></div>}
            {Object.entries(trace.commitPasses).length && at >= total ? (
              <div style={{ marginTop: '0.4rem' }} data-commit-passes>
                <Translated vi="Địa chỉ commit, theo từng lượt:">Commit addresses, pass by pass:</Translated>{' '}
                {Object.entries(trace.commitPasses).map(([b, passes]) => (
                  <span key={b} style={{ marginRight: '0.6rem' }}><Translated vi="khối">block</Translated> {b}: {passes.map((p) => `[${p.join(', ')}]`).join(' ')}</span>
                ))}
                <div className="muted"><Translated vi="Trong mỗi lượt, các luồng liên tiếp ghi vào các địa chỉ liên tiếp, nên trên GPU các lần ghi của một warp được hợp nhất (coalescing).">In each pass consecutive threads write consecutive addresses, so on a GPU a warp's stores coalesce.</Translated></div>
              </div>
            ) : null}
          </div>

          <div className="panel" style={{ marginBottom: '0.8rem' }} data-tally>
            <b><Translated vi="Các phép atomic trên bộ đếm chung">Atomics on the public counter</Translated></b>
            <div>
              <Translated
                vi={
                  <>
                    đến giờ: <b data-global-atomics>{snap.globalAtomics}</b> ({snap.overflows} lần nối do tràn và{' '}
                    {reserved.length} lần đặt chỗ); Hình 18.12 sẽ thực hiện {newSoFar} lần, mỗi đỉnh mới một lần.
                  </>
                }
              >
                so far: <b data-global-atomics>{snap.globalAtomics}</b> ({snap.overflows} overflow append{snap.overflows === 1 ? '' : 's'} and{' '}
                {reserved.length} reservation{reserved.length === 1 ? '' : 's'}); Fig. 18.12 would have made {newSoFar}, one per new vertex.
              </Translated>
            </div>
            {at >= total ? (
              <div data-final-tally={`${trace.tally.fig1815}/${trace.tally.fig1812}/${trace.tally.overflows}`}>
                <Translated
                  vi={
                    <>
                      cả mức: <b>{trace.tally.fig1815}</b> so với <b>{trace.tally.fig1812}</b>: mỗi khối một lần ({trace.tally.reservations}) cộng
                      mỗi lần tràn một lần ({trace.tally.overflows}).
                    </>
                  }
                >
                  whole level: <b>{trace.tally.fig1815}</b> against <b>{trace.tally.fig1812}</b>: one per block ({trace.tally.reservations}) plus one per
                  overflow ({trace.tally.overflows}).
                </Translated>
              </div>
            ) : null}
          </div>

          <div className="panel" data-races={raced.map((r) => r.sites.join('|')).join(' ')}>
            <b><Translated vi="Kiểm tra race, cả ba thứ tự luồng">Race check, all three thread orders</Translated></b>
            {raced.length === 0 ? (
              <div><span className="chip ok"><Translated vi="không có race">no race</Translated></span></div>
            ) : (
              <ul style={{ margin: '0.3rem 0 0', paddingLeft: '1.2rem' }}>
                {raced.map((r) => (
                  <li key={r.sites.join('|')} data-race={r.sites.join('|')}>
                    <span className="chip fail"><Translated vi="race">race</Translated></span>{' '}
                    {r.sites[0] === r.sites[1]
                      ? <Translated vi={<>dòng {listingLine(shape, r.sites[0])}, {SITE_TEXT[r.sites[0]]}, trong hai {r.example.a.block !== r.example.b.block ? 'khối' : 'luồng'}</>}><>line {listingLine(shape, r.sites[0])}, {SITE_TEXT[r.sites[0]]}, in two {r.example.a.block !== r.example.b.block ? 'blocks' : 'threads'}</></Translated>
                      : <Translated vi={<>dòng {listingLine(shape, r.sites[0])}, {SITE_TEXT[r.sites[0]]}, đối với dòng {listingLine(shape, r.sites[1])}, {SITE_TEXT[r.sites[1]]}</>}><>line {listingLine(shape, r.sites[0])}, {SITE_TEXT[r.sites[0]]}, against line {listingLine(shape, r.sites[1])}, {SITE_TEXT[r.sites[1]]}</></Translated>}
                    {' '}<Translated vi={<>(trên {r.example.addr})</>}>(on {r.example.addr})</Translated>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {at >= total ? (
            <p style={{ margin: '0.9rem 0 0' }} data-outcome={trace.outcome.kind}>
              {trace.outcome.kind === 'correct' ? (
                <span className="chip ok" style={WRAP} data-outcome-chip>
                  <Translated vi={<>Đúng: {trace.outcome.size} đỉnh{trace.outcome.grouped ? ', phần commit của mỗi khối nằm trong một khoảng liên tục' : ''}.</>}>
                    Correct: {trace.outcome.size} vertices{trace.outcome.grouped ? ', each block\'s commit in one contiguous range' : ''}.
                  </Translated>
                </span>
              ) : trace.outcome.kind === 'crash' ? (
                <span className="chip fail" style={WRAP} data-outcome-chip><Translated vi={<>Dừng: {trace.outcome.crash}.</>}>Stopped: {trace.outcome.crash}.</Translated></span>
              ) : (
                <span className="chip fail" style={WRAP} data-outcome-chip><Translated vi={<>Sai: {summary(trace.outcome)}.</>}>Wrong: {summary(trace.outcome)}.</Translated></span>
              )}
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

