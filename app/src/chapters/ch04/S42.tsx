import { useEffect, useMemo, useRef, useState, type RefObject } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  FIG42,
  capacity,
  completionOrder,
  equalDurations,
  schedule,
  stateAt,
  unevenDurations,
  waitingAtLaunch,
  type SchedEvent,
  type Schedule,
} from '../../sim/blocksched';

// The grid of chapter 3's 62 x 76 picture: 4 rows of 5 blocks of 16 x 16 threads (ch. 3, p. 49).
const PICTURE_BLOCKS = 20;
const MAX_BLOCKS = 24;
const clampInt = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, Math.round(Number.isFinite(v) ? v : lo)));

/** One sentence for the event that has just happened. */
function describe(s: Schedule, n: number): string {
  const e: SchedEvent = s.events[n - 1];
  const prev = n >= 2 ? s.events[n - 2] : undefined;
  if (e.kind === 'complete')
    return tr(
      `Time ${e.time}: block ${e.block} finishes on SM ${e.unit} and gives up its slot.`,
      `Thời điểm ${e.time}: khối ${e.block} chạy xong trên SM ${e.unit} và nhường lại chỗ của nó.`,
    );
  if (prev && prev.kind === 'complete' && prev.unit === e.unit && prev.slot === e.slot)
    return tr(
      `Time ${e.time}: block ${e.block}, next on the list, takes the slot on SM ${e.unit} that block ${prev.block} has just left. All of its threads go there together.`,
      `Thời điểm ${e.time}: khối ${e.block}, kế tiếp trong danh sách, lấy chỗ trên SM ${e.unit} mà khối ${prev.block} vừa rời đi. Mọi luồng của nó cùng đến đó một lúc.`,
    );
  return tr(
    `Time ${e.time}: block ${e.block} is assigned to SM ${e.unit}, slot ${e.slot}. All of its threads go there together.`,
    `Thời điểm ${e.time}: khối ${e.block} được giao cho SM ${e.unit}, chỗ ${e.slot}. Mọi luồng của nó cùng đến đó một lúc.`,
  );
}

/**
 * Width of an element in CSS pixels, so a drawing can use it as its viewBox and keep
 * its text at reading size on a phone. (Chapter 1 has the same helper; a shared one
 * would serve both.)
 */
function useWidth(initial: number, min = 280, max = 640): [RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(initial);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const read = () => setW(Math.max(min, Math.min(max, Math.round(el.clientWidth))));
    read();
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, [min, max]);
  return [ref, w];
}

/** A strip per block: when it ran, and on which SM. */
function Strips({ s, n, now }: { s: Schedule; n: number; now: number }) {
  const blocks = s.spec.blocks;
  const started = new Set(s.events.slice(0, n).filter((e) => e.kind === 'assign').map((e) => e.block));
  const [box, W] = useWidth(520);
  const L = 30;
  const row = 16;
  const H = blocks * row + 24;
  const span = Math.max(1, s.makespan);
  const x = (t: number) => L + (t / span) * (W - L - 8);
  const most = Math.max(2, Math.floor((W - L) / 34)); // ticks at least 34 px apart
  const ticks = Array.from({ length: Math.floor(span) + 1 }, (_, i) => i).filter((t) => span <= most || t % Math.ceil(span / most) === 0);
  return (
    <div ref={box}>
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('When each block runs', 'Khi nào mỗi khối chạy')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={0} y2={blocks * row} stroke="var(--rule)" />
          <text x={x(t)} y={blocks * row + 13} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{t}</text>
        </g>
      ))}
      <text x={W - 8} y={blocks * row + 23} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi="thời gian">time</Translated>
      </text>
      {s.placements.map((p) => {
        const y = p.block * row;
        const on = started.has(p.block);
        const upto = Math.min(p.end, Math.max(p.start, now));
        // The label goes inside the strip when it fits, after it when it does not; it is
        // light only where the dark part of the strip is under all of it.
        const LABEL = 30;
        const inside = x(p.end) - x(p.start) >= LABEL + 4;
        const light = inside && x(upto) - x(p.start) >= LABEL + 4;
        const after = x(p.end) + 3 + LABEL <= W;
        const lx = inside ? x(p.start) + 4 : after ? x(p.end) + 3 : x(p.start) - 3;
        return (
          <g key={p.block} data-strip={p.block} data-on={on ? '1' : '0'}>
            <text x={L - 5} y={y + 11.5} fontSize="10.5" textAnchor="end" style={{ fill: on ? 'var(--ink)' : 'var(--ink-3)' }}>{p.block}</text>
            {on ? (
              <>
                <rect x={x(p.start)} y={y + 1.5} width={x(p.end) - x(p.start) - 1} height={row - 3} rx="2" fill="var(--cell)" stroke="var(--ink-3)" />
                <rect x={x(p.start)} y={y + 1.5} width={Math.max(0, x(upto) - x(p.start) - 1)} height={row - 3} rx="2" fill="var(--ink-2)" />
                <text x={lx} y={y + 11.5} fontSize="10" textAnchor={inside || after ? 'start' : 'end'} style={{ fill: light ? 'var(--page)' : 'var(--ink-2)' }}>SM {p.unit}</text>
              </>
            ) : null}
          </g>
        );
      })}
      <line x1={x(now)} x2={x(now)} y1={0} y2={blocks * row} stroke="var(--ink)" strokeWidth="1.4" />
    </svg>
    </div>
  );
}

function BlockStepper() {
  const [units, setUnits] = useState<number>(FIG42.units);
  const [slots, setSlots] = useState<number>(FIG42.slotsPerUnit);
  const [blocks, setBlocks] = useState(PICTURE_BLOCKS);
  const [uneven, setUneven] = useState(false);
  const [n, setN] = useState(0);
  const [playing, setPlaying] = useState(false);

  const s = useMemo(
    () => schedule({ blocks, units, slotsPerUnit: slots, durations: uneven ? unevenDurations(blocks) : equalDurations(blocks) }),
    [blocks, units, slots, uneven],
  );
  const total = s.events.length;
  const at = Math.min(n, total);
  const st = useMemo(() => stateAt(s, at), [s, at]);
  const filled = Math.min(blocks, s.capacity); // events that fill the device at launch
  const order = completionOrder(s);
  const inLaunchOrder = order.every((b, i) => b === i);

  useEffect(() => { setN(0); setPlaying(false); }, [blocks, units, slots, uneven]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) { setPlaying(false); return; }
    const id = window.setTimeout(() => setN((x) => x + 1), 380);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);
  const go = (k: number) => { setPlaying(false); setN(Math.max(0, Math.min(total, k))); };

  return (
    <Bench
      title={tr('Blocks onto SMs', 'Khối lên SM')}
      task={
        <Translated vi="Chạy từng bước qua lần khởi chạy. Liệu bạn có bao giờ có nhiều khối đang chạy hơn SM × chỗ, hay một khối nằm trên hai SM?">
          Step through the launch. Can you ever get more blocks running than SMs × slots, or one block on two SMs?
        </Translated>
      }
      foot={
        <Translated vi="Thời gian của khối tính bằng đơn vị tùy ý. Với thời gian không đều, mỗi khối mất 1, 2, 3 hoặc 4 đơn vị, cố định bằng một seed để lần chạy lặp lại được.">
          Block times are in arbitrary units. With uneven times each block takes 1, 2, 3 or 4 units, fixed by a seed so that the run repeats.
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="SM">SMs</Translated>
          <select value={units} onChange={(e) => setUnits(+e.target.value)} aria-label={tr('SMs', 'SM')}>
            {[1, 2, 3, 4].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Số khối một SM giữ cùng lúc">Blocks an SM holds at once</Translated>
          <select value={slots} onChange={(e) => setSlots(+e.target.value)} aria-label={tr('Blocks an SM holds at once', 'Số khối một SM giữ cùng lúc')}>
            {[1, 2, 3, 4].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Số khối trong lưới">Blocks in the grid</Translated>
          <input type="number" min={1} max={MAX_BLOCKS} step={1} value={blocks} aria-label={tr('Blocks in the grid', 'Số khối trong lưới')}
            onChange={(e) => setBlocks(clampInt(+e.target.value, 1, MAX_BLOCKS))} />
        </label>
        <span className="seg" role="group" aria-label={tr('Block times', 'Thời gian của khối')}>
          <button aria-pressed={!uneven} onClick={() => setUneven(false)}>
            <Translated vi="thời gian khối bằng nhau">equal block times</Translated>
          </button>
          <button aria-pressed={uneven} onClick={() => setUneven(true)}>
            <Translated vi="thời gian khối không đều">uneven block times</Translated>
          </button>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => go(0)} disabled={at === 0}>
          <Translated vi="Bắt đầu lại">Restart</Translated>
        </button>
        <button onClick={() => go(at - 1)} disabled={at === 0}>
          <Translated vi="Lùi lại">Back</Translated>
        </button>
        <button className="primary" onClick={() => go(at + 1)} disabled={at >= total}>
          <Translated vi="Bước tiếp">Step</Translated>
        </button>
        <button onClick={() => go(filled)} disabled={at >= filled}>
          <Translated vi="Lấp đầy thiết bị">Fill the device</Translated>
        </button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Phát')}</button>
        <button onClick={() => go(total)} disabled={at >= total}>
          <Translated vi="Chạy đến cuối">Run to the end</Translated>
        </button>
        <span className="muted" aria-live="polite">
          <Translated vi={<>sự kiện {at} trên {total}</>}>event {at} of {total}</Translated>
        </span>
      </div>
      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}>
        {at === 0 ? (
          <Translated
            vi={
              <>
                Chưa có gì được giao. Lưới có {blocks} khối. Thiết bị có {units} SM, mỗi SM có chỗ cho {slots} khối như vậy,
                nên {s.capacity} khối có thể chạy cùng lúc. Bấm Bước tiếp.
              </>
            }
          >
            Nothing is assigned yet. The grid has {blocks} blocks. The device has {units} SM{units > 1 ? 's' : ''} with
            room for {slots} of these blocks each, so {s.capacity} can be in progress at once. Press Step.
          </Translated>
        ) : (
          describe(s, at)
        )}
        {at >= total ? (
          <Translated
            vi={
              <>
                {' '}Mọi khối đều đã chạy. Chúng kết thúc{' '}
                {inLaunchOrder ? 'theo đúng thứ tự được khởi chạy' : 'theo một thứ tự khác với thứ tự được khởi chạy'}.
              </>
            }
          >
            {' '}Every block has run. They finished {inLaunchOrder ? 'in the order they were launched' : 'in a different order from the one they were launched in'}.
          </Translated>
        ) : null}
      </div>
      <p className="counts" style={{ margin: '0 0 0.9rem' }}>
        <span className="chip" data-k="waiting">
          <Translated vi={<>{st.pending.length} đang chờ</>}>{st.pending.length} waiting</Translated>
        </span>{' '}
        <span className="chip" data-k="resident">
          <Translated vi={<>{st.resident} trên {s.capacity} chỗ đang dùng</>}>
            {st.resident} of {s.capacity} slots in use
          </Translated>
        </span>{' '}
        <span className="chip" data-k="done">
          <Translated vi={<>{st.done.length} đã xong</>}>{st.done.length} finished</Translated>
        </span>
      </p>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 5fr) minmax(0, 6fr)' }}>
        <div>
          <div className="mx-label">
            <Translated vi="Chưa bắt đầu (danh sách của runtime, khối kế tiếp đứng đầu)">
              Not started (the runtime's list, next first)
            </Translated>
          </div>
          <div className="row pending" style={{ gap: '0.3rem', minHeight: '1.9rem', marginBottom: '0.9rem' }}>
            {st.pending.length ? st.pending.map((b) => <span key={b} className="chip">{b}</span>) : <span className="muted">
                <Translated vi="trống">empty</Translated>
              </span>}
          </div>
          <div className="row" style={{ alignItems: 'stretch', gap: '0.7rem', marginBottom: '0.9rem' }}>
            {st.slots.map((unit, u) => (
              <div key={u} className="panel sm" data-sm={u} style={{ padding: '0.5rem 0.6rem', minWidth: '6.4rem' }}>
                <div className="mx-label" style={{ marginBottom: '0.35rem' }}>SM {u}</div>
                {unit.map((b, k) => (
                  <div key={k} data-slot={k} data-block={b ?? ''}
                    style={{
                      border: b == null ? '1px dashed var(--ink-3)' : '1.5px solid var(--ink)',
                      borderRadius: 3, padding: '0.2rem 0.4rem', marginBottom: '0.3rem', minHeight: '2.5rem',
                      background: b == null ? 'transparent' : 'var(--cell)', color: b == null ? 'var(--ink-3)' : 'var(--ink)',
                    }}>
                    {b == null ? tr('free', 'trống') : (
                      <>
                        <b>
                          <Translated vi={<>khối {b}</>}>block {b}</Translated>
                        </b>
                        <div className="muted" style={{ fontSize: '0.7rem', letterSpacing: '0.04em' }}>
                          <Translated vi="t0 t1 t2 … tất cả ở đây">t0 t1 t2 … all here</Translated>
                        </div>
                      </>
                    )}
                  </div>
                ))}
              </div>
            ))}
          </div>
          <div className="mx-label">
            <Translated vi="Đã xong, theo thứ tự kết thúc">Finished, in the order they finished</Translated>
          </div>
          <div className="row finished" style={{ gap: '0.3rem', minHeight: '1.9rem' }}>
            {st.done.length ? st.done.map((b) => <span key={b} className="chip ok">{b}</span>) : <span className="muted">
                <Translated vi="chưa có">none yet</Translated>
              </span>}
          </div>
        </div>
        <div>
          <div className="mx-label">
            <Translated vi="Mỗi khối một dải: chạy khi nào, và ở đâu">One strip per block: when it runs, and where</Translated>
          </div>
          <Strips s={s} n={at} now={st.time} />
        </div>
      </div>
    </Bench>
  );
}

export const questionsS42: Record<string, Question> = {
  'ch04/4.2/q-crossblock': {
    kind: 'choice',
    id: 'ch04/4.2/q-crossblock',
    prompt: (
      <Translated
        vi={
          <>
            Mỗi khối của một kernel ghi tổng phần dữ liệu của riêng nó vào <code>out[blockIdx.x]</code>. Sau đó, trong cùng
            kernel, khối <code>b</code> đọc <code>out[b - 1]</code> để biến các tổng thành tổng cộng dồn. Bài kiểm thử chạy
            đúng trên GPU của tác giả. Kết luận?
          </>
        }
      >
        Each block of a kernel writes the sum of its own slice to <code>out[blockIdx.x]</code>. Then, in the same
        kernel, block <code>b</code> reads <code>out[b - 1]</code> to turn the sums into a running total. The test
        passes on the author's GPU. Verdict?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Đúng: các khối được giao theo thứ tự chỉ số, nên khối b − 1 đã ghi xong trước khi khối b đọc.">
            Correct: blocks are handed out in index order, so block b − 1 has written before block b reads.
          </Translated>
        ),
        why: (
          <Translated vi="Danh sách có thể được giao theo thứ tự chỉ số mà khối b vẫn đến lệnh đọc của nó trước, hoặc chạy cùng lúc trên một SM khác.">
            The list may be handed out in index order and block b can still reach its read first, or run at the same moment on another SM.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Sai: không có gì sắp thứ tự các khối, nên khối b có thể đọc out[b − 1] trước khi khối b − 1 ghi nó.">
            Wrong: nothing orders blocks, so block b may read out[b − 1] before block b − 1 has written it.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chỉ sai trên thiết bị có nhiều SM; trên một SM duy nhất các khối chạy lần lượt.">
            Wrong only on a device with several SMs; on a single SM the blocks run one after another.
          </Translated>
        ),
        why: (
          <Translated vi="Một SM giữ nhiều khối cùng lúc, và không nhà cung cấp nào hứa một thứ tự, kể cả khi chạy tuần tự.">
            One SM holds several blocks at once, and no vendor promises an order even in series.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Đúng khi đặt một __syncthreads() trước lệnh đọc, để lệnh ghi đã hoàn tất.">
            Correct once a __syncthreads() is placed before the read, so that the write has landed.
          </Translated>
        ),
        why: (
          <Translated vi="Rào chắn đó dừng ở ranh giới của khối. Nó không thể bắt một khối chờ một khối khác.">
            That barrier stops at the edge of the block. It cannot make one block wait for another.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một khối được giao bất cứ khi nào có SM còn chỗ, và nó chạy bên cạnh bất cứ thứ gì khác đang thường trú. Vì vậy
            không có thời điểm nào mà "khối b − 1 đã xong" là đúng đối với khối b. Bài kiểm thử chạy đúng là nhờ cách một
            thiết bị tình cờ xếp các khối vào lần đó. Cách sửa là một bước thứ hai bắt đầu sau khi mọi tổng đã được ghi, ví
            dụ một kernel thứ hai khởi chạy sau đó, hoặc một trong các thuật toán scan của chương 11, vốn tồn tại chính vì
            việc này.
          </p>
        }
      >
        <p>
          A block is assigned whenever some SM has room, and it runs beside whatever else is resident. So there is no
          moment at which "block b − 1 is done" is true for block b. The test passes because of how one device happened to
          place the blocks that time. The fix is a second step that starts after every sum is written, for example a
          second kernel launched afterwards, or one of the scan algorithms of chapter 11, which exist for exactly this.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.2, p. 69 and §4.3, p. 73; NVIDIA\'s wording in notes/today/ch04-facts.md A5',
    vi: { source: 'ch. 4 §4.2, p. 69 và §4.3, p. 73; cách diễn đạt của NVIDIA trong notes/today/ch04-facts.md A5' },
  },
  'ch04/4.2/q-waiting': {
    kind: 'numeric',
    id: 'ch04/4.2/q-waiting',
    prompt: (
      <Translated
        vi={
          <>
            Một thiết bị có 40 SM. Với kernel và kích thước khối của bạn, mỗi SM có chỗ cho 6 khối. Bạn khởi chạy một lưới
            1000 khối. Ngay sau khi khởi chạy, còn bao nhiêu khối đang chờ trong danh sách của runtime?
          </>
        }
      >
        A device has 40 SMs. For your kernel and block size, each SM has room for 6 blocks. You launch a grid of 1000
        blocks. How many blocks are still waiting on the runtime's list just after launch?
      </Translated>
    ),
    answer: waitingAtLaunch(1000, capacity(40, 6)),
    unit: 'blocks',
    explain: (
      <Translated
        vi={
          <p>
            {capacity(40, 6)} khối có thể chạy cùng lúc (40 × 6).
            1000 − {capacity(40, 6)} = {waitingAtLaunch(1000, capacity(40, 6))} khối còn lại phải chờ, và mỗi khối bắt đầu
            khi một khối trước đó kết thúc. Con số 6 không phải là thuộc tính của riêng thiết bị: nó phụ thuộc vào những gì
            kernel của bạn đòi hỏi ở một SM, đó là mục 4.7.
          </p>
        }
      >
        <p>
          {capacity(40, 6)} blocks can be in progress at once (40 × 6). The other 1000 − {capacity(40, 6)} ={' '}
          {waitingAtLaunch(1000, capacity(40, 6))} wait, and each of them starts when some earlier block ends. The number
          6 is not a property of the device alone: it depends on what your kernel asks of an SM, which is section 4.7.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.2, p. 69 (the device and grid are ours)',
    vi: { unit: 'khối', source: 'ch. 4 §4.2, p. 69 (thiết bị và lưới là của khóa học này)' },
  },
  'ch04/4.2/q-whole': {
    kind: 'choice',
    id: 'ch04/4.2/q-whole',
    prompt: (
      <Translated vi={<>Vì sao runtime không bao giờ chia một khối ra hai SM?</>}>
        <>Why does the runtime never split one block across two SMs?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một khối là một warp, và phần cứng chạy một warp như một đơn vị, nên không thể chia nó ra.">
            A block is one warp, and the hardware runs a warp as one unit, so it cannot be divided.
          </Translated>
        ),
        why: (
          <Translated vi="Một khối thường gồm nhiều warp. Mục 4.4 sẽ cắt nó ra.">
            A block is usually many warps. Section 4.4 cuts it up.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các luồng của nó có thể gặp nhau tại một rào chắn và chia sẻ bộ nhớ trên chip, và cả hai thứ đó đều nằm trong một SM.">
            Its threads may meet at a barrier and share on-chip memory, and both live inside one SM.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Có thể chia, nhưng chuyển luồng giữa các SM thì chậm, nên bộ lập lịch tránh việc đó khi có thể.">
            It could, but moving threads between SMs is slow, so the scheduler avoids it when it can.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là một bảo đảm của mô hình, không phải sở thích của bộ lập lịch.">
            It is a guarantee of the model, not a preference of the scheduler.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một khối có tối đa 1024 luồng và một SM giữ được nhiều hơn thế, nên câu hỏi không bao giờ xuất hiện.">
            A block has at most 1024 threads and an SM holds more than that, so the question never comes up.
          </Translated>
        ),
        why: (
          <Translated vi="Một thiết bị nhỏ hơn, hoặc một kernel cần nhiều tài nguyên hơn cho mỗi luồng, sẽ khiến câu hỏi xuất hiện ngay.">
            A smaller device, or a kernel that needs more per thread, would make it come up at once.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Việc giao trọn khối là nền tảng của hai dịch vụ trong cùng khối. Bộ nhớ chia sẻ (shared memory) là bộ nhớ trên
            chip của một SM, nên các luồng dùng chung nó phải ở đó. Một rào chắn cần mọi luồng của khối đều có thể đến được,
            nên tất cả chúng phải có tài nguyên trước khi luồng đầu tiên chạy. Chia một khối ra thì không thứ nào trong hai
            thứ đó có thể được cung cấp.
          </p>
        }
      >
        <p>
          Whole-block assignment is what the two same-block services rest on. Shared memory is on-chip memory of one SM,
          so the threads that share it must be there. A barrier needs every thread of the block to be able to arrive, so
          all of them must have their resources before the first one runs. Split a block and neither can be offered.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.2, p. 69 and §4.3, pp. 72–73',
    vi: { source: 'ch. 4 §4.2, p. 69 và §4.3, pp. 72–73' },
  },
};

export function S42() {
  const q = questionsS42;
  const cap = capacity(FIG42.units, FIG42.slotsPerUnit);
  const waiting = waitingAtLaunch(PICTURE_BLOCKS, cap);
  return (
    <>
      <Translated
        vi={
          <p>
            Một lần khởi chạy có thể đòi hỏi nhiều hơn hẳn những gì bất kỳ thiết bị nào có. Bức ảnh 1500 × 2000 của chương 3
            cần 11,750 khối 256 luồng, khoảng ba triệu luồng chỉ từ một câu lệnh. Không thiết bị nào chạy ngần ấy luồng cùng
            lúc. Vì vậy runtime không khởi động cả lưới; nó khởi động phần vừa với thiết bị, rồi giao phần còn lại khi phần
            cứng rảnh ra.
          </p>
        }
      >
        <p>
          A launch can ask for far more than any device has. Chapter 3's 1500 × 2000 picture needed 11,750 blocks of 256
          threads, about three million threads from one statement. No device runs that many at once. So the runtime does
          not start a grid; it starts as much of it as fits, and hands out the rest as hardware comes free.
        </p>
      </Translated>
      <Source section="4.2" pages="69" />
      <Translated
        vi={
          <p>
            Đơn vị được giao là <b>khối (block)</b>. Mọi luồng của một khối được giao cho cùng một SM vào cùng một lúc. Một
            SM có thể nhận nhiều khối cùng lúc, nhiều đến mức tài nguyên của nó cho phép với kernel này; hình của sách vẽ ba
            khối trên mỗi SM trong hai SM.
          </p>
        }
      >
        <p>
          The unit it hands out is the <b>block</b>. All threads of a block are assigned to the same SM at the same time.
          An SM can take several blocks at once, as many as its resources allow for this kernel; the book's picture shows
          three on each of two SMs.
        </p>
      </Translated>
      <Source
        section="4.2"
        pages="69"
        figs={tr('Fig. 4.2, redrawn in the stepper below', 'Fig. 4.2, được vẽ lại trong trình chạy từng bước bên dưới')}
      />
      <Translated
        vi={
          <p>
            Vì vậy một thiết bị có một trần cho số khối đang chạy: số SM nhân với số khối mỗi SM giữ được. Một lưới thường có
            nhiều hơn thế rất nhiều. Runtime giữ một danh sách các khối chưa bắt đầu và giao khối kế tiếp mỗi khi một khối
            đang chạy kết thúc.
          </p>
        }
      >
        <p>
          So a device has a ceiling on blocks in progress: the number of SMs times the blocks each SM holds. A grid
          usually has many more. The runtime keeps a list of the blocks that have not started and assigns the next one
          whenever a block that was running finishes.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              <b>Có vẻ là lỗi in.</b> Sách nói các giới hạn về số khối mỗi SM được bàn ở "Section 4.6". Thực ra chúng nằm ở{' '}
              <a href={href('ch04', '4.7')}>mục 4.7</a>.
            </p>
          }
        >
          <p>
            <b>Apparent erratum.</b> The book says the limits on blocks per SM are discussed in "Section 4.6". They are
            in <a href={href('ch04', '4.7')}>section 4.7</a>.
          </p>
        </Translated>
      </Aside>
      <Predict
        id="ch04/4.2/p-pending"
        prompt={
          <Translated
            vi={
              <>
                Bức ảnh 62 × 76 của chương 3 được phủ bằng một lưới {PICTURE_BLOCKS} khối. Hãy khởi chạy nó trên thiết bị
                đồ chơi trong Fig. 4.2 của sách: {FIG42.units} SM, mỗi SM có chỗ cho {FIG42.slotsPerUnit} khối. Ngay sau
                khi khởi chạy, bao nhiêu khối đang chờ?
              </>
            }
          >
            Chapter 3's 62 × 76 picture was covered by a grid of {PICTURE_BLOCKS} blocks. Launch it on the toy device of
            the book's Fig. 4.2: {FIG42.units} SMs, each with room for {FIG42.slotsPerUnit} blocks. How many blocks are
            waiting right after the launch?
          </Translated>
        }
        unit={tr('blocks', 'khối')}
      >
        {(v) => (
          <Translated
            vi={
              <p>
                <b>{waiting}</b>. Thiết bị chạy {FIG42.units} × {FIG42.slotsPerUnit} = {cap} khối cùng lúc, nên{' '}
                {PICTURE_BLOCKS} − {cap} = {waiting} khối phải chờ.{' '}
                {v === waiting
                  ? 'Bạn đã đúng.'
                  : v === cap
                    ? `${cap} là số khối bắt đầu chạy; câu hỏi là bao nhiêu khối không bắt đầu.`
                    : 'Hãy chạy từng bước bên dưới.'}
              </p>
            }
          >
            <p>
              <b>{waiting}</b>. The device runs {FIG42.units} × {FIG42.slotsPerUnit} = {cap} blocks at once, so{' '}
              {PICTURE_BLOCKS} − {cap} = {waiting} wait.{' '}
              {v === waiting ? 'You had it.' : v === cap ? `${cap} is the number that start; the question was how many do not.` : 'Step through it below.'}
            </p>
          </Translated>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Trình chạy từng bước bên dưới bắt đầu với chính lần khởi chạy đó. Hãy chạy từng bước và xem danh sách khối đang
            chờ vơi dần khi các khối đang chạy kết thúc.
          </p>
        }
      >
        <p>
          The stepper below starts on that launch. Step it and watch the list of waiting blocks drain as running blocks
          finish.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Ở đây một khối là một nguyên tử chiếm một chỗ. "Số chỗ mỗi SM" đại diện cho bất kỳ giới hạn thực nào đang ràng
              buộc (mục 4.7). Khối nhận chỗ trống nào là lựa chọn của chúng tôi: không nhà cung cấp nào công bố chính sách
              này. Thời lượng là bịa ra; không gì ở đây dự đoán thời gian.
            </p>
          }
        >
          <p>
            A block here is an atom that takes one slot. "Slots per SM" stands for whichever real limit binds
            (section 4.7). Which free slot a block gets is our choice: no vendor documents the policy. Durations are
            made up; nothing here predicts time.
          </p>
        </Translated>
      </Simplification>
      <BlockStepper />
      <Translated
        vi={
          <p>
            Có ba điều cần rút ra. Số khối đang chạy không bao giờ vượt quá SM × chỗ, dù danh sách dài đến đâu. Một khối nằm
            trong một chỗ của một SM từ bước đầu tiên đến bước cuối cùng của nó. Và một khi các khối mất thời gian khác nhau,
            thứ tự kết thúc không còn là thứ tự bắt đầu, nên không gì ở phần sau chương trình của bạn được phép phụ thuộc
            vào thứ tự nào trong hai thứ tự đó.
          </p>
        }
      >
        <p>
          Three things to take from it. The count of blocks in progress never passes SMs × slots, however long the list.
          A block is in one slot of one SM from its first step to its last. And once blocks take different times, the
          order in which they finish is no longer the order in which they started, so nothing later in your program may
          depend on either.
        </p>
      </Translated>
      <Ask q={q['ch04/4.2/q-waiting']} label={tr('Work it out', 'Tự tính')} />

      <h2>
        <Translated vi="Một khối được gì khi được giao trọn vẹn">What a block gets for being assigned whole</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Vì các luồng của một khối ở trên một SM vào cùng một lúc, chúng có thể làm hai việc mà các luồng của những khối
            khác nhau không làm được. Chúng có thể chờ nhau tại một <b>rào chắn (barrier)</b>, là nội dung của{' '}
            <a href={href('ch04', '4.3')}>mục tiếp theo</a>. Và chúng có thể trao đổi dữ liệu qua <b>bộ nhớ chia sẻ (shared memory)</b>,
            một bộ nhớ nhanh bên trong SM, là nội dung của chương 5. Cả hai đều không tồn tại giữa các khối, vì lý do mà
            trình chạy từng bước cho thấy: hai khối của cùng một lưới có thể nằm trên các SM khác nhau, hoặc hoàn toàn không
            chạy cùng lúc.
          </p>
        }
      >
        <p>
          Because the threads of a block are on one SM at one time, they can do two things that threads of different
          blocks cannot. They can wait for each other at a <b>barrier</b>, which is{' '}
          <a href={href('ch04', '4.3')}>the next section</a>. And they can exchange data through <b>shared memory</b>, a
          fast memory inside the SM, which is chapter 5. Neither exists across blocks, for the reason the stepper shows:
          two blocks of one grid may be on different SMs, or not running at the same time at all.
        </p>
      </Translated>
      <Source section="4.2" pages="69" />
      <Today checked="2026-11-20 (notes/today/ch04-facts.md, A5)">{/* i18n-ignore: a date and a citation */}
        <Translated
          vi={
            <p>
              Hướng dẫn của NVIDIA nói một khối chạy đến khi hoàn tất trên SM của nó "trong hầu hết các trường hợp": với
              dynamic parallelism, một khối có thể bị tạm dừng và đưa ra bộ nhớ. Hướng dẫn đó cũng nói rằng khối nào chạy trên
              SM nào thì không thể điều khiển cũng không thể truy vấn, và không có thứ tự nào được hứa.
            </p>
          }
        >
          <p>
            NVIDIA's guide says a block runs to completion on its SM "in most cases": with dynamic parallelism a block can
            be suspended to memory. It also says that which block runs on which SM can be neither controlled nor queried,
            and that no order is promised.
          </p>
        </Translated>
      </Today>
      <Ask q={q['ch04/4.2/q-whole']} />
      <Ask q={q['ch04/4.2/q-crossblock']} />

      <h2>
        <Translated vi="Cụm khối luồng (thread block cluster)">Thread block clusters</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Từ thế hệ Hopper, CUDA có thêm một cấp tùy chọn nằm giữa khối và lưới: <b>cụm (cluster)</b>, một nhóm khối. SM
            đối với khối thế nào thì GPC đối với cụm như thế: các khối của một cụm được lập lịch cùng nhau trên một GPC. Đổi
            lại, các luồng của chúng có thể gặp nhau tại một rào chắn toàn cụm và có thể truy cập bộ nhớ chia sẻ của nhau, mà
            sách gọi là distributed shared memory.
          </p>
        }
      >
        <p>
          With the Hopper generation CUDA gained an optional level between block and grid: the <b>cluster</b>, a group of
          blocks. What an SM is to a block, a GPC is to a cluster: the blocks of a cluster are scheduled together on one
          GPC. In return their threads can meet at a cluster-wide barrier and can reach each other's shared memory, which
          the book calls distributed shared memory.
        </p>
      </Translated>
      <Source section="4.2" pages="69–70" />
      <Translated
        vi={
          <p>
            Ở đây sách chỉ mô tả cụm bằng lời và không đưa ra mã; chương 5 sẽ quay lại với chúng. Với chương này, chỉ cần biết
            rằng cấp này tồn tại và là tùy chọn.
          </p>
        }
      >
        <p>
          The book describes clusters in prose only here and gives no code; chapter 5 comes back to them. For this
          chapter it is enough to know that the level exists and that it is optional.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Trên CUDA, cụm cần compute capability 9.0 trở lên. HIP 7.17 có tài liệu về một <code>cluster_group</code> và
              một cờ <code>clusterLaunch</code> theo từng thiết bị; những thiết bị AMD nào bật cờ này thì unverified. Metal
              không có cấp nào như vậy (notes/today/ch04-facts.md, A7, B11, C3).
            </p>
          }
        >
          <p>
            On CUDA, clusters need compute capability 9.0 or higher. HIP 7.17 documents a <code>cluster_group</code> and a
            per-device <code>clusterLaunch</code> flag; which AMD devices set it is unverified. Metal has no such level
            (notes/today/ch04-facts.md, A7, B11, C3).
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Trên các backend khác">On other backends</Translated>
      </h2>
      <Translated
        vi={
          <p>
            AMD công bố cùng bảo đảm đó và cùng sự vắng mặt đó cho HIP: mọi warp của một khối thực thi trên cùng một compute
            unit, và cả cách ánh xạ khối vào compute unit lẫn thứ tự của chúng đều không được định nghĩa. Apple có tài liệu
            về threadgroup, rào chắn và bộ nhớ của chúng, nhưng không nói về đơn vị nào mà một threadgroup được đặt lên, và
            chúng tôi không tìm thấy khẳng định nào về thứ tự các threadgroup chạy. Ở đó cũng hãy coi thứ tự là không được
            quy định.
          </p>
        }
      >
        <p>
          AMD documents the same guarantee and the same absence of one for HIP: all warps of a block execute on the same
          compute unit, and neither the mapping of blocks to compute units nor their order is defined. Apple documents
          threadgroups, their barrier and their memory, but no unit that a threadgroup is placed on, and we found no
          statement about the order in which threadgroups run. Treat order as unspecified there too.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cách diễn đạt của AMD nằm trong notes/today/ch04-facts.md, B2. Câu về Metal ghi nhận một sự vắng mặt trong các
              tài liệu đã đọc cho khóa học này (C1, C2): unverified.
            </p>
          }
        >
          <p>
            AMD's wording is in notes/today/ch04-facts.md, B2. The Metal sentence reports an absence in the documents
            read for this course (C1, C2): unverified.
          </p>
        </Translated>
      </Beyond>
    </>
  );
}
