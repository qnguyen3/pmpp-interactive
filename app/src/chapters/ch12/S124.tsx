import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { href } from '../../lib/router';
import { region } from '../../sim/histpriv';
import {
  BLOCK_ORDER_LABEL,
  BLOCKS,
  BOOK_KEEP,
  CORRECT,
  GARBAGE,
  N,
  SCHEDULES,
  SCHEDULE_LABEL,
  SITE_TEXT,
  THREADS,
  UNWRITTEN,
  VARIANT,
  blockOrderOf,
  keyName,
  keyValue,
  kernelBody,
  outcomeTable,
  publicAtomics,
  simulate,
  statementLines,
  summary,
  type BlockOrder,
  type Schedule,
  type Shape,
  type Snapshot,
  type Step,
} from '../../sim/privfilter';

// Section 12.4: the privatised unstable filter (Figs. 12.5 and 12.6, pp. 295-297).
// The stepper and the outcome table run on src/sim/privfilter.ts, whose kernels are
// labs/ch12/snippets/s124_privatised_variants.cu (pinned by privfilter.test.ts, which also
// runs that file on the emulator and its race flavour).

const TIERS: Record<string, Tier> = {
  output_s: 'shared', outputSize_s: 'shared', blockStart: 'shared', j_s: 'shared', outputSize_s_ref: 'shared',
  input: 'global', output: 'global', outputSize: 'global', outputSize_ref: 'global',
  i: 'reg', val: 'reg', slot: 'reg', j: 'reg', tid: 'reg', count: 'reg',
};

const cellText = (v: number) => (v === GARBAGE ? 'CD' : v === UNWRITTEN ? '' : keyName(v));

/** Public slots whose value changed in this step. */
function changed(prev: Snapshot, now: Snapshot): Set<number> {
  const s = new Set<number>();
  now.output.forEach((v, j) => {
    if (v !== prev.output[j]) s.add(j);
  });
  return s;
}

function Stepper() {
  const [keep, setKeep] = useState<boolean[]>([...BOOK_KEEP]);
  const [schedule, setSchedule] = useState<Schedule>('forward');
  const [order, setOrder] = useState<BlockOrder>('book');
  const [seed, setSeed] = useState(1);
  const [shape, setShape] = useState<Shape>(CORRECT);
  const [n, setN] = useState(0);
  const [playing, setPlaying] = useState(false);

  const blockOrder = useMemo(() => blockOrderOf(order, seed), [order, seed]);
  const locale = useLocale();
  const trace = useMemo(() => simulate(shape, schedule, keep, blockOrder), [shape, schedule, keep, blockOrder, locale]);
  const total = trace.steps.length;
  const at = Math.min(n, total);
  const cur: Step | undefined = at > 0 ? trace.steps[at - 1] : undefined;
  const snap = cur ? cur.after : trace.initial;
  const prev = at > 1 ? trace.steps[at - 2].after : trace.initial;
  const fresh = cur ? changed(prev, snap) : new Set<number>();
  const block = cur ? cur.block : blockOrder[0];
  const listing = kernelBody(shape, true);
  const kept = keep.filter(Boolean).length;

  useEffect(() => setN(0), [shape, schedule, keep, blockOrder]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setN((x) => x + 1), 380);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);

  const nextBarrier = () => {
    for (let k = at; k < total; k++) if (trace.steps[k].op === 'release') return setN(k + 1);
    setN(total);
  };
  const toggle = (i: number) => setKeep((k) => k.map((x, j) => (j === i ? !x : x)));
  const flag = (key: keyof Shape, label: string) => (
    <label className="check">
      <input type="checkbox" checked={shape[key]} onChange={(e) => setShape({ ...shape, [key]: e.target.checked })} /> {label}
    </label>
  );
  const keptValues = new Set(keep.flatMap((k, i) => (k ? [keyValue(i, true)] : [])));
  const seen = new Map<number, number>();
  for (const v of snap.output) seen.set(v, (seen.get(v) ?? 0) + 1);

  return (
    <Bench
      title={tr('Step through the privatised filter', 'Chạy từng bước bộ lọc riêng tư hóa')}
      task={tr(
        'Find which barrier protects which step, then make one defect fail in one order and pass in another.',
        'Tìm xem rào chắn nào bảo vệ bước nào, rồi làm cho một lỗi thất bại ở thứ tự này và vượt qua ở thứ tự khác.',
      )}
      foot={
        <Translated
          vi={
            <>
              Bốn khối, mỗi khối bốn luồng, một khóa mỗi luồng, mười sáu khóa của sách. Bấm vào một khóa để giữ hoặc bỏ nó.
              Bộ nhớ chia sẻ bắt đầu bằng các byte 0xCD (hiển thị là CD), như trên trình giả lập; một ô công khai trống là ô
              chưa bao giờ được ghi. Con số nhỏ trong một ô công khai là khối đã ghi nó; một ô được đánh dấu khi nó chứa một
              khóa hai lần, không chứa khóa được giữ nào, hoặc nằm quá bộ đếm công khai.
            </>
          }
        >
          <>
            Four blocks of four threads, one key per thread, the book's sixteen keys. Click a key to keep or drop it.
            Shared memory starts as 0xCD bytes (shown as CD), as on the emulator; an empty public slot was never written.
            The small number in a public slot is the block that wrote it; a slot is marked when it holds a key twice, holds no
            kept key, or lies past the public counter.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Thứ tự luồng trong một khối">Thread order in a block</Translated>
          <select value={schedule} aria-label={tr('Thread order', 'Thứ tự luồng')} onChange={(e) => setSchedule(e.target.value as Schedule)}>
            {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Thứ tự các khối chạy và đặt trước">Order in which blocks run and reserve</Translated>
          <select value={order} aria-label={tr('Block order', 'Thứ tự khối')} onChange={(e) => setOrder(e.target.value as BlockOrder)}>
            {(['book', 'in-order', 'random'] as BlockOrder[]).map((o) => <option key={o} value={o}>{BLOCK_ORDER_LABEL[o]}</option>)}
          </select>
        </label>
        {order === 'random' ? <button className="quiet" onClick={() => setSeed((s) => s + 1)}><Translated vi="Một thứ tự ngẫu nhiên khác">Another random order</Translated></button> : null}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        {flag('barrier1', tr('barrier 1', 'rào chắn 1'))}
        {flag('barrier2', tr('barrier 2', 'rào chắn 2'))}
        {flag('barrier3', tr('barrier 3', 'rào chắn 3'))}
        {flag('atomicCounter', tr('private counter is atomic', 'bộ đếm riêng là atomic'))}
        <button className="quiet" onClick={() => { setShape(CORRECT); setKeep([...BOOK_KEEP]); }}><Translated vi="Đặt lại">Reset</Translated></button>
      </div>

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setN(0); }} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => { setPlaying(false); setN(at - 1); }} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => { setPlaying(false); setN(at + 1); }} disabled={at >= total}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => { setPlaying(false); nextBarrier(); }} disabled={at >= total}><Translated vi="Chạy tới rào chắn kế tiếp">Run to next barrier</Translated></button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Phát')}</button>
        {trace.firstFailure >= 0 ? (
          <button onClick={() => { setPlaying(false); setN(trace.firstFailure + 1); }}><Translated vi="Tới lỗi đầu tiên">Go to the first failure</Translated></button>
        ) : null}
        <span className="muted" aria-live="polite"><Translated vi={<>bước {at} / {total}</>}>step {at} of {total}</Translated></span>
      </div>

      <div className={'narrate' + (cur?.violation ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {cur ? (
          <>
            {cur.text}
            {cur.violation ? <><br /><b className="t-fail"><Translated vi="Lỗi:">Failure:</Translated> {cur.violation.message}.</b></> : null}
            {cur.note ? <><br /><span className="muted">{cur.note}</span></> : null}
          </>
        ) : (
          <Translated
            vi={
              <>
                Chưa có gì chạy. {kept} trong {N} khóa qua cond(). Các khối chạy lần lượt theo thứ tự{' '}
                {blockOrder.join(', ')}; trong một khối, một luồng chạy cho đến khi gặp rào chắn, rồi luồng kế tiếp tiếp quản. Bấm
                Bước.
              </>
            }
          >
            <>
              Nothing has run yet. {kept} of the {N} keys pass cond(). Blocks run one after another in the order{' '}
              {blockOrder.join(', ')}; inside a block, a thread runs until it reaches a barrier, then the next thread
              takes over. Press Step.
            </>
          </Translated>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 7fr) minmax(0, 5fr)' }}>
        <div>
          <Code tabs={[{ label: tr('kernel body (BLOCK_DIM = 4)', 'thân kernel (BLOCK_DIM = 4)'), code: listing, tiers: TIERS }]} highlight={cur ? statementLines(shape, true, cur.line) : []} />
          <table className="threads" aria-label={tr('Threads of the current block', 'Các luồng của khối hiện tại')}>
            <thead>
              <tr><th><Translated vi={<>khối {block}</>}>block {block}</Translated></th><th><Translated vi="khóa">key</Translated></th><th>cond()</th><th><Translated vi="đang làm">doing</Translated></th></tr>
            </thead>
            <tbody>
              {Array.from({ length: THREADS }, (_, t) => {
                const i = block * THREADS + t;
                return (
                  <tr key={t} className={cur?.thread === t ? 'cur' : ''}>
                    <td><Translated vi={<>luồng {t}</>}>thread {t}</Translated></td>
                    <td>k{i}</td>
                    <td>{keep[i] ? tr('kept', 'được giữ') : tr('dropped', 'bị bỏ')}</td>
                    <td>{snap.blocks[block].status[t]}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div>
          <div className="mx-label t-global"><Translated vi="đầu vào (bộ nhớ toàn cục): bấm vào một khóa để giữ hoặc bỏ nó">input (global memory): click a key to keep or drop it</Translated></div>
          <div className="row" style={{ gap: '0.5rem', marginBottom: '0.9rem' }}>
            {Array.from({ length: BLOCKS }, (_, b) => (
              <div key={b}>
                <div className="muted" style={{ fontSize: '0.72rem' }}><Translated vi={<>khối {b}</>}>block {b}</Translated></div>
                <div className="mx global" style={{ gridTemplateColumns: `repeat(${THREADS}, auto)` }} data-input-block={b}>
                  {Array.from({ length: THREADS }, (_, t) => {
                    const i = b * THREADS + t;
                    const hot = cur && cur.block === b && cur.thread === t && (cur.op === 'load' || cur.op === 'count' || cur.op === 'store');
                    return (
                      <span
                        key={i} role="button" tabIndex={0} aria-pressed={keep[i]} data-key={i}
                        className={hot ? 'hot' : keep[i] ? 'on' : ''}
                        style={{ cursor: 'pointer', fontWeight: keep[i] ? 650 : 400 }}
                        onClick={() => toggle(i)}
                        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(i); } }}
                        title={keep[i] ? tr('kept: click to drop', 'được giữ: bấm để bỏ') : tr('dropped: click to keep', 'bị bỏ: bấm để giữ')}
                      >
                        k{i}
                      </span>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>

          <div className="mx-label t-shared"><Translated vi="bộ nhớ chia sẻ của mỗi khối">each block's shared memory</Translated></div>
          <div className="row" style={{ gap: '0.7rem', alignItems: 'flex-start', marginBottom: '0.9rem' }}>
            {snap.blocks.map((sb, b) => (
              <div key={b} data-shared-block={b} style={{ opacity: cur && cur.block !== b ? 0.6 : 1 }}>
                <div className="muted" style={{ fontSize: '0.72rem', fontWeight: cur?.block === b ? 650 : 400 }}><Translated vi={<>khối {b}</>}>block {b}</Translated></div>
                <div className="mx shared" style={{ gridTemplateColumns: `repeat(${THREADS}, auto)` }} aria-label={tr(`output_s of block ${b}`, `output_s của khối ${b}`)}>
                  {sb.output_s.map((v, k) => (
                    <span key={k} data-slot={k} data-value={cellText(v)} className={!sb.written.output_s[k] ? 'void' : cur?.block === b && cur.op === 'store' && cur.text.includes(`output_s[${k}]`) ? 'hot' : 'on'}>
                      {cellText(v)}<sub>{k}</sub>
                    </span>
                  ))}
                </div>
                <div style={{ fontSize: '0.72rem', marginTop: '0.25rem', fontFamily: 'var(--sans)' }}>
                  <div data-counter={b}>outputSize_s{/* i18n-ignore: variable name */} <b className={sb.written.outputSize_s ? 't-shared' : 'muted'}>{sb.written.outputSize_s ? sb.outputSize_s : 'CD'}</b></div>
                  <div data-start={b}>blockStart <b className={sb.written.blockStart ? 't-shared' : 'muted'}>{sb.written.blockStart ? sb.blockStart : 'CD'}</b></div>
                </div>
              </div>
            ))}
          </div>

          <div className="mx-label t-global">
            <Translated vi="đầu ra công khai (bộ nhớ toàn cục), bộ đếm">public output (global memory), counter</Translated>{' '}
            <span data-public-size>{snap.outputSize}</span>
          </div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(8, auto)` }} aria-label={tr('public output', 'đầu ra công khai')}>
            {snap.output.map((v, j) => {
              // A slot is suspect if it holds no kept key, a key that appears twice, or lies past the public counter.
              const bad = v !== UNWRITTEN && (!keptValues.has(v) || (seen.get(v) ?? 0) > 1 || j >= snap.outputSize);
              return (
                <span key={j} data-public={j} data-value={cellText(v)} className={fresh.has(j) ? 'hot' : bad ? 'bad' : v === UNWRITTEN ? 'void' : 'on'}>
                  {cellText(v)}
                  {snap.writer[j] >= 0 ? <sub>b{snap.writer[j]}</sub> : null}
                </span>
              );
            })}
          </div>
          {at >= total ? (
            <p style={{ margin: '0.9rem 0 0' }} data-outcome={trace.outcome.kind}>
              {trace.outcome.kind === 'correct' ? (
                <span className="chip ok"><Translated vi={<>Đúng: {trace.outcome.size} khóa, danh sách của mỗi khối nằm trong một đoạn liền kề.</>}>Correct: {trace.outcome.size} keys, each block's list in one contiguous chunk.</Translated></span>
              ) : trace.outcome.kind === 'crash' ? (
                <span className="chip fail"><Translated vi="Dừng:">Stopped:</Translated> {trace.outcome.crash}.</span>
              ) : (
                <span className="chip fail"><Translated vi="Sai:">Wrong:</Translated> {summary(trace.outcome)}.</span>
              )}
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

function Outcomes() {
  const locale = useLocale();
  const table = useMemo(() => outcomeTable(), [locale]);
  return (
    <Bench
      title={tr('Every order at once', 'Mọi thứ tự cùng lúc')}
      task={tr(
        'Before you read the table: which row passes every emulator column and is still wrong?',
        'Trước khi đọc bảng: hàng nào vượt qua mọi cột của trình giả lập mà vẫn sai?',
      )}
      foot={
        <Translated
          vi={
            <>
              Mỗi ô chạy toàn bộ lần khởi chạy trong mô hình ở trên với các khóa của sách và các khối theo thứ tự của trình
              giả lập. Ba thứ tự của trình giả lập cho đúng các kết quả này (privfilter.test.ts chạy các kernel trên nó). Cột
              cuối là phép kiểm tra race của mô hình: hai lần truy cập vào cùng một vị trí là race khi một trong hai ghi, chúng
              không cùng là atomic, và không có rào chắn hay thứ tự chương trình nào đặt cái này trước cái kia.
            </>
          }
        >
          <>
            Each cell runs the whole launch in the model above with the book's keys and blocks in the emulator's order. The
            emulator's three orders give exactly these results (privfilter.test.ts runs the kernels on it). The last
            column is the model's race check: two accesses to one location race when one of them writes, they are not both
            atomic, and no barrier or program order puts one before the other.
          </>
        </Translated>
      }
    >
      <table className="data" style={{ background: 'var(--page)', margin: 0 }}>
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}><Translated vi="Kernel">Kernel</Translated></th>
            {SCHEDULES.map((s) => <th key={s}>{SCHEDULE_LABEL[s]}</th>)}
            <th><Translated vi="kiểm tra race">race check</Translated></th>
          </tr>
        </thead>
        <tbody>
          {table.map((r) => (
            <tr key={r.variant} data-variant={r.variant}>
              <td style={{ paddingLeft: '0.6rem' }}>{VARIANT[r.variant].label}</td>
              {SCHEDULES.map((s) => {
                const o = r.cells[s];
                return (
                  <td key={s} data-outcome={o.kind}>
                    {o.kind === 'correct' ? <span className="chip ok"><Translated vi="đúng">correct</Translated></span> : <span className="chip fail">{o.kind === 'crash' ? tr('stopped', 'dừng') : tr('wrong', 'sai')}</span>}
                    {o.kind !== 'correct' ? <div className="muted">{summary(o)}</div> : null}
                  </td>
                );
              })}
              <td data-race={r.races.length ? 'reported' : 'none'}>
                {r.races.length ? <span className="chip fail"><Translated vi="có báo race">race reported</Translated></span> : <span className="chip ok"><Translated vi="không có">none</Translated></span>}
                {r.races.slice(0, 2).map((x) => (
                  <div key={x.sites.join('|')} className="muted">
                    {x.sites[0] === x.sites[1]
                      ? tr(
                          `${SITE_TEXT[x.sites[0]]}, in two ${x.example.a.block !== x.example.b.block ? 'blocks' : 'threads'}`,
                          `${SITE_TEXT[x.sites[0]]}, trong hai ${x.example.a.block !== x.example.b.block ? 'khối' : 'luồng'}`,
                        )
                      : tr(`${SITE_TEXT[x.sites[0]]} and ${SITE_TEXT[x.sites[1]]}`, `${SITE_TEXT[x.sites[0]]} và ${SITE_TEXT[x.sites[1]]}`)}
                  </div>
                ))}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Bench>
  );
}

const PA = publicAtomics(2 ** 20, 256, 0.01);

export const questionsS124: Record<string, Question> = {
  'ch12/12.4/q-scope': {
    kind: 'choice',
    id: 'ch12/12.4/q-scope',
    prompt: (
      <Translated
        vi={
          <>
            <code>atomic_ref</code> của bộ đếm riêng có thể dùng <code>cuda::thread_scope_device</code> thay cho{' '}
            <code>cuda::thread_scope_block</code> không?
          </>
        }
      >
        <>
          Could the private counter's <code>atomic_ref</code> use <code>cuda::thread_scope_device</code> instead of{' '}
          <code>cuda::thread_scope_block</code>?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không: phạm vi device chỉ áp dụng cho bộ nhớ toàn cục, nên không thể dùng cho một bộ đếm nằm trong bộ nhớ chia sẻ.">
            No: device scope applies to global memory only, so it cannot be used on a counter that lives in shared memory.
          </Translated>
        ),
        why: (
          <Translated vi="Phạm vi (scope) cho biết những luồng nào phải thấy phép toán một cách nhất quán, không phải nó được chạm vào bộ nhớ nào.">
            Scope says which threads must see the operation consistently, not which memory it may touch.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không: với phạm vi device, các luồng của khối không còn thấy các lần tăng của nhau theo thứ tự, nên chúng sẽ gặp race.">
            No: with device scope the block's threads no longer see each other's increments in order, so they would race.
          </Translated>
        ),
        why: (
          <Translated vi="Một phạm vi rộng hơn bao gồm mọi luồng mà phạm vi hẹp hơn bao gồm, nên không có gì đã có thứ tự lại mất thứ tự.">
            A wider scope includes every thread a narrower one does, so nothing that was ordered stops being ordered.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Có. Phạm vi block là phạm vi đúng hẹp nhất, không phải là phạm vi duy nhất.">
            Yes. Block scope is the narrowest correct scope, not the only one.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Phạm vi gọi tên tập các luồng phải quan sát phép atomic một cách nhất quán. Mọi luồng chạm vào{' '}
            <code>outputSize_s</code> đều nằm trong khối, nên phạm vi block là đủ, và đó là lựa chọn đúng hẹp nhất, cũng là lựa
            chọn của sách (p. 296). Phạm vi device bao gồm khối, nên nó cũng đúng. Phạm vi rộng hơn có tốn hơn trên một GPU cụ
            thể hay không thì NVIDIA không ghi nhận cho trường hợp này: hướng dẫn của họ so sánh phạm vi block với phạm vi system
            (ch09-facts A2). Nếu điều đó quan trọng, hãy đo.
          </p>
        }
      >
        <p>
          The scope names the set of threads that must observe the atomic consistently. Every thread that touches{' '}
          <code>outputSize_s</code> is in the block, so block scope is enough, and it is the narrowest correct choice, which is
          the one the book makes (p. 296). Device scope includes the block, so it is also correct. Whether the wider scope
          costs more on a given GPU is not something NVIDIA documents for this case: its guide compares block scope with
          system scope (ch09-facts A2). If it matters, measure it.
        </p>
      </Translated>
    ),
    source: 'ch. 12 §12.4, p. 296; notes/today/ch09-facts.md A2',
    vi: { source: 'ch. 12 §12.4, p. 296; notes/today/ch09-facts.md, mục A2' },
  },
  'ch12/12.4/q-barrier2': {
    kind: 'choice',
    id: 'ch12/12.4/q-barrier2',
    prompt: (
      <Translated
        vi={
          <>
            Bỏ rào chắn giữa bước lọc và lần đặt trước công khai của luồng 0. Với lịch chạy theo thứ tự của trình giả lập (luồng
            0 chạy cho đến khi gặp rào chắn, rồi tới luồng 1, và cứ thế), điều gì hỏng?
          </>
        }
      >
        <>
          Remove the barrier between the filtering step and thread 0's public reservation. Under the emulator's in-order
          schedule (thread 0 runs until it reaches a barrier, then thread 1, and so on), what goes wrong?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không gì cả: bước commit vẫn chờ ở rào chắn 3, nên mọi khóa đã nằm trong danh sách riêng trước khi có gì được chép.">
            Nothing: the commit still waits at barrier 3, so every key is in the private list before anything is copied.
          </Translated>
        ),
        why: (
          <Translated vi="Rào chắn 3 sắp thứ tự cho lần đọc blockStart. Nó không thể đổi số đếm mà luồng 0 đã dùng để đặt trước.">
            Barrier 3 orders the read of blockStart. It cannot change the count thread 0 already used for the reservation.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Luồng 0 đặt trước khi các luồng khác kịp đếm, nên khối đặt trước quá ít ô công khai và đoạn của nó chồng lên đoạn của khối kế tiếp.">
            Thread 0 reserves before the other threads have counted, so the block reserves too few public slots and its chunk overlaps the next block's.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Danh sách riêng chưa đầy đủ khi được chép, vì các luồng chạy sau luồng 0 chưa lưu khóa của mình.">
            The private list is incomplete when it is copied, because threads that run after thread 0 have not stored their keys yet.
          </Translated>
        ),
        why: (
          <Translated vi="Việc chép diễn ra sau rào chắn 3, lúc đó mọi luồng đã lưu khóa của mình. Danh sách đầy đủ; phần đặt trước bị làm quá nhỏ.">
            The copy happens after barrier 3, by which time every thread has stored its key. The list is complete; the reservation was made too small.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Theo thứ tự, luồng 0 chạy trước: nó đếm khóa của chính nó (nếu được giữ), rồi đi thẳng tới bước đặt trước và cộng
            bất cứ giá trị nào <code>outputSize_s</code> đang giữ lúc đó. Với các khóa của sách, luồng 0 của khối 0 không giữ gì
            và đặt trước 0 ô; luồng 0 của khối 1 đặt trước 1 ô cho hai khóa. Bước commit sau đó chép đầy đủ các danh sách riêng
            vào các phạm vi chồng lên nhau, các khóa ghi đè lên nhau, và bộ đếm công khai dừng ở 2 thay vì 8. Đảo ngược, luồng 0
            chạy cuối và cùng kernel đó vượt qua.
          </p>
        }
      >
        <p>
          In order, thread 0 runs first: it counts its own key (if kept), then goes straight to the reservation and adds
          whatever <code>outputSize_s</code> holds at that moment. With the book's keys, block 0's thread 0 keeps nothing and
          reserves 0 slots; block 1's reserves 1 for two keys. The commit later copies the full private lists into ranges that
          overlap, keys overwrite each other, and the public counter ends at 2 instead of 8. Reversed, thread 0 runs last
          and the same kernel passes.
        </p>
      </Translated>
    ),
    source: 'ch. 12 §12.4, pp. 296–297; src/sim/privfilter.ts',
    vi: { source: 'ch. 12 §12.4, pp. 296–297; mô hình src/sim/privfilter.ts' },
  },
  'ch12/12.4/q-barrier3': {
    kind: 'choice',
    id: 'ch12/12.4/q-barrier3',
    prompt: (
      <Translated vi={<>Chỉ bỏ rào chắn sau khi luồng 0 ghi <code>blockStart</code>. Thứ tự luồng nào của trình giả lập làm lộ lỗi?</>}>
        <>Remove only the barrier after thread 0 writes <code>blockStart</code>. Which of the emulator's thread orders exposes it?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Chỉ theo thứ tự.">
            In order only.
          </Translated>
        ),
        why: (
          <Translated vi="Theo thứ tự, luồng 0 chạy trước và đã ghi blockStart trước khi bất kỳ luồng nào khác đọc nó.">
            In order, thread 0 runs first and has written blockStart before any other thread reads it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ đảo ngược.">
            Reversed only.
          </Translated>
        ),
        why: (
          <Translated vi="Đảo ngược có làm lộ lỗi, nhưng thứ tự xáo trộn cũng vậy: ở đó cũng có một luồng có khóa cần chép chạy trước luồng 0.">
            Reversed does expose it, but so does the shuffled order: there too a thread with a key to copy runs before thread 0.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Đảo ngược và xáo trộn; theo thứ tự thì vượt qua.">
            Reversed and shuffled; in order passes.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Cả ba: ở mọi thứ tự đều có luồng đọc blockStart trước khi luồng 0 ghi nó.">
            All three: in every order some thread reads blockStart before thread 0 writes it.
          </Translated>
        ),
        why: (
          <Translated vi="Theo thứ tự, luồng 0 tình cờ chạy trước.">
            In order, thread 0 happens to go first.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đảo ngược, các luồng 3, 2 và 1 tới bước commit trước khi luồng 0 đặt trước. Khối 0 giữ hai khóa, nên luồng 3 và 2
            không có gì để chép, nhưng luồng 1 thì có: nó đọc <code>blockStart</code>, thứ chưa ai ghi (các byte 0xCD của trình
            giả lập, 3,452,816,845), và ghi ra xa ngoài đầu ra; AddressSanitizer dừng lần chạy. Thứ tự xáo trộn chạy các luồng
            của khối 0 theo thứ tự 1, 3, 2, 0, nên luồng 1 chạy trước và thực hiện cùng lần ghi đó. Theo thứ tự thì kernel vượt
            qua, vì luồng 0 chạy trước. Một thứ tự vượt qua không chứng minh gì, đó là lý do trình giả lập chạy ba thứ tự.
          </p>
        }
      >
        <p>
          Reversed, threads 3, 2 and 1 reach the commit before thread 0 has reserved. Block 0 kept two keys, so threads 3 and
          2 have nothing to copy, but thread 1 does: it reads <code>blockStart</code>, which nobody has written (the emulator's
          0xCD bytes, 3,452,816,845), and stores far outside the output; AddressSanitizer stops the run. The shuffled order
          runs block 0's threads as 1, 3, 2, 0, so thread 1 goes first and makes the same store. In order the kernel passes,
          because thread 0 runs first. A passing order proves nothing, which is why the emulator runs three.
        </p>
      </Translated>
    ),
    source: 'ch. 12 §12.4, p. 297; labs/README.md (the emulator\'s orders)',
    vi: { source: 'ch. 12 §12.4, p. 297; labs/README.md (các thứ tự của trình giả lập)' },
  },
  'ch12/12.4/q-commit': {
    kind: 'choice',
    id: 'ch12/12.4/q-commit',
    prompt: (
      <Translated vi={<>Trong bước commit, vì sao luồng <i>k</i> chép phần tử thứ <i>k</i> của danh sách riêng thay vì khóa của chính nó?</>}>
        <>In the commit, why does thread <i>k</i> copy the <i>k</i>-th entry of the private list instead of its own key?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Để các luồng liên tiếp ghi vào các địa chỉ liên tiếp và các lần ghi được hợp nhất, kể cả với những luồng có khóa của chính mình bị bỏ.">
            So that consecutive threads write consecutive addresses and the stores coalesce, even for threads whose own key was dropped.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Để giữ các khóa theo thứ tự đầu vào: phần tử k của danh sách riêng chứa khóa được giữ thứ k của khối, nên chép theo vị trí sẽ giữ được thứ tự.">
            To keep the keys in input order: entry k of the private list holds the block's k-th kept key, so copying by position preserves it.
          </Translated>
        ),
        why: (
          <Translated vi="Danh sách riêng theo thứ tự các phép atomic riêng hoàn thành, nên kết quả vẫn không ổn định.">
            The private list is in the order the private atomics completed, so the result is still unstable.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Vì một luồng không thể đọc bộ nhớ chia sẻ của luồng khác.">
            Because a thread cannot read another thread's shared memory.
          </Translated>
        ),
        why: (
          <Translated vi="Bộ nhớ chia sẻ được dùng chung bởi cả khối; đó chính là mục đích của nó.">
            Shared memory is shared by the whole block; that is the point of it.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Khóa của chính một luồng, nếu được giữ, nằm ở bất kỳ ô nào phép atomic của nó trả về. Chép theo vị trí thay vào đó
            biến bước commit thành kiểu ghi đều đặn nhất có thể: luồng <i>k</i> ghi <code>output[blockStart + k]</code>. Đó là
            lợi ích thứ hai mà sách nêu (p. 297). Thứ tự bên trong đoạn vẫn là thứ tự mà các luồng của khối thắng phép atomic
            riêng.
          </p>
        }
      >
        <p>
          A thread's own key, if kept, sits at whatever slot its atomic returned. Copying by position instead turns the commit
          into the most regular store there is: thread <i>k</i> writes <code>output[blockStart + k]</code>. That is the second
          benefit the book names (p. 297). The order inside the chunk is still the order in which the block's threads won
          the private atomic.
        </p>
      </Translated>
    ),
    source: 'ch. 12 §12.4, p. 297',
  },
  'ch12/12.4/q-public': {
    kind: 'numeric',
    id: 'ch12/12.4/q-public',
    prompt: (
      <Translated
        vi={
          <>
            2<sup>20</sup> khóa, khối 256 luồng, một khóa mỗi luồng, 1 phần trăm số khóa được giữ. Hình 12.6 phát bao nhiêu phép
            atomic trên bộ đếm công khai?
          </>
        }
      >
        <>
          2<sup>20</sup> keys, blocks of 256 threads, one key per thread, 1 percent of the keys kept. How many atomics does
          Fig. 12.6 issue on the public counter?
        </>
      </Translated>
    ),
    answer: PA.perBlock,
    explain: (
      <Translated
        vi={
          <p>
            Một phép mỗi khối, bất kể tỷ lệ: 2<sup>20</sup> / 256 = {PA.perBlock.toLocaleString('en-US')}. Luồng 0 của sách đặt
            trước ngay cả cho một khối không giữ gì (nó cộng 0). Để so sánh ở tỷ lệ này, Hình 12.2 phát một phép mỗi khóa được
            giữ, khoảng {PA.perKey.toLocaleString('en-US')}, còn một phép atomic mỗi warp có giữ khóa nào đó thì vào khoảng{' '}
            {(Math.round(PA.perWarp / 100) * 100).toLocaleString('en-US')} nếu các khóa được giữ rải ngẫu nhiên (phép tính của
            khóa học này).
          </p>
        }
      >
        <p>
          One per block, whatever the fraction: 2<sup>20</sup> / 256 = {PA.perBlock.toLocaleString('en-US')}. The book's thread 0
          reserves even for a block that kept nothing (it adds 0). For comparison at this fraction, Fig. 12.2 issues one per
          kept key, about {PA.perKey.toLocaleString('en-US')}, and one atomic per warp that keeps anything comes to about{' '}
          {(Math.round(PA.perWarp / 100) * 100).toLocaleString('en-US')} if the kept keys are scattered at random (our arithmetic).
        </p>
      </Translated>
    ),
    source: 'ch. 12 §12.4, p. 296; src/sim/privfilter.ts (publicAtomics)',
    vi: { source: 'ch. 12 §12.4, p. 296; mô hình src/sim/privfilter.ts (publicAtomics)' },
  },
  'ch12/12.4/q-vote-private': {
    kind: 'choice',
    id: 'ch12/12.4/q-vote-private',
    prompt: (
      <Translated
        vi={
          <>
            Bạn giữ Hình 12.6 nhưng thay lần tăng <code>outputSize_s</code> bằng cách bỏ phiếu warp của mục 12.3: một phép atomic
            trên bộ nhớ chia sẻ mỗi warp, với hạng của mỗi lane lấy từ một popcount có mặt nạ. Điều gì thay đổi?
          </>
        }
      >
        <>
          You keep Fig. 12.6 but replace the increment of <code>outputSize_s</code> with section 12.3's warp vote: one
          shared-memory atomic per warp, with each lane's rank from a masked popcount. What changes?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Đầu ra trở nên ổn định, vì hạng của mỗi lane giờ theo thứ tự lane thay vì thứ tự các phép atomic hoàn thành.">
            The output becomes stable, since each lane's rank now follows lane order rather than the order the atomics complete.
          </Translated>
        ),
        why: (
          <Translated vi="Trong một warp các hạng theo thứ tự lane, nhưng các warp vẫn đặt trước phần của mình trong danh sách riêng theo bất kỳ thứ tự nào mà các phép atomic của chúng hoàn thành.">
            Within a warp the ranks follow lane order, but warps still reserve their pieces of the private list in whatever order their atomics complete.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Ít phép atomic hơn trên outputSize_s, một phép mỗi warp có việc; vẫn một phép atomic công khai mỗi khối, và vẫn không ổn định.">
            Fewer atomics on outputSize_s, one per busy warp; still one public atomic per block, and still unstable.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Số phép atomic công khai giảm về không, vì lane dẫn đầu của mỗi warp giờ có thể trực tiếp đặt trước phần của nó trong đầu ra công khai.">
            The public atomics drop to zero, because each warp's leader can now reserve its piece of the public output directly.
          </Translated>
        ),
        why: (
          <Translated vi="Khối vẫn phải đặt trước phạm vi của nó trong đầu ra công khai, một lần.">
            The block still has to reserve its range of the public output, once.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hai kỹ thuật cộng dồn được: bỏ phiếu loại bỏ tranh chấp bên trong mỗi warp, riêng tư hóa loại bỏ tranh chấp giữa các
            khối. Trên HIP việc bỏ phiếu cần một mặt nạ có kiểu 64 bit và một popcount 64 bit (ch12-facts B1), như kernel warp của
            lab 12.1 cho thấy. Và trên AMD trình biên dịch có thể đã tự gộp theo warp giúp bạn cho một phép atomic cùng địa chỉ,
            kể cả LDS (B4), nên hãy đo trước và sau.
          </p>
        }
      >
        <p>
          The two techniques stack: the vote removes contention inside each warp, privatisation removes contention between
          blocks. On HIP the vote needs a 64-bit typed mask and a 64-bit popcount (ch12-facts B1), as lab 12.1's warp kernel
          shows. And on AMD the compiler may already have done the warp aggregation for you on a uniform-address atomic, LDS
          included (B4), so measure before and after.
        </p>
      </Translated>
    ),
    source: 'ch. 12 §§12.3–12.4; notes/today/ch12-facts.md B1, B4',
    vi: { source: 'ch. 12 §§12.3–12.4; notes/today/ch12-facts.md, mục B1, B4' },
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

export function S124() {
  const status = useLabStatus('ch12/lab1_unstable');
  const cu = useLabFile('ch12/lab1_unstable/solution/kernel.cu');
  const metal = useLabFile('ch12/lab1_unstable/solution/kernel.metal');
  const part = (src: string | null) => (src ? (region(src, 'private')?.text ?? src) : null);
  const cuPart = part(cu);
  const metalPart = part(metal);
  return (
    <>
      <Translated
        vi={
          <p>
            Mục 12.2 và 12.3 để mọi luồng của lưới nhắm vào một bộ đếm trong bộ nhớ toàn cục: một phép atomic mỗi khóa được giữ,
            hoặc tốt nhất là một phép mỗi warp. Chương 6 và 9 đã có lời giải cho dạng tranh chấp này. Cho mỗi nhóm luồng một bản
            sao riêng của đầu ra, để nhóm cập nhật bản sao của mình, và chạm vào bản công khai một lần mỗi nhóm ở cuối.
          </p>
        }
      >
        <p>
          Sections 12.2 and 12.3 left every thread of the grid aiming at one counter in global memory: one atomic per kept
          key, or one per warp at best. Chapters 6 and 9 already have the answer to that shape of contention. Give each
          group of threads a private copy of the output, let the group update its copy, and touch the public copy once per
          group at the end.
        </p>
      </Translated>
      <Source section="12.4" pages="295" />
      <Translated
        vi={
          <p>
            Với một bộ lọc, bản sao riêng là một danh sách. Mỗi khối giữ một mảng <code>BLOCK_DIM</code> ô và một bộ đếm trong
            bộ nhớ chia sẻ, và các luồng của nó nối các khóa được giữ vào danh sách đó bằng một phép atomic trên bộ đếm riêng.
            Khi khối xong việc, một luồng cộng số đếm của khối vào bộ đếm công khai bằng một phép atomic duy nhất, giúp đặt
            trước một phạm vi liền kề của đầu ra công khai, và cả khối chép danh sách của mình vào phạm vi đó.
          </p>
        }
      >
        <p>
          For a filter the private copy is a list. Each block keeps an array of <code>BLOCK_DIM</code> slots and a counter
          in shared memory, and its threads append kept keys to that list with an atomic on the private counter. When the
          block is done, one thread adds the block's count to the public counter with a single atomic, which reserves a
          contiguous range of the public output, and the whole block copies its list into that range.
        </p>
      </Translated>
      <Source section="12.4" pages="295" figs={tr('Fig. 12.5', 'Hình 12.5')} />

      <h2><Translated vi="Kernel">The kernel</Translated></h2>
      <Translated
        vi={
          <p>
            Hãy đọc nó theo ba giai đoạn: đặt bộ đếm riêng về 0, lọc vào danh sách riêng, rồi đặt trước và commit. Mỗi giai đoạn
            kết thúc ở một rào chắn.
          </p>
        }
      >
        <p>
          Read it in three phases: zero the private counter, filter into the private list, then reserve and commit. Each
          phase ends at a barrier.
        </p>
      </Translated>
      {cuPart && metalPart ? (
        <Code
          tabs={[
            { label: 'CUDA', code: cuPart, tiers: TIERS, status: status.cu },
            { label: tr('Metal kernel', 'Kernel Metal'), code: metalPart, tiers: TIERS, status: status.metal },
          ]}
          caption={
            <Translated
              vi={
                <>
                  Kernel riêng tư hóa trong lời giải tham chiếu của lab 12.1, tải từ{' '}
                  <code>labs/ch12/lab1_unstable/solution/</code>. Nó khác trang sách ở hai chỗ. <code>cond()</code> của sách chỉ
                  là chỗ giữ chỗ; ở đây một khóa được giữ khi nó nhỏ hơn một ngưỡng. Và sách dùng tên <code>j</code> cho hai thứ
                  khác nhau: ô mà một luồng nhận được trong danh sách riêng, một biến cục bộ, và điểm bắt đầu của phạm vi của khối
                  trong danh sách công khai, một biến dùng chung. Lab đặt tên biến dùng chung là <code>j_s</code>; trình chạy từng
                  bước bên dưới gọi chúng là <code>slot</code> và <code>blockStart</code>.
                </>
              }
            >
              <>
                The privatised kernel of lab 12.1's reference solution, loaded from{' '}
                <code>labs/ch12/lab1_unstable/solution/</code>. It differs from the page in two places. The book's{' '}
                <code>cond()</code> is a placeholder; here a key is kept when it is below a threshold. And the book uses the
                name <code>j</code> for two different things: the slot a thread gets in the private list, a local, and the
                start of the block's range in the public list, a shared variable. The lab names the shared one{' '}
                <code>j_s</code>; the stepper below calls them <code>slot</code> and <code>blockStart</code>.
              </>
            </Translated>
          }
        />
      ) : null}
      <Translated
        vi={
          <p>
            Ngoài các rào chắn, có hai chi tiết quan trọng. <code>atomic_ref</code> của bộ đếm riêng có{' '}
            <code>thread_scope_block</code>: chỉ các luồng của một khối chạm vào nó, nên khối là phạm vi hẹp nhất mà vẫn đúng.
            Bộ đếm công khai giữ <code>thread_scope_device</code>, vì mọi khối đều cộng vào nó.
          </p>
        }
      >
        <p>
          Two details matter beyond the barriers. The private counter's <code>atomic_ref</code> has{' '}
          <code>thread_scope_block</code>: only the threads of one block touch it, so the block is the narrowest scope that
          is still correct. The public counter keeps <code>thread_scope_device</code>, because every block adds to it.
        </p>
      </Translated>
      <Source section="12.4" pages="296" figs={tr('Fig. 12.6', 'Hình 12.6')} />
      <Ask q={questionsS124['ch12/12.4/q-scope']} />

      <h2><Translated vi="Chạy từng bước">Step through it</Translated></h2>
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy một bản thu nhỏ của kernel: bốn khối, mỗi khối bốn luồng, trên mười sáu khóa của sách, với
            k1, k3, k4, k6, k8, k10, k11 và k14 được giữ, như trong Hình 12.5. Nó mở đầu với các khối đặt trước theo thứ tự trong
            hình của sách. Bấm <i>Chạy tới rào chắn kế tiếp</i> vài lần và xem danh sách của khối 1 hình thành trong bộ nhớ chia
            sẻ, rồi rơi vào đầu ra công khai thành một đoạn.
          </p>
        }
      >
        <p>
          The stepper runs a shrunken copy of the kernel: four blocks of four threads over the book's sixteen keys, with
          k1, k3, k4, k6, k8, k10, k11 and k14 kept, as in Fig. 12.5. It opens with the blocks reserving in the order of
          the book's figure. Press <i>Run to next barrier</i> a few times and watch block 1's list form in shared memory,
          then land in the public output as one chunk.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi lần một câu lệnh nguồn, chạy theo cách trình giả lập của khóa học chạy một khối: một luồng chạy cho đến khi
              gặp rào chắn, rồi luồng kế tiếp theo thứ tự đã chọn tiếp quản. Các khối chạy lần lượt; trên GPU chúng chạy cùng
              lúc và đặt trước theo bất kỳ thứ tự nào mà luồng 0 của chúng tới phép atomic công khai. Không có warp, không có
              thời gian.
            </p>
          }
        >
          <p>
            One source statement at a time, run the way the course's emulator runs a block: a thread goes until it reaches a
            barrier, then the next thread in the chosen order takes over. Blocks run one after another; on a GPU they run
            at once and reserve in whatever order their thread 0 reaches the public atomic. No warps, no timing.
          </p>
        </Translated>
      </Simplification>
      <Stepper />
      <Translated
        vi={
          <p>
            Giờ đổi thứ tự khối thành 0, 1, 2, 3. Cùng bốn đoạn đó rơi vào một cách sắp xếp khác: đoạn của một khối đi đâu chỉ
            phụ thuộc vào lúc luồng 0 của nó tới bộ đếm công khai. Sau đó đặt thứ tự luồng là đảo ngược: bên trong mỗi đoạn các
            khóa đổi chỗ, vì danh sách riêng theo thứ tự các luồng của khối thắng phép atomic riêng. Hình 12.5 vẽ các khối theo
            thứ tự xanh lá, cam, vàng và xanh dương, tức là 1, 3, 2, 0. Đó là một cách sắp xếp trong nhiều cách; không gì trong
            kernel ưu tiên nó.
          </p>
        }
      >
        <p>
          Now change the block order to 0, 1, 2, 3. The same four chunks land in a different arrangement: where a block's
          chunk goes depends only on when its thread 0 reached the public counter. Then set the thread order to reversed:
          inside each chunk the keys swap places, because the private list is in the order the block's threads won the
          private atomic. Fig. 12.5 shows the blocks in the order green, orange, yellow and blue, that is 1, 3, 2, 0. That
          is one arrangement among many; nothing in the kernel prefers it.
        </p>
      </Translated>
      <Source section="12.4" pages="295" figs={tr('Fig. 12.5', 'Hình 12.5')} />

      <h2><Translated vi="Mỗi rào chắn bảo vệ điều gì">What each barrier protects</Translated></h2>
      <Translated
        vi={
          <p>
            <b>Rào chắn 1</b>, sau khi luồng 0 đặt <code>outputSize_s</code> về 0: không luồng nào được tăng bộ đếm trước khi nó
            bằng 0. Bộ nhớ chia sẻ bắt đầu với bất cứ thứ gì có sẵn ở đó, và một số đếm bắt đầu từ giá trị còn sót sẽ phát ra các
            ô vượt xa cuối danh sách.
          </p>
        }
      >
        <p>
          <b>Barrier 1</b>, after thread 0 zeroes <code>outputSize_s</code>: no thread may increment the counter before it
          is zero. Shared memory starts with whatever was there before, and a count that starts from leftovers hands out
          slots far beyond the end of the list.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>Rào chắn 2</b>, sau bước lọc: luồng 0 đặt trước <code>outputSize_s</code> ô công khai, nên số đếm phải đầy đủ khi
            nó đọc. Không có rào chắn, nó có thể đọc một số đếm còn thiếu khóa của những luồng chưa chạy.
          </p>
        }
      >
        <p>
          <b>Barrier 2</b>, after the filtering: thread 0 reserves <code>outputSize_s</code> public slots, so the count must
          be complete when it reads it. Without the barrier it can read a count that is missing the keys of threads that
          have not run yet.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>Rào chắn 3</b>, sau bước đặt trước: <code>blockStart</code> chỉ do luồng 0 ghi và được mọi luồng đọc trong bước
            commit. Không có rào chắn, một luồng có thể đọc nó trước khi nó được ghi.
          </p>
        }
      >
        <p>
          <b>Barrier 3</b>, after the reservation: <code>blockStart</code> is written by thread 0 alone and read by every
          thread in the commit. Without the barrier a thread can read it before it is written.
        </p>
      </Translated>
      <Source section="12.4" pages="296–297" />
      <Translated
        vi={
          <p>
            Cả ba nằm ngoài mọi câu lệnh điều kiện, vì mọi luồng của khối phải tới cùng các rào chắn. Trong trình chạy từng bước,
            bỏ chọn một rào chắn, chọn một thứ tự luồng, và bấm <i>Tới lỗi đầu tiên</i>. Sau đó thử các thứ tự khác.
          </p>
        }
      >
        <p>
          All three sit outside any conditional, because every thread of the block must reach the same barriers. In the
          stepper, untick one barrier, pick a thread order, and press <i>Go to the first failure</i>. Then try the other
          orders.
        </p>
      </Translated>
      <Outcomes />
      <Ask q={questionsS124['ch12/12.4/q-barrier2']} />
      <Ask q={questionsS124['ch12/12.4/q-barrier3']} />
      <Translated
        vi={
          <p>
            Hai điều trong bảng đáng xem lại. Cùng một lỗi thất bại ở thứ tự này và vượt qua ở thứ tự khác: không có rào chắn 3,
            lần chạy theo thứ tự là đúng vì luồng 0 tình cờ chạy trước, và không có rào chắn 2, lần chạy đảo ngược là đúng vì
            luồng 0 tình cờ chạy cuối. Và có một hàng vượt qua mọi thứ tự mà trình giả lập có.
          </p>
        }
      >
        <p>
          Two things in the table are worth a second look. The same defect fails under one order and passes under another:
          without barrier 3 the in-order run is correct because thread 0 happens to go first, and without barrier 2 the
          reversed run is correct because thread 0 happens to go last. And one row passes every order the emulator has.
        </p>
      </Translated>

      <h2><Translated vi="Race mà chạy từng luồng một không thấy được">The race one thread at a time cannot see</Translated></h2>
      <Translated
        vi={
          <p>
            Hàng đó là bộ đếm riêng được tăng bằng <code>outputSize_s++</code>. Khi các luồng lần lượt chạy, một lần đọc thường
            rồi một lần ghi cho đúng ô mà một phép atomic sẽ cho, nên các lần chạy từng luồng một của trình giả lập đều đúng.
            Trên GPU các luồng của một khối chạy cùng lúc: hai luồng có thể đọc cùng một số đếm và lấy cùng một ô của{' '}
            <code>output_s</code>, bộ đếm thiếu một, và một trong hai khóa bị ghi đè và mất. Cột race của bảng áp dụng phép kiểm
            tra mà một trình phát hiện race thực hiện, và nó báo cặp lần tăng đó.
          </p>
        }
      >
        <p>
          That row is the private counter bumped with <code>outputSize_s++</code>. When threads take turns, a plain read
          followed by a write gives exactly the slot an atomic would, so the emulator's one-thread-at-a-time runs are all
          correct. On a GPU the threads of a block run at once: two of them can read the same count and take the same slot
          of <code>output_s</code>, the counter ends one short, and one of the two keys is overwritten and lost. The table's
          race column applies the check a race detector makes, and it reports the pair of increments.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Backend <code>race</code> của khóa học chạy các luồng đồng thời dưới ThreadSanitizer chính vì loại lỗi này
              (labs/README.md; ch09-facts E1). Trên kernel này nó chỉ thỉnh thoảng bắt được lần tăng: nó coi mọi rào chắn của
              một khối là một điểm đồng bộ hóa duy nhất, nên một luồng rời rào chắn 1 muộn bị coi là xếp sau các luồng đang chờ ở
              rào chắn 2. Đó là một giới hạn của trình giả lập của khóa học này, đã được báo để sửa; phép kiểm tra của mô hình ở
              trên không có giới hạn đó.
            </p>
          }
        >
          <p>
            The course's <code>race</code> backend runs threads concurrently under ThreadSanitizer for exactly this kind of
            bug (labs/README.md; ch09-facts E1). On this kernel it catches the increment only now and then: it treats all of
            a block's barriers as one synchronisation point, so a thread that leaves barrier 1 late is taken to be ordered
            after threads already waiting at barrier 2. That is a limitation of this course's emulator, reported for fixing;
            the model's check above does not have it.
          </p>
        </Translated>
      </Aside>

      <h2><Translated vi="Một bước commit được hợp nhất">A coalesced commit</Translated></h2>
      <Translated
        vi={
          <p>
            Nhìn lại dòng commit: luồng <i>k</i> chép <code>output_s[k]</code>, không phải khóa của chính nó. Vì vậy các luồng
            liên tiếp ghi vào các địa chỉ công khai liên tiếp, mẫu ghi mà chương 6 cho thấy phần cứng có thể gộp lại. Đó là lợi
            ích thứ hai của riêng tư hóa ở đây. Điều nó không mang lại là thứ tự: danh sách riêng theo thứ tự các phép atomic
            riêng hoàn thành, nên bộ lọc vẫn không ổn định.
          </p>
        }
      >
        <p>
          Look at the commit line again: thread <i>k</i> copies <code>output_s[k]</code>, not its own key. Consecutive
          threads therefore write consecutive public addresses, the store pattern chapter 6 showed the hardware can combine.
          That is the second benefit of privatisation here. What it does not buy is order: the private list is in the order
          the private atomics completed, so the filter is still unstable.
        </p>
      </Translated>
      <Source section="12.4" pages="297" />
      <Ask q={questionsS124['ch12/12.4/q-commit']} />

      <h2><Translated vi="Giờ nó tốn gì">What it costs now</Translated></h2>
      <Translated
        vi={
          <p>
            Bộ đếm công khai giờ nhận một phép atomic mỗi khối. Luồng 0 của sách đặt trước ngay cả khi khối của nó không giữ gì,
            cộng 0; một phép kiểm tra trên số đếm sẽ bỏ qua những lần đó với cái giá là một nhánh rẽ.
          </p>
        }
      >
        <p>
          The public counter now takes one atomic per block. The book's thread 0 reserves even when its block kept nothing,
          adding 0; a test on the count would skip those at the price of a branch.
        </p>
      </Translated>
      <Source section="12.4" pages="296" />
      <Ask q={questionsS124['ch12/12.4/q-public']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Tranh chấp đã dời chỗ, chứ chưa biến mất. Mỗi khóa được giữ vẫn thực hiện một phép atomic trên{' '}
            <code>outputSize_s</code>, một word duy nhất mà mọi luồng của khối nhắm vào. Một phép atomic trên bộ nhớ chia sẻ rẻ
            hơn một phép trên bộ nhớ toàn cục (hướng dẫn của NVIDIA nói rõ điều đó; ch09-facts A2), nhưng nó vẫn là một phép
            atomic cùng địa chỉ, bị tuần tự hóa giữa các luồng của khối. Cách bỏ phiếu của mục 12.3 cũng áp dụng ở đây, và trên
            AMD trình biên dịch có thể đã áp dụng nó giúp bạn.
          </p>
        }
      >
        <p>
          The contention has moved, not gone. Every kept key still does one atomic on <code>outputSize_s</code>, a single
          word that all the block's threads aim at. A shared-memory atomic is cheaper than a global one (NVIDIA's guide says
          so plainly; ch09-facts A2), but it is still a same-address atomic, serialised among the threads of the block.
          Section 12.3's vote applies here too, and on AMD the compiler may already apply it for you.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch12-facts.md, B4)', '2026-11-20 (notes/today/ch12-facts.md, mục B4)')}>
        <Translated
          vi={
            <p>
              Backend AMD của LLVM upstream tự thực hiện việc bỏ phiếu này cho một phép atomic cùng địa chỉ, kể cả LDS, ở mức
              -O1 trở lên. Mặc định trong bản phát hành của ROCm chưa được xác nhận.
            </p>
          }
        >
          <p>
            Upstream LLVM's AMD backend does this vote itself for a uniform-address atomic, LDS included, at -O1 and
            above. ROCm's shipped default was not confirmed.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS124['ch12/12.4/q-vote-private']} />

      <h2><Translated vi="HIP và Metal">HIP and Metal</Translated></h2>
      <Translated
        vi={
          <p>
            Chuyển tab trong đoạn mã ở trên cùng: kernel Metal có cùng ba giai đoạn và cùng ba rào chắn. Khác biệt nằm ở cách
            khai báo bộ đếm.
          </p>
        }
      >
        <p>
          Switch tabs in the listing at the top: the Metal kernel has the same three phases and the same three barriers.
          The differences are in how the counter is declared.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Đã đối chiếu với notes/today/infra-facts.md (B1, B2, C1) và ch09-facts.md (B1, C1). Sách chỉ dùng CUDA.
            </p>
          }
        >
          <p>
            Checked against notes/today/infra-facts.md (B1, B2, C1) and ch09-facts.md (B1, C1). The book is CUDA-only.
          </p>
        </Translated>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th><Translated vi="Trong Hình 12.6">In Fig. 12.6</Translated></th><th>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <Translated vi={<><code>cuda::atomic_ref&lt;unsigned int, cuda::thread_scope_block&gt;</code> trên một bộ đếm dùng chung</>}>
                <code>cuda::atomic_ref&lt;unsigned int, cuda::thread_scope_block&gt;</code> on a shared counter
              </Translated>
            </td>
            <td>
              <Translated
                vi={
                  <>
                    Cùng cách viết qua <code>&lt;cuda/atomic&gt;</code> của libhipcxx (ROCm 10.1 trở lên). Không có{' '}
                    <code>atomicAdd_block</code>; <code>atomicAdd</code> kiểu cũ hoạt động trên bộ nhớ chia sẻ và toàn cục, và
                    lab 12.1 dùng nó trên HIP. Bản dựng HIP của khóa học liên kết <code>libhipcxx::libhipcxx</code> khi CMake
                    tìm thấy nó. Không có nó, mã include <code>&lt;cuda/atomic&gt;</code> không dựng được. Có nó, chưa máy ROCm
                    nào dựng mã của khóa học này (unverified). Lời gọi kiểu cũ không cần thư viện bổ sung.
                  </>
                }
              >
                The same spelling through libhipcxx's <code>&lt;cuda/atomic&gt;</code> (ROCm 10.1 and later). There is no <code>atomicAdd_block</code>; the legacy <code>atomicAdd</code> works on shared and global memory, and lab 12.1 uses it on HIP. The course's HIP build links <code>libhipcxx::libhipcxx</code> when CMake finds it. Without it, code that includes <code>&lt;cuda/atomic&gt;</code> does not build. With it, no ROCm machine has built this course's code yet (unverified). The legacy call needs no extra library.
              </Translated>
            </td>
            <td>
              <Translated
                vi={
                  <>
                    <code>threadgroup atomic_uint outputSize_s;</code> và{' '}
                    <code>atomic_fetch_add_explicit(&amp;outputSize_s, 1u, memory_order_relaxed)</code>. Không có đối số phạm
                    vi: không gian địa chỉ cho biết ai dùng chung bộ đếm.
                  </>
                }
              >
                <code>threadgroup atomic_uint outputSize_s;</code> and <code>atomic_fetch_add_explicit(&amp;outputSize_s, 1u, memory_order_relaxed)</code>. No scope argument: the address space says who shares the counter.
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="Đặt về 0 và đọc bộ đếm bằng các truy cập thường">Zeroing and reading the counter with plain accesses</Translated></td>
            <td><Translated vi="Giống vậy">The same</Translated></td>
            <td>
              <Translated vi={<>Một kiểu atomic chỉ chấp nhận các hàm atomic: <code>atomic_store_explicit</code> để đặt về 0, <code>atomic_load_explicit</code> để đọc.</>}>
                An atomic type admits only atomic functions: <code>atomic_store_explicit</code> to zero it, <code>atomic_load_explicit</code> to read it.
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="Bộ đếm công khai phạm vi device">The device-scope public counter</Translated></td>
            <td><Translated vi={<>Cùng cách viết, hoặc <code>atomicAdd</code> kiểu cũ</>}>The same spelling, or legacy <code>atomicAdd</code></Translated></td>
            <td>
              <Translated vi={<>Tham số buffer được khai báo là <code>device atomic_uint*</code>; host cấp phát các số nguyên 32 bit thường.</>}>
                The buffer parameter is declared <code>device atomic_uint*</code>; the host allocates plain 32-bit integers.
              </Translated>
            </td>
          </tr>
          <tr>
            <td><code>__syncthreads()</code></td>
            <td><Translated vi="Giống vậy">The same</Translated></td>
            <td><code>threadgroup_barrier(mem_flags::mem_threadgroup)</code></td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Danh sách riêng tốn <code>BLOCK_DIM</code> × 4 byte bộ nhớ chia sẻ mỗi khối: 1 KiB ở 256 luồng. Đọc giới hạn mỗi khối
            của thiết bị bạn từ <code>python3 labs/pmpp.py doctor</code> và so sánh. Mục 12.6 trung chuyển nhiều khóa mỗi luồng, và ở đó ngân sách bắt đầu quan trọng.
          </p>
        }
      >
        <p>
          The private list costs <code>BLOCK_DIM</code> × 4 bytes of shared memory per block: 1 KiB at 256 threads. Read
          your device's limit per block from <code>python3 labs/pmpp.py doctor</code> and compare. Section 12.6 stages
          several keys per thread, and there the budget starts to matter.
        </p>
      </Translated>

      <h2><Translated vi="Điều này xuất hiện ở đâu trong một bộ lấy mẫu">Where this shows up in a sampler</Translated></h2>
      <Translated
        vi={
          <p>
            Cho việc lấy mẫu theo batch một khối mỗi hàng logits và riêng tư hóa ánh xạ thẳng lên đó: danh sách của khối trong bộ
            nhớ chia sẻ là danh sách ứng viên của hàng đó, và bộ đếm riêng của nó là số ứng viên của hàng. Nếu mỗi hàng còn sở
            hữu một vùng dung lượng cố định của đầu ra, thì không còn bộ đếm công khai nào phải dùng chung: offset của hàng là
            chỉ số hàng nhân với dung lượng. Đó là thiết kế của <a href={href('ch12', 'labs')}>lab 12.4</a>, nén gọn ứng viên cho
            mốc dự án M5.
          </p>
        }
      >
        <p>
          Give batched sampling one block per row of logits and privatisation maps straight onto it: the block's list in
          shared memory is that row's candidate list, and its private counter is the row's candidate count. If each row also owns a
          fixed-capacity region of the output, there is no public counter left to share: the row's offset is its row
          index times the capacity. That is the design of{' '}
          <a href={href('ch12', 'labs')}>lab 12.4</a>, candidate compaction for the M5 milestone.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cách ánh xạ sang lấy mẫu là của khóa học này. Top-k CUDA của llama.cpp giữ một bộ đếm riêng cho mỗi hàng, mỗi bộ
              đếm trên một cache line 128 byte riêng, "so the gather atomics of different rows do not compete" (ch12-facts D1).
            </p>
          }
        >
          <p>
            The mapping to sampling is ours. llama.cpp's CUDA top-k keeps a separate counter per row, each on its own
            128-byte cache line, "so the gather atomics of different rows do not compete" (ch12-facts D1).
          </p>
        </Translated>
      </Beyond>
    </>
  );
}
