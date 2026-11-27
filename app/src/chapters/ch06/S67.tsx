import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, MATMUL_TIERS, useLabFile } from '../../components/Code';
import { defineLabs, solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useReports } from '../../lib/reports';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  KIND_LABEL,
  SCHEDULES,
  SCHEDULE_LABEL,
  VARIANTS,
  VARIANT_LABEL,
  firstFailure,
  isDouble,
  loopSource,
  nextRelease,
  run,
  stateAt,
  type Op,
  type Schedule,
  type Step,
  type Variant,
} from '../../sim/doublebuf';
import { DoubleBufferScene } from './DoubleBufferScene';

// labs/ch06/snippets/s67_exchange.cu, which ./verify.sh front-end-checks.
// snippets-s67.test.ts fails if this copy drifts from the file.
const EXCHANGE = `// Section 6.7, pp. 146-147: the book's exchange loop, made compilable, once with one
// shared buffer and once with two. Each thread reads its right-hand neighbour's slot
// and writes that value plus 10 into its own slot, n times. Launch one block of
// BLOCK threads. The book does not show declarations, the computation or swap
// (notes item 15); they are ours.
#define BLOCK 256

// Exchanges the calling thread's own two pointers. Every thread must call it.
__device__ void swapPointers(float*& a, float*& b) {
    float* t = a;
    a = b;
    b = t;
}

__global__ void exchangeSingle(const float* in, float* out, unsigned int n) {
    __shared__ float buffer[BLOCK];
    unsigned int me = threadIdx.x;
    unsigned int other = (threadIdx.x + 1) % blockDim.x;
    buffer[me] = in[me];
    __syncthreads();
    for (unsigned int i = 0; i < n; ++i) {
        float v = buffer[other];
        __syncthreads();            // after the read
        buffer[me] = v + 10.0f;
        __syncthreads();            // after the write
    }
    out[me] = buffer[me];
}

__global__ void exchangeDouble(const float* in, float* out, unsigned int n) {
    __shared__ float bufferA[BLOCK];
    __shared__ float bufferB[BLOCK];
    float* inBuffer = bufferA;      // each thread holds its own copy of both pointers
    float* outBuffer = bufferB;
    unsigned int me = threadIdx.x;
    unsigned int other = (threadIdx.x + 1) % blockDim.x;
    inBuffer[me] = in[me];
    __syncthreads();
    for (unsigned int i = 0; i < n; ++i) {
        float v = inBuffer[other];
        outBuffer[me] = v + 10.0f;
        __syncthreads();            // after the write
        swapPointers(inBuffer, outBuffer);
    }
    out[me] = inBuffer[me];
}`;

const EXCHANGE_TIERS = {
  buffer: 'shared', bufferA: 'shared', bufferB: 'shared', in: 'global', out: 'global',
  v: 'reg', me: 'reg', other: 'reg', i: 'reg', inBuffer: 'reg', outBuffer: 'reg',
} as const;

const [LAB65] = defineLabs('ch06', [{ num: '6.5', dir: 'lab5_doublebuf', name: 'double buffering', vi: { name: 'buffer kép' } }]);
const [LAB52] = defineLabs('ch05', [{ num: '5.2', dir: 'lab2_tiled', name: 'tiled', vi: { name: 'phân ô' } }]);

const show = (v: number) => (Number.isNaN(v) ? '·' : String(v));
const bufName = (variant: Variant, b: number) => (isDouble(variant) ? (b === 0 ? 'A' : 'B') : '');

function describe(s: Step, variant: Variant, prevOp: Op | 'release' | undefined): string {
  if (s.thread < 0)
    return prevOp === 'barrierRead'
      ? tr(
          'The barrier after the read opens: every thread has read. Now it is safe to overwrite slots.',
          'Rào chắn sau lần đọc mở ra: mọi luồng đều đã đọc. Giờ ghi đè các ô đã an toàn.',
        )
      : tr(
          'The barrier after the write opens: every thread has written. Now it is safe to read the new values.',
          'Rào chắn sau lần ghi mở ra: mọi luồng đều đã ghi. Giờ đọc các giá trị mới đã an toàn.',
        );
  const who = tr(`Thread ${s.thread}, iteration ${s.iter}`, `Luồng ${s.thread}, lần lặp ${s.iter}`);
  const where = (b: number, slot: number) =>
    isDouble(variant)
      ? tr(`buffer ${bufName(variant, b)}, slot ${slot}`, `buffer ${bufName(variant, b)}, ô ${slot}`)
      : tr(`slot ${slot}`, `ô ${slot}`);
  switch (s.op) {
    case 'read': {
      const r = s.read!;
      const from =
        r.version < 0
          ? tr('nothing written yet', 'chưa có gì được ghi')
          : r.version === 0
            ? tr('the initial value', 'giá trị ban đầu')
            : tr(`written in iteration ${r.version - 1}`, `được ghi ở lần lặp ${r.version - 1}`);
      return tr(
        `${who}: reads its neighbour's ${where(r.buf, r.slot)} and finds ${show(r.value)} (${from}).`,
        `${who}: đọc ${where(r.buf, r.slot)} của luồng láng giềng và thấy ${show(r.value)} (${from}).`,
      );
    }
    case 'write':
      return tr(
        `${who}: writes ${show(s.write!.value)}, the value it read plus 10, into its own ${where(s.write!.buf, s.write!.slot)}.`,
        `${who}: ghi ${show(s.write!.value)}, giá trị nó đọc cộng 10, vào ${where(s.write!.buf, s.write!.slot)} của chính nó.`,
      );
    case 'barrierRead':
      return tr(
        `${who}: reaches the barrier after the read and waits for the others.`,
        `${who}: tới rào chắn sau lần đọc và chờ các luồng khác.`,
      );
    case 'barrierWrite':
      return tr(
        `${who}: reaches the barrier after the write and waits for the others.`,
        `${who}: tới rào chắn sau lần ghi và chờ các luồng khác.`,
      );
    case 'swap':
      return tr(
        `${who}: swaps its own two pointers. From now on it reads buffer ${bufName(variant, s.pointers!.inBuf)} and writes buffer ${bufName(variant, s.pointers!.outBuf)}.`,
        `${who}: hoán đổi hai con trỏ của chính nó. Từ giờ nó đọc buffer ${bufName(variant, s.pointers!.inBuf)} và ghi buffer ${bufName(variant, s.pointers!.outBuf)}.`,
      );
    default:
      return '';
  }
}

// ---------------------------------------------------------------- the stepper
function Stepper() {
  const [variant, setVariant] = useState<Variant>('single-both');
  const [schedule, setSchedule] = useState<Schedule>('in-order');
  const [P, setP] = useState(4);
  const [N, setN] = useState(3);
  const [n, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  const locale = useLocale();
  const trace = useMemo(() => run({ threads: P, iterations: N, variant }, schedule), [P, N, variant, schedule, locale]);
  const total = trace.steps.length;
  const at = Math.min(n, total);
  const st = useMemo(() => stateAt(trace, at), [trace, at]);
  const cur = at > 0 ? trace.steps[at - 1] : undefined;
  const prev = at > 1 ? trace.steps[at - 2] : undefined;
  const bad = firstFailure(trace);
  const src = loopSource(variant);
  const dbl = isDouble(variant);
  const hl = cur ? (cur.op === 'release' ? (prev && prev.op !== 'release' && prev.op in src.line ? [src.line[prev.op as Op]] : []) : [src.line[cur.op]]) : [];

  useEffect(() => setStep(0), [P, N, variant, schedule]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setStep((x) => x + 1), 380);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);

  const slotCls = (b: number, s: number) => {
    if (cur?.write && cur.write.buf === b && cur.write.slot === s) return 'hot';
    if (cur?.read && cur.read.buf === b && cur.read.slot === s) return cur.read.ok ? 'hot' : 'bad';
    return '';
  };

  return (
    <Bench
      title={tr('Step through the exchange', 'Đi qua từng bước phép trao đổi')}
      task={tr(
        'Remove one barrier at a time, keep the order where each thread runs as far as it can, and go to the first failure.',
        'Mỗi lần bỏ một rào chắn, giữ thứ tự trong đó mỗi luồng chạy xa nhất có thể, và đi tới lỗi đầu tiên.',
      )}
      foot={
        <Translated
          vi={
            <>
              Con số nhỏ cạnh một giá trị là lần lặp đã ghi nó; dấu gạch đánh dấu giá trị ban đầu. Luồng t bắt đầu với t; sau
              n lần lặp nó phải giữ 10n cộng giá trị ban đầu của luồng t + n, đếm vòng quanh.
            </>
          }
        >
          <>A small number beside a value is the iteration that wrote it; a dash marks a starting value. Thread t starts with t; after n iterations it should hold 10n plus the starting value of thread t + n, counted round the ring.</>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Kernel">Kernel</Translated>
          <select value={variant} onChange={(e) => setVariant(e.target.value as Variant)} aria-label={tr('kernel', 'kernel')} style={{ maxWidth: '100%' }}>
            {VARIANTS.map((v) => (
              <option key={v} value={v}>
                <Translated
                  vi={
                    {
                    'single-both': 'một buffer, cả hai rào chắn',
                    'single-no-read-barrier': 'một buffer, không có rào chắn sau lần đọc',
                    'single-no-write-barrier': 'một buffer, không có rào chắn sau lần ghi',
                    double: 'hai buffer, một rào chắn',
                    'double-no-swap': 'hai buffer, quên hoán đổi',
                    'double-no-barrier': 'hai buffer, không có rào chắn',
                  }[v]
                  }
                >
                  {VARIANT_LABEL[v]}
                </Translated>
              </option>
            ))}
          </select>
        </label>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Thứ tự luồng">Thread order</Translated>
          <select value={schedule} onChange={(e) => setSchedule(e.target.value as Schedule)} aria-label={tr('thread order', 'thứ tự luồng')} style={{ maxWidth: '100%' }}>
            {SCHEDULES.map((s) => (
              <option key={s} value={s}>
                <Translated
                  vi={
                    {
                    lockstep: 'mọi luồng chạy đồng bộ từng bước (lockstep)',
                    'in-order': 'mỗi luồng chạy xa nhất có thể, luồng 0 trước',
                    reversed: 'mỗi luồng chạy xa nhất có thể, luồng cuối trước',
                    shuffled: 'xen kẽ ngẫu nhiên',
                  }[s]
                  }
                >
                  {SCHEDULE_LABEL[s]}
                </Translated>
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <Translated vi="Số luồng">Threads</Translated>
          <select value={P} onChange={(e) => setP(+e.target.value)} aria-label={tr('threads', 'số luồng')}>
            {[4, 5, 6, 7, 8].map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Số lần lặp">Iterations</Translated>
          <select value={N} onChange={(e) => setN(+e.target.value)} aria-label={tr('iterations', 'số lần lặp')}>
            {[1, 2, 3, 4].map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
      </div>

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setStep(0); }} disabled={at === 0}>
          <Translated vi="Bắt đầu lại">Restart</Translated>
        </button>
        <button onClick={() => { setPlaying(false); setStep(at - 1); }} disabled={at === 0}>
          <Translated vi="Lùi">Back</Translated>
        </button>
        <button className="primary" onClick={() => { setPlaying(false); setStep(at + 1); }} disabled={at >= total}>
          <Translated vi="Bước">Step</Translated>
        </button>
        <button onClick={() => { setPlaying(false); setStep(nextRelease(trace, at)); }} disabled={at >= total}>
          <Translated vi="Chạy tới rào chắn kế tiếp">Run to next barrier</Translated>
        </button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Chạy')}</button>
        {bad ? (
          <button onClick={() => { setPlaying(false); setStep(bad); }}>
            <Translated vi="Đi tới lỗi đầu tiên">Go to the first failure</Translated>
          </button>
        ) : null}
        <span className="muted" aria-live="polite">
          <Translated vi={<>bước {at} trên {total}</>}>
            <>step {at} of {total}</>
          </Translated>
        </span>
      </div>

      <div className={'narrate' + (cur?.violation ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {cur ? (
          <>
            {describe(cur, variant, prev?.op)}
            {cur.violation ? (
              <Translated
                vi={
                  <>
                    <br />
                    <b className="t-fail">
                      Lỗi (
                      {{
                    'read-after-write': 'đọc-sau-ghi',
                    'write-after-read': 'ghi-sau-đọc',
                    'wrong-buffer': 'sai buffer (không có race)',
                  }[cur.violation.kind]}
                      ):{' '}
                      {cur.violation.kind === 'wrong-buffer'
                        ? `luồng ${cur.violation.thread} đọc ở lần lặp ${cur.violation.iter} từ buffer mà giá trị của lần lặp trước không bao giờ được ghi vào`
                        : cur.violation.kind === 'read-after-write'
                          ? `luồng ${cur.violation.thread} đọc ở lần lặp ${cur.violation.iter} trước khi luồng láng giềng ghi giá trị của lần lặp đó vào ô`
                          : `luồng ${cur.violation.thread} vẫn cần giá trị cũ ở lần lặp ${cur.violation.iter}, nhưng luồng láng giềng đã ghi đè nó`}
                      .
                    </b>
                  </>
                }
              >
                <><br /><b className="t-fail">Failure ({KIND_LABEL[cur.violation.kind]}): {cur.violation.message}.</b></>
              </Translated>
            ) : null}
          </>
        ) : (
          <Translated
            vi={
              <>
                Chưa có gì chạy. Mọi luồng sẽ đọc ô của luồng bên phải nó (luồng {P - 1} đọc ô 0). Nhấn Bước.
              </>
            }
          >
            <>Nothing has run yet. Every thread will read the slot of the thread to its right (thread {P - 1} reads slot 0). Press Step.</>
          </Translated>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 5fr) minmax(0, 6fr)' }}>
        <div>
          <Code tabs={[{ label: tr('loop body', 'thân vòng lặp'), code: src.code, tiers: EXCHANGE_TIERS }]} highlight={hl} />
        </div>
        <div>
          {(dbl ? [0, 1] : [0]).map((b) => (
            <div key={b} style={{ marginBottom: '0.8rem' }}>
              <div className="mx-label t-shared">
                <Translated vi={<>{dbl ? `buffer ${bufName(variant, b)}` : 'buffer'} (chia sẻ)</>}>
                  <>{dbl ? `buffer ${bufName(variant, b)}` : 'buffer'} (shared)</>
                </Translated>
              </div>
              <div className="mx shared" data-buffer={b} style={{ gridTemplateColumns: `repeat(${P}, auto)` }}>
                {st.buffers[b].map((s, i) => (
                  <span key={i} className={slotCls(b, i)}>
                    {show(s.value)}
                    {s.version >= 0 ? <sub>{s.version === 0 ? '–' : s.version - 1}</sub> : null}
                  </span>
                ))}
              </div>
            </div>
          ))}
          <table className="threads" aria-label={tr('threads', 'các luồng')}>
            <thead>
              <tr>
                <th>
                  <Translated vi="luồng">thread</Translated>
                </th>
                <th>
                  <Translated vi="đọc ô">reads slot</Translated>
                </th>
                {dbl ? (
                  <th>
                    <Translated vi="đọc, ghi">reads, writes</Translated>
                  </th>
                ) : null}
                <th>
                  <Translated vi="đang làm">doing</Translated>
                </th>
                <th>v</th>
              </tr>
            </thead>
            <tbody>
              {st.threads.map((th, t) => (
                <tr key={t} className={cur?.thread === t ? 'cur' : ''}>
                  <td>{t}</td>
                  <td>{(t + 1) % P}</td>
                  {dbl ? <td>{bufName(variant, th.inBuf)}, {bufName(variant, th.outBuf)}</td> : null}
                  <td>
                    {th.done
                      ? tr('finished', 'đã xong')
                      : th.waiting
                        ? tr('waiting at the barrier', 'đang chờ ở rào chắn')
                        : th.last
                          ? tr(`iteration ${th.iter}`, `lần lặp ${th.iter}`)
                          : tr('not started', 'chưa bắt đầu')}
                  </td>
                  <td className="t-reg">{show(th.v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      {at >= total ? (
        <p style={{ margin: '0.9rem 0 0' }} data-k="result">
          {trace.correct ? (
            <span className="chip ok">
              <Translated vi="Mọi luồng đều trả về giá trị đúng.">Every thread hands back the right value.</Translated>
            </span>
          ) : (
            <span className="chip fail">
              <Translated vi={<>{trace.wrong.length} trên {P} luồng trả về giá trị sai.</>}>
                <>{trace.wrong.length} of {P} threads hand back a wrong value.</>
              </Translated>
            </span>
          )}{' '}
          <span className="muted">
            <Translated
              vi={
                <>
                  Trả về: {trace.output.join(', ')}; kỳ vọng: {trace.reference.join(', ')}.
                  {trace.violations.length ? ` ${trace.violations.length} lần đọc sai.` : ''}
                </>
              }
            >
              <>
                Handed back: {trace.output.join(', ')}; expected: {trace.reference.join(', ')}.
                {trace.violations.length ? ` ${trace.violations.length} bad read${trace.violations.length === 1 ? '' : 's'}.` : ''}
              </>
            </Translated>
          </span>
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the outcome table
function Outcomes() {
  const [P, setP] = useState(4);
  const [guess, setGuess] = useState<Set<Variant>>(new Set());
  const [shown, setShown] = useState(false);
  const N = 3;
  const results = useMemo(
    () => VARIANTS.map((v) => ({ v, cells: SCHEDULES.map((s) => run({ threads: P, iterations: N, variant: v }, s)) })),
    [P],
  );
  const everywhere = results.filter((r) => r.cells.every((c) => c.correct)).map((r) => r.v);
  const toggle = (v: Variant) => {
    const g = new Set(guess);
    if (g.has(v)) g.delete(v);
    else g.add(v);
    setGuess(g);
  };
  const same = shown && everywhere.length === guess.size && everywhere.every((v) => guess.has(v));
  return (
    <Bench
      title={tr('Every thread order at once', 'Mọi thứ tự luồng cùng lúc')}
      task={tr(
        'Tick the kernels you expect to be correct under every order, then reveal the table.',
        'Đánh dấu các kernel bạn nghĩ sẽ đúng dưới mọi thứ tự, rồi mở bảng.',
      )}
      foot={tr(
        `Each cell runs the loop ${N} times with the model above and compares what every thread hands back with the expected value.`,
        `Mỗi ô chạy vòng lặp ${N} lần với mô hình ở trên và so giá trị mọi luồng trả về với giá trị kỳ vọng.`,
      )}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }} role="group" aria-label={tr('your prediction', 'dự đoán của bạn')}>
        {VARIANTS.map((v) => (
          <label key={v} className="check">
            <input type="checkbox" checked={guess.has(v)} disabled={shown} onChange={() => toggle(v)} />{' '}
            <Translated
              vi={
                {
                    'single-both': 'một buffer, cả hai rào chắn',
                    'single-no-read-barrier': 'một buffer, không có rào chắn sau lần đọc',
                    'single-no-write-barrier': 'một buffer, không có rào chắn sau lần ghi',
                    double: 'hai buffer, một rào chắn',
                    'double-no-swap': 'hai buffer, quên hoán đổi',
                    'double-no-barrier': 'hai buffer, không có rào chắn',
                  }[v]
              }
            >
              {VARIANT_LABEL[v]}
            </Translated>
          </label>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        {shown ? null : (
          <button className="primary" onClick={() => setShown(true)}>
            <Translated vi="Mở bảng">Reveal the table</Translated>
          </button>
        )}
        <label className="field">
          <Translated vi="Số luồng">Threads</Translated>
          <select value={P} onChange={(e) => setP(+e.target.value)} aria-label={tr('threads in the table', 'số luồng trong bảng')}>
            {[4, 6, 8].map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
      </div>
      {shown ? (
        <>
          <p style={{ margin: '0 0 0.8rem' }} data-k="prediction">
            <span className={'chip ' + (same ? 'ok' : 'fail')}>{same ? tr('as you predicted', 'đúng như bạn dự đoán') : tr('not quite', 'chưa hẳn')}</span>{' '}
            <Translated
              vi={
                <>
                  Đúng dưới mọi thứ tự:{' '}
                  {everywhere
                    .map(
                      (v) =>
                        ({
                    'single-both': 'một buffer, cả hai rào chắn',
                    'single-no-read-barrier': 'một buffer, không có rào chắn sau lần đọc',
                    'single-no-write-barrier': 'một buffer, không có rào chắn sau lần ghi',
                    double: 'hai buffer, một rào chắn',
                    'double-no-swap': 'hai buffer, quên hoán đổi',
                    'double-no-barrier': 'hai buffer, không có rào chắn',
                  }[v]),
                    )
                    .join('; ')}
                  .
                </>
              }
            >
              <>Correct under every order: {everywhere.map((v) => VARIANT_LABEL[v]).join('; ')}.</>
            </Translated>
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table className="data" style={{ background: 'var(--page)', margin: 0 }}>
              <thead>
                <tr>
                  <th style={{ paddingLeft: '0.6rem' }}>
                    <Translated vi="Kernel">Kernel</Translated>
                  </th>
                  {SCHEDULES.map((s) => (
                    <th key={s}>
                      <Translated
                        vi={
                          {
                    lockstep: 'mọi luồng chạy đồng bộ từng bước (lockstep)',
                    'in-order': 'mỗi luồng chạy xa nhất có thể, luồng 0 trước',
                    reversed: 'mỗi luồng chạy xa nhất có thể, luồng cuối trước',
                    shuffled: 'xen kẽ ngẫu nhiên',
                  }[s]
                        }
                      >
                        {SCHEDULE_LABEL[s]}
                      </Translated>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.map(({ v, cells }) => (
                  <tr key={v} data-variant={v}>
                    <td style={{ paddingLeft: '0.6rem' }}>
                      <Translated
                        vi={
                          {
                    'single-both': 'một buffer, cả hai rào chắn',
                    'single-no-read-barrier': 'một buffer, không có rào chắn sau lần đọc',
                    'single-no-write-barrier': 'một buffer, không có rào chắn sau lần ghi',
                    double: 'hai buffer, một rào chắn',
                    'double-no-swap': 'hai buffer, quên hoán đổi',
                    'double-no-barrier': 'hai buffer, không có rào chắn',
                  }[v]
                        }
                      >
                        {VARIANT_LABEL[v]}
                      </Translated>
                    </td>
                    {cells.map((r, k) => {
                      const kinds = [...new Set(r.violations.map((x) => x.kind))];
                      return (
                        <td key={k} data-outcome={r.correct ? 'correct' : 'wrong'} data-kinds={kinds.join(' ')}>
                          {r.correct ? (
                            <span className="chip ok">
                              <Translated vi="đúng">correct</Translated>
                            </span>
                          ) : (
                            <span className="chip fail">
                              <Translated vi={<>{r.wrong.length} trên {P} sai</>}>
                                <>{r.wrong.length} of {P} wrong</>
                              </Translated>
                            </span>
                          )}
                          {kinds.map((x) => (
                            <div key={x} className="muted">
                              <Translated
                                vi={
                                  {
                    'read-after-write': 'đọc-sau-ghi',
                    'write-after-read': 'ghi-sau-đọc',
                    'wrong-buffer': 'sai buffer (không có race)',
                  }[x]
                                }
                              >
                                {KIND_LABEL[x]}
                              </Translated>
                            </div>
                          ))}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the tiled kernel, double-buffered
function TiledListing() {
  const { reports } = useReports();
  const ch5 = useLabFile('ch05/lab2_tiled/solution/kernel.cu');
  const cu = useLabFile('ch06/lab5_doublebuf/solution/kernel.cu');
  const metal = useLabFile('ch06/lab5_doublebuf/solution/kernel.metal');
  if (!ch5 || !cu || !metal) return null;
  // The CUDA tabs use the labs page's words for the emulator, CUDA and HIP together.
  return (
    <Code
      tabs={[
        { label: tr('Chapter 5: two barriers', 'Chương 5: hai rào chắn'), code: ch5, tiers: MATMUL_TIERS, status: solutionTabStatus(reports, LAB52, 'kernel.cu') },
        { label: tr('Double-buffered (CUDA)', 'Buffer kép (CUDA)'), code: cu, tiers: MATMUL_TIERS, status: solutionTabStatus(reports, LAB65, 'kernel.cu') },
        { label: tr('Double-buffered (Metal)', 'Buffer kép (Metal)'), code: metal, tiers: MATMUL_TIERS, status: solutionTabStatus(reports, LAB65, 'kernel.metal') },
      ]}
      caption={
        <Translated
          vi={
            <>
              Lời giải tham chiếu của lab 5.2 bên cạnh lời giải của lab 6.5, nạp từ <code>labs/ch05/lab2_tiled/solution/</code>{' '}
              và <code>labs/ch06/lab5_doublebuf/solution/</code>. Các trạng thái lấy từ báo cáo của máy này. Hãy thử{' '}
              <a href={href('ch06', 'labs')}>lab</a> trước khi đọc tab thứ hai.
            </>
          }
        >
          <>
            Lab 5.2's reference solution beside lab 6.5's, loaded from <code>labs/ch05/lab2_tiled/solution/</code> and{' '}
            <code>labs/ch06/lab5_doublebuf/solution/</code>. The statuses come from this machine's reports. Try the{' '}
            <a href={href('ch06', 'labs')}>lab</a> before you read the second tab.
          </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- the reader's shared-memory limit
/** Per-block shared (threadgroup) memory each reporting GPU allows, against `need` bytes. */
function SmemLimits({ need }: { need: number }) {
  const { reports, loading } = useReports();
  if (loading) return null;
  const rows = Object.entries(reports)
    .filter(([b]) => b !== 'emu' && b !== 'race')
    .flatMap(([b, run]) => {
      const key = run.device.maxThreadgroupMemoryLength != null ? 'maxThreadgroupMemoryLength' : 'sharedMemPerBlock';
      const limit = Number(run.device[key]);
      return isFinite(limit) && limit > 0 ? [{ b, key, limit, device: String(run.device.name ?? b) }] : [];
    });
  if (!rows.length)
    return (
      <p data-smem-limit="none">
        <Translated
          vi={
            <>
              Chưa thiết bị nào của bạn báo cáo giới hạn của nó. Chạy <code>python3 labs/pmpp.py doctor</code> trên một máy có
              GPU, rồi tải lại.
            </>
          }
        >
          <>
            None of your devices has reported its limit yet. Run <code>python3 labs/pmpp.py doctor</code> on a machine with
            a GPU, then reload.
          </>
        </Translated>
      </p>
    );
  return (
    <>
      {rows.map((r) => (
        <p key={r.b} data-smem-limit={r.limit} data-backend={r.b}>
          <Translated
            vi={
              <>
                {r.device} ({r.b}) báo cáo <code>{r.key}</code> = {r.limit.toLocaleString('en-US')} byte:{' '}
                {need <= r.limit
                  ? `${need.toLocaleString('en-US')} byte vừa đủ, còn dư ${(r.limit - need).toLocaleString('en-US')}.`
                  : `${need.toLocaleString('en-US')} byte không vừa.`}
              </>
            }
          >
            <>
              {r.device} ({r.b}) reports <code>{r.key}</code> = {r.limit.toLocaleString('en-US')} bytes:{' '}
              {need <= r.limit
                ? `the ${need.toLocaleString('en-US')} bytes fit, with ${(r.limit - need).toLocaleString('en-US')} to spare.`
                : `the ${need.toLocaleString('en-US')} bytes do not fit.`}
            </>
          </Translated>
        </p>
      ))}
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS67: Record<string, Question> = {
  'ch06/6.7/q-which': {
    kind: 'choice',
    id: 'ch06/6.7/q-which',
    prompt: (
      <Translated
        vi={
          <>
            Trong vòng lặp một buffer (đọc, rào chắn, ghi, rào chắn), hai buffer làm cho rào chắn nào trở nên không cần
            thiết?
          </>
        }
      >
        <>
          In the one-buffer loop (read, barrier, write, barrier), which barrier do two buffers make unnecessary?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Rào chắn sau lần đọc, thứ canh giữ phụ thuộc giả.">
            The one after the read, which guards the false dependence.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Rào chắn sau lần ghi, thứ canh giữ phụ thuộc đọc-sau-ghi.">
            The one after the write, which guards the read-after-write dependence.
          </Translated>
        ),
        why: (
          <Translated vi="Rào chắn đó bắt những luồng đọc của lần lặp sau chờ những luồng ghi của lần lặp này. Chúng cần các giá trị mới; không buffer thứ hai nào cung cấp chúng sớm hơn.">
            That one makes the next iteration's readers wait for this iteration's writers. They need the new values; no second buffer supplies them sooner.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Cả hai: với hai buffer, đọc và ghi không bao giờ dùng chung một vị trí.">
            Both: with two buffers, reads and writes never share a location.
          </Translated>
        ),
        why: (
          <Translated vi="Một trong hai phụ thuộc là thật bất kể bố cục bộ nhớ: một luồng đọc vẫn cần giá trị mà một luồng ghi tạo ra.">
            One of the two dependences is real whatever the memory layout: a reader still needs the value a writer produces.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không cái nào: hai buffer tiết kiệm lưu lượng bộ nhớ, không tiết kiệm rào chắn.">
            Neither: two buffers save memory traffic, not barriers.
          </Translated>
        ),
        why: (
          <Translated vi="Chúng không tiết kiệm lưu lượng nào. Một rào chắn tồn tại chỉ vì lần ghi rơi vào chỗ mà một lần đọc chưa diễn ra, và hai buffer bỏ được nó.">
            They save no traffic. One barrier exists only because the write lands where a read has yet to happen, and two buffers remove it.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Rào chắn sau lần đọc ngăn một luồng ghi đè một ô mà luồng láng giềng của nó chưa đọc. Khi các lần ghi đi tới buffer kia, không có gì để ghi đè, nên rào chắn đó có thể bỏ. Trang 147 nói rào chắn "thứ hai" có thể được loại bỏ;
            phần phân tích trước đó, đoạn mã sau đó và câu tiếp theo đều có ý nói rào chắn thứ nhất, sau lần đọc. Ghi chú của
            khóa học này ghi lại điều đó như một lỗi in.
          </p>
        }
      >
        <p>
          The barrier after the read stops a writer from overwriting a slot its neighbour has not read yet. With the writes
          going to the other buffer, there is nothing to overwrite, so that barrier can go. Page 147 says the "second"
          barrier can be eliminated; the analysis before it, the code after it and the sentence that follows all mean the
          first one, after the read. This course's notes record it as an erratum.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.7, pp. 146–147; erratum in notes/ch06.md, item 2',
    vi: { source: 'ch. 6 §6.7, pp. 146–147; lỗi in trong notes/ch06.md, mục 2' },
  },
  'ch06/6.7/q-swap': {
    kind: 'choice',
    id: 'ch06/6.7/q-swap',
    prompt: (
      <Translated vi={<>Vòng lặp buffer kép quên phép hoán đổi. Chuyện gì xảy ra?</>}>
        <>The double-buffered loop forgets the swap. What happens?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mỗi lần lặp lại đọc các giá trị ban đầu: sai trên mọi thiết bị, mọi lần, mà không có race nào.">
            Every iteration reads the starting values again: wrong on every device, every time, with no race.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Một race chỉ một số thứ tự luồng mới để lộ, vì giờ đọc và ghi chồng lên nhau.">
            A race that only some thread orders show, since reads and writes now overlap.
          </Translated>
        ),
        why: (
          <Translated vi="Hãy thử nó trong trình chạy từng bước ở chế độ lockstep: nó cũng sai ở đó. Một race phụ thuộc vào thời điểm; cái này thì không.">
            Try it in the stepper under lockstep: it is wrong there too. A race depends on timing; this does not.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không có gì: rào chắn sau lần ghi dù sao cũng làm các giá trị mới hiển thị.">
            Nothing: the barrier after the write makes the new values visible anyway.
          </Translated>
        ),
        why: (
          <Translated vi="Rào chắn sắp thứ tự các lần đọc và ghi; nó không cho lần lặp sau biết các giá trị mới nằm ở đâu.">
            The barrier orders the reads and writes; it does not tell the next iteration where the new values are.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Không có phép hoán đổi, mỗi luồng đọc buffer A và ghi buffer B ở mọi lần lặp, nên mọi lần lặp lại bắt đầu từ các
            giá trị ban đầu. Bảng kết quả cho thấy nó là hàng duy nhất cũng thất bại dưới lockstep. Điều đó khiến nó là một
            lỗi logic: bộ phát hiện race không có gì để báo, và kết quả sai trên mọi thiết bị.
          </p>
        }
      >
        <p>
          Without the swap each thread reads buffer A and writes buffer B in every iteration, so every iteration starts from
          the initial values again. The outcome table shows it as the one row that fails under lockstep too. That makes it a
          logic error: the race detector has nothing to report, and the result is wrong on every device.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.7, p. 147; the outcome above (our model)',
    vi: { source: 'ch. 6 §6.7, p. 147; bảng kết quả ở trên (mô hình của chúng tôi)' },
  },
  'ch06/6.7/q-swap-all': {
    kind: 'choice',
    id: 'ch06/6.7/q-swap-all',
    prompt: (
      <Translated
        vi={
          <>
            Có phải mọi luồng đều phải thực thi <code>swap(inBuffer, outBuffer)</code>?
          </>
        }
      >
        <>
          Must every thread execute <code>swap(inBuffer, outBuffer)</code>?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Đúng: mỗi luồng giữ bản sao riêng của hai con trỏ.">
            Yes: each thread holds its own copy of the two pointers.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không: một luồng hoán đổi cho cả khối, vì các buffer là chia sẻ.">
            No: one thread swaps for the whole block, since the buffers are shared.
          </Translated>
        ),
        why: (
          <Translated vi="Các con trỏ là biến cục bộ bình thường, mỗi luồng một cặp, trong thanh ghi. Không điều gì luồng khác làm thay đổi được cặp của bạn.">
            The pointers are ordinary local variables, one pair per thread, in registers. Nothing another thread does changes yours.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ luồng 0 của mỗi warp, vì một warp chạy lockstep.">
            Only thread 0 of each warp, since a warp runs in lockstep.
          </Translated>
        ),
        why: (
          <Translated vi="Biến cục bộ cũng không có bản sao theo warp, và dù lockstep hay không, mỗi lane có thanh ghi riêng.">
            There is no per-warp copy of a local variable either, and lockstep or not, each lane has its own registers.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Các buffer là chia sẻ; con trỏ tới chúng thì không, nên một luồng bỏ qua phép hoán đổi sẽ đọc và ghi sai buffer.
            Sách không cho thấy <code>swap</code> được viết thế nào (notes mục 15), và điều đó che mất điểm mấu chốt: nó hoán
            đổi chính các con trỏ của luồng gọi nó. Đoạn mã ở trên truyền chúng theo tham chiếu cho <code>swapPointers</code>,
            và mọi luồng đều gọi nó.
          </p>
        }
      >
        <p>
          The buffers are shared; the pointers to them are not, so a thread that skipped the swap would read and write
          the wrong buffers. The book does not show how <code>swap</code> is written
          (notes item 15), and that hides the point: it exchanges the calling thread's own pointers. The snippet above
          passes them by reference to <code>swapPointers</code>, and every thread calls it.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.7, p. 147; notes/ch06.md, item 15',
    vi: { source: 'ch. 6 §6.7, p. 147; notes/ch06.md, mục 15' },
  },
  'ch06/6.7/q-smem': {
    kind: 'numeric',
    id: 'ch06/6.7/q-smem',
    prompt: (
      <Translated
        vi={
          <>
            Một phép nhân ma trận phân ô với <code>TILE_WIDTH = 32</code> giữ một ô của M và một ô của N trong bộ nhớ chia sẻ,
            dạng <code>float</code>. Các ô của phiên bản buffer kép cần bao nhiêu byte?
          </>
        }
      >
        <>
          A tiled matrix multiplication with <code>TILE_WIDTH = 32</code> keeps one tile of M and one of N in shared memory,
          as <code>float</code>. How many bytes do the tiles of the double-buffered version need?
        </>
      </Translated>
    ),
    answer: 16384,
    unit: 'bytes',
    explain: (
      <Translated
        vi={
          <>
            <p>
              Hai cặp × hai ô × 32 × 32 float × 4 byte = 16,384 byte, gấp đôi 8192 của kernel một buffer. Lượng bộ nhớ đó bị
              tính vào occupancy như mọi lượng khác (mục 5.6), và trước hết nó phải vừa giới hạn mỗi khối của thiết bị bạn.{' '}
              <code>python3 labs/pmpp.py doctor</code> in giới hạn đó là <code>sharedMemPerBlock</code> trên CUDA và HIP và{' '}
              <code>maxThreadgroupMemoryLength</code> trên Metal.
            </p>
            <SmemLimits need={16384} />
          </>
        }
      >
        <>
          <p>
            Two pairs × two tiles × 32 × 32 floats × 4 bytes = 16,384 bytes, twice the 8192 of the single-buffered kernel.
            That memory counts against occupancy like any other (section 5.6), and it has to fit your device&rsquo;s
            per-block limit at all. <code>python3 labs/pmpp.py doctor</code> prints that limit as{' '}
            <code>sharedMemPerBlock</code> on CUDA and HIP and <code>maxThreadgroupMemoryLength</code> on Metal.
          </p>
          <SmemLimits need={16384} />
        </>
      </Translated>
    ),
    source: 'our arithmetic; the limits are the properties in notes/today/ch05-facts.md A1, B2, C3, read from your reports',
    vi: {
      unit: 'byte',
      source: 'phép tính của chúng tôi; các giới hạn là các thuộc tính trong notes/today/ch05-facts.md A1, B2, C3, đọc từ báo cáo của bạn',
    },
  },
  'ch06/6.7/q-async': {
    kind: 'choice',
    id: 'ch06/6.7/q-async',
    prompt: (
      <Translated
        vi={
          <>
            Trên một GPU NVIDIA có compute capability 8.0 trở lên, một khối hai warp, các luồng t = 0 đến 63, điền ô 64 float
            kế tiếp bằng các phép sao chép bất đồng bộ (LDGSTS): luồng t chép phần tử t. Sau đó mỗi luồng chờ các phép sao
            chép của nó hoàn tất và, không có rào chắn, đọc <code>tile[t]</code> và <code>tile[(t + 1) % 64]</code>. Những lần
            đọc nào có thể thấy dữ liệu cũ?
          </>
        }
      >
        <>
          On an NVIDIA GPU of compute capability 8.0 or later, a block of two warps, threads t = 0 to 63, fills the next
          64-float tile with asynchronous copies (LDGSTS): thread t copies element t. Each thread then waits for its
          copies to complete and, with no barrier, reads <code>tile[t]</code> and <code>tile[(t + 1) % 64]</code>. Which
          reads can see stale data?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mọi lần đọc tile[(t + 1) % 64]: mỗi lần đọc dữ liệu do một luồng khác chép, và việc chờ của một luồng chỉ bao phủ các lần sao chép của chính nó.">
            Every read of tile[(t + 1) % 64]: each is another thread's copy, and a thread's wait covers only its own.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không lần nào: khi việc chờ trả về, cả ô đã tới nơi.">
            None: once the wait returns, the whole tile has landed.
          </Translated>
        ),
        why: (
          <Translated vi="NVIDIA nói điều ngược lại: mặc định mỗi luồng chỉ chờ các lần sao chép của chính nó.">
            NVIDIA says the opposite: by default each thread only waits for its own copies.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ hai lần đọc vượt sang warp kia, của luồng 31 và 63.">
            Only the two reads that cross into the other warp, by threads 31 and 63.
          </Translated>
        ),
        why: (
          <Translated vi="Quy tắc tính theo luồng, không theo warp. Luồng 4 đọc phần tử 5 là đọc dữ liệu do luồng 5 chép, và việc chờ của chính nó không nói gì về lần sao chép đó.">
            The rule is per thread, not per warp. Thread 4 reading element 5 reads thread 5's copy, and its own wait says nothing about that copy.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Cả 128 lần đọc, kể cả tile[t]: các lần sao chép bất đồng bộ chỉ hiển thị sau một rào chắn.">
            All 128 reads, tile[t] included: asynchronous copies are visible only after a barrier.
          </Translated>
        ),
        why: (
          <Translated vi="Việc chờ của một luồng chính là điều bảo đảm các lần sao chép của chính nó. Rào chắn dành cho dữ liệu mà các luồng khác đã chép.">
            A thread's wait is exactly what guarantees its own copies. The barrier is for data other threads copied.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            NVIDIA: &ldquo;By default, each thread only waits for its own LDGSTS copies,&rdquo; (mặc định mỗi luồng chỉ chờ
            các lần sao chép LDGSTS của chính nó) nên dữ liệu nạp trước cho các luồng khác cần một <code>__syncthreads()</code> sau
            việc chờ. Các lần đọc <code>tile[t]</code> là an toàn: mỗi lần đọc dữ liệu do chính luồng đó chép. Một ô được nhiều
            luồng điền là dữ liệu chia sẻ, nên khối vẫn cần một rào chắn giữa việc chờ và lần đọc đầu tiên phần tử của một
            luồng khác. Đây là một phụ thuộc thật (đọc sau ghi), loại mà buffer kép không bao giờ loại bỏ. Hàm trợ giúp của
            llama.cpp cho các phép sao chép này có một chú thích nói điều tương tự.
          </p>
        }
      >
        <p>
          NVIDIA: &ldquo;By default, each thread only waits for its own LDGSTS copies,&rdquo; so data prefetched for other
          threads needs a <code>__syncthreads()</code> after the wait. The reads of <code>tile[t]</code> are safe: each
          is the thread&rsquo;s own copy. A tile filled by many threads is shared data, so the
          block still needs a barrier between the wait and the first read of another thread&rsquo;s element. This is a true
          dependence (read after write), the kind double buffering never removes. llama.cpp&rsquo;s helper for these copies
          carries a comment saying the same.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch06-facts.md A16, D1 (not in the book)',
    vi: { source: 'notes/today/ch06-facts.md A16, D1 (không có trong sách)' },
  },
  'ch06/6.7/q-race': {
    kind: 'choice',
    id: 'ch06/6.7/q-race',
    prompt: (
      <Translated vi={<>Kernel buffer kép của bạn chạy sạch trên backend race. Bạn đã chứng tỏ được điều gì?</>}>
        <>Your double-buffered kernel runs clean on the race backend. What have you shown?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không có truy cập thiếu đồng bộ nào xảy ra trong các lịch thực thi đã chạy: bằng chứng mạnh, không phải chứng minh.">
            No unsynchronised access occurred in the schedules that ran: strong evidence, not a proof.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Kernel không có race trên mọi GPU, vì mọi truy cập đều đã được kiểm tra.">
            The kernel is race-free on every GPU, since every access was checked.
          </Translated>
        ),
        why: (
          <Translated vi="ThreadSanitizer chỉ xem xét các kiểu xen kẽ thực sự đã xảy ra trong lần chạy đó.">
            ThreadSanitizer examines only the interleavings that actually happened in that run.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không gì cả: backend race không thấy được bộ nhớ chia sẻ.">
            Nothing: the race backend cannot see shared memory.
          </Translated>
        ),
        why: (
          <Translated vi="Nó thấy mọi truy cập mà các luồng giả lập thực hiện, kể cả bộ nhớ chia sẻ; thứ nó không thấy được là một lịch thực thi chưa từng xảy ra.">
            It sees every access the emulated threads make, shared memory included; what it cannot see is a schedule that never occurred.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Backend race chạy các luồng của một khối như các luồng hệ điều hành thật dưới ThreadSanitizer. Nó báo hai truy cập
            thiếu đồng bộ vào cùng một chỗ kèm cả hai dòng mã nguồn, ngay cả khi lần chạy tình cờ cho ra con số đúng. Một lần
            chạy sạch cho biết lần này không có cặp nào như vậy xảy ra. Kết hợp với một lập luận vì sao nó không thể xảy ra,
            như lập luận trong mục này, đó là mức tốt nhất mà kiểm thử có thể đạt.
          </p>
        }
      >
        <p>
          The race backend runs the threads of a block as real operating-system threads under ThreadSanitizer. It reports two
          unsynchronised accesses to the same place with both source lines, even when the run happened to produce the right
          numbers. A clean run says no such pair occurred this time. Combined with an argument for why it cannot occur, like
          the one in this section, that is as good as testing gets.
        </p>
      </Translated>
    ),
    source: 'labs/README.md, "The race-detection flavour"', // i18n-ignore: file and heading citation
  },
};

export function S67() {
  return (
    <>
      <Translated
        vi={
          <p>
            Giả sử các luồng của một khối truyền giá trị cho nhau qua một buffer chia sẻ, bên trong một vòng lặp. Mỗi luồng
            đọc một giá trị mà một luồng khác đã ghi ở lần lặp trước, rồi ghi giá trị của riêng nó. Phiên bản của sách có hai
            rào chắn (barrier) trong vòng lặp: một sau lần đọc, một sau lần ghi.
          </p>
        }
      >
        <p>
          Suppose the threads of a block pass values to each other through one shared buffer, inside a loop. Each thread
          reads a value another thread wrote in the previous iteration, then writes its own. The book's version has two
          barriers in the loop: one after the read, one after the write.
        </p>
      </Translated>
      <Source section="6.7" pages="146" />
      <Code
        tabs={[{ label: 's67_exchange.cu', code: EXCHANGE, tiers: EXCHANGE_TIERS, status: 'front-end-checked; unverified on CUDA and HIP hardware' }]} // i18n-ignore: file name and verification status
        highlight={[21, 22, 23, 24, 25, 26]}
        caption={
          <Translated
            vi={
              <>
                Vòng lặp của sách (p. 146) dưới dạng một kernel, từ <code>labs/ch06/snippets/</code>, với phiên bản hai buffer
                (p. 147) bên dưới. Phép tính, các khai báo và <code>swapPointers</code> là của chúng tôi; sách chỉ cho thấy
                vòng lặp.
              </>
            }
          >
            <>
              The book's loop (p. 146) as a kernel, from <code>labs/ch06/snippets/</code>, with the two-buffer version
              (p. 147) below it. The computation, the declarations and <code>swapPointers</code> are ours; the book shows only
              the loop.
            </>
          </Translated>
        }
      />

      <h2>
        <Translated vi="Hai rào chắn, hai loại phụ thuộc">Two barriers, two kinds of dependence</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Rào chắn sau lần ghi bắt mọi luồng ghi xong trước khi bất kỳ ai đọc ở lần lặp sau. Luồng đọc cần giá trị mà luồng
            ghi tạo ra: một phụ thuộc <b>đọc-sau-ghi</b> (read-after-write). Nó được gọi là phụ thuộc <b>thật</b> (true) vì
            không cách sắp xếp lại bộ nhớ nào loại bỏ được nó.
          </p>
        }
      >
        <p>
          The barrier after the write makes every thread finish writing before anyone reads in the next iteration. The reader
          needs the value the writer produces: a <b>read-after-write</b> dependence. It is called a <b>true</b> dependence
          because no rearrangement of memory removes it.
        </p>
      </Translated>
      <Source section="6.7" pages="146–147" />
      <Translated
        vi={
          <p>
            Rào chắn sau lần đọc bắt mọi luồng đọc xong trước khi bất kỳ ai ghi đè một ô. Ở đây luồng ghi không cần gì từ
            luồng đọc. Chúng va chạm chỉ vì lần đọc của lần lặp này và lần ghi giá trị kế tiếp dùng cùng một vị trí: một phụ
            thuộc <b>ghi-sau-đọc</b> (write-after-read), được gọi là <b>giả</b> (false). Đặt lần ghi ở chỗ khác thì nó biến
            mất. Đây là cùng hai loại như rào chắn 1 và 2 của{' '}
            <a href={href('ch05', '5.4')}>kernel phân ô ở chương 5</a>.
          </p>
        }
      >
        <p>
          The barrier after the read makes every thread finish reading before anyone overwrites a slot. Here the writer needs
          nothing from the reader. They collide only because this iteration's read and the next value's write use the same
          location: a <b>write-after-read</b> dependence, called <b>false</b>. Put the write somewhere else and it is gone.
          These are the same two kinds as barriers 1 and 2 of <a href={href('ch05', '5.4')}>chapter 5's tiled kernel</a>.
        </p>
      </Translated>
      <Source section="6.7" pages="147" />
      <Translated
        vi={
          <p>
            Trong trình chạy từng bước, đặt kernel là "một buffer, không có rào chắn sau lần đọc" với mỗi luồng chạy xa nhất
            có thể, và nhấn <i>Đi tới lỗi đầu tiên</i>. Rồi thử "không có rào chắn sau lần ghi", rồi lockstep.
          </p>
        }
      >
        <p>
          In the stepper, set the kernel to "no barrier after the read" with each thread running as far as it can, and press{' '}
          <i>Go to the first failure</i>. Then try "no barrier after the write", and then lockstep.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một luồng tiến từng thao tác ở mức mã nguồn; luồng thật xen kẽ ở mức lệnh, theo warp. "Lockstep" là một warp
              đơn lý tưởng hóa. Không có thời gian: mô hình nói luồng có thể thấy những giá trị nào, không nói việc gì tốn bao
              lâu.
            </p>
          }
        >
          <p>
            A thread advances one source-level operation at a time; real threads interleave at the instruction level, in
            warps. "Lockstep" is an idealised single warp. There is no timing: the model says which values a thread can see,
            not how long anything takes.
          </p>
        </Translated>
      </Simplification>
      <Stepper />
      <Outcomes />

      <h2>
        <Translated vi="Hai buffer">Two buffers</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một buffer mới cho mỗi lần lặp sẽ loại bỏ phụ thuộc giả, nhưng nó dùng bộ nhớ tỉ lệ với số lần lặp. Hai buffer là
            đủ: đọc từ một cái, ghi vào cái kia, và hoán đổi vai trò của chúng mỗi lần lặp. Còn lại một rào chắn, sau lần ghi,
            cho phụ thuộc thật. Đó là <b>buffer kép</b> (double buffering).
          </p>
        }
      >
        <p>
          A fresh buffer for every iteration would remove the false dependence, but it uses memory in proportion to the trip
          count. Two buffers are enough: read from one, write into the other, and swap their roles each iteration. One
          barrier remains, after the write, for the true dependence. That is <b>double buffering</b>.
        </p>
      </Translated>
      <Source section="6.7" pages="147" />
      <Ask q={questionsS67['ch06/6.7/q-which']} />
      <Translated
        vi={
          <p>
            Phép hoán đổi là phần dễ làm sai. Sau rào chắn, các giá trị mới nhất nằm trong <code>outBuffer</code>, nên lần lặp
            sau phải đọc ở đó: mọi luồng hoán đổi cặp con trỏ của riêng nó ở cuối thân vòng lặp.
          </p>
        }
      >
        <p>
          The swap is the easy part to get wrong. After the barrier, the newest values are in <code>outBuffer</code>, so the
          next iteration must read there: every thread swaps its own pair of pointers at the end of the body.
        </p>
      </Translated>
      <Ask q={questionsS67['ch06/6.7/q-swap']} />
      <Ask q={questionsS67['ch06/6.7/q-swap-all']} />
      <DoubleBufferScene />

      <h2>
        <Translated vi="Quay lại kernel phân ô">Back to the tiled kernel</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Kernel phân ô của chương 5 ghi trước và đọc sau trong mỗi pha: nạp các ô, rào chắn 1, tính từ chúng, rào chắn 2.
            Rào chắn 2 là rào chắn giả. Nó chỉ ngăn một luồng nhanh nạp ô của pha kế tiếp đè lên những giá trị mà một luồng
            láng giềng chậm vẫn đang đọc. Hai cặp ô, luân phiên theo pha, loại bỏ nó, và lab 6.5 yêu cầu bạn làm điều đó. Lời
            giải tham chiếu của lab chọn cặp theo số pha thay vì hoán đổi con trỏ: cùng một kiểu luân phiên.
          </p>
        }
      >
        <p>
          Chapter 5's tiled kernel writes first and reads second in each phase: load the tiles, barrier 1, compute from them,
          barrier 2. Barrier 2 is the false one. It only stops a fast thread from loading the next phase's tile over values a
          slow neighbour is still reading. Two pairs of tiles, alternating by phase, remove it, and lab 6.5 asks you to do
          that. The lab's reference picks the pair with the phase number instead of swapping pointers: the same alternation.
        </p>
      </Translated>
      <Source section="6.7" pages="146–147" />
      <TiledListing />
      <Translated
        vi={
          <p>
            Cái giá là bộ nhớ chia sẻ: gấp đôi số ô, bị tính vào occupancy như mọi bộ nhớ chia sẻ khác (
            <a href={href('ch05', '5.6')}>mục 5.6</a>).
          </p>
        }
      >
        <p>
          The price is shared memory: twice the tiles, which count against occupancy like any other shared memory (
          <a href={href('ch05', '5.6')}>section 5.6</a>).
        </p>
      </Translated>
      <Ask q={questionsS67['ch06/6.7/q-smem']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Lab 6.5 cũng chạy trên backend race, backend chạy các luồng của một khối đồng thời dưới ThreadSanitizer và chỉ ra
            cả hai dòng mã nguồn của bất kỳ cặp truy cập thiếu đồng bộ nào nó thấy. Mã khởi đầu (starter) là kernel phân ô với
            rào chắn 2 đã bị xóa; trang lab cho thấy mỗi backend đã tìm thấy gì trên máy này.
          </p>
        }
      >
        <p>
          Lab 6.5 also runs on the race backend, which runs the threads of a block concurrently under ThreadSanitizer and
          names both source lines of any unsynchronised pair of accesses it sees. The starter is the tiled kernel with
          barrier 2 deleted; the labs page shows what each backend found on this machine.
        </p>
      </Translated>
      <Ask q={questionsS67['ch06/6.7/q-race']} />

      <h2>
        <Translated vi="Nơi bạn sẽ gặp nó">Where you will meet it</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách gọi ví dụ này là một bản minh họa; công dụng thật của nó đến ở chương 11 (scan Kogge-Stone) và chương 15
            (phép nhân ma trận).
          </p>
        }
      >
        <p>
          The book calls this example a toy; its real uses come in chapter 11 (the Kogge-Stone scan) and chapter 15 (matrix
          multiplication).
        </p>
      </Translated>
      <Source section="6.8" pages="152" />
      <Translated
        vi={
          <p>
            Ngày nay ý tưởng này thường đi cùng các phép sao chép bất đồng bộ: một buffer được điền trong khi các luồng tính
            trên buffer kia, điều đi xa hơn một bước so với việc bỏ một rào chắn.
          </p>
        }
      >
        <p>
          Today the idea usually arrives together with asynchronous copies: one buffer is filled while the threads compute
          on the other, which goes a step further than removing a barrier.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md, A16, B8, C7, D1, D3)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <>
              <p>
                Ngày nay NVIDIA chủ yếu mô tả buffer kép cùng với các phép sao chép bất đồng bộ vào bộ nhớ chia sẻ (LDGSTS từ
                compute capability 8.0, TMA, <code>cuda::memcpy_async</code>): một buffer được điền trong khi buffer kia được
                dùng. Mỗi luồng chỉ chờ các lần sao chép của chính nó, nên vẫn cần <code>__syncthreads()</code>.
              </p>
              <p>
                HIP chỉ mô tả một phép sao chép bất đồng bộ từ bộ nhớ toàn cục sang LDS cho CDNA 3 và 4, qua{' '}
                <code>__builtin_amdgcn_async_work_group_copy</code> (chưa được xác nhận trong mã nguồn trình biên dịch); nó
                không có <code>cuda::pipeline</code>. Metal không mô tả cái nào, nên dạng hai mảng đồng bộ là dạng di động
                (portable).
              </p>
            </>
          }
        >
          <>
            <p>
              NVIDIA now documents double buffering mostly together with asynchronous copies into shared memory (LDGSTS from
              compute capability 8.0, TMA, <code>cuda::memcpy_async</code>): one buffer is filled while the other is used.
              Each thread waits only for its own copies, so <code>__syncthreads()</code> is still needed.
            </p>
            <p>
              HIP documents an asynchronous global-to-LDS copy only for CDNA 3 and 4, through{' '}
              <code>__builtin_amdgcn_async_work_group_copy</code> (not confirmed in compiler sources); it has no{' '}
              <code>cuda::pipeline</code>. Metal documents none, so the synchronous two-array form is the portable one.
            </p>
          </>
        </Translated>
      </Today>
      <Ask q={questionsS67['ch06/6.7/q-async']} />
      <Translated vi={<p>Mã bạn sẽ đọc cho thấy cả hai đầu của dải đó.</p>}>
        <p>The code you will read shows both ends of that range.</p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Mã thật, đọc vào 2026-11-20: kernel flash-attention CUDA của llama.cpp chạy hai tầng pipeline với các phép sao
              chép bất đồng bộ nơi phần cứng hỗ trợ. Vòng lặp GEMM Metal chính của MLX dùng một buffer với hai rào chắn mỗi
              bước, tức vòng lặp "trước" của sách.
            </p>
          }
        >
          <p>
            Real code, read on 2026-11-20: llama.cpp's CUDA flash-attention kernel runs two pipeline stages with
            asynchronous copies where the hardware has them. MLX's main Metal GEMM loop is single-buffered with two
            barriers per step, the book's "before" loop.
          </p>
        </Translated>
      </Beyond>
    </>
  );
}
