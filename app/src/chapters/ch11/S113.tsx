import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source } from '../../components/notes';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { benchRows, fmt, num, realBackends, useReports, type BenchRow } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  EXERCISE_1,
  FIG_11_3,
  FIG_11_5,
  KIND_LABEL,
  SCHEDULES,
  SCHEDULE_LABEL,
  barriersPerScan,
  iterations,
  kernelSource,
  resultBuffer,
  run,
  sharedBytes,
  swaps,
  type Config,
  type Schedule,
} from '../../sim/ksblock';
import { functionSource } from '../../sim/toolchain';
import { TraceView } from './KsGrid';

// ---------------------------------------------------------------- two buffers, one barrier
type VariantId = 'double' | 'no-copy' | 'single';
const VARIANTS: { id: VariantId; readonly label: string; cfg: Omit<Config, 'threads'> }[] = [
  {
    id: 'double',
    get label() {
      return tr('two buffers, one barrier (Fig. 11.5)', 'hai buffer, một rào chắn (Fig. 11.5)');
    },
    cfg: FIG_11_5,
  },
  {
    id: 'no-copy',
    get label() {
      return tr('two buffers, copy branch left out', 'hai buffer, bỏ nhánh sao chép');
    },
    cfg: { ...FIG_11_5, copy: false },
  },
  {
    id: 'single',
    get label() {
      return tr('one buffer, barrier 1 only', 'một buffer, chỉ có rào chắn 1');
    },
    cfg: { ...FIG_11_3, barrier2: false },
  },
];

function TwoBuffers() {
  const [variant, setVariant] = useState<VariantId>('double');
  const [schedule, setSchedule] = useState<Schedule>('in-order');
  const v = VARIANTS.find((x) => x.id === variant)!;
  const locale = useLocale();
  const trace = useMemo(() => run({ ...v.cfg, threads: 8 }, EXERCISE_1, schedule), [v, schedule, locale]);
  return (
    <Bench
      title={tr('Two buffers, one barrier', 'Hai buffer, một rào chắn')}
      task={
        <Translated vi="Chạy từng kernel đến cuối theo từng thứ tự. Kernel nào đúng ở mọi thứ tự, và kernel nào sai ở mọi thứ tự?">
          Run each kernel to the end in each order. Which one is right in every order, and which is wrong in every order?
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Các hàng luân phiên giữa hai buffer vật lý, như trong Fig. 11.4: buffer1_s chứa dữ liệu nạp vào, buffer2_s
              chứa những gì lần lặp 0 đã ghi, và cứ thế. Mũi tên nét đứt là các lần sao chép. Với một buffer, mỗi hàng là
              buffer_s sau bước nhảy (stride) đó.
            </>
          }
        >
          <>
            Rows alternate between the two physical buffers, as in Fig. 11.4: buffer1_s holds the load, buffer2_s what
            iteration 0 wrote, and so on. Dashed arrows are copies. For one buffer, a row is buffer_s after that stride.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Kernel">Kernel</Translated>
          <select value={variant} onChange={(e) => setVariant(e.target.value as VariantId)} aria-label={tr('Kernel', 'Kernel')}>
            {VARIANTS.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </label>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Thứ tự luồng">Thread order</Translated>
          <select value={schedule} onChange={(e) => setSchedule(e.target.value as Schedule)} aria-label={tr('Thread order', 'Thứ tự luồng')}>
            {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
          </select>
        </label>
      </div>
      <TraceView trace={trace} />
    </Bench>
  );
}

function Outcomes() {
  return (
    <Bench title={tr('Three kernels, every order', 'Ba kernel, mọi thứ tự')} task={<Translated vi="Lỗi nào phụ thuộc vào thứ tự, và lỗi nào không?">Which failures depend on the order, and which do not?</Translated>}
      foot={tr("Each cell runs the whole block of 8 threads in the model above on exercise 1's values.", 'Mỗi ô chạy toàn bộ khối 8 luồng trong mô hình ở trên với các giá trị của bài tập 1.')}>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-part="outcomes">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}><Translated vi="Kernel">Kernel</Translated></th>
              {SCHEDULES.map((s) => <th key={s}>{SCHEDULE_LABEL[s]}</th>)}
            </tr>
          </thead>
          <tbody>
            {VARIANTS.map((v) => (
              <tr key={v.id} data-variant={v.id}>
                <td style={{ paddingLeft: '0.6rem' }}>{v.label}</td>
                {SCHEDULES.map((s) => {
                  const r = run({ ...v.cfg, threads: 8 }, EXERCISE_1, s);
                  const kinds = [...new Set(r.violations.map((x) => x.kind))];
                  return (
                    <td key={s} data-outcome={r.correct ? 'correct' : 'wrong'} data-wrong={r.wrong.join(' ')}>
                      {r.correct ? <span className="chip ok"><Translated vi="đúng">correct</Translated></span> : <span className="chip fail">{r.wrong.length} <Translated vi="trên 8 sai">of 8 wrong</Translated></span>}
                      {kinds.map((k) => <div key={k} className="muted">{KIND_LABEL[k]}</div>)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- where is the answer
function SwapReplay({ committed }: { committed: number }) {
  const [B, setB] = useState(256);
  const sw = swaps(B);
  const res = resultBuffer(B);
  return (
    <div data-readout="buffer" data-result={res}>
      <Translated
        vi={
          <p>
            Một khối {B} luồng chạy {sw.length} lần lặp (bước nhảy từ 1 đến {B / 2}). Mỗi lần kết thúc bằng một lần hoán
            đổi, nên inBuffer_s lần lượt chỉ tới:{' '}
            <span className="mono">
              buffer1_s{sw.map((s) => ` → buffer${s.outBuf + 1}_s`).join('')}
            </span>
            . Sau vòng lặp nó chỉ tới <b>buffer{res}_s</b>, và đáp án nằm ở đó.{' '}
            {B === 256 ? (committed === res ? 'Bạn đã đoán đúng.' : 'Một số chẵn lần hoán đổi đưa nó trở về buffer1_s.') : ''}
          </p>
        }
      >
        <p>
          A block of {B} threads runs {sw.length} iterations (strides 1 to {B / 2}). Each ends with a swap, so inBuffer_s
          names, in turn:{' '}
          <span className="mono">
            buffer1_s{sw.map((s) => ` → buffer${s.outBuf + 1}_s`).join('')}{/* i18n-ignore: array name */}
          </span>
          . After the loop it names <b>buffer{res}_s</b>, which is where the answer is.{' '}
          {B === 256 ? (committed === res ? 'You had it.' : 'An even number of swaps brings it back to buffer1_s.') : ''}
        </p>
      </Translated>
      <div className="row">
        <span className="muted"><Translated vi="Thử một kích thước khối khác:">Try another block size:</Translated></span>
        <span className="seg" role="group" aria-label={tr('Block size', 'Kích thước khối')}>
          {[8, 256, 512, 1024].map((b) => (
            <button key={b} aria-pressed={B === b} onClick={() => setB(b)}>{b}</button>
          ))}
        </span>
      </div>
      <Translated
        vi={
          <p style={{ marginBottom: 0 }}>
            Kernel không bao giờ cần biết: nó luôn đọc <code>inBuffer_s</code>, và tính chẵn lẻ của log₂B quyết định đó là
            mảng nào.
          </p>
        }
      >
        <p style={{ marginBottom: 0 }}>
          The kernel never needs to know: it always reads <code>inBuffer_s</code>, and the parity of log₂B decides which
          array that is.
        </p>
      </Translated>
    </div>
  );
}

// ---------------------------------------------------------------- measured: two barriers against two buffers
// Lab 11.1 writes rows named "scan" with a "kernel" key (two_barrier, double_buffer, ...) and a "block" size.
const SIZE_KEYS = ['block', 'threads', 'seg', 'seg_size', 'block_size', 'B'];
const pickRow = (rows: BenchRow[], name: string) => {
  const named = rows.filter((r) => r.kernel === name || r.name === name);
  return named.find((r) => SIZE_KEYS.some((k) => r[k] === 1024)) ?? named[0];
};
/** How many times faster b ran than a: from times if both have one, else from a throughput. */
function speedup(a: BenchRow, b: BenchRow): number | undefined {
  for (const k of ['seconds', 'ms', 'us', 'time_ms', 'time_us']) {
    const x = num(a[k]);
    const y = num(b[k]);
    if (x && y) return x / y;
  }
  for (const k of Object.keys(a).filter((k) => /per_s$|gflops|gb_s/.test(k))) {
    const x = num(a[k]);
    const y = num(b[k]);
    if (x && y) return y / x;
  }
  return undefined;
}
function MeasuredDouble({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = realBackends(reports).flatMap((be) => {
    const labs = [...new Set(reports[be].results.map((r) => r.lab).filter((l) => l.startsWith('ch11/lab1')))];
    for (const lab of labs) {
      const rows = benchRows(reports[be], lab);
      const two = pickRow(rows, 'two_barrier');
      const dbl = pickRow(rows, 'double_buffer');
      const s = two && dbl ? speedup(two, dbl) : undefined;
      if (s) return [{ be, device: String(reports[be].device.name ?? be), s }];
    }
    return [];
  });
  if (!found.length)
    return (
      <Translated
        vi={
          <p data-readout="measured" data-found="0">
            Máy này chưa có số đo nào của lab 11.1. Hãy chạy nó với <code>--bench</code> từ{' '}
            <a href={href('ch11', 'labs')}>trang lab</a>, ghi lại bằng <code>./verify.sh</code>, rồi quay lại: ô này sẽ so
            sánh dự đoán của bạn với những gì thiết bị của bạn đã làm.
          </p>
        }
      >
        <p data-readout="measured" data-found="0">
          No measurement of lab 11.1 on this machine yet. Run it with <code>--bench</code> from the{' '}
          <a href={href('ch11', 'labs')}>labs page</a>, record it with <code>./verify.sh</code>, and come back: this box will
          compare your prediction with what your device did.
        </p>
      </Translated>
    );
  return (
    <>
      {found.map((f) => (
        <Translated
          key={f.be}
          vi={
            <p data-readout="measured" data-found="1">
              Trên <b>{f.device}</b> ({f.be}), kernel buffer kép chạy nhanh gấp <b>{fmt(f.s)}×</b> kernel hai rào chắn.{' '}
              {predicted > f.s * 1.5 ? 'Bạn đã kỳ vọng nhiều hơn.' : predicted < f.s / 1.5 ? 'Bạn đã kỳ vọng ít hơn.' : 'Kết quả này gần với dự đoán của bạn.'}
            </p>
          }
        >
          <p key={f.be} data-readout="measured" data-found="1">
            On <b>{f.device}</b> ({f.be}) the double-buffered kernel ran <b>{fmt(f.s)}×</b> as fast as the two-barrier
            kernel.{' '}
            {predicted > f.s * 1.5 ? 'You expected more.' : predicted < f.s / 1.5 ? 'You expected less.' : 'That is close to your prediction.'}
          </p>
        </Translated>
      ))}
      <p className="muted"><Translated vi="Đây là số đo từ máy của bạn, không phải giá trị kỳ vọng; các thiết bị khác có thể khác theo cả hai hướng.">A measurement from your machine, not an expected value; other devices can differ either way.</Translated></p>
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS113: Record<string, Question> = {
  'ch11/11.3/q-which-barrier': {
    kind: 'choice',
    id: 'ch11/11.3/q-which-barrier',
    prompt: <Translated vi={<>Buffer kép loại bỏ rào chắn nào của Fig. 11.3?</>}><>Which barrier of Fig. 11.3 does double buffering remove?</></Translated>,
    options: [
      { label: <Translated vi="Dòng 14, vốn chỉ bảo vệ một quan hệ ghi-sau-đọc.">Line 14, which guards only a write-after-read.</Translated>, correct: true },
      {
        label: <Translated vi="Dòng 10, ở đầu vòng lặp: nó chỉ ngăn bên ghi đè lên những giá trị vẫn đang được đọc.">Line 10, at the top of the loop: it only keeps writers from overwriting values still being read.</Translated>,
        why: <Translated vi="Đó là mô tả của dòng 14. Dòng 10 bắt bên đọc chờ kết quả của lần lặp trước, một phụ thuộc thật, và hai buffer không thay đổi điều đó.">That describes line 14. Line 10 makes readers wait for the previous iteration's results, a true dependence, and two buffers do not change that.</Translated>,
      },
      {
        label: <Translated vi="Cả hai: với hai buffer, không luồng nào đọc một ô mà luồng khác ghi.">Both: with two buffers no thread reads a slot that another thread writes.</Translated>,
        why: <Translated vi="Bên đọc vẫn cần những giá trị mà các luồng khác đã ghi ở lần lặp trước, và một trong hai rào chắn bảo vệ đúng điều đó.">Readers still need the values other threads wrote in the previous iteration, and one of the two barriers guards exactly that.</Translated>,
      },
      { label: <Translated vi="Không cái nào; nó chỉ tiết kiệm lưu lượng bộ nhớ chia sẻ, không bớt rào chắn.">Neither; it only saves shared-memory traffic, not barriers.</Translated>, why: <Translated vi="Nó bớt được một rào chắn mỗi lần lặp, và tốn thêm bộ nhớ chia sẻ.">It saves a barrier per iteration, and costs shared memory.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Dòng 14 áp đặt một phụ thuộc ghi-sau-đọc, hay phụ thuộc giả: lệnh ghi phải chờ những luồng đọc giá trị cũ chỉ
            vì giá trị cũ và mới dùng chung một ô. Với buffer đầu vào và đầu ra tách riêng thì không còn như vậy. Dòng 10
            áp đặt một phụ thuộc thật (đọc-sau-ghi): bên đọc cần giá trị mà một luồng khác đã ghi ở lần lặp trước.
            Fig. 11.5 giữ rào chắn đó, ở dòng 09 của nó.
          </p>
        }
      >
        <p>
          Line 14 enforces a write-after-read, or false, dependence: the store must wait for readers of the old value
          only because old and new share a slot. With separate input and output buffers they do not. Line 10 enforces a
          true (read-after-write) dependence: a reader needs the value another thread wrote in the previous iteration.
          Fig. 11.5 keeps that one, as its line 09.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.3, pp. 258–261',
  },
  'ch11/11.3/q-copy': {
    kind: 'choice',
    id: 'ch11/11.3/q-copy',
    prompt: <Translated vi={<>Trong Fig. 11.5, các luồng không cộng bỏ qua bước sao chép vào outBuffer_s. Điều gì hỏng?</>}><>In Fig. 11.5, the threads that are not adding skip the copy into outBuffer_s. What goes wrong?</></Translated>,
    options: [
      { label: <Translated vi="Các ô đầu ra của chúng giữ giá trị cũ, và các lần lặp sau đọc chúng như giá trị cuối cùng.">Their output slots keep stale values, which later iterations read as final.</Translated>, correct: true },
      {
        label: <Translated vi="Không gì cả: các vị trí đó đã là giá trị cuối cùng trong buffer đầu vào, nên kết quả không đổi.">Nothing: those positions are final in the input buffer, so the result is unchanged.</Translated>,
        why: <Translated vi="Chúng là giá trị cuối cùng trong buffer đầu vào. Lần lặp tiếp theo đọc buffer kia.">They are final in the input buffer. The next iteration reads the other buffer.</Translated>,
      },
      {
        label: <Translated vi="Một race condition: không có bước sao chép, bên đọc của lần lặp tiếp theo không được sắp sau bên ghi.">A race: without the copy, readers of the next iteration are not ordered after the writers.</Translated>,
        why: <Translated vi="Kết quả sai theo cùng một cách ở mọi thứ tự: không có yếu tố thời điểm nào ở đây.">The result is wrong the same way in every order: no timing is involved.</Translated>,
      },
      { label: <Translated vi="Chỉ đầu ra cuối cùng bị sai.">Only the last output is wrong.</Translated>, why: <Translated vi="Những vị trí rảnh việc sớm nhất mới là những vị trí bị ảnh hưởng.">The positions that go idle first are the ones hit.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Với một buffer, một giá trị đã xong cứ nằm yên tại chỗ. Với hai buffer, mỗi lần lặp ghi vào một buffer mới, và
            chỗ nào không được ghi thì chứa thứ mà buffer đó giữ từ hai lần lặp trước. Luồng 0 ở lần lặp 0 phải sao chép
            x0 sang, nếu không ô 0 của buffer2_s sẽ không bao giờ được ghi (p. 260). Trình chạy từng bước cho thấy điều
            đó: kết quả sai ở mọi thứ tự luồng, và sai y hệt nhau.
          </p>
        }
      >
        <p>
          With one buffer, a finished value simply stays where it is. With two, each iteration writes a fresh buffer, and
          whatever is not written there is whatever that buffer held two iterations ago. Thread 0 in iteration 0 must copy
          x0 across, or slot 0 of buffer2_s is never written at all (p. 260). The stepper shows it: the result is wrong in
          every thread order, identically.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.3, pp. 259–260',
  },
  'ch11/11.3/q-buffer': {
    kind: 'choice',
    id: 'ch11/11.3/q-buffer',
    prompt: <Translated vi={<>Với một khối 512 luồng, mảng nào chứa kết quả của Fig. 11.5 sau vòng lặp?</>}><>For a 512-thread block, which array holds the result of Fig. 11.5 after the loop?</></Translated>,
    options: [
      { label: 'buffer1_s', why: <Translated vi="Hãy đếm số lần hoán đổi: điều đó cần một số chẵn lần lặp.">Count the swaps: that would need an even number of iterations.</Translated> }, // i18n-ignore: array name
      { label: 'buffer2_s', correct: true }, // i18n-ignore: array name
      { label: <Translated vi="Phụ thuộc vào thứ tự luồng.">It depends on the thread order.</Translated>, why: <Translated vi="Mọi luồng hoán đổi con trỏ của mình y hệt nhau, mỗi lần lặp một lần.">Every thread swaps its pointers identically, once per iteration.</Translated> },
      { label: <Translated vi="Cả hai đều chứa nó.">Both hold it.</Translated>, why: <Translated vi="Chỉ buffer đầu ra cuối cùng có các giá trị cuối cùng.">Only the last output buffer has the final values.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Vòng lặp chạy 9 lần lặp, bước nhảy từ 1 đến 256, mỗi lần kết thúc bằng một lần hoán đổi. Sau một số lẻ lần
            hoán đổi, inBuffer_s chỉ tới buffer2_s. Kernel đọc inBuffer_s, nên nó không bao giờ phải biết.
          </p>
        }
      >
        <p>
          The loop runs 9 iterations, strides 1 to 256, each ending in a swap. After an odd number of swaps inBuffer_s
          points at buffer2_s. The kernel reads inBuffer_s, so it never has to know.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.3, p. 260 (the count is ours)',
    vi: { source: 'ch. 11 §11.3, p. 260 (phép đếm là của khóa học này)' },
  },
  'ch11/11.3/q-bytes': {
    kind: 'numeric',
    id: 'ch11/11.3/q-bytes',
    prompt: <Translated vi={<>Một khối của Fig. 11.5 dùng bao nhiêu byte bộ nhớ chia sẻ với SEG_SIZE = 1024 số float?</>}><>How many bytes of shared memory does one block of Fig. 11.5 use with SEG_SIZE = 1024 floats?</></Translated>,
    answer: sharedBytes(1024, 'double'),
    unit: 'bytes',
    explain: <Translated vi={<p>Hai mảng gồm 1024 số float bốn byte: 8192 byte, gấp đôi 4096 của Fig. 11.3.</p>}><p>Two arrays of 1024 four-byte floats: 8192 bytes, twice Fig. 11.3's 4096.</p></Translated>,
    source: 'ch. 11 Figs. 11.3 and 11.5, pp. 256, 260',
    vi: { unit: 'byte', source: 'ch. 11 Figs. 11.3 và 11.5, pp. 256, 260' },
  },
  'ch11/11.3/q-fig119': {
    kind: 'choice',
    id: 'ch11/11.3/q-fig119',
    prompt: (
      <Translated
        vi={
          <>
            Fig. 11.9, ở phần tiếp theo, có hai rào chắn: một sau khi các warp ghi tổng của mình vào bộ nhớ chia sẻ, một sau
            khi warp 0 đã scan các tổng đó. Buffer kép có loại bỏ được rào chắn nào không?
          </>
        }
      >
        <>
          Fig. 11.9, in the next section, has two barriers: one after the warps write their sums to shared memory, one after
          warp 0 has scanned those sums. Could double buffering remove either?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Không: cả hai đều bảo vệ phụ thuộc thật.">No: both guard true dependences.</Translated>, correct: true },
      {
        label: <Translated vi="Có, rào chắn thứ nhất: nó chỉ bảo vệ tổng của các warp khỏi bị ghi đè.">Yes, the first: it only protects the warp sums from being overwritten.</Translated>,
        why: <Translated vi="Nó bắt warp 0 chờ những tổng mà các warp khác đã ghi. Đó là đọc sau ghi.">It makes warp 0 wait for the sums the other warps wrote. That is a read after a write.</Translated>,
      },
      {
        label: <Translated vi="Có, rào chắn thứ hai: warp 0 ghi ngược vào chính các ô nó vừa đọc.">Yes, the second: warp 0 writes back into the slots it has just read.</Translated>,
        why: <Translated vi="Mỗi lane ghi lại ô mà chính nó đã đọc, việc này không cần rào chắn. Rào chắn bắt các warp khác chờ những lần ghi đó: lại là đọc sau ghi.">Each lane rewrites the slot it read itself, which needs no barrier. The barrier makes the other warps wait for those writes: a read after a write again.</Translated>,
      },
      { label: <Translated vi="Có, cả hai: hai buffer loại bỏ mọi rào chắn giữa các giai đoạn.">Yes, both: two buffers remove every barrier between the stages.</Translated>, why: <Translated vi="Buffer kép chỉ loại bỏ xung đột ghi-sau-đọc.">Double buffering removes only write-after-read conflicts.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi rào chắn ở đó tách một lần ghi khỏi những lần đọc cần nó: tổng của các warp được ghi trước khi warp 0 đọc
            chúng, các tổng đã scan được ghi trước khi các warp khác đọc chúng. Không rào chắn nào bảo vệ một giá trị cũ
            khỏi bị ghi đè, nên buffer thứ hai không có gì để sửa.
          </p>
        }
      >
        <p>
          Each barrier there separates a write from the reads that need it: warp sums written before warp 0 reads them,
          scanned sums written before the other warps read them. Neither protects an old value from being overwritten, so
          a second buffer has nothing to fix.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.4, p. 264 (Fig. 11.9); §11.3, pp. 258–259',
    vi: { source: 'ch. 11 §11.4, p. 264 (Fig. 11.9); và §11.3, pp. 258–259' },
  },
  'ch11/11.3/q-final-store': {
    kind: 'choice',
    id: 'ch11/11.3/q-final-store',
    prompt: <Translated vi={<>Vì sao Fig. 11.5 không cần rào chắn giữa cuối vòng lặp và lệnh ghi cuối cùng?</>}><>Why does Fig. 11.5 need no barrier between the end of the loop and the final store?</></Translated>,
    options: [
      { label: <Translated vi="Mỗi luồng ghi ra ô mà chính nó đã ghi.">Each thread stores the slot it wrote itself.</Translated>, correct: true },
      {
        label: <Translated vi="Lệnh ghi bộ nhớ toàn cục tự động được sắp sau các lần ghi bộ nhớ chia sẻ.">Global stores are ordered after shared-memory writes automatically.</Translated>,
        why: <Translated vi="Câu hỏi là về lần đọc bộ nhớ chia sẻ trước lệnh ghi.">The question is about the shared-memory read before the store.</Translated>,
      },
      {
        label: <Translated vi="Vòng lặp kết thúc bằng một rào chắn mà mọi luồng vượt qua trước lệnh ghi.">The loop ends with a barrier that every thread passes before the store.</Translated>,
        why: <Translated vi="Nó bắt đầu bằng một rào chắn; sau lần ghi và lần hoán đổi cuối cùng thì không có rào chắn nào.">It starts with one; after the last write and swap there is none.</Translated>,
      },
      { label: <Translated vi="Có cần; sách đã quên, và race condition hiếm khi lộ ra.">It does need one; the book forgot it, and the race rarely shows.</Translated>, why: <Translated vi="Hãy xem lệnh ghi đọc lần ghi của ai.">Check whose write the store reads.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Cần rào chắn khi một luồng đọc thứ mà luồng khác đã ghi. Ở đây luồng <i>t</i> đọc ô <i>t</i>, tức lần ghi của
            chính nó ở lần lặp cuối (một phép cộng hoặc một lần sao chép), nên thứ tự chương trình đã bảo đảm điều đó. Đó là
            nhận xét của chúng tôi; sách không nói điều này.
          </p>
        }
      >
        <p>
          A barrier is needed when a thread reads what another thread wrote. Here thread <i>t</i> reads slot <i>t</i>, its
          own write from the last iteration (an addition or a copy), so program order already guarantees it. That is our
          observation; the book does not say it.
        </p>
      </Translated>
    ),
    source: 'ch. 11 Fig. 11.5, p. 260; notes/ch11.md, Fig. 11.5 "Care"',
    vi: { source: 'ch. 11 Fig. 11.5, p. 260; notes/ch11.md, Fig. 11.5 mục "Care"' },
  },
};

// ---------------------------------------------------------------- the section
export function S113() {
  const file = useLabFile('ch11/snippets/s112_kogge_stone.cu');
  const fn = file ? functionSource(file, 'double_buffer_scan_kernel') : null;
  const lines = kernelSource(FIG_11_5).line;
  const q = questionsS113;
  return (
    <>
      <Translated
        vi={
          <p>
            Mỗi lần lặp của Fig. 11.3 gọi <code>__syncthreads()</code> hai lần, cho một phép cộng mỗi luồng. Rào chắn
            (barrier) là một trong những chi phí chính của kernel đó, nên ba phần tiếp theo lần lượt loại bỏ chúng, mỗi
            phần một cách.
          </p>
        }
      >
        <p>
          Every iteration of Fig. 11.3 calls <code>__syncthreads()</code> twice, for one addition per thread. Barriers are
          one of the main costs of that kernel, so the next three sections remove them, one way after another.
        </p>
      </Translated>
      <Source section="11.3" pages="258" />
      <Translated
        vi={
          <p>
            Hai rào chắn không áp đặt cùng một điều. Rào chắn thứ nhất (dòng 10) bắt bên đọc chờ cho đến khi các giá trị
            của lần lặp trước đã được ghi: một phụ thuộc <b>thật</b>, hay đọc-sau-ghi. Rào chắn thứ hai (dòng 14) bắt bên
            ghi chờ cho đến khi mọi luồng đã đọc giá trị cũ: một phụ thuộc <b>giả</b>, hay ghi-sau-đọc. Nó tồn tại chỉ vì
            giá trị cũ và giá trị mới của một vị trí dùng chung một ô. Bạn đã gặp cặp này trong phép nhân ma trận phân ô
            (<a href={href('ch05', '5.4')}>phần 5.4</a>) và cách chữa nó ở{' '}
            <a href={href('ch06', '6.7')}>phần 6.7</a>.
          </p>
        }
      >
        <p>
          The two barriers do not enforce the same thing. The first (line 10) makes readers wait until the previous
          iteration's values have been written: a <b>true</b>, or read-after-write, dependence. The second (line 14) makes
          writers wait until everyone has read the old values: a <b>false</b>, or write-after-read, dependence. It exists
          only because the old and the new value of a position share one slot. You met the same pair in the tiled matrix
          multiplication (<a href={href('ch05', '5.4')}>section 5.4</a>) and its cure in{' '}
          <a href={href('ch06', '6.7')}>section 6.7</a>.
        </p>
      </Translated>
      <Source section="11.3" pages="258–259" />
      <Ask q={q['ch11/11.3/q-which-barrier']} />

      <h2><Translated vi="Đọc từ một buffer, ghi vào buffer kia">Read from one buffer, write the other</Translated></h2>
      <Translated
        vi={
          <p>
            <b>Buffer kép</b> đặt đầu ra của mỗi lần lặp ở chỗ khác với đầu vào của nó. Một buffer mới cho mỗi lần lặp
            cũng làm được, nhưng phí bộ nhớ chia sẻ (shared memory). Một khi đầu ra của một lần lặp đã có, đầu vào của nó
            không còn dùng nữa, nên hai buffer là đủ: lần lặp 0 đọc buffer1 và ghi buffer2, lần lặp 1 đọc buffer2 và ghi
            buffer1, và cứ thế. Bên ghi không còn làm hỏng giá trị mà ai đó vẫn đang đọc, nên rào chắn chống lại điều đó
            được bỏ đi.
          </p>
        }
      >
        <p>
          <b>Double buffering</b> puts each iteration's outputs somewhere other than its inputs. A fresh buffer per
          iteration would do it but waste shared memory. Once an iteration's outputs exist, its inputs are dead, so two
          buffers are enough: iteration 0 reads buffer1 and writes buffer2, iteration 1 reads buffer2 and writes buffer1,
          and so on. A writer can no longer spoil a value someone still reads, so the barrier against that goes.
        </p>
      </Translated>
      <Source section="11.3" pages="259" figs="Fig. 11.4" />
      <Translated
        vi={
          <p>
            Lấy lần lặp 0. Luồng 2 đọc x1 từ ô 1 của buffer1 trong khi luồng 1 ghi Σx0..x1 vào ô 1 của buffer2. Với một buffer, luồng 1 sẽ phải chờ cho đến khi luồng 2 đọc xong; với hai buffer, thứ tự giữa chúng không còn quan trọng.
          </p>
        }
      >
        <p>
          Take iteration 0. Thread 2 reads x1 from slot 1 of buffer1 while thread 1 writes Σx0..x1 into slot 1 of buffer2.
          With one buffer, thread 1 would have to wait until thread 2 had read; with two, the order between them no longer
          matters.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Một chỗ nhầm ở p. 259: sách ghi kết quả lần lặp 0 của luồng 1 là Σx1..x2 và cho luồng 1 đọc x1. Luồng 1 ghi
              Σx0..x1; luồng 2 đọc x1 và ghi Σx1..x2.
            </p>
          }
        >
          <p>
            A slip on p. 259: it gives thread 1's iteration-0 result as Σx1..x2 and has thread 1 read x1. Thread 1 writes
            Σx0..x1; thread 2 reads x1 and writes Σx1..x2.
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Có một hệ quả dễ bị bỏ sót. Với một buffer, một giá trị đã xong cứ nằm yên tại chỗ. Với hai buffer, lần lặp
            tiếp theo đọc buffer kia, nên một luồng lẽ ra đã rảnh việc, chẳng hạn luồng 0 ở lần lặp 0, giờ phải sao chép
            giá trị của mình sang.
          </p>
        }
      >
        <p>
          One consequence is easy to miss. With one buffer, a finished value stays where it is. With two, the next
          iteration reads the other buffer, so a thread that would have gone idle, thread 0 in iteration 0 for one, must now
          copy its value across.
        </p>
      </Translated>
      <Source section="11.3" pages="259–260" />
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy Fig. 11.5 từng luồng một trên các giá trị của bài tập 1. Hãy tắt bước sao chép và xem
            những gì còn lại trong các ô không ai ghi; sau đó thử một buffer với chỉ rào chắn 1.
          </p>
        }
      >
        <p>
          The stepper runs Fig. 11.5 thread by thread on exercise 1's values. Switch the copy off and look at what is left
          in the slots nobody wrote; then try one buffer with only barrier 1.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Như ở phần 11.2: mỗi bước là một thao tác ở mức mã nguồn, không có warp, không có thời gian, và bộ nhớ chia sẻ
              chưa ai ghi được đọc ra là NaN. Mỗi luồng hoán đổi con trỏ của riêng nó, như trong kernel.
            </p>
          }
        >
          <p>
            As in section 11.2: one source-level operation per step, no warps, no timing, and shared memory nobody wrote
            reads as NaN. Each thread swaps its own pointers, as in the kernel.
          </p>
        </Translated>
      </Simplification>
      <TwoBuffers />
      <Outcomes />
      <Ask q={q['ch11/11.3/q-copy']} />

      <h2><Translated vi="Kernel">The kernel</Translated></h2>
      <Translated
        vi={
          <p>
            Fig. 11.5 giữ một rào chắn, ở đầu vòng lặp (dòng 09). Các luồng đang hoạt động đọc <code>inBuffer_s</code> và
            ghi tổng vào <code>outBuffer_s</code> (dòng 11 đến 12); các luồng còn lại sao chép giá trị của mình sang (dòng
            14). Sau đó mọi luồng hoán đổi hai con trỏ của mình (dòng 16 đến 18). Sau lần hoán đổi cuối cùng, các giá trị
            mới nhất nằm trong <code>inBuffer_s</code>, vì vậy lệnh ghi cuối cùng đọc nó.
          </p>
        }
      >
        <p>
          Fig. 11.5 keeps one barrier, at the top of the loop (line 09). Active threads read <code>inBuffer_s</code> and
          write the sum to <code>outBuffer_s</code> (lines 11 to 12); the others copy their value across (line 14). Then
          every thread swaps its two pointers (lines 16 to 18). After the last swap the freshest values are in{' '}
          <code>inBuffer_s</code>, which is why the final store reads it.
        </p>
      </Translated>
      <Source section="11.3" pages="260–261" figs="Fig. 11.5" />
      {fn ? (
        <Code
          tabs={[{ label: 'CUDA', code: fn.text, status: 'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware' }]}
          highlight={[...(lines.barrier1 ?? []), ...(lines.copy ?? [])]}
          caption={
            <Translated
              vi={
                <>
                  Fig. 11.5 lấy từ labs/ch11/snippets/s112_kogge_stone.cu, đánh số dòng như trên trang sách; rào chắn và bước
                  sao chép được đánh dấu. Sách gọi nó là <code>scan_kernel</code>, giống Fig. 11.3.
                </>
              }
            >
              <>
                Fig. 11.5 from labs/ch11/snippets/s112_kogge_stone.cu, numbered as on the page; the barrier and the copy are
                marked. The book calls it <code>scan_kernel</code>, like Fig. 11.3.
              </>
            </Translated>
          }
        />
      ) : null}
      <Translated
        vi={
          <p>
            Giữa vòng lặp và lệnh ghi cũng không cần rào chắn: mỗi luồng ghi ra ô mà chính nó đã ghi ở lần lặp cuối. Với một
            khối 1024 luồng, kernel giờ thực thi {barriersPerScan(1024, 'double')} rào chắn thay vì{' '}
            {barriersPerScan(1024, 'single')}.
          </p>
        }
      >
        <p>
          No barrier is needed between the loop and the store either: each thread stores the slot it wrote itself in the
          last iteration. For a block of 1024 threads the kernel now executes {barriersPerScan(1024, 'double')} barriers
          instead of {barriersPerScan(1024, 'single')}.
        </p>
      </Translated>
      <Aside>
        <Translated vi={<p>Nhận xét về lệnh ghi cuối cùng là của chúng tôi (notes, Fig. 11.5); sách không bàn đến nó.</p>}>
          <p>The remark about the final store is ours (notes, Fig. 11.5); the book does not discuss it.</p>
        </Translated>
      </Aside>
      <Ask q={q['ch11/11.3/q-final-store']} />

      <h2><Translated vi="Đáp án nằm ở đâu?">Where is the answer?</Translated></h2>
      <Translated
        vi={
          <p>
            Các con trỏ hoán đổi ở cuối mỗi lần lặp, nên mảng nào trong hai mảng cuối cùng chứa kết quả scan phụ thuộc vào
            số lần lặp đã chạy. Hãy tự tính ra trước khi phần phát lại làm thay bạn.
          </p>
        }
      >
        <p>
          The pointers swap at the end of every iteration, so which of the two arrays ends up holding the scan depends on
          how many iterations ran. Work it out before the replay does.
        </p>
      </Translated>
      <Predict
        id="ch11/11.3/p-buffer"
        prompt={
          <Translated vi={<>Một khối 256 luồng chạy Fig. 11.5. Sau vòng lặp, mảng nào chứa kết quả: 1 cho buffer1_s hay 2 cho buffer2_s?</>}>
            <>A block of 256 threads runs Fig. 11.5. After the loop, which array holds the result: 1 for buffer1_s or 2 for buffer2_s?</>
          </Translated>
        }
      >
        {(v) => <SwapReplay committed={v} />}
      </Predict>
      <Ask q={q['ch11/11.3/q-buffer']} />

      <h2><Translated vi="Cái giá phải trả">What it costs</Translated></h2>
      <Translated
        vi={
          <p>
            Hai buffer cần gấp đôi bộ nhớ chia sẻ: {sharedBytes(1024, 'double')} byte mỗi khối 1024 số float thay vì{' '}
            {sharedBytes(1024, 'single')}. Bộ nhớ chia sẻ là một trong các ngân sách giới hạn số khối mà một SM giữ thường
            trú (<a href={href('ch05', '5.6')}>phần 5.6</a>), nên rào chắn tiết kiệm được có thể phải trả bằng occupancy
            (mức chiếm dụng). Nó có phải là lợi trên thiết bị của bạn hay không là chuyện đo đạc, không phải quy tắc: lab
            11.1 đo thời gian cả hai kernel ở {iterations(1024)} lần lặp mỗi khối.
          </p>
        }
      >
        <p>
          Two buffers need twice the shared memory: {sharedBytes(1024, 'double')} bytes per block of 1024 floats instead of{' '}
          {sharedBytes(1024, 'single')}. Shared memory is one of the budgets that cap how many blocks an SM keeps resident
          (<a href={href('ch05', '5.6')}>section 5.6</a>), so the saved barrier can be paid for with occupancy. Whether it is
          a gain on your device is a measurement, not a rule: lab 11.1 times both kernels at {iterations(1024)} iterations
          per block.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Cái giá về occupancy là nhận định của chúng tôi; sách chỉ nói buffer thứ hai tốn bộ nhớ chia sẻ.</p>}>
          <p>The occupancy price is ours; the book only says the second buffer costs shared memory.</p>
        </Translated>
      </Beyond>
      <Ask q={q['ch11/11.3/q-bytes']} label={tr('Work it out', 'Tự tính ra')} />
      <Predict
        id="ch11/11.3/p-double"
        prompt={
          <Translated
            vi={
              <>
                Ở 1024 luồng mỗi khối, kernel buffer kép sẽ chạy nhanh gấp bao nhiêu lần kernel hai rào chắn trên thiết bị{' '}
                <em>của bạn</em>? Hãy chốt một con số trước khi chạy lab 11.1.
              </>
            }
          >
            <>
              At 1024 threads per block, how many times faster than the two-barrier kernel will the double-buffered kernel
              run on <em>your</em> device? Commit a number before running lab 11.1.
            </>
          </Translated>
        }
        unit={tr('× as fast', '× nhanh hơn')}
      >
        {(v) => <MeasuredDouble predicted={v} />}
      </Predict>

      <h2><Translated vi="Những gì còn lại">What is left</Translated></h2>
      <Translated
        vi={
          <p>
            Buffer kép vẫn trả một rào chắn mỗi lần lặp, và ba lần truy cập bộ nhớ chia sẻ (hai lần đọc, một lần ghi) cho
            mỗi phép cộng. Vòng lặp bị chi phối bởi đồng bộ hóa và bộ nhớ chia sẻ. Phần{' '}
            <a href={href('ch11', '11.4')}>11.4</a> chuyển phần lớn việc scan vào thanh ghi (register) và warp shuffle, và
            đưa cả phép scan khối xuống còn hai rào chắn.
          </p>
        }
      >
        <p>
          Double buffering still pays one barrier per iteration, and three shared-memory accesses (two reads, one write)
          for every single addition. The loop is dominated by synchronisation and shared memory. Section{' '}
          <a href={href('ch11', '11.4')}>11.4</a> moves most of the scan into registers and warp shuffles, and gets the
          whole block scan down to two barriers.
        </p>
      </Translated>
      <Source section="11.4" pages="261" />
      <Ask q={q['ch11/11.3/q-fig119']} />
    </>
  );
}
