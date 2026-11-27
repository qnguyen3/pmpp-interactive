import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import { NoteGroup } from './NoteGroup';
import {
  BOOK_LIST,
  FOLD_OPS,
  HUGE_ROW,
  NEGATIVE_ROW,
  START_LABEL,
  f32short,
  f64short,
  foldSequential,
  randomRow,
  randomRowTest,
  type FoldOpId,
  type StartId,
} from '../../sim/reducetree';

/** A caption above a group of buttons (a <label> would forward clicks on the caption to the first button). */
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
/** Long button groups wrap on a narrow screen instead of running off the bench. */
const WRAP: CSSProperties = { flexWrap: 'wrap' };
const OPS: FoldOpId[] = ['sum', 'product', 'max', 'min', 'argmax'];
const STARTS: StartId[] = ['identity', 'zero', 'first', 'negmax'];
const PRESETS: { label: string; values: number[] }[] = [
  { get label() { return tr("the book's list", 'danh sách của sách'); }, values: BOOK_LIST },
  { get label() { return tr('logits, all negative', 'logits, toàn số âm'); }, values: NEGATIVE_ROW },
  { get label() { return tr('one huge logit', 'một logit khổng lồ'); }, values: HUGE_ROW },
];
const listText = (v: number[]) => v.map((x) => (Number.isInteger(x) ? x.toFixed(1) : String(x))).join(', ');
function parseList(s: string): number[] | null {
  const parts = s.split(/[\s,]+/).filter(Boolean);
  if (parts.length < 1 || parts.length > 16) return null;
  const v = parts.map((p) => Number(p.replace('−', '-')));
  return v.every((x) => Number.isFinite(x)) ? v : null;
}

// ---------------------------------------------------------------- the fold explorer
function FoldExplorer() {
  const [op, setOp] = useState<FoldOpId>('sum');
  const [start, setStart] = useState<StartId>('identity');
  const [values, setValues] = useState<number[]>(BOOK_LIST);
  const [text, setText] = useState(listText(BOOK_LIST));
  const [seed, setSeed] = useState(1);
  const trace = useMemo(() => foldSequential(values, op, start, 'f32'), [values, op, start]);
  const n = trace.steps.length;
  const [shown, setShown] = useState(n);
  const [playing, setPlaying] = useState(false);
  // Any change of list, operator or start value shows the whole run again.
  useEffect(() => {
    setShown(n);
    setPlaying(false);
  }, [values, op, start, n]);
  useEffect(() => {
    if (!playing) return;
    if (shown >= n) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => setShown((k) => k + 1), 650);
    return () => clearTimeout(t);
  }, [playing, shown, n]);
  const random = useMemo(() => randomRowTest(op, start), [op, start]);
  const bad = parseList(text) === null;
  const pick = (v: number[]) => {
    setValues(v);
    setText(listText(v));
  };
  const isArg = op === 'argmax';
  const pair = (v: number, i: number) => `(${f32short(v)}, ${i < 0 ? tr('none', 'không có') : i})`;
  const accText = (v: number, i: number) => (isArg ? pair(v, i) : f32short(v));
  const done = shown >= n;
  const identity = FOLD_OPS[op].identityLabel;

  return (
    <Bench
      title={tr('One loop, any operator', 'Một vòng lặp, mọi toán tử')}
      task={tr('Pick an operator, a list and a starting value. Run the loop and compare its answer with the right one.', 'Chọn một toán tử, một danh sách và một giá trị khởi đầu. Chạy vòng lặp và so đáp án của nó với đáp án đúng.')}
      foot={
        <>
          <Translated
            vi={
              <>
                Vòng lặp chạy bằng float32, làm tròn sau mỗi phép toán, giống một vòng lặp <code>float</code> trong C. Đáp án
                đúng được tính riêng bằng float64 với phần tử đơn vị của chính toán tử đó; với argmax, đó là chỉ số đầu tiên của
                giá trị lớn nhất. "Đúng" với một phép tổng hay tích nghĩa là sai khác trong phạm vi làm tròn float32.
              </>
            }
          >
            The loop runs in float32, rounding after every operation, as a C <code>float</code> loop does. The right
            answer is computed separately in float64 with the operator's own identity; for argmax it is the first index
            of the largest value. "Right" for a sum or product means within float32 rounding.
          </Translated>
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Toán tử">Operator</Translated>
          <span className="seg" role="group" aria-label={tr('Operator', 'Toán tử')} style={WRAP}>
            {OPS.map((o) => (
              <button key={o} aria-pressed={op === o} onClick={() => setOp(o)}>
                {FOLD_OPS[o].label}
              </button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          <Translated vi="Khởi đầu biến tích lũy tại">Start the accumulator at</Translated>
          <span className="seg" role="group" aria-label={tr('Start value', 'Giá trị khởi đầu')} style={WRAP}>
            {STARTS.map((s) => (
              <button key={s} aria-pressed={start === s} onClick={() => setStart(s)}>
                {s === 'identity' ? tr(`the identity, ${identity}`, `phần tử đơn vị, ${identity}`) : START_LABEL[s]}
              </button>
            ))}
          </span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ flex: '1 1 18rem' }}>
          <Translated vi="Danh sách (tối đa 16 số)">The list (up to 16 numbers)</Translated>
          <input
            type="text"
            value={text}
            aria-label={tr('The list', 'Danh sách')}
            aria-invalid={bad}
            style={{ width: '100%', fontVariantNumeric: 'tabular-nums', borderColor: bad ? 'var(--fail)' : undefined }}
            onChange={(e) => {
              setText(e.target.value);
              const v = parseList(e.target.value);
              if (v) setValues(v);
            }}
          />
        </label>
        <span className="seg" role="group" aria-label={tr('Lists', 'Các danh sách')} style={WRAP}>
          {PRESETS.map((p) => (
            <button key={p.label} onClick={() => pick(p.values)}>
              {p.label}
            </button>
          ))}
          <button
            onClick={() => {
              setSeed(seed + 1);
              pick(randomRow(8, seed));
            }}
          >
            <Translated vi="một hàng ngẫu nhiên">a random row</Translated>
          </button>
        </span>
      </div>
      {bad ? (
        <p className="t-fail" style={{ marginTop: 0 }}>
          <Translated vi="Gõ từ 1 đến 16 số, cách nhau bằng dấu phẩy. Trình khám phá tiếp tục dùng danh sách hợp lệ gần nhất.">Type between 1 and 16 numbers, separated by commas. The explorer keeps using the last list that parsed.</Translated>
        </p>
      ) : null}

      <div className="cols" style={{ alignItems: 'start' }}>
        <div className="panel" style={{ overflowX: 'auto' }}>
          <div className="row" style={{ marginBottom: '0.6rem' }}>
            <button onClick={() => { setPlaying(false); setShown(0); }} disabled={shown === 0}>
              <Translated vi="Bắt đầu lại">Restart</Translated>
            </button>
            <button onClick={() => { setPlaying(false); setShown(Math.min(n, shown + 1)); }} disabled={done}>
              <Translated vi="Bước">Step</Translated>
            </button>
            <button className="primary" onClick={() => { if (done) setShown(0); setPlaying(!playing); }}>
              {playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Chạy')}
            </button>
          </div>
          <table className="data" style={{ margin: 0 }} data-part="trace">
            <thead>
              <tr>
                <th><Translated vi="Vòng lặp">Iteration</Translated></th>
                <th className="num">input[i]</th>
                <th className="num"><Translated vi={<>{isArg ? '(giá trị, chỉ số)' : 'acc'} sau đó</>}>{isArg ? '(value, index)' : 'acc'} after it</Translated></th>
              </tr>
            </thead>
            <tbody>
              <tr data-row="start">
                <td><Translated vi="bắt đầu">start</Translated></td>
                <td className="num muted">{trace.from === 1 ? tr('element 0', 'phần tử 0') : ''}</td>
                <td className="num">
                  <span className="chip reg">{accText(trace.init.value, trace.init.index)}</span>
                </td>
              </tr>
              {trace.steps.slice(0, shown).map((s) => (
                <tr key={s.i} data-row={s.i}>
                  <td>i = {s.i}</td>
                  <td className="num">{f32short(s.x)}</td>
                  <td className="num">
                    <span className="chip reg">{accText(s.acc, s.index)}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.5rem 0 0' }} data-readout="progress">
            {trace.from === 1
              ? tr(`Started from element 0, so ${values.length} elements take ${n} iterations.`, `Bắt đầu từ phần tử 0, nên ${values.length} phần tử cần ${n} vòng lặp.`)
              : tr(`${values.length} elements, ${n} iterations.`, `${values.length} phần tử, ${n} vòng lặp.`)}{' '}
            {done ? '' : tr(`Shown: ${shown} of ${n}.`, `Đã hiện: ${shown} trên ${n}.`)}
          </p>
        </div>
        <div className="panel" aria-live="polite" data-readout="result" data-ok={done ? (trace.ok ? '1' : '0') : ''}>
          {done ? (
            <>
              <p style={{ marginTop: 0 }}>
                <b><Translated vi="Đáp án của vòng lặp:">The loop's answer:</Translated></b>{' '}
                <span data-value="result">{accText(trace.result, trace.resultIndex)}</span>
              </p>
              <p>
                <b><Translated vi="Đáp án đúng:">The right answer:</Translated></b>{' '}
                <span data-value="truth">
                  {isArg ? tr(`index ${trace.truthIndex} (value ${f64short(trace.truth)})`, `chỉ số ${trace.truthIndex} (giá trị ${f64short(trace.truth)})`) : f64short(trace.truth)}
                </span>
              </p>
              <p>
                {trace.ok ? <span className="chip ok"><Translated vi="đúng">right</Translated></span> : <span className="chip fail"><Translated vi="sai">wrong</Translated></span>}{' '}
                {trace.ok
                  ? trace.start === 'identity' || trace.start === 'first'
                    ? tr('As it must be with this start.', 'Đúng như phải thế với giá trị khởi đầu này.')
                    : tr('On this list. Try the other lists.', 'Trên danh sách này. Hãy thử các danh sách khác.')
                  : explainWrong(op, start)}
              </p>
            </>
          ) : (
            <p style={{ margin: 0 }} className="muted">
              <Translated vi="Vòng lặp đang chạy. Đáp án của nó xuất hiện sau vòng lặp cuối cùng.">The loop is running. Its answer appears after the last iteration.</Translated>
            </p>
          )}
          <p style={{ marginBottom: 0 }} data-readout="random">
            <Translated
              vi={
                <>
                  Cùng vòng lặp và giá trị khởi đầu đó trên <b>{random.rows}</b> hàng ngẫu nhiên gồm 8 logits: sai trên <b>{random.wrong}</b>.
                  {random.wrong > 0 && random.wrong < 50 ? ' Một bài kiểm tra trên dữ liệu ngẫu nhiên rất dễ bỏ sót nó.' : ''}
                </>
              }
            >
              The same loop and start on <b>{random.rows}</b> random rows of 8 logits: wrong on <b>{random.wrong}</b>.
              {random.wrong > 0 && random.wrong < 50 ? ' A test on random data could easily miss it.' : ''}
            </Translated>
          </p>
        </div>
      </div>
    </Bench>
  );
}

function explainWrong(op: FoldOpId, start: StartId): string {
  if (start === 'zero' && (op === 'max' || op === 'argmax')) return tr('Every value is below 0, so the start value wins and an answer that is not in the list comes out.', 'Mọi giá trị đều dưới 0, nên giá trị khởi đầu thắng và một đáp án không có trong danh sách lọt ra.');
  if (start === 'zero' && op === 'min') return tr('Every value is above 0, so the start value wins.', 'Mọi giá trị đều trên 0, nên giá trị khởi đầu thắng.');
  if (start === 'zero' && op === 'product') return tr('Anything times 0 is 0.', 'Bất cứ thứ gì nhân 0 cũng bằng 0.');
  if (start === 'negmax' && op === 'min') return tr('−FLT_MAX is below every value, so it wins every comparison.', '−FLT_MAX nhỏ hơn mọi giá trị, nên nó thắng mọi phép so sánh.');
  if (start === 'negmax' && op === 'sum') return tr('The start value is part of the sum, and it is enormous.', 'Giá trị khởi đầu là một phần của tổng, và nó khổng lồ.');
  if (start === 'negmax' && op === 'product') return tr('The start value multiplies everything.', 'Giá trị khởi đầu nhân vào mọi thứ.');
  return tr('The start value is not neutral for this operator.', 'Giá trị khởi đầu không trung hòa đối với toán tử này.');
}

// ---------------------------------------------------------------- questions
export const questionsS101: Record<string, Question> = {
  'ch10/10.1/q-identity-max': {
    kind: 'choice',
    id: 'ch10/10.1/q-identity-max',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel max-logit khởi đầu biến tích lũy của mỗi luồng tại <code>0.0f</code> rồi lấy max với các phần tử của
            nó. Đáp án của nó sai trên hàng logits nào?
          </>
        }
      >
        <>
          A max-logit kernel starts each thread's accumulator at <code>0.0f</code> and then takes the max with its
          elements. On which row of logits is its answer wrong?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Bất kỳ hàng nào chứa một logit dương.">Any row that contains a positive logit.</Translated>, why: <Translated vi="Khi đó có giá trị thắng 0 và giá trị khởi đầu không bao giờ lộ ra.">Then some value beats 0 and the start never shows.</Translated> },
      { label: <Translated vi="Một hàng mà mọi logit đều âm.">A row whose logits are all negative.</Translated>, correct: true },
      { label: <Translated vi="Một hàng chứa một NaN.">A row that contains one NaN.</Translated>, why: <Translated vi="Đó là một vấn đề khác (mục 10.2). Giá trị khởi đầu sai ngay cả trên một hàng không có NaN nào.">That is a separate problem (section 10.2). The start value is wrong on a row with no NaN at all.</Translated> },
      { label: <Translated vi="Không hàng nào: 0 là phần tử đơn vị của max.">None: 0 is the identity of max.</Translated>, why: <Translated vi="0 là phần tử đơn vị của phép cộng. max(0, x) bằng 0, không phải x, mỗi khi x âm.">0 is the identity of addition. max(0, x) is 0, not x, whenever x is negative.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trên một hàng toàn số âm, kernel trả về 0, một giá trị không có trong hàng. Giá trị khởi đầu của max phải bằng
            hoặc nhỏ hơn mọi đầu vào có thể: −∞ trong sách, hoặc −FLT_MAX cho float hữu hạn. Những hàng có ít nhất một logit
            dương che giấu lỗi này, và phần lớn các hàng đều có, nên một bài kiểm tra trên logits ngẫu nhiên vẫn có thể qua.
          </p>
        }
      >
        <p>
          On an all-negative row the kernel returns 0, a value that is not in the row. The start of a max has to be at or
          below every possible input: −∞ in the book, or −FLT_MAX for finite floats. Rows with at least one positive
          logit hide the bug, and most rows have one, which is why a test on random logits can pass.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.1, p. 222',
  },
  'ch10/10.1/q-finite-identity': {
    kind: 'choice',
    id: 'ch10/10.1/q-finite-identity',
    prompt: (
      <Translated
        vi={
          <>
            Trên một backend nơi vô cực không được định nghĩa (Metal dưới fast math), giá trị khởi đầu nào an toàn cho một
            phép max trên float hữu hạn, ở mọi luồng của một kernel?
          </>
        }
      >
        <>
          On a backend where infinities are undefined (Metal under fast math), which start value is safe for a max over
          finite floats, in every thread of a kernel?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="0.0f, như cho phép tổng theo sau max">0.0f, as for the sum that follows the max</Translated>, why: <Translated vi="Sai trên mọi hàng toàn số âm.">Wrong on any row that is all negative.</Translated> },
      { label: <Translated vi="-INFINITY, phần tử đơn vị mà sách đưa ra cho max">-INFINITY, the identity the book gives for max</Translated>, why: <Translated vi="Chính là giá trị mà trình biên dịch được phép giả định không bao giờ xuất hiện ở chế độ fast.">Exactly the value the compiler may assume never occurs in fast mode.</Translated> },
      { label: <Translated vi="-FLT_MAX, số float hữu hạn âm nhất">-FLT_MAX, the most negative finite float</Translated>, correct: true },
      { label: <Translated vi="Phần tử đầu tiên của luồng (chú thích 1)">The thread's first element (footnote 1)</Translated>, why: <Translated vi="Ổn với một vòng lặp trên danh sách không rỗng. Một luồng hay lane không có phần tử nào thì không có gì để bắt đầu.">Fine for one loop over a non-empty list. A thread or lane that has no element has nothing to start from.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mọi float hữu hạn đều bằng hoặc lớn hơn −FLT_MAX, nên max(−FLT_MAX, x) = x với tất cả chúng, và không điều gì ở
            đây dựa vào vô cực. Bắt đầu từ phần tử đầu tiên (chú thích 1, p. 222) chạy được trong một vòng lặp tuần tự, nhưng
            trong một kernel, các luồng nằm quá cuối đầu vào, và các lane đệm của một warp, không có phần tử đầu tiên. Chúng
            vẫn phải đóng góp một thứ gì đó trung hòa.
          </p>
        }
      >
        <p>
          Every finite float is at or above −FLT_MAX, so max(−FLT_MAX, x) = x for all of them, and nothing about it relies
          on infinities. Starting from the first element (footnote 1, p. 222) works in a sequential loop, but in a kernel
          the threads past the end of the input, and the padded lanes of a warp, have no first element. They still have
          to contribute something neutral.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.1, p. 222; Metal: notes/today/ch10-facts.md C3', // i18n-ignore: citation only
  },
  'ch10/10.1/q-reduce-order': {
    kind: 'choice',
    id: 'ch10/10.1/q-reduce-order',
    prompt: (
      <Translated
        vi={
          <>
            <code>std::accumulate</code> và <code>std::reduce</code> cùng cộng một <code>std::vector&lt;float&gt;</code>.
            Hàm nào được phép cho đáp án khác nhau giữa các lần chạy, và vì sao?
          </>
        }
      >
        <>
          <code>std::accumulate</code> and <code>std::reduce</code> sum the same <code>std::vector&lt;float&gt;</code>.
          Which one is allowed to give a different answer from run to run, and why?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="std::accumulate, vì biến tích lũy của nó lấy kiểu của giá trị ban đầu.">std::accumulate, because its accumulator takes the type of the initial value.</Translated>, why: <Translated vi="Kiểu đó cố định lúc biên dịch, và phép gộp chạy từ trái sang phải theo một thứ tự cố định: vòng lặp của Hình 10.2.">That type is fixed when the code is compiled, and the fold runs left to right in one fixed order: the loop of Fig. 10.2.</Translated> },
      {
        label: <Translated vi="std::reduce: nó được phép nhóm lại và đổi thứ tự các phần tử, mà phép cộng float không có tính kết hợp.">std::reduce: it may regroup and reorder the elements, and float addition is not associative.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Không hàm nào: cả hai cộng đúng cùng những con số, nên phải cho cùng kết quả.">Neither: both add exactly the same numbers, so they must agree.</Translated>, why: <Translated vi="Cùng những con số nhưng nhóm khác thì làm tròn khác.">The same numbers in a different grouping round differently.</Translated> },
      { label: <Translated vi="Cả hai, vì phép cộng float làm tròn khác nhau giữa các lần chạy.">Both, because float addition rounds differently from one run to the next.</Translated>, why: <Translated vi="Mỗi phép cộng là tất định. Chỉ cách nhóm mới có thể thay đổi.">Each addition is deterministic. Only the grouping can change.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>std::accumulate</code> là vòng lặp tuần tự. <code>std::reduce</code> được phép nhóm và sắp xếp lại các
            phần tử theo bất kỳ thứ tự nào, và thư viện C++ ghi trong tài liệu rằng kết quả của nó là không tất định với phép
            cộng dấu phẩy động. Đó chính là hai tính chất mà mục 10.2 đòi hỏi ở một toán tử trước khi một cây song song được
            phép tính nó.
          </p>
        }
      >
        <p>
          <code>std::accumulate</code> is the sequential loop. <code>std::reduce</code> may group and rearrange the
          elements in any order, and the C++ library documents its result as non-deterministic for floating-point
          addition. Those are the two properties section 10.2 asks of an operator before a parallel tree may compute it.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.1, p. 223 and §10.2, p. 224; cppreference: notes/today/ch10-facts.md A9',
    vi: { source: 'ch. 10 §10.1, p. 223 và §10.2, p. 224; cppreference: notes/today/ch10-facts.md A9' },
  },
};

// ---------------------------------------------------------------- the section
export function S101() {
  const fold = useLabFile('ch10/snippets/fold.h');
  return (
    <>
      <Translated
        vi={
          <p>
            Một <b>phép rút gọn (reduction)</b> biến một danh sách giá trị thành một giá trị duy nhất bằng cách áp dụng một
            toán tử hai ngôi: một phép toán nhận hai đầu vào và trả về một. Phép tổng là ví dụ của sách: danh sách 7.0, 2.1,
            5.3, 9.0, 11.2 rút gọn thành 34.6. Các giá trị cũng có thể là số nguyên, float nửa độ chính xác hay ký tự. Cấu trúc
            của phép tính là như nhau với tất cả, và mọi điều chương này làm để song song hóa nó cũng vậy.
          </p>
        }
      >
        <p>
          A <b>reduction</b> turns a list of values into one value by applying a binary operator: an operation that takes
          two inputs and returns one. Summing is the book's example: the list 7.0, 2.1, 5.3, 9.0, 11.2 reduces to 34.6.
          The values could as well be integers, half-precision floats or characters. The structure of the computation is
          the same for all of them, and so is everything this chapter does to make it parallel.
        </p>
      </Translated>
      <Source section="10.1" pages="221–222" />

      <h2><Translated vi="Một vòng lặp cho mọi phép rút gọn">One loop for every reduction</Translated></h2>
      <Translated
        vi={
          <p>
            Tuần tự, một phép tổng là vòng lặp bạn viết mà không cần nghĩ: khởi đầu một biến tại 0, lần lượt cộng từng phần tử
            vào nó. Giá trị của nó sau mỗi vòng lặp là 7.0, 9.1, 14.4, 23.4 và cuối cùng là 34.6. Sau đó sách tách toán tử ra
            khỏi vòng lặp. Khởi đầu một biến tích lũy tại một giá trị gọi là <code>IDENTITY</code>, và ở mỗi vòng lặp thay nó
            bằng <code>Operator(acc, input[i])</code>. Tổng, tích, max và min đều là một vòng lặp này, với một toán tử khác và
            một giá trị khởi đầu khác. N phần tử cần N vòng lặp.
          </p>
        }
      >
        <p>
          Sequentially, a sum is the loop you would write without thinking: start a variable at 0, add each element to it
          in turn. Its value after each iteration is 7.0, 9.1, 14.4, 23.4 and finally 34.6. The book then pulls the
          operator out of the loop. Start an accumulator at a value called <code>IDENTITY</code>, and in each iteration
          replace it by <code>Operator(acc, input[i])</code>. Sum, product, max and min are all this one loop, with a
          different operator and a different starting value. N elements take N iterations.
        </p>
      </Translated>
      <Source section="10.1" pages="222–223" figs={tr('Figs. 10.1 and 10.2', 'Hình 10.1 và 10.2')} />
      <Code
        tabs={[{ label: 'C++', code: fold ?? '// loading labs/ch10/snippets/fold.h' }]}
        highlight={[8, 9, 10]}
        caption={
          <Translated
            vi={
              <>
                Hình 10.2 viết thành một hàm C++, cùng với hai phép rút gọn mà giải mã tham lam cần: logit lớn nhất và argmax
                (labs/ch10/snippets/fold.h). Các unit test biên dịch và chạy nó trên CPU và so từng số float với trình khám phá
                bên dưới, từng bit một.
              </>
            }
          >
            <>
              Fig. 10.2 written as a C++ function, plus the two reductions greedy decoding needs: the largest logit and
              the argmax (labs/ch10/snippets/fold.h). The unit tests compile and run it on the CPU and compare every float
              with the explorer below, bit for bit.
            </>
          </Translated>
        }
      />

      <h2><Translated vi="Biến tích lũy bắt đầu từ đâu">Where the accumulator starts</Translated></h2>
      <Translated
        vi={
          <p>
            Giá trị khởi đầu là <b>phần tử đơn vị</b> của toán tử: giá trị mà khi kết hợp với bất kỳ đầu vào nào sẽ trả lại
            chính đầu vào đó. Với phép cộng nó là 0, với phép nhân là 1. Với min nó là +∞ và với max là −∞: mọi giá trị đều
            không vượt quá +∞ và không nhỏ hơn −∞, nên min(+∞, x) = x và max(−∞, x) = x. Phần tử đơn vị là một sự tiện lợi,
            không phải một yêu cầu. Một toán tử không có phần tử đơn vị có thể khởi đầu biến tích lũy tại phần tử 0 và bắt
            đầu vòng lặp từ phần tử 1.
          </p>
        }
      >
        <p>
          The starting value is the operator's <b>identity</b>: the value that, combined with any input, gives that input
          back. For addition it is 0, for multiplication 1. For min it is +∞ and for max −∞: every value is at most +∞
          and at least −∞, so min(+∞, x) = x and max(−∞, x) = x. An identity is a convenience, not a requirement. An
          operator without one can start the accumulator at element 0 and begin the loop at element 1.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="10.1" pages="222" />
        <Aside>
          <Translated
            vi={
              <p>
                Hai chỗ sơ suất ở p. 222: phần tử đơn vị của toán tử "minimum (minimum)" nên đọc là "minimum (maximum)", và
                "after iteration 5" nghĩa là vòng lặp thứ năm, được đánh số 4 khi đếm từ 0. Cả hai đều được liệt kê ở{' '}
                <a href={href('ch10', 'wrap')}>trang tóm tắt</a>.
              </p>
            }
          >
            <p>
              Two slips on p. 222: the identity of the "minimum (minimum)" operator should read "minimum (maximum)", and
              "after iteration 5" means the fifth iteration, numbered 4 when counting from 0. Both are listed on the{' '}
              <a href={href('ch10', 'wrap')}>summary page</a>.
            </p>
          </Translated>
        </Aside>
      </NoteGroup>
      <Translated
        vi={
          <p>
            Trong một vòng lặp của riêng bạn, một giá trị khởi đầu sai rất dễ nhận ra. Trong một kernel nó ẩn đi: giá trị 0
            thay cho −∞ cho ra max đúng trên gần như mọi hàng bạn có khả năng dùng để kiểm tra. Hãy tìm những hàng mà nó sai.
          </p>
        }
      >
        <p>
          In a loop of your own, a wrong starting value is easy to spot. In a kernel it hides: the value 0 instead of
          −∞ gives the right max on almost every row you are likely to test with. Find the rows where it does not.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một luồng, không GPU: float32 được mô phỏng bằng cách làm tròn về float32 sau mỗi phép toán. Max và min coi NaN là
              giá trị bị thiếu, giống <code>fmaxf</code> của CUDA; danh sách không thể chứa NaN.
            </p>
          }
        >
          <p>
            One thread, no GPU: float32 is imitated by rounding to float32 after every operation. Max and min treat NaN as
            missing, like CUDA's <code>fmaxf</code>; the list cannot hold a NaN.
          </p>
        </Translated>
      </Simplification>
      <FoldExplorer />
      <Translated
        vi={
          <p>
            Có ba điều lẽ ra bạn đã rút ra từ trình khám phá. Bất kỳ giá trị khởi đầu nào bằng hoặc nhỏ hơn mọi đầu vào đều
            dùng được cho max, đó là lý do −FLT_MAX, số float hữu hạn âm nhất, tốt ngang −∞ trên logits thật. Giá trị khởi đầu
            0 chỉ sai trên những hàng toàn số âm, thứ mà dữ liệu kiểm tra ngẫu nhiên gần như không bao giờ tạo ra. Và bắt đầu
            từ phần tử 0 luôn chạy được trong một vòng lặp, nhưng nó giả định có tồn tại phần tử 0.
          </p>
        }
      >
        <p>
          Three things should have come out of the explorer. Any start at or below every input works for max, which is why
          −FLT_MAX, the most negative finite float, is as good as −∞ on real logits. A start of 0 fails only on rows that
          are entirely negative, which random test data almost never produces. And starting from element 0 always works
          in a loop, but it assumes there is an element 0.
        </p>
      </Translated>
      <Ask q={questionsS101['ch10/10.1/q-identity-max']} />

      <h2><Translated vi="Phần tử đơn vị trở thành gì trong một kernel">What the identity becomes in a kernel</Translated></h2>
      <Translated
        vi={
          <p>
            Một softmax trên một hàng logits là hai phép rút gọn liên tiếp. Đầu tiên là max, với phần tử đơn vị −∞; rồi một
            phép tổng của exp(x − max), với phần tử đơn vị 0; rồi một phép chia cho mỗi phần tử. Argmax là một phép max trên
            các cặp (giá trị, chỉ số). Trong một kernel GPU, phần tử đơn vị không còn là sự tiện lợi. Hàng nghìn luồng, mỗi
            luồng khởi đầu một biến tích lũy, và những luồng có chỉ số nằm quá cuối hàng, hay các lane của một warp không có
            phần tử nào, vẫn phải trao một thứ gì đó cho phép rút gọn. Thứ đó phải trung hòa: phần tử đơn vị. Mục{' '}
            <a href={href('ch10', '10.7')}>10.7</a> và <a href={href('ch10', '10.9')}>10.9</a> phụ thuộc vào điều này.
          </p>
        }
      >
        <p>
          A softmax over a row of logits is two reductions in a row. First a max, with identity −∞; then a sum of exp(x
          − max), with identity 0; then one division per element. Argmax is a max over (value, index) pairs. In a GPU
          kernel the identity stops being a convenience. Thousands of threads each start an accumulator, and the threads
          whose index lies past the end of the row, or the lanes of a warp that have no element, must still hand
          something to the reduction. That something has to be neutral: the identity. Sections{' '}
          <a href={href('ch10', '10.7')}>10.7</a> and <a href={href('ch10', '10.9')}>10.9</a> depend on it.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Softmax, argmax và vai trò của phần tử đơn vị cho các luồng rảnh là của khóa học này; chương của sách chỉ dùng phép
              tổng.
            </p>
          }
        >
          <p>
            Softmax, argmax and the role of the identity for idle threads are ours; the book's chapter uses only sums.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Trên Metal, phần tử đơn vị hiển nhiên của max là một cái bẫy. Dưới fast math của Metal, trình biên dịch được phép
            giả định không giá trị nào là vô cực, nên hành vi của −∞ là không xác định, và Apple mô tả chế độ toán mặc định theo
            ba cách khác nhau trong tài liệu. Vì vậy các kernel Metal của khóa học này khởi đầu phép max tại −FLT_MAX. Ngược lại,
            tùy chọn fast-math của CUDA không liệt kê giả định nào như vậy.
          </p>
        }
      >
        <p>
          On Metal, the obvious identity for max is a trap. Under Metal's fast math the compiler may assume no value is
          infinite, so the behaviour of −∞ is undefined, and Apple documents the default math mode three different ways.
          This course's Metal kernels therefore start a max at −FLT_MAX. CUDA's fast-math option, by contrast, lists no
          such assumption.
        </p>
      </Translated>
      <Today checked="2026-11-20">
        <Translated
          vi={
            <p>
              MSL 4.1, §8.1: "Nếu fast math được bật thì hành vi xử lý NaN hoặc INF (làm đầu vào hoặc đầu ra) là không xác định."
            </p>
          }
        >
          <p>
            MSL 4.1, §8.1: "If fast math is enabled the behavior of handling NaN or INF (as inputs or outputs) is
            undefined."
          </p>
        </Translated>
        <p className="muted">ch10-facts C3, A11; ch05-facts C7</p>{/* i18n-ignore: citation only */}
      </Today>
      <Ask q={questionsS101['ch10/10.1/q-finite-identity']} />

      <h2><Translated vi="Những phép rút gọn bạn không tự viết">Reductions you do not write yourself</Translated></h2>
      <Translated
        vi={
          <p>
            Phép rút gọn phổ biến đến mức thư viện nào cũng có. C++ có <code>std::accumulate</code> và{' '}
            <code>std::reduce</code>, hai biến thể với những lời hứa khác nhau. Trên GPU NVIDIA, các thư viện Thrust và CUB
            cung cấp những phép rút gọn được tối ưu kỹ. Dù vậy sách vẫn tự viết, vì con đường đến một phép rút gọn nhanh đi qua
            gần như mọi kỹ thuật trong phần đầu của sách.
          </p>
        }
      >
        <p>
          Reduction is common enough that every library has it. C++ has <code>std::accumulate</code> and{' '}
          <code>std::reduce</code>, two variants with different promises. On NVIDIA GPUs, the Thrust and CUB libraries
          offer heavily optimised reductions. The book writes its own anyway, because the road to a fast one passes
          through nearly every technique in the first part of the book.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="10.1" pages="223" />
        <Today checked="2026-11-20 (ch10-facts A9, A12)">{/* i18n-ignore: date and citation */}
          <Translated
            vi={
              <p>
                cppreference: với phép cộng float, "kết quả là không tất định" của <code>std::reduce</code>. Thrust và CUB giờ nằm
                dưới <code>nvidia.github.io/cccl/unstable/</code>.
              </p>
            }
          >
            <p>
              cppreference: for float addition, <code>std::reduce</code>'s "result is non-deterministic". Thrust and CUB
              now live under <code>nvidia.github.io/cccl/unstable/</code>.
            </p>
          </Translated>
        </Today>
      </NoteGroup>
      <Translated
        vi={
          <p>
            Trên AMD, các bản tương ứng là hipCUB và rocPRIM, với tên của CUB nằm trong một namespace và header khác, nên một
            phép rút gọn viết theo CUB không biên dịch nguyên vẹn được cho HIP. Trên Metal, các khối xây dựng có tài liệu là{' '}
            <code>simd_sum</code> và <code>simd_max</code>, rút gọn trên một SIMD-group; ở cấp toàn bộ tensor, Metal
            Performance Shaders Graph cung cấp các phép rút gọn như <code>reductionArgMaximum</code>.
          </p>
        }
      >
        <p>
          On AMD the counterparts are hipCUB and rocPRIM, with CUB's names in a different namespace and header, so a
          reduction written against CUB does not compile unchanged for HIP. On Metal, the documented building blocks are{' '}
          <code>simd_sum</code> and <code>simd_max</code>, which reduce across one SIMD-group; at the level of whole
          tensors, Metal Performance Shaders Graph offers reductions such as <code>reductionArgMaximum</code>.
        </p>
      </Translated>
      <Today checked="2026-11-20 (infra-facts B11; ch10-facts C1, C5)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              hipCUB: "một thư viện bọc mỏng, chỉ gồm header, cho rocPRIM". argmax của MPSGraph không ghi rõ cách phân xử khi hòa.
            </p>
          }
        >
          <p>
            hipCUB: "a thin, header-only wrapper library for rocPRIM". MPSGraph's argmax does not document its tie-break.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS101['ch10/10.1/q-reduce-order']} />
    </>
  );
}
