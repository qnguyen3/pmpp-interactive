import { useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { tieGroups, type Item } from '../../sim/sortnet';
import {
  BATCH,
  BATCH_PER_ROW,
  BATCH_ROWS,
  CANDIDATE_PRESETS,
  cascade,
  rowChecks,
  sortCandidates,
  type BatchItem,
  type PassKey,
  type SortKind,
} from '../../sim/sortstable';

/** A row of cells that wraps on a narrow screen. */
const WRAP_ROW = { display: 'flex', flexWrap: 'wrap', gap: 2, marginBottom: '0.8rem' } as const;
/** 0.15, 0.20, 0.125: two decimals unless a third is needed. */
const fmtP = (p: number) => p.toFixed(3).replace(/0$/, '');

/** One candidate: probability over token id; marked when it sits in a tie group. */
function Cand({ it, mark }: { it: Item; mark?: 'good' | 'bad' }) {
  return (
    <span className={mark ?? ''} style={{ height: 'auto', minWidth: '3.3rem', padding: '0.2rem 0.25rem', lineHeight: 1.25 }}>
      <b>{fmtP(it.key)}</b>
      <small className="muted">t{it.id}</small>
    </span>
  );
}

function TiesExplorer() {
  const [preset, setPreset] = useState(0);
  const [kind, setKind] = useState<SortKind>('network');
  const [byIndex, setByIndex] = useState(false);
  const input = CANDIDATE_PRESETS[preset].items;
  const out = sortCandidates(input, kind, byIndex);
  const groups = tieGroups(out);
  const markOf = (it: Item) => {
    const g = groups.find((x) => x.key === it.key);
    return g ? (g.increasing ? 'good' : 'bad') : undefined;
  };
  return (
    <Bench
      title={tr('Ties in a candidate list', 'Giá trị trùng trong một danh sách ứng viên')}
      task={
        <Translated vi="Tìm thiết lập làm đảo thứ tự các token trùng giá trị, rồi tìm một thay đổi duy nhất sửa được nó mà không đổi phép sắp xếp.">
          Find the setting that reorders tied tokens, then the one change that fixes it without changing the sort.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Tám ứng viên, xác suất trên token id, đến theo thứ tự token tăng dần như một hàng của softmax. Đầu ra xếp
              lớn nhất trước. Các ô trùng giá trị được đánh dấu: thành công khi các token trùng ra theo thứ tự token tăng
              dần, thất bại khi không.
            </>
          }
        >
          <>
            Eight candidates, probability over token id, arriving in increasing token order as a row of a softmax does. The
            output is largest first. Tied cells are marked: success when the tied tokens come out in increasing token
            order, failure when they do not.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Candidates', 'Ứng viên')}>
          {CANDIDATE_PRESETS.map((p, k) => (
            <button key={p.id} aria-pressed={k === preset} onClick={() => setPreset(k)}>{tr(p.label, p.id === 'two' ? 'trùng ở hai giá trị' : p.id === 'equal' ? 'tất cả bằng nhau' : 'không trùng')}</button>
          ))}
        </span>
        <span className="seg" role="group" aria-label={tr('Sort', 'Sắp xếp')}>
          <button aria-pressed={kind === 'stable'} onClick={() => setKind('stable')}><Translated vi="sắp xếp ổn định">stable sort</Translated></button>
          <button aria-pressed={kind === 'network'} onClick={() => setKind('network')}><Translated vi="mạng bitonic (14.9)">bitonic network (14.9)</Translated></button>
        </span>
        <label className="check">
          <input type="checkbox" checked={byIndex} onChange={(e) => setByIndex(e.target.checked)} />{' '}
          <Translated vi="phá thế trùng bằng chỉ số token: sắp xếp các cặp (value, index)">
            break ties by token
            index: sort (value, index) pairs
          </Translated>
        </label>
      </div>
      <div className="mx-label"><Translated vi="Đầu vào">Input</Translated></div>
      <div className="mx" data-row="input" style={WRAP_ROW}>
        {input.map((it) => <Cand key={it.id} it={it} />)}
      </div>
      <div className="mx-label"><Translated vi="Đã sắp xếp, lớn nhất trước">Sorted, largest first</Translated></div>
      <div className="mx" data-row="output" style={{ ...WRAP_ROW, marginBottom: '0.9rem' }}>
        {out.map((it) => <Cand key={it.id} it={it} mark={markOf(it)} />)}
      </div>
      {groups.length ? (
        <table className="threads" aria-label={tr('Tie groups', 'Các nhóm trùng')}>
          <thead>
            <tr>
              <th><Translated vi="giá trị">value</Translated></th>
              <th><Translated vi="các token, theo thứ tự đầu ra">tokens, in output order</Translated></th>
              <th><Translated vi="theo thứ tự token?">in token order?</Translated></th>
            </tr>
          </thead>
          <tbody>
            {groups.map((g) => (
              <tr key={g.key} data-group={fmtP(g.key)} data-ok={g.increasing ? 'yes' : 'no'}>
                <td>{fmtP(g.key)}</td>
                <td>{g.ids.map((i) => 't' + i).join(', ')}</td>
                <td>{g.increasing ? <span className="chip ok"><Translated vi="có">yes</Translated></span> : <span className="chip fail"><Translated vi="bị đảo">reordered</Translated></span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="muted" data-group="none" style={{ margin: 0 }}>
          <Translated vi="Không có hai giá trị nào bằng nhau, nên không có thứ tự trùng nào cần giữ: mọi phép sắp xếp đúng đều cho đầu ra này.">
            No two values are equal, so there is no tie order to keep: every correct sort gives this output.
          </Translated>
        </p>
      )}
    </Bench>
  );
}

const ROW_KEY_LABEL: Record<PassKey, string> = {
  get value() {
    return tr('by value, largest first', 'theo giá trị, lớn nhất trước');
  },
  get row() {
    return tr('by row id', 'theo row id');
  },
};

function BatchCells({ items, label, name }: { items: BatchItem[]; label: string; name: string }) {
  const cell = (t: BatchItem) => (
    <span key={`${t.row}-${t.token}`} style={{ height: 'auto', minWidth: '3.1rem', padding: '0.2rem 0.25rem', lineHeight: 1.25 }}>
      <b>{fmtP(t.p)}</b>
      <small className="muted">r{t.row} t{t.token}</small>
    </span>
  );
  // Positions 0 to 5 and 6 to 11: the blocks row 0 and row 1 should end up in.
  return (
    <div style={{ marginBottom: '0.8rem' }}>
      <div className="mx-label">{label}</div>
      <div data-row={name} style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem 0.9rem' }}>
        {[0, 1].map((half) => (
          <div key={half} className="mx" style={{ ...WRAP_ROW, marginBottom: 0 }}>
            {items.slice(half * BATCH_PER_ROW, (half + 1) * BATCH_PER_ROW).map(cell)}
          </div>
        ))}
      </div>
    </div>
  );
}

type Guess = '' | 'ok' | 'not';

function TwoPasses() {
  const [first, setFirst] = useState<PassKey>('row');
  const [stable2, setStable2] = useState(false);
  const [guess, setGuess] = useState<Guess[]>(['', '']);
  const [applied, setApplied] = useState(false);
  const second: PassKey = first === 'value' ? 'row' : 'value';
  const steps = cascade(BATCH, [{ key: first, stable: true }, { key: second, stable: stable2 }]);
  const checks = rowChecks(steps[2], BATCH_ROWS, BATCH_PER_ROW);
  const reset = () => setApplied(false);
  const ready = guess.every((g) => g !== '');
  return (
    <Bench
      title={tr('Two passes for a batch', 'Hai lượt cho một batch')}
      task={
        <Translated vi="Chọn hai lượt, dự đoán từng hàng, rồi áp dụng. Tìm tổ hợp duy nhất cho kết quả đúng.">
          Choose the two passes, predict each row, then apply. Find the only combination that works.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Hai hàng, mỗi hàng sáu token, được làm phẳng theo từng hàng như một tensor (rows, vocabulary) nằm trong bộ
              nhớ. Mục tiêu: token của mỗi hàng nằm trong khối sáu ô của riêng nó (các hàng theo thứ tự), lớn nhất trước.
              Lượt 1 luôn dùng phép sắp xếp ổn định; phép không ổn định là mạng bitonic của mục 14.9.
            </>
          }
        >
          <>
            Two rows of six tokens, flattened row by row as a (rows, vocabulary) tensor is in memory. The goal: each row's
            tokens in their own block of six (rows in order), largest first. Pass 1 always uses the stable sort; the unstable
            one is the bitonic network of section 14.9.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Lượt 1">Pass 1</Translated>
          <select value={first} onChange={(e) => { setFirst(e.target.value as PassKey); reset(); }}>
            <option value="value">{ROW_KEY_LABEL.value}</option>
            <option value="row">{ROW_KEY_LABEL.row}</option>
          </select>
        </label>
        <span className="ui">
          <Translated vi={<>Lượt 2: {ROW_KEY_LABEL[second]}, dùng</>}>
            Pass 2: {ROW_KEY_LABEL[second]}, with
          </Translated>
        </span>
        <span className="seg" role="group" aria-label={tr('Pass 2 sort', 'Phép sắp xếp của lượt 2')}>
          <button aria-pressed={stable2} onClick={() => { setStable2(true); reset(); }}><Translated vi="một phép sắp xếp ổn định">a stable sort</Translated></button>
          <button aria-pressed={!stable2} onClick={() => { setStable2(false); reset(); }}><Translated vi="một phép sắp xếp không ổn định">an unstable sort</Translated></button>
        </span>
      </div>
      <BatchCells items={BATCH} label={tr('Input (row 0 | row 1)', 'Đầu vào (hàng 0 | hàng 1)')} name="input" />{/* i18n-ignore: data-row key */}
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="ui"><Translated vi="Dự đoán của bạn:">Your prediction:</Translated></span>
        {[0, 1].map((r) => (
          <label key={r} className="field">
            <Translated vi={<>Hàng {r}</>}>Row {r}</Translated>
            <select value={guess[r]} onChange={(e) => { const g = guess.slice(); g[r] = e.target.value as Guess; setGuess(g); reset(); }}>
              <option value="">{tr('choose', 'chọn')}</option>
              <option value="ok">{tr('contiguous and sorted', 'liền nhau và đã sắp xếp')}</option>
              <option value="not">{tr('not both', 'không đủ cả hai')}</option>
            </select>
          </label>
        ))}
        <button className="primary" disabled={!ready} onClick={() => setApplied(true)}><Translated vi="Áp dụng cả hai lượt">Apply both passes</Translated></button>
      </div>
      {applied ? (
        <>
          <BatchCells items={steps[1]} label={tr(`After pass 1 (${ROW_KEY_LABEL[first]}, stable)`, `Sau lượt 1 (${ROW_KEY_LABEL[first]}, ổn định)`)} name="pass1" />{/* i18n-ignore: data-row key */}
          <BatchCells items={steps[2]} label={tr(`After pass 2 (${ROW_KEY_LABEL[second]}, ${stable2 ? 'stable' : 'unstable'})`, `Sau lượt 2 (${ROW_KEY_LABEL[second]}, ${stable2 ? 'ổn định' : 'không ổn định'})`)} name="pass2" />{/* i18n-ignore: data-row key */}
          <table className="threads" aria-label={tr('Row checks', 'Kiểm tra từng hàng')}>
            <thead>
              <tr>
                <th><Translated vi="hàng">row</Translated></th>
                <th><Translated vi="liền nhau, theo thứ tự hàng">contiguous, in row order</Translated></th>
                <th><Translated vi="đã sắp xếp, lớn nhất trước">sorted, largest first</Translated></th>
                <th><Translated vi="dự đoán của bạn">your prediction</Translated></th>
              </tr>
            </thead>
            <tbody>
              {checks.map((c) => {
                const right = (guess[c.row] === 'ok') === c.ok;
                return (
                  <tr key={c.row} data-rowcheck={c.row} data-ok={c.ok ? 'yes' : 'no'} data-contiguous={c.contiguous ? 'yes' : 'no'} data-sorted={c.sorted ? 'yes' : 'no'}>
                    <td><Translated vi={<>hàng {c.row}</>}>row {c.row}</Translated></td>
                    <td>{c.contiguous ? <span className="chip ok"><Translated vi="có">yes</Translated></span> : <span className="chip fail"><Translated vi="không">no</Translated></span>}</td>
                    <td>{c.sorted ? <span className="chip ok"><Translated vi="có">yes</Translated></span> : <span className="chip fail"><Translated vi="không">no</Translated></span>}</td>
                    <td data-guess={right ? 'right' : 'wrong'}>{right ? tr('right', 'đúng') : tr('wrong', 'sai')}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      ) : (
        <p className="muted" style={{ margin: 0 }}>
          <Translated vi="Hãy dự đoán cho cả hai hàng để áp dụng các lượt.">Make a prediction for both rows to apply the passes.</Translated>
        </p>
      )}
    </Bench>
  );
}

export const questionsS141: Record<string, Question> = {
  'ch14/14.1/q-cascade': {
    kind: 'choice',
    id: 'ch14/14.1/q-cascade',
    prompt: (
      <Translated
        vi={
          <>
          Lấy danh sách các bộ (age, income) của sách, [(30,150), (32,80), (22,45), (29,80)]. Sắp xếp nó theo tuổi, tăng dần.
          Rồi sắp xếp kết quả theo thu nhập, không tăng, bằng một phép sắp xếp ổn định. Kết quả là gì?
          </>
        }
      >
        <>
          Take the book's list of (age, income) tuples, [(30,150), (32,80), (22,45), (29,80)]. Sort it by age, ascending.
          Then sort the result by income, nonincreasing, with a stable sort. What comes out?
        </>
      </Translated>
    ),
    options: [
      {
        label: '(30,150), (32,80), (29,80), (22,45)',
        why: (
          <Translated vi="Đó là phép sắp xếp ổn định của danh sách gốc chỉ theo thu nhập: hai giá trị 80 theo thứ tự đầu vào. Lượt đầu đã đổi thứ tự đó.">
            That is the stable sort of the original list by income alone: the 80s in input order. The first pass changed that order.
          </Translated>
        ),
      },
      { label: '(30,150), (29,80), (32,80), (22,45)', correct: true },
      {
        label: (
          <Translated vi="Hai giá trị 80 theo thứ tự nào cũng được; phép sắp xếp ổn định không hứa gì ở đây.">
            Either order of the two 80s; a stable sort makes no promise here.
          </Translated>
        ),
        why: (
          <Translated vi="Phép sắp xếp ổn định hứa đúng điều này: các thu nhập bằng nhau giữ thứ tự chúng có khi vào, thứ tự mà lượt 1 đã đặt theo tuổi.">
            A stable sort makes exactly this promise: equal incomes keep the order they had going in, which pass 1 set by age.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
          Sau lượt 1, thứ tự là (22,45), (29,80), (30,150), (32,80). Lượt 2 theo thu nhập; hai thu nhập 80 là các khóa bằng nhau,
          nên phép sắp xếp ổn định giữ chúng như lượt 1 để lại: 29 trước 32. Kết quả được sắp theo thu nhập, và trong cùng thu
          nhập thì theo tuổi. Đó là chuỗi lượt (cascade): khóa phụ trước, khóa chính sau cùng.
          </p>
        }
      >
        <p>
          After pass 1 the order is (22,45), (29,80), (30,150), (32,80). Pass 2 is by income; the two incomes of 80 are equal
          keys, so a stable sort keeps them as pass 1 left them: 29 before 32. The result is sorted by income, and within
          equal income by age. That is the cascade: the secondary key first, the primary key last.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.1, p. 330',
  },
  'ch14/14.1/q-ties': {
    kind: 'numeric',
    id: 'ch14/14.1/q-ties',
    prompt: (
      <Translated
        vi={
          <>
          Một bộ lấy mẫu giữ một hàng xác suất bf16, mỗi giá trị trong [0, 1], trên một bộ từ vựng 151,936 token. Ít nhất bao nhiêu
          token phải có cùng giá trị với một token nào đó đứng trước trong hàng?
          </>
        }
      >
        <>
          A sampler holds a row of bf16 probabilities, each in [0, 1], over a vocabulary of 151,936 tokens. At least how many
          tokens must have the same value as some token earlier in the row?
        </>
      </Translated>
    ),
    answer: 135679,
    explain: (
      <Translated
        vi={
          <>
          <p>
          bf16 biểu diễn được tối đa 16,257 giá trị phân biệt trong [0, 1]: các mẫu bit 0x0000 đến 0x3F80, mẫu cuối là 1.0 (số
          không âm thêm một mẫu bit nhưng không thêm giá trị). 151,936 token vào 16,257 giá trị: ít nhất 151,936 − 16,257 =
          135,679 token lặp lại một giá trị đã gặp. Một hàng thực tế có ít giá trị phân biệt hơn thế nhiều, vì phần lớn xác suất
          rất nhỏ.
          </p>
          <p>
          Vì vậy giá trị trùng là điều chắc chắn chứ không hiếm, và thứ tự của chúng là một phần kết quả của bộ lấy mẫu: sau khi
          sắp xếp, token nào nằm ở một hạng cho trước quyết định token nào được một lần rút ngẫu nhiên cho trước chọn.
          </p>
          </>
        }
      >
        <>
          <p>
            bf16 can represent at most 16,257 distinct values in [0, 1]: the bit patterns 0x0000 to 0x3F80, the last being
            1.0 (negative zero adds a pattern but not a value). 151,936 tokens into 16,257 values: at least 151,936 − 16,257 =
            135,679 of them repeat a value already seen. A real row has far fewer distinct values than that, since most
            probabilities are tiny.
          </p>
          <p>
            Ties are therefore guaranteed, not rare, and their order is part of what a sampler produces: after a sort, which
            token sits at a given rank decides which token a given random draw selects.
          </p>
        </>
      </Translated>
    ),
    source: 'our arithmetic; vocabulary size from notes/today/ch09-facts.md D7',
    vi: { source: 'phép tính của khóa học này; kích thước bộ từ vựng lấy từ notes/today/ch09-facts.md D7' },
  },
  'ch14/14.1/q-unstable-second': {
    kind: 'choice',
    id: 'ch14/14.1/q-unstable-second',
    prompt: (
      <Translated
        vi={
          <>
          Bạn sắp xếp một batch (rows, vocabulary) đã làm phẳng trong hai lượt: theo giá trị trên toàn bộ, rồi theo row id. Lượt
          theo row id dùng một phép sắp xếp không ổn định. Kết quả là gì?
          </>
        }
      >
        <>
          You sort a flattened (rows, vocabulary) batch in two passes: by value over everything, then by row id. The row-id
          pass uses an unstable sort. What comes out?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Các hàng liền nhau và theo thứ tự hàng; bên trong mỗi hàng, các giá trị không theo thứ tự nào được hứa.">
            Rows contiguous and in row order; inside each row, values in no promised order.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Hoàn toàn đúng: row id là số nguyên nhỏ, không có phần thập phân để so sánh, nên lượt theo row id không thể sai.">
            Fully correct: row ids are small integers with no fractions to compare, so the row-id pass cannot go wrong.
          </Translated>
        ),
        why: (
          <Translated vi="Nó không thể làm sai các hàng. Nhưng mọi token của một hàng có cùng row id, nên trong một hàng mọi khóa đều trùng nhau, và một phép sắp xếp không ổn định có thể xáo trộn chúng.">
            It cannot get the rows wrong. But every token of a row has the same row id, so inside a row all keys tie, and an unstable sort may shuffle them.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các hàng xen kẽ: một lượt không ổn định có thể đưa một token vào giữa các token của hàng khác.">
            Rows interleaved: an unstable pass may move a token in among another row's tokens.
          </Translated>
        ),
        why: (
          <Translated vi="Lượt cuối sắp xếp theo row id; dù làm gì khác, nó vẫn gom các hàng lại.">
            The last pass sorts by row id; whatever else it does, it groups the rows.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
          Lượt cuối quyết định thứ tự chính (các hàng), và chỉ tính ổn định của nó mới mang được thứ tự của lượt trước (các giá
          trị) đi qua. Trong một hàng, mọi khóa của lượt theo row id đều bằng nhau: một nhóm trùng gồm cả hàng. Trong trình khám
          phá ở trên, hãy chọn giá trị trước và một lượt thứ hai không ổn định: cả hai hàng ra liền nhau nhưng không được sắp xếp.
          </p>
        }
      >
        <p>
          The last pass decides the primary order (rows), and only its stability carries the earlier pass's order (values)
          through. Inside one row every key of the row-id pass is equal: a tie group of the whole row. In the explorer above,
          choose value first and an unstable second pass: both rows come out contiguous and unsorted.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.1, p. 330; notes/today/ch14-facts.md D5',
    vi: { source: 'ch. 14 §14.1, p. 330; notes/today/ch14-facts.md mục D5' },
  },
  'ch14/14.1/q-bound': {
    kind: 'choice',
    id: 'ch14/14.1/q-bound',
    prompt: (
      <Translated vi={<>Vì sao radix sort có thể làm tốt hơn cận cỡ N log N phép so sánh?</>}>
        <>Why can radix sort do better than the bound of order N log N comparisons?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Nó dùng nhiều luồng hơn: với một luồng cho mỗi khóa, các phép so sánh được trải ra trên cả lưới.">
            It uses more threads: with one thread per key, the comparisons are spread over the whole grid.
          </Translated>
        ),
        why: (
          <Translated vi="Cận này là về số phép so sánh, bất kể bao nhiêu luồng thực hiện chúng.">
            The bound is on comparisons, however many threads make them.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó không bao giờ so sánh hai khóa: nó đặt mỗi khóa dựa vào các chữ số của khóa, còn cận kia chỉ đếm phép so sánh.">
            It never compares two keys: it places each key from its digits, and the bound counts only comparisons.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Cận này dành cho mã tuần tự; một phép sắp xếp song song có thể thực hiện các phép so sánh trong ít bước hơn.">
            The bound is for sequential code; a parallel sort can make its comparisons in fewer steps.
          </Translated>
        ),
        why: (
          <Translated vi="Nó đúng với mọi phép sắp xếp chỉ biết về thứ tự bằng cách so sánh từng cặp.">
            It holds for any sort that learns about the order only by comparing pairs.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không làm được: radix sort chạy log N lượt, mỗi lượt N bước, nên cũng là N log N.">
            It does not: radix sort makes log N passes of N steps each, so it is N log N too.
          </Translated>
        ),
        why: (
          <Translated vi="Với độ rộng khóa cố định, số lượt là cố định và mỗi lượt tuyến tính theo N.">
            For a fixed key width, the number of passes is fixed and each pass is linear in N.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
          Một phép sắp xếp so sánh biết về thứ tự từng phép so sánh một, và đó là thứ cận kia đếm. Radix sort không bao giờ hỏi
          "a có nhỏ hơn b không?": nó đọc một chữ số và tính xem khóa đi đâu. Cách đó chỉ đúng với những khóa mà các bit, khi đọc
          như một số nguyên không dấu, có thứ tự giống thứ tự của giá trị. Số nguyên không dấu thỏa điều này; số float thì không,
          và từ mục 14.4 trở đi phải xử lý chuyện đó.
          </p>
        }
      >
        <p>
          A comparison sort learns about the order one comparison at a time, and that is what the bound counts. Radix sort
          never asks "is a less than b?": it reads a digit and computes where the key goes. That works only for keys whose
          bits, read as an unsigned integer, order the way the values do. Unsigned integers do; floats do not, which section
          14.4 onward has to deal with.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.1, p. 330',
  },
};

export function S141() {
  return (
    <>
      <Translated
        vi={
          <p>
          Một phép sắp xếp có hai nhiệm vụ. Đầu ra phải theo thứ tự, không giảm hoặc không tăng, và phải là một hoán vị của đầu
          vào: mọi phần tử được giữ, không phần tử nào được bịa thêm. Nhiệm vụ thứ hai nghe có vẻ hiển nhiên, nhưng lại là thứ
          một phép sắp xếp song song cẩu thả làm hỏng đầu tiên, như mục tiếp theo cho thấy.
          </p>
        }
      >
        <p>
          A sort has two jobs. The output must be in order, nondecreasing or nonincreasing, and it must be a permutation of
          the input: every element kept, none invented. The second job sounds trivial and is the one a careless parallel
          sort breaks first, as the next section shows.
        </p>
      </Translated>
      <Source section="14.1" pages="330" />
      <Translated
        vi={
          <p>
          Trường hợp đáng chú ý của sách là các phần tử có một khóa và một giá trị: các bộ (age, income), sắp theo thu nhập.
          Trong một bộ lấy mẫu, khóa là một xác suất và giá trị là token id của nó. Một argsort chính là như vậy: một phép sắp
          xếp khóa-giá trị có các giá trị khởi đầu là 0, 1, ..., V − 1, để sau khi sắp xếp, các giá trị cho biết token nào nằm ở
          mỗi hạng.
          </p>
        }
      >
        <p>
          The book's interesting case is elements with a key and a value: tuples of (age, income), sorted by income. In a
          sampler the key is a probability and the value is its token id. An argsort is exactly this: a key-value sort whose
          values start as 0, 1, ..., V − 1, so that after sorting the values say which token sits at each rank.
        </p>
      </Translated>

      <h2><Translated vi="Tính ổn định, và vì sao bộ lấy mẫu cần nó">Stability, and why a sampler needs it</Translated></h2>
      <Translated
        vi={
          <p>
          Sắp xếp danh sách của sách [(30,150), (32,80), (22,45), (29,80)] theo thu nhập, lớn nhất trước. Cả (32,80) và (29,80)
          đều có khóa 80. Một phép sắp xếp <b>ổn định</b> hứa rằng các khóa bằng nhau giữ thứ tự đầu vào, nên (32,80) vẫn đứng
          trước. Một phép sắp xếp không ổn định chỉ hứa thứ tự đúng của các khóa; giá trị 80 nào đứng trước cũng được, và hai lần
          chạy có thể cho kết quả khác nhau.
          </p>
        }
      >
        <p>
          Sort the book's list [(30,150), (32,80), (22,45), (29,80)] by income, largest first. Both (32,80) and (29,80) have
          the key 80. A <b>stable</b> sort promises that equal keys keep their input order, so (32,80) stays first. An
          unstable sort promises only a correct order of keys; either 80 may come first, and two runs may disagree.
        </p>
      </Translated>
      <Source section="14.1" pages="330" />
      <Translated
        vi={
          <p>
          Với một bộ lấy mẫu giữ xác suất ở dạng bf16, giá trị trùng không phải trường hợp hiếm. Một bộ từ vựng lớn có nhiều token
          hơn hẳn số giá trị bf16 có trong [0, 1], nên nhiều token phải chung một giá trị. Thứ tự của chúng sau khi sắp xếp quyết
          định token nào nằm ở mỗi hạng, và do đó token nào được một số ngẫu nhiên cho trước chọn. Phép sắp xếp ổn định cố định
          thứ tự đó theo chỉ số token tăng dần; phép không ổn định để nó tùy vào cách cài đặt.
          </p>
        }
      >
        <p>
          For a sampler that holds its probabilities in bf16, ties are not an edge case. A large vocabulary has far more
          tokens than bf16 has values in [0, 1], so many tokens must share a value. Their order after the sort decides which
          token sits at each rank, and therefore which token a given random number selects. A stable sort fixes that order
          to increasing token index; an unstable one leaves it to the implementation.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
            Cách nhìn từ góc bộ lấy mẫu là của khóa học này. Sách bàn về tính ổn định cho sắp xếp nhiều khóa và, ở mục 14.4, cho
            tính đúng đắn của radix sort.
            </p>
          }
        >
          <p>
            The sampler reading is ours. The book discusses stability for multi-key sorting and, in section 14.4, for radix
            sort's correctness.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS141['ch14/14.1/q-ties']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
          Trình khám phá sắp xếp tám ứng viên theo thứ tự lớn nhất trước, bằng một phép sắp xếp ổn định hoặc bằng một phép sắp xếp
          không ổn định cụ thể: mạng bitonic bạn sẽ gặp ở mục 14.9. Hãy bắt đầu với trùng ở hai giá trị và mạng bitonic.
          </p>
        }
      >
        <p>
          The explorer sorts eight candidates largest first, with a stable sort or with one particular unstable sort: the
          bitonic network you will meet in section 14.9. Start with ties at two values and the network.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
            Phép sắp xếp không ổn định ở đây là một mạng cố định. Phép sắp xếp không ổn định không hứa gì, nên một phép khác có thể
            đảo các giá trị trùng theo cách khác hoặc, nhờ may mắn, không đảo gì cả. Xác suất là các số thập phân ngắn, không phải
            mẫu bit bf16.
            </p>
          }
        >
          <p>
            The unstable sort here is one fixed network. An unstable sort promises nothing, so another one could reorder
            ties differently or, by luck, not at all. Probabilities are short decimals, not bf16 bit patterns.
          </p>
        </Translated>
      </Simplification>
      <TiesExplorer />
      <Translated
        vi={
          <p>
          Có ba điều cần thấy. Mạng bitonic đảo một nhóm trùng và để yên nhóm kia. Đánh dấu tùy chọn chỉ số làm mạng cho kết quả
          giống phép sắp xếp ổn định: so sánh các cặp (value, index) nghĩa là không bao giờ có hai khóa bằng nhau, nên mọi phép
          sắp xếp đúng đều cho cùng một đầu ra. Và khi cả tám giá trị bằng nhau, mạng giữ nguyên thứ tự đầu vào, vì không bộ so
          sánh nào của nó thấy một cặp sai thứ tự: ổn định nhờ may mắn, không phải nhờ lời hứa.
          </p>
        }
      >
        <p>
          Three things to see. The network reorders one tie group and leaves the other alone. Ticking the index option
          makes the network agree with the stable sort: comparing (value, index) pairs means no two keys are ever equal, so
          any correct sort produces the same output. And with all eight values equal, the network keeps the input order,
          because none of its comparators ever finds a pair out of order: stability by luck, not by promise.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch14-facts.md, D4 and D5)', '2026-11-20 (notes/today/ch14-facts.md, mục D4 và D5)')}>
        <Translated
          vi={
            <p>
            <code>torch.sort</code> của PyTorch mặc định là <code>stable=False</code>, và trên CUDA sắp xếp các lát cắt từ 32 phần
            tử trở xuống bằng một mạng bitonic không ổn định. Tài liệu của <code>torch.topk</code> ghi rằng các chỉ số trùng "are
            not guaranteed to be stable" (không được bảo đảm ổn định). Tài liệu MLX ghi <code>mlx.core.sort</code> là ổn định.
            </p>
          }
        >
          <p>
            PyTorch's <code>torch.sort</code> defaults to <code>stable=False</code>, and on CUDA sorts slices of 32 or fewer
            with an unstable bitonic network. <code>torch.topk</code> documents that tied indices "are not guaranteed to be
            stable". MLX documents <code>mlx.core.sort</code> as stable.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Sắp xếp theo hai khóa">Sorting by two keys</Translated></h2>
      <Translated
        vi={
          <p>
          Tính ổn định là thứ làm cho sắp xếp nhiều khóa hoạt động. Để sắp theo một khóa chính và, trong đó, theo một khóa phụ,
          hãy sắp theo khóa phụ trước, rồi theo khóa chính bằng một phép sắp xếp ổn định. Lượt thứ hai sắp theo khóa chính và,
          giữa các khóa chính bằng nhau, giữ thứ tự mà lượt đầu đã tạo.
          </p>
        }
      >
        <p>
          Stability is what makes multi-key sorting work. To sort by a primary key and, within it, by a secondary key, sort
          by the secondary key first, then by the primary key with a stable sort. The second pass orders by the primary key
          and, among equal primary keys, keeps the order the first pass made.
        </p>
      </Translated>
      <Source section="14.1" pages="330" />
      <Ask q={questionsS141['ch14/14.1/q-cascade']} />
      <Translated
        vi={
          <p>
          Cùng chuỗi lượt đó có thể sắp mọi hàng của một batch cùng lúc: làm phẳng các hàng thành một mảng và sắp nó hai lần, một
          lần theo giá trị và một lần theo row id. Theo thứ tự nào, và lượt nào phải ổn định? Trình khám phá dưới đây cho bạn hai
          hàng; hãy dự đoán từng hàng trước khi áp dụng các lượt.
          </p>
        }
      >
        <p>
          The same cascade can sort every row of a batch at once: flatten the rows into one array and sort it twice, once by
          value and once by row id. In which order, and which pass must be stable? The explorer below gives you two rows;
          predict each one before you apply the passes.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
            Hai hàng sáu phần tử, được sắp trên CPU bằng mô hình của khóa học. Lượt không ổn định là cùng mạng cố định như ở trên,
            nên một phép sắp xếp không ổn định khác có thể xáo trộn các giá trị trùng theo cách khác.
            </p>
          }
        >
          <p>
            Two rows of six, sorted on the CPU by the course's model. The unstable pass is the same fixed network as above,
            so another unstable sort could scramble the ties differently.
          </p>
        </Translated>
      </Simplification>
      <TwoPasses />
      <Translated
        vi={
          <p>
          Chỉ một tổ hợp cho kết quả đúng: giá trị trước, row id sau cùng, và lượt cuối ổn định. Sắp theo hàng trước thì hỏng dù
          lượt thứ hai ổn định đến đâu, vì một lượt theo giá trị không để ý gì đến các hàng.
          </p>
        }
      >
        <p>
          Only one combination works: values first, row id last, and the last pass stable. Rows first fails however stable
          the second pass is, because a pass by value pays no attention to rows.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch14-facts.md, D5)', '2026-11-20 (notes/today/ch14-facts.md, mục D5)')}>
        <Translated
          vi={
            <p>
            PyTorch trên CUDA sắp các hàng của một tensor theo cách này trong <code>segmented_sort_pairs_by_full_sort</code>{' '}
            (<code>SortStable.cu</code>): "First we sort by values, globally ... Then we stable sort by segment id" (trước hết sắp
            theo giá trị trên toàn cục, rồi sắp ổn định theo segment id). Cách này được dùng cho 2 đến 127 đoạn ngắn hơn một triệu
            phần tử. Một đoạn duy nhất, hoặc các đoạn rất dài, dùng radix sort theo từng đoạn; từ 128 đoạn trở lên, dùng segmented
            radix sort của CUB.
            </p>
          }
        >
          <p>
            PyTorch on CUDA sorts the rows of a tensor this way in <code>segmented_sort_pairs_by_full_sort</code>{' '}
            (<code>SortStable.cu</code>): "First we sort by values, globally ... Then we stable sort by segment id". It is
            used for 2 to 127 segments shorter than a million elements. One segment, or very long ones, take a per-segment
            radix sort; 128 segments or more, CUB's segmented radix sort.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS141['ch14/14.1/q-unstable-second']} />

      <h2><Translated vi="Sắp xếp so sánh và sắp xếp không so sánh">Comparison and non-comparison sorts</Translated></h2>
      <Translated
        vi={
          <p>
          Một phép sắp xếp chỉ biết về thứ tự bằng cách so sánh từng cặp khóa cần cỡ N log N phép so sánh cho N khóa; không phép
          sắp xếp so sánh nào làm tốt hơn. Một phép sắp xếp không so sánh thì làm được, với một cái giá: nó không dùng được cho
          kiểu khóa bất kỳ. Cả hai loại đều có thể song song hóa.
          </p>
        }
      >
        <p>
          A sort that learns about the order only by comparing pairs of keys needs on the order of N log N comparisons for N
          keys; no comparison sort does better. A non-comparison sort can, at a price: it does not work for arbitrary key
          types. Both kinds can be parallelised.
        </p>
      </Translated>
      <Source section="14.1" pages="330" />
      <Translated
        vi={
          <p>
          Radix sort, chủ đề chính của chương, thuộc loại không so sánh. Nó đọc mỗi khóa từng chữ số một và tính xem khóa đi đâu,
          điều này đòi hỏi các khóa mà bit của chúng, khi đọc như một số nguyên không dấu, có thứ tự giống thứ tự của giá trị. Số
          nguyên không dấu thỏa điều kiện. Số float thì không: một số float âm có bit dấu được bật, nên khi đọc như số nguyên không
          dấu nó trông lớn hơn mọi số dương. Xác suất, vốn không bao giờ âm, tình cờ có thứ tự đúng; logit thì không, và phép biến
          đổi sửa chúng đi kèm với radix sort.
          </p>
        }
      >
        <p>
          Radix sort, the chapter's main subject, is the non-comparison kind. It reads each key one digit at a time and
          computes where the key goes, which needs keys whose bits order the way the values do when read as an unsigned
          integer. Unsigned integers qualify. Floats do not: a negative float has its sign bit set, so read as an unsigned
          integer it looks larger than every positive one. Probabilities, which are never negative, happen to order
          correctly; logits do not, and the transform that fixes them comes with radix sort.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
            Ví dụ về float là của khóa học này. Phép biến đổi được CUB ghi trong tài liệu cho radix sort của nó
            (notes/today/ch14-facts.md A3).
            </p>
          }
        >
          <p>
            The float example is ours. The transform is documented by CUB for its radix sort (notes/today/ch14-facts.md A3).
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS141['ch14/14.1/q-bound']} />

      <h2><Translated vi="Bản đồ của chương">The chapter's map</Translated></h2>
      <Translated
        vi={
          <p>
          Chương này dùng sắp xếp như một nghiên cứu điển hình về song song hóa và tối ưu hóa, theo Satish, Harris và Garland. Nó
          đề cập ngắn gọn hai phép sắp xếp so sánh, <a href={href('ch14', '14.2')}>sắp xếp chẵn-lẻ</a> và{' '}
          <a href={href('ch14', '14.3')}>sắp xếp trộn (merge)</a>, và đi sâu vào một phép sắp xếp không so sánh,{' '}
          <a href={href('ch14', '14.4')}>radix sort</a> (các mục 14.4 đến 14.8): một lượt là một phép phân hoạch ổn định dựng
          từ một scan loại trừ, và các tối ưu hóa chính là những tối ưu của bộ lọc ổn định ở chương 12.{' '}
          <a href={href('ch14', '14.9')}>Mục 14.9</a> điểm qua phần còn lại, và là nơi liệt kê các lời gọi thư viện mà bình
          thường bạn nên dùng thay thế.
          </p>
        }
      >
        <p>
          The chapter uses sorting as a case study in parallelisation and optimisation, after Satish, Harris and Garland. It
          covers two comparison sorts, <a href={href('ch14', '14.2')}>odd-even sort</a> and{' '}
          <a href={href('ch14', '14.3')}>merge sort</a>, briefly, and one non-comparison sort,{' '}
          <a href={href('ch14', '14.4')}>radix sort</a>, at length (sections 14.4 to 14.8): a pass is a stable partition
          built from an exclusive scan, and the optimisations are those of chapter 12's stable filter.{' '}
          <a href={href('ch14', '14.9')}>Section 14.9</a> surveys the rest, and is where the library calls you should
          normally use instead are listed.
        </p>
      </Translated>
      <Source section="14.1" pages="330–331" />
    </>
  );
}
