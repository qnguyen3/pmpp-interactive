import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Aside, Bench, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import { EXP_LIMIT_F32, attention, differing, dotTerms, makeAttnProblem, visible } from '../../sim/attnref';

export const questionsS202: Record<string, Question> = {
  'ch20/20.2/q-s-bytes': {
    kind: 'numeric',
    id: 'ch20/20.2/q-s-bytes',
    prompt: (
      <Translated vi={<>Một head, N = 4096 token, độ rộng head d = 128, fp32. S chiếm bao nhiêu MiB?</>}>
        <>One head, N = 4096 tokens, head width d = 128, fp32. How many MiB does S take?</>
      </Translated>
    ),
    answer: 64,
    unit: 'MiB',
    explain: (
      <Translated
        vi={
          <p>
            4096 × 4096 × 4 B = 64 MiB. Q là 4096 × 128 × 4 B = 2 MiB, nên S lớn gấp N / d = 32 lần ma trận sinh ra nó, và nó
            tăng theo bình phương của ngữ cảnh trong khi Q tăng tuyến tính.
          </p>
        }
      >
        <p>
          4096 × 4096 × 4 B = 64 MiB. Q is 4096 × 128 × 4 B = 2 MiB, so S is N / d = 32 times the size of the matrix it
          came from, and it grows with the square of the context while Q grows linearly.
        </p>
      </Translated>
    ),
    source: 'ch. 20 pp. 483–484 (shapes); our arithmetic',
    vi: { source: 'ch. 20 pp. 483–484 (kích thước); phép tính của chúng tôi' },
  },
  'ch20/20.2/q-row-nonzeros': {
    kind: 'numeric',
    id: 'ch20/20.2/q-row-nonzeros',
    prompt: (
      <Translated vi={<>N = 8, mặt nạ nhân quả. Hàng 5 của P có bao nhiêu phần tử khác 0 (hàng và cột đếm từ 0)?</>}>
        <>N = 8, causal mask. How many non-zero entries does row 5 of P have (rows and columns counted from 0)?</>
      </Translated>
    ),
    answer: 6,
    explain: (
      <Translated
        vi={
          <p>
            Các cột 0 đến 5: token 5 chú ý tới mọi token trước nó <em>và tới chính nó</em>. Sách viết &ldquo;các token đã sinh
            trước đó&rdquo;; mặt nạ giữ lại đường chéo.
          </p>
        }
      >
        <p>
          Columns 0 to 5: token 5 attends to every earlier token <em>and to itself</em>. The book&rsquo;s words are
          &ldquo;previously generated tokens&rdquo;; the mask keeps the diagonal.
        </p>
      </Translated>
    ),
    source: 'ch. 20 p. 484',
  },
  'ch20/20.2/q-no-scale': {
    kind: 'choice',
    id: 'ch20/20.2/q-no-scale',
    prompt: (
      <Translated vi={<>Bạn quên 1/√d với d = 128. Mỗi hàng của P sẽ ra sao?</>}>
        <>You forget the 1/√d with d = 128. What happens to each row of P?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Logits lớn lên khoảng 11.3 lần; mỗi hàng gần thành one-hot.">{'Logits grow about 11.3 times; each row gets close to one-hot.'}</Translated>, correct: true },
      { label: <Translated vi="Không gì cả: softmax không quan tâm đến tỷ lệ của đầu vào.">{'Nothing: softmax does not care about the scale of its inputs.'}</Translated>, why: <Translated vi="Softmax bỏ qua một hằng số được cộng vào mọi logit của một hàng, không bỏ qua một hệ số nhân với chúng.">{'Softmax ignores a constant added to every logit of a row, not a factor multiplying them.'}</Translated> },
      { label: <Translated vi="Các hàng không còn có tổng bằng 1, nên O bị phóng to khoảng 11.3 lần.">{'The rows no longer sum to 1, so O is scaled up by about 11.3.'}</Translated>, why: <Translated vi="Hàng softmax nào cũng có tổng bằng 1; tỷ lệ thay đổi cách phân chia khối lượng, không thay đổi tổng lượng.">{'Any softmax row sums to 1; the scale changes how the mass is shared, not how much there is.'}</Translated> },
      { label: <Translated vi="Các phần tử bị che trở nên khác 0, vì mặt nạ cũng bị co giãn.">{'Masked entries become non-zero, because the mask is scaled too.'}</Translated>, why: <Translated vi="Âm vô cực nhân với bất kỳ hệ số dương nào vẫn là âm vô cực.">{'Minus infinity times any positive factor is still minus infinity.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            √128 ≈ 11.3, nên mọi logit lớn lên theo hệ số đó. Hiệu giữa các logit cũng lớn lên theo và logit lớn nhất áp đảo:
            hàng sắc nét dần về one-hot. Lý do sách đưa ra cho hệ số này là huấn luyện: các hàng softmax bão hòa cho gradient
            triệt tiêu. Bỏ chọn &ldquo;co giãn theo 1/√d&rdquo; ở trên để xem điều đó ở d = 6.
          </p>
        }
      >
        <p>
          √128 ≈ 11.3, so every logit grows by that factor. Differences between logits grow with them and the largest
          dominates: the row sharpens towards one-hot. The book&rsquo;s reason for the factor is training: saturated
          softmax rows give vanishing gradients. Untick &ldquo;scale by 1/√d&rdquo; above to watch it at d = 6.
        </p>
      </Translated>
    ),
    source: 'ch. 20 p. 484',
  },
  'ch20/20.2/q-shift': {
    kind: 'choice',
    id: 'ch20/20.2/q-shift',
    prompt: (
      <Translated
        vi={
          <>
            Vì sao trừ đi giá trị lớn nhất của mỗi hàng không làm P thay đổi, trong khi trừ một hằng số khác nhau ở mỗi{' '}
            <em>cột</em> thì có?
          </>
        }
      >
        <>
          Why does subtracting each row&rsquo;s max leave P unchanged, while subtracting a different constant from each{' '}
          <em>column</em> would not?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Dịch một hàng co giãn tử số và tổng của nó như nhau; dịch một cột thì không.">{'A row\'s shift scales its numerators and sum alike; a column shift does not.'}</Translated>, correct: true },
      { label: <Translated vi="Vì chỉ giá trị lớn nhất của hàng mới giữ mọi số mũ hữu hạn trong float32.">{'Because only the row max keeps every exponent finite in float32.'}</Translated>, why: <Translated vi="Bất kỳ hằng số nào ít nhất bằng giá trị lớn nhất cũng giữ chúng hữu hạn; tính hữu hạn không phải lý do P không đổi.">{'Any constant at least the max would also keep them finite; finiteness is not why P is unchanged.'}</Translated> },
      { label: <Translated vi="Vì softmax áp dụng lên các cột, nên chỉ phép dịch cột mới tác động tới nó.">{'Because softmax is applied to columns, so only a column shift reaches it.'}</Translated>, why: <Translated vi="Softmax chuẩn hóa từng hàng của S + M (Eq. 20.2): mỗi hàng có tổng bằng 1.">{'Softmax normalises each row of S + M (Eq. 20.2): a row sums to 1.'}</Translated> },
      { label: <Translated vi="Nó có làm P thay đổi chút ít, do làm tròn số float của phép trừ.">{'It does change P slightly, by the float rounding of the subtraction.'}</Translated>, why: <Translated vi="Làm tròn là chi tiết của float32; câu hỏi nói về đại số, nơi thay đổi đúng bằng không.">{'Rounding is a float32 detail; the question is about the algebra, where the change is exactly none.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trừ m<sub>r</sub> khỏi mọi logit của hàng r nhân mỗi tử số e<sup>l</sup> và tổng của chúng với cùng một hệ số
            e<sup>−m<sub>r</sub></sup>, và hệ số này triệt tiêu. Một phép dịch khác nhau giữa các cột nhân các số hạng của một
            hàng với các hệ số khác nhau, nên không triệt tiêu. Trong trình xem, hãy cộng +90 vào mọi logit: P không nhúc nhích
            (đó là một phép dịch cho mỗi hàng, như nhau cho tất cả).
          </p>
        }
      >
        <p>
          Subtracting m<sub>r</sub> from every logit of row r multiplies each numerator e<sup>l</sup> and their sum by the
          same factor e<sup>−m<sub>r</sub></sup>, which cancels. A shift that differs between columns multiplies the terms
          of one row by different factors, which does not cancel. In the viewer, offset every logit by +90: P does not
          move (it is one shift per row, the same for all).
        </p>
      </Translated>
    ),
    source: 'ch. 20 Eq. 20.2, p. 484',
  },
  'ch20/20.2/q-rows-x': {
    kind: 'numeric',
    id: 'ch20/20.2/q-rows-x',
    prompt: (
      <Translated
        vi={
          <>
            Ví dụ đồ chơi của sách: mỗi từ là một token, không có system prompt, prompt là &ldquo;The title of this book is
            Programming&rdquo;. Trong cách làm cơ sở này, X có bao nhiêu hàng ở vòng sinh phát ra &ldquo;Processors&rdquo;?
          </>
        }
      >
        <>
          The book&rsquo;s toy example: whole words as tokens, no system prompt, the prompt &ldquo;The title of this book
          is Programming&rdquo;. In this baseline, how many rows does X have in the generation iteration that emits
          &ldquo;Processors&rdquo;?
        </>
      </Translated>
    ),
    answer: 9,
    unit: 'rows',
    vi: { unit: 'hàng', source: 'ch. 20 pp. 485–486, Hình 20.3' },
    explain: (
      <Translated
        vi={
          <p>
            Bảy token của prompt, cộng &ldquo;Massively&rdquo; (do giai đoạn tóm tắt phát ra) và &ldquo;Parallel&rdquo; (vòng
            sinh 1): 9 hàng ở vòng 2, vòng phát ra &ldquo;Processors&rdquo;. Cách làm cơ sở tính lại mọi hàng của K, V, Q, S, P
            và O ở mọi lớp, mọi vòng. Mục{' '}
            <a href={href('ch20', '20.4')}>20.4</a> cho thấy chỉ có hàng cuối là mới.
          </p>
        }
      >
        <p>
          Seven prompt tokens, plus &ldquo;Massively&rdquo; (emitted by the summarization phase) and &ldquo;Parallel&rdquo;
          (generation iteration 1): 9 rows in iteration 2, which emits &ldquo;Processors&rdquo;. The baseline recomputes
          every row of K, V, Q, S, P and O in every layer, every iteration. Section{' '}
          <a href={href('ch20', '20.4')}>20.4</a> shows that only the last row is new.
        </p>
      </Translated>
    ),
    source: 'ch. 20 pp. 485–486, Fig. 20.3',
  },
};

// ---------------------------------------------------------------- one head, every matrix
const HOT: CSSProperties = { background: 'var(--ink)', color: 'var(--page)', borderColor: 'var(--ink)', fontWeight: 650 };
const ON: CSSProperties = { background: 'var(--rule-soft)', borderColor: 'var(--ink-2)' };
const num = (x: number, digits = 2) => {
  if (Number.isNaN(x)) return 'NaN';
  if (x === Infinity) return '∞';
  if (x === -Infinity) return '−∞';
  const s = Number(x.toFixed(digits)).toString();
  return s.replace('-', '−');
};

function Grid({
  label,
  rows,
  cols,
  value,
  style,
  cls,
  onPick,
  name,
}: {
  label: string;
  rows: number;
  cols: number;
  value: (r: number, c: number) => string;
  style?: (r: number, c: number) => CSSProperties | undefined;
  cls?: (r: number, c: number) => string;
  onPick?: (r: number, c: number) => void;
  name: string;
}) {
  return (
    <div>
      <div className="mx-label">{label}</div>
      <div className="mx" data-grid={name} style={{ gridTemplateColumns: `repeat(${cols}, auto)` }}>
        {Array.from({ length: rows * cols }, (_, k) => {
          const r = Math.floor(k / cols);
          const c = k % cols;
          return (
            <span
              key={k}
              data-r={r}
              data-c={c}
              className={cls?.(r, c) ?? ''}
              style={{ ...style?.(r, c), cursor: onPick ? 'pointer' : undefined }}
              onClick={onPick ? () => onPick(r, c) : undefined}
              onMouseEnter={onPick ? () => onPick(r, c) : undefined}
              role={onPick ? 'button' : undefined}
              tabIndex={onPick ? 0 : undefined}
              onKeyDown={onPick ? (e) => (e.key === 'Enter' || e.key === ' ') && onPick(r, c) : undefined}
            >
              {value(r, c)}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function Viewer() {
  const [n, setN] = useState(6);
  const [d, setD] = useState(4);
  const [seed, setSeed] = useState(1);
  const [scale, setScale] = useState(true);
  const [mask, setMask] = useState(true);
  const [subtract, setSubtract] = useState(true);
  const [offset, setOffset] = useState(0);
  const [pick, setPick] = useState<[number, number]>([3, 1]);
  const prob = useMemo(() => makeAttnProblem(n, d, seed), [n, d, seed]);
  const opts = { scale, mask, subtractMax: subtract, offset, precision: 'f32' as const };
  const res = useMemo(() => attention(prob, opts), [prob, scale, mask, subtract, offset]); // eslint-disable-line react-hooks/exhaustive-deps
  const causal = useMemo(() => attention(prob, { ...opts, mask: true }), [prob, scale, subtract, offset]); // eslint-disable-line react-hooks/exhaustive-deps
  const leaks = mask ? [] : differing(res.O, causal.O, 1e-6, true);
  const leakRows = new Set(leaks.map(([r]) => r));
  const [pi, pj] = [Math.min(pick[0], n - 1), Math.min(pick[1], n - 1)];
  const vis = !mask || visible(pi, pj, n, n);
  const dot = dotTerms(prob, pi, pj, 'f32');
  const nanCount = res.P.flat().filter(Number.isNaN).length;
  const live = res.live[pi];
  const rowNaN = res.P[pi].some(Number.isNaN);

  const setSize = (nn: number, dd: number) => {
    setN(nn);
    setD(dd);
    setPick(([r, c]) => [Math.min(r, nn - 1), Math.min(c, nn - 1)]);
  };

  return (
    <Bench
      title={tr("One head, every matrix", 'Một head, mọi ma trận')}
      task={
        <Translated vi="Chỉ vào một ô của P. Rồi lần lượt tắt mặt nạ, hệ số co giãn và phép trừ giá trị lớn nhất, mỗi lần một thứ.">
          Point at a cell of P. Then switch the mask, the scale and the max subtraction off, one at a time.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Q và K chứa các giá trị từ {'{−1, −0.5, 0, 0.5, 1}'} và V các số nguyên từ −3 đến 3, nên bạn có thể nhẩm một tích vô
              hướng. Mọi phép tính đều là float32. Di chuột hoặc chạm vào một ô của S hoặc P.
            </>
          }
        >
          <>
            Q and K hold values from {'{−1, −0.5, 0, 0.5, 1}'} and V integers from −3 to 3, so you can check a dot product
            in your head. All arithmetic is float32. Hover or tap a cell of S or P.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field" style={{ minWidth: '8rem' }}>
          <Translated vi="số token N: ">tokens N: </Translated><b>{n}</b>
          <input type="range" min={2} max={8} value={n} onChange={(e) => setSize(+e.target.value, d)} aria-label="N" />
        </label>
        <label className="field" style={{ minWidth: '8rem' }}>
          <Translated vi="độ rộng head d: ">head width d: </Translated><b>{d}</b>
          <input type="range" min={2} max={6} value={d} onChange={(e) => setSize(n, +e.target.value)} aria-label="d" />
        </label>
        <button className="quiet" onClick={() => setSeed((s) => s + 1)}>
          <Translated vi="Số mới">New numbers</Translated>
        </button>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="check">
          <input type="checkbox" checked={scale} onChange={(e) => setScale(e.target.checked)} /> <Translated vi="co giãn theo 1/√d">scale by 1/√d</Translated>
        </label>
        <label className="check">
          <input type="checkbox" checked={mask} onChange={(e) => setMask(e.target.checked)} /> <Translated vi="mặt nạ nhân quả">causal mask</Translated>
        </label>
        <label className="check">
          <input type="checkbox" checked={subtract} onChange={(e) => setSubtract(e.target.checked)} /> <Translated vi="trừ giá trị lớn nhất của hàng">subtract the row max</Translated>
        </label>
        <button aria-pressed={offset === 90} onClick={() => setOffset(offset === 90 ? 0 : 90)}>
          {offset === 90 ? tr('Logits offset by +90', 'Logits đã cộng +90') : tr('Offset every logit by +90', 'Cộng +90 vào mọi logit')}
        </button>
      </div>

      <div className="narrate" role="status" data-narrate style={{ marginBottom: '0.9rem' }}>
        <Translated
          vi={
            <>
              {vis ? (
                <>
                  P[{pi}][{pj}]: q<sub>{pi}</sub> · k<sub>{pj}</sub> ={' '}
                  {dot.terms.map((t, k) => (
                    <span key={k}>
                      {k ? ' + ' : ''}({num(t.q, 1)})({num(t.k, 1)})
                    </span>
                  ))}{' '}
                  = {num(dot.value, 2)}
                  {scale ? <>; × 1/√{d} = {num(res.S[pi][pj] * res.scale, 3)}</> : null}
                  {offset ? <>; + 90 = {num(res.logits[pi][pj], 3)}</> : null}. Giá trị lớn nhất của hàng {pi} là {num(res.m[pi], 3)}
                  {subtract ? (
                    <>
                      , nên tử số là exp({num(res.logits[pi][pj], 3)} − {num(res.m[pi], 3)}) = {num(res.E[pi][pj], 4)}
                    </>
                  ) : (
                    <>
                      , không bị trừ, nên tử số là exp({num(res.logits[pi][pj], 3)}) = {num(res.E[pi][pj], 4)}
                      {res.E[pi][pj] === Infinity ? <> (vượt quá float32 khi lớn hơn {num(EXP_LIMIT_F32, 2)})</> : null}
                    </>
                  )}
                  ; tổng của hàng là {num(res.sum[pi], 4)}, và p = <b data-p>{num(res.P[pi][pj], 4)}</b>.
                </>
              ) : (
                <>
                  P[{pi}][{pj}] bị che: cột {pj} là một token đứng sau token {pi}, M cộng −∞ vào logit của nó, và exp(−∞) đúng bằng{' '}
                  <b data-p>0</b>. Hàng {pi} có {live} phần tử khác 0.
                </>
              )}
              {nanCount ? (
                <>
                  <br />
                  <b className="t-fail" data-nan={nanCount}>
                    {nanCount} phần tử của P là NaN: tử số của chúng tràn thành vô cực, và các tổng cũng vậy.
                  </b>
                </>
              ) : null}
              {leakRows.size ? (
                <>
                  <br />
                  <b className="t-fail" data-leak={[...leakRows].join(',')}>
                    Không có mặt nạ, các hàng {[...leakRows].join(', ')} của O trộn vào giá trị của các token đến sau: tương lai rò rỉ
                    vào quá khứ.
                  </b>
                </>
              ) : null}
            </>
          }
        >
          {vis ? (
            <>
              P[{pi}][{pj}]: q<sub>{pi}</sub> · k<sub>{pj}</sub> ={' '}
              {dot.terms.map((t, k) => (
                <span key={k}>
                  {k ? ' + ' : ''}({num(t.q, 1)})({num(t.k, 1)})
                </span>
              ))}{' '}
              = {num(dot.value, 2)}
              {scale ? <>; × 1/√{d} = {num(res.S[pi][pj] * res.scale, 3)}</> : null}
              {offset ? <>; + 90 = {num(res.logits[pi][pj], 3)}</> : null}. Row {pi}&rsquo;s max is {num(res.m[pi], 3)}
              {subtract ? (
                <>
                  , so the numerator is exp({num(res.logits[pi][pj], 3)} − {num(res.m[pi], 3)}) = {num(res.E[pi][pj], 4)}
                </>
              ) : (
                <>
                  , not subtracted, so the numerator is exp({num(res.logits[pi][pj], 3)}) = {num(res.E[pi][pj], 4)}
                  {res.E[pi][pj] === Infinity ? <> (beyond float32 above {num(EXP_LIMIT_F32, 2)})</> : null}
                </>
              )}
              ; the row&rsquo;s sum is {num(res.sum[pi], 4)}, and p = <b data-p>{num(res.P[pi][pj], 4)}</b>.
            </>
          ) : (
            <>
              P[{pi}][{pj}] is masked: column {pj} is a token after token {pi}, M adds −∞ to its logit, and exp(−∞) is exactly{' '}
              <b data-p>0</b>. Row {pi} has {live} non-zero entries.
            </>
          )}
          {nanCount ? (
            <>
              <br />
              <b className="t-fail" data-nan={nanCount}>
                {nanCount} entries of P are NaN: their numerators overflowed to infinity, and so did the sums.
              </b>
            </>
          ) : null}
          {leakRows.size ? (
            <>
              <br />
              <b className="t-fail" data-leak={[...leakRows].join(',')}>
                Without the mask, rows {[...leakRows].join(', ')} of O mix in values of later tokens: the future leaks into the
                past.
              </b>
            </>
          ) : null}
        </Translated>
      </div>

      <div className="row" style={{ alignItems: 'flex-start', gap: '1.2rem', marginBottom: '1rem' }}>
        <Grid
          name="Q"
          label={`Q (${n} × ${d})`}
          rows={n}
          cols={d}
          value={(r, c) => num(prob.Q[r][c], 1)}
          style={(r) => (r === pi ? ON : undefined)}
        />
        <Grid
          name="KT"
          label={`K^T (${d} × ${n})`}
          rows={d}
          cols={n}
          value={(r, c) => num(prob.K[c][r], 1)}
          style={(_, c) => (c === pj ? ON : undefined)}
        />
      </div>
      <div className="row" style={{ alignItems: 'flex-start', gap: '1.2rem', marginBottom: '1rem' }}>
        <Grid
          name="S"
          label={scale ? 'S / √d' : tr('S (unscaled)', 'S (chưa co giãn)')}
          rows={n}
          cols={n}
          value={(r, c) => num(res.S[r][c] * res.scale, 2)}
          cls={(r, c) => (mask && !visible(r, c, n, n) ? 'void' : '')}
          style={(r, c) => (r === pi && c === pj ? HOT : undefined)}
          onPick={(r, c) => setPick([r, c])}
        />
        <Grid
          name="M"
          label="M"
          rows={n}
          cols={n}
          value={(r, c) => (mask && !visible(r, c, n, n) ? '−∞' : '0')}
          cls={(r, c) => (mask && !visible(r, c, n, n) ? 'void' : '')}
        />
      </div>
      <div className="row" style={{ alignItems: 'flex-start', gap: '1.2rem' }}>
        <Grid
          name="P"
          label="P = softmax(S/√d + M)"
          rows={n}
          cols={n}
          value={(r, c) => num(res.P[r][c], 2)}
          cls={(r, c) => (Number.isNaN(res.P[r][c]) ? 'bad' : res.P[r][c] === 0 ? 'void' : '')}
          style={(r, c) => (r === pi && c === pj ? HOT : r === pi ? ON : undefined)}
          onPick={(r, c) => setPick([r, c])}
        />
        <Grid
          name="V"
          label={`V (${n} × ${d})`}
          rows={n}
          cols={d}
          value={(r, c) => num(prob.V[r][c], 0)}
          style={(r) => (res.P[pi][r] !== 0 ? ON : undefined)}
        />
        <Grid
          name="O"
          label={`O = P V (${n} × ${d})`}
          rows={n}
          cols={d}
          value={(r, c) => num(res.O[r][c], 2)}
          cls={(r, c) => (Number.isNaN(res.O[r][c]) || leaks.some(([a, b]) => a === r && b === c) ? 'bad' : '')}
          style={(r) => (r === pi ? ON : undefined)}
        />
      </div>
      <p className="muted" style={{ margin: '0.8rem 0 0' }} data-rowinfo>
        {rowNaN
          ? tr(`Row ${pi} of P has no usable entries: it overflowed. Row ${pi} of O is NaN with it.`, `Hàng ${pi} của P không có phần tử nào dùng được: nó đã tràn số. Hàng ${pi} của O cũng thành NaN theo.`)
          : tr(`Row ${pi} of P has ${live} non-zero entries; row ${pi} of O is the weighted mean of the highlighted rows of V.`, `Hàng ${pi} của P có ${live} phần tử khác 0; hàng ${pi} của O là trung bình có trọng số của các hàng được tô sáng của V.`)}
      </p>
    </Bench>
  );
}

export function S202() {
  return (
    <>
      <Translated
        vi={
          <p>
            Attention của một lớp là một tập các head không trao đổi với nhau. Mỗi head có trọng số riêng và tính đầu ra riêng từ
            cùng một đầu vào, và các đầu ra chỉ được nối lại ở cuối. Với GPU, điều đó có nghĩa các head là thêm một chiều của
            lưới, nên sách thiết kế một head duy nhất và phần còn lại của chương cũng vậy. Trong huấn luyện, các head tách dần ra
            và chuyên môn hóa (có head theo dõi khoảng cách, có head theo cú pháp, có head theo chủ đề), và đó là lý do có nhiều
            head.
          </p>
        }
      >
        <p>
          A layer&rsquo;s attention is a set of heads that do not talk to each other. Each has its own weights and
          computes its own output from the same input, and the outputs are only concatenated at the end. For a GPU that
          means the heads are one more grid dimension, so the book designs a single head and the rest of the chapter does
          too. In training the heads drift apart and specialise (some track distance, some syntax, some topic), which is
          why there are many of them.
        </p>
      </Translated>
      <Source section="20.2" pages="482" />
      <Translated
        vi={
          <p>
            Các lab prefill (20.1 và 20.2) đánh chỉ số head theo cách lời gọi attention của PyTorch làm: batch trước, rồi head,
            rồi token, rồi độ rộng head, <code>[batch, heads, N, d]</code>. Khi đó một head của một chuỗi là một ma trận N × d
            liền khối. Các lab decode (20.4 và 20.5) thì giữ cache theo dạng{' '}
            <code>[batch, max_seq, h_kv, d]</code>, token trước head, đó là cách bố trí tham chiếu của Meta (mục{' '}
            <a href={href('ch20', '20.4')}>20.4</a>).
          </p>
        }
      >
        <p>
          The prefill labs (20.1 and 20.2) index heads the way PyTorch&rsquo;s attention call does: batch first, then
          heads, then tokens, then the head width, <code>[batch, heads, N, d]</code>. One head of one sequence is then a
          contiguous N × d matrix. The decode labs (20.4 and 20.5) keep the cache as{' '}
          <code>[batch, max_seq, h_kv, d]</code> instead, tokens before heads, which is Meta&rsquo;s reference layout
          (section <a href={href('ch20', '20.4')}>20.4</a>).
        </p>
      </Translated>

      <h2><Translated vi="Mọi ma trận của một head">Every matrix of one head</Translated></h2>
      <Translated
        vi={
          <p>
            X có kích thước N × d: mỗi token cho đến giờ một hàng, dù nó đến từ system prompt, đầu vào của bạn, tài liệu được truy
            xuất hay đầu ra của chính mô hình. N là độ dài chuỗi, còn gọi là độ dài ngữ cảnh, và nó tăng theo cuộc hội thoại. Ba
            ma trận d × d đã học chiếu X: Q = X W<sub>Q</sub>, K = X W<sub>K</sub>, V = X W
            <sub>V</sub>, mỗi ma trận N × d. (Sách vẽ chúng vuông cho đơn giản; tổng quát chúng là d<sub>x</sub> ×
            d<sub>q</sub> và cứ thế, và Q K<sup>T</sup> buộc d<sub>q</sub> = d<sub>k</sub>.)
          </p>
        }
      >
        <p>
          X is N × d: one row per token so far, whether it came from the system prompt, your input, retrieved documents or
          the model&rsquo;s own output. N is the sequence length, also called the context length, and it grows as the
          conversation does. Three learned d × d matrices project X: Q = X W<sub>Q</sub>, K = X W<sub>K</sub>, V = X W
          <sub>V</sub>, each N × d. (The book draws them square for simplicity; in general they are d<sub>x</sub> ×
          d<sub>q</sub> and so on, and Q K<sup>T</sup> forces d<sub>q</sub> = d<sub>k</sub>.)
        </p>
      </Translated>
      <Source section="20.2" pages="482–483" figs={tr("Fig. 20.2", 'Hình 20.2')} />
      <Translated
        vi={
          <p>
            Rồi đến Eq. 20.1: O = softmax(Q K<sup>T</sup> / √d + M) V. Phần tử (i, j) của S = Q K<sup>T</sup> là tích vô hướng
            của query của token i với key của token j, mà sách hiểu là độ tương đồng của chúng. S có kích thước N × N. M cộng −∞
            vào mọi phần tử nằm trên đường chéo, nên không token nào nhìn được token đến sau. P, softmax theo hàng, có kích thước
            N × N, và O = P V lại là N × d, sẵn sàng cho Add &amp; Norm. Như trước, d ở đây là độ rộng head.
          </p>
        }
      >
        <p>
          Then Eq. 20.1: O = softmax(Q K<sup>T</sup> / √d + M) V. Entry (i, j) of S = Q K<sup>T</sup> is the inner product
          of token i&rsquo;s query with token j&rsquo;s key, which the book reads as their similarity. S is N × N. M adds −∞
          to every entry above the diagonal, so no token can look at a later one. P, the row-wise softmax, is N × N, and
          O = P V is N × d again, ready for the Add &amp; Norm. As before, d here is the head width.
        </p>
      </Translated>
      <Source section="20.2" pages="483–484" />
      <Ask q={questionsS202['ch20/20.2/q-s-bytes']} label={tr("Work it out", 'Tự tính')} />

      <h2><Translated vi="Mặt nạ, hệ số co giãn và giá trị lớn nhất">The mask, the scale and the max</Translated></h2>
      <Translated
        vi={
          <p>
            Ba chi tiết của Eq. 20.1 và 20.2 quan trọng với một kernel, và trình xem cho bạn tắt từng thứ. Hệ số 1/√d giữ cho
            tích vô hướng không lớn lên theo độ rộng head; Hình 20.2 bỏ nó đi, còn mã thì có. Mặt nạ biến P thành ma trận tam giác
            dưới: hàng i của P là một phân phối xác suất trên các token 0 đến i, và hàng i của O là trung bình có trọng số của các
            hàng 0 đến i của V. Và softmax trừ đi giá trị lớn nhất m<sub>r</sub> của mỗi hàng trước khi lấy mũ (Eq. 20.2).
          </p>
        }
      >
        <p>
          Three details of Eq. 20.1 and 20.2 matter for a kernel, and the viewer lets you switch each one off. The
          factor 1/√d keeps dot products from growing with the head width; Fig. 20.2 leaves it out, the code has it.
          The mask turns P lower-triangular: row i of P is a probability distribution over tokens 0 to i, and row i of O
          is a weighted mean of rows 0 to i of V. And the softmax subtracts each row&rsquo;s maximum m<sub>r</sub> before
          exponentiating (Eq. 20.2).
        </p>
      </Translated>
      <Source section="20.2" pages="484–485" />
      <Translated
        vi={
          <p>
            Phép trừ đó nhân các tử số và tổng của một hàng với cùng một hệ số e<sup>−m<sub>r</sub></sup>, nên P không thể thay
            đổi. Cái thay đổi là miền giá trị: mọi số mũ nhiều nhất là 0, nên không gì tràn số, và các logit bị che cho
            e<sup>−∞</sup> = 0 chính xác. Hãy dự đoán trước khi thử: khi tắt phép trừ và cộng 90 vào mọi logit, hàng 0 của P
            hiện ra gì?
          </p>
        }
      >
        <p>
          That subtraction multiplies the numerators and the sum of a row by the same factor e<sup>−m<sub>r</sub></sup>,
          so P cannot change. What changes is the range: every exponent is at most 0, so nothing overflows, and masked
          logits give e<sup>−∞</sup> = 0 exactly. Predict before you try it: with the subtraction off and every logit
          raised by 90, what does row 0 of P show?
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Vài token và một head. Float32 được giả lập bằng cách làm tròn mỗi phép toán một lần (Math.fround); hàm exp của GPU
              làm tròn hơi khác, nên các chữ số cuối có thể khác.
            </p>
          }
        >
          <p>
            A few tokens and one head. Float32 is emulated by rounding every operation once (Math.fround); a GPU&rsquo;s
            own exp rounds slightly differently, so last digits can differ.
          </p>
        </Translated>
      </Simplification>
      <Viewer />
      <Ask q={questionsS202['ch20/20.2/q-row-nonzeros']} />
      <Ask q={questionsS202['ch20/20.2/q-no-scale']} />
      <Ask q={questionsS202['ch20/20.2/q-shift']} />

      <h2><Translated vi="Hai giai đoạn">Two phases</Translated></h2>
      <Translated
        vi={
          <p>
            Hình 20.3 chạy prompt đồ chơi &ldquo;The title of this book is Programming&rdquo;, bảy token là các từ. Giai đoạn{' '}
            <b>tóm tắt</b> (summarization; prefill, như mục 20.4 đổi tên) đẩy cả bảy hàng qua mọi lớp dưới dạng ma trận đầy đủ và
            phát ra &ldquo;Massively&rdquo;. Rồi giai đoạn <b>sinh</b> (generation) bắt đầu: token mới được embedding, nối vào X
            thành thêm một hàng, và mọi lớp chạy lại, phát ra &ldquo;Parallel&rdquo;, rồi &ldquo;Processors&rdquo;, rồi token
            kết thúc chuỗi.
          </p>
        }
      >
        <p>
          Fig. 20.3 runs the toy prompt &ldquo;The title of this book is Programming&rdquo;, seven word-tokens. The{' '}
          <b>summarization</b> phase (prefill, as section 20.4 renames it) pushes all seven rows through every layer as
          full matrices and emits &ldquo;Massively&rdquo;. Then the <b>generation</b> phase begins: the new token is
          embedded, appended to X as one more row, and every layer runs again, emitting &ldquo;Parallel&rdquo;, then
          &ldquo;Processors&rdquo;, then the end-of-sequence token.
        </p>
      </Translated>
      <Source section="20.2" pages="485–486" figs={tr("Fig. 20.3", 'Hình 20.3')} />
      <Translated
        vi={
          <p>
            Trong cách làm cơ sở này, mỗi vòng tính lại K, V, Q, S, P và O cho mọi hàng, dù mọi hàng trừ hàng cuối đều giống lần
            trước. Mục <a href={href('ch20', '20.4')}>20.4</a> loại bỏ sự lãng phí đó; mục{' '}
            <a href={href('ch20', '20.3')}>20.3</a> trước hết viết softmax thành một kernel.
          </p>
        }
      >
        <p>
          In this baseline each iteration recomputes K, V, Q, S, P and O for every row, although all rows but the last are
          the same as last time. Section <a href={href('ch20', '20.4')}>20.4</a> removes that waste; section{' '}
          <a href={href('ch20', '20.3')}>20.3</a> first writes the softmax as a kernel.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>Giai đoạn sinh còn được gọi là giai đoạn tự hồi quy (autoregression) hoặc decode (p. 485).</p>
          }
        >
          <p>Generation is also called the autoregression or decoding phase (p. 485).</p>
        </Translated>
      </Aside>
      <Ask q={questionsS202['ch20/20.2/q-rows-x']} label={tr("Count", 'Đếm')} />

      <h2><Translated vi="Điều này là gì trong PyTorch">What this is in PyTorch</Translated></h2>
      <Translated
        vi={
          <p>
            PyTorch ghi trong tài liệu ngữ nghĩa tham chiếu của lời gọi attention đúng bằng các bước này: một mặt nạ boolean tam
            giác dưới, <code>masked_fill</code> với âm vô cực, tích có co giãn của query với key chuyển vị, một softmax, và một
            phép nhân với V. Grouped-query attention được biểu diễn bằng cách lặp lại mỗi head K và V cho nhóm head query của nó
            (<code>repeat_interleave</code>), điều mà mục 20.7 quay lại. Đó cũng là thứ mà một tham chiếu CPU cho các lab của
            chương này phải tính.
          </p>
        }
      >
        <p>
          PyTorch documents the reference semantics of its attention call as exactly these steps: a lower-triangular
          boolean mask, <code>masked_fill</code> with minus infinity, the scaled product of the query with the transposed
          key, a softmax, and a product with V. Grouped-query attention is expressed by repeating each K and V head for its
          group of query heads (<code>repeat_interleave</code>), which section 20.7 returns to. It is also what a CPU
          reference for this chapter&rsquo;s labs has to compute.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch20-facts.md D5)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              <code>torch.nn.functional.scaled_dot_product_attention</code>, tài liệu PyTorch 2.15: mã tham chiếu dùng{' '}
              <code>tril</code>, <code>masked_fill_(…, float(&quot;-inf&quot;))</code> và{' '}
              <code>repeat_interleave</code> trên chiều head.
            </p>
          }
        >
          <p>
            <code>torch.nn.functional.scaled_dot_product_attention</code>, PyTorch 2.15 documentation: the reference code
            uses <code>tril</code>, <code>masked_fill_(…, float(&quot;-inf&quot;))</code> and{' '}
            <code>repeat_interleave</code> on the head dimension.
          </p>
        </Translated>
      </Today>
    </>
  );
}
