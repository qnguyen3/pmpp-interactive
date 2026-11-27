import { useEffect, useMemo, useState, type KeyboardEvent } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr as trText } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  BOOK_F,
  BOOK_LAYER,
  BOOK_X,
  LENET,
  batchTraffic,
  changedByFlip,
  fAt,
  forward,
  hOut,
  lessonRegion,
  perInputMap,
  shapes,
  terms,
  wOut,
  type LayerShape,
  type Stage,
} from '../../sim/cnnlayer';

// Section 19.1: the convolutional layer as the book defines it. The worked example of
// Fig. 19.2 as a term-by-term stepper, LeNet-5's three convolution layers with their
// shapes and batch traffic, and the sequential reference from lab 19.1's driver.

const S = BOOK_LAYER;
const LAB_MAIN = 'ch19/lab1_convgemm/main.cpp';
const big = (v: number) => v.toLocaleString('en-US');
const pct = (v: number) => (v >= 0.1 ? (v * 100).toFixed(1) : v >= 0.001 ? (v * 100).toFixed(2) : '<0.1') + '%';
const dims = (d: number[]) => d.join(' × ');
const bytes = (b: number) => (b >= 1 << 20 ? `${(b / (1 << 20)).toFixed(1)} MiB` : b >= 1024 ? `${(b / 1024).toFixed(1)} KiB` : `${b} B`); // i18n-ignore: byte units

// ---------------------------------------------------------------- questions

export const questionsS191: Record<string, Question> = {
  'ch19/19.1/q-flip': {
    kind: 'numeric',
    id: 'ch19/19.1/q-flip',
    prompt: (
      <Translated
        vi={
          <>
            Giả sử một thư viện xoay mọi bộ lọc 180° trước khi trượt nó, đúng như nghĩa của từ tích chập (convolution) trong
            xử lý tín hiệu. Khi đó <code>Y[0, 0, 0]</code> trong ví dụ của sách sẽ thành bao nhiêu?
          </>
        }
      >
        <>
          Suppose a library rotated every filter by 180° before sliding it, which is what the word convolution means in
          signal processing. What would <code>Y[0, 0, 0]</code> of the book&rsquo;s example become?
        </>
      </Translated>
    ),
    answer: 15,
    explain: (
      <Translated
        vi={
          <>
            <p>
              Khi xoay, F[0, 0] = [1 1; 2 2] thành [2 2; 1 1], và nhân với vùng [1 2; 1 1] cho 2 + 4 + 1 + 1 = 8 thay vì 7.
              F[0, 1] (toàn số một) và F[0, 2] ([0 1; 1 0]) trông y hệt sau nửa vòng quay, nên vẫn cho 5 và 2. Tổng là 15.
              Hãy bật công tắc xoay trong trình khám phá để xem những đầu ra nào khác thay đổi.
            </p>
            <p>
              Sách và PyTorch thống nhất với nhau: không bên nào lật. Phép lật làm thay đổi kết quả mỗi khi một bộ lọc không
              đối xứng qua nửa vòng quay, nên trọng số được huấn luyện theo quy ước này sẽ cho kết quả sai theo quy ước kia.
            </p>
          </>
        }
      >
        <>
          <p>
            Rotated, F[0, 0] = [1 1; 2 2] becomes [2 2; 1 1], and against the patch [1 2; 1 1] it gives 2 + 4 + 1 + 1 = 8
            instead of 7. F[0, 1] (all ones) and F[0, 2] ([0 1; 1 0]) look the same after a half turn, so they still give 5
            and 2. Total 15. Tick the rotation switch in the explorer to see which other outputs move.
          </p>
          <p>
            The book and PyTorch agree with each other: neither flips. A flip changes the result whenever a filter is not
            symmetric under a half turn, so weights trained under one convention give wrong answers under the other.
          </p>
        </>
      </Translated>
    ),
    source: 'ch. 19 §19.1, p. 457; notes/today/ch07-facts.md D1 (PyTorch, cuDNN, MIOpen compute cross-correlation); ch19-facts.md C3 (MPSGraph documents neither choice)',
    vi: { source: 'ch. 19 §19.1, p. 457; notes/today/ch07-facts.md D1 (PyTorch, cuDNN, MIOpen tính tương quan chéo); ch19-facts.md C3 (MPSGraph không ghi rõ lựa chọn nào)' },
  },
  'ch19/19.1/q-outsize': {
    kind: 'numeric',
    id: 'ch19/19.1/q-outsize',
    prompt: (
      <Translated
        vi={
          <>
            <code>nn.Conv2d(6, 16, 5)</code>, với mọi đối số khác để mặc định, nhận một đầu vào 14 × 14. Chiều cao đầu ra
            của nó là bao nhiêu?
          </>
        }
      >
        <>
          <code>nn.Conv2d(6, 16, 5)</code> with every other argument left at its default receives a 14 × 14 input. What
          is the height of its output?
        </>
      </Translated>
    ),
    answer: 10,
    explain: (
      <Translated
        vi={
          <p>
            Mặc định <code>padding=0</code> chính là lớp của sách không có ô ma (ghost cell): H − K + 1 = 14 − 5 + 1 = 10.
            Đây là C3 của LeNet. Để giữ 14 × 14, bạn sẽ yêu cầu <code>padding=2</code>, tức các ô ma bằng 0 của chương 7.
          </p>
        }
      >
        <p>
          The default <code>padding=0</code> is the book&rsquo;s layer with no ghost cells: H − K + 1 = 14 − 5 + 1 = 10.
          This is LeNet&rsquo;s C3. To keep 14 × 14 you would ask for <code>padding=2</code>, chapter 7&rsquo;s zero ghost
          cells.
        </p>
      </Translated>
    ),
    source: 'ch. 19 §19.1, p. 457; notes/today/ch07-facts.md D1 (PyTorch 2.15 Conv2d defaults)',
    vi: { source: 'ch. 19 §19.1, p. 457; notes/today/ch07-facts.md D1 (giá trị mặc định của Conv2d trong PyTorch 2.15)' },
  },
  'ch19/19.1/q-weights': {
    kind: 'numeric',
    id: 'ch19/19.1/q-weights',
    prompt: (
      <Translated
        vi={
          <>
            C3 của LeNet biến 6 map đầu vào 14 × 14 thành 16 map đầu ra 10 × 10 với các bộ lọc 5 × 5. F chứa bao nhiêu
            trọng số?
          </>
        }
      >
        <>
          LeNet&rsquo;s C3 maps 6 input maps of 14 × 14 to 16 output maps of 10 × 10 with 5 × 5 filters. How many weights
          does F hold?
        </>
      </Translated>
    ),
    answer: 2400,
    unit: 'weights',
    vi: { unit: 'trọng số' },
    explain: (
      <Translated
        vi={
          <p>
            Mỗi cặp (map đầu ra, map đầu vào) một bộ lọc: 16 · 6 = 96 bộ lọc, mỗi bộ 5 · 5 = 25 trọng số, vậy là 2400. Kích
            thước map không xuất hiện: cùng các trọng số đó trượt qua mọi vị trí. Kích thước của F là 16 × 6 × 5 × 5.
          </p>
        }
      >
        <p>
          One filter per (output map, input map) pair: 16 · 6 = 96 filters of 5 · 5 = 25 weights, so 2400. The map size
          does not appear: the same weights slide over every position. F&rsquo;s shape is 16 × 6 × 5 × 5.
        </p>
      </Translated>
    ),
    source: 'ch. 19 §19.1, pp. 456, 458', // i18n-ignore: citation
  },
  'ch19/19.1/q-batch': {
    kind: 'choice',
    id: 'ch19/19.1/q-batch',
    prompt: (
      <Translated
        vi={
          <>
            Trong lớp có batch, X và Y có thêm chiều N còn F thì không. Khi N tăng, lưu lượng cho F <em>trên mỗi mẫu</em>{' '}
            có thể thay đổi thế nào?
          </>
        }
      >
        <>
          In the batched layer, X and Y gain an N dimension and F does not. As N grows, what can happen to the traffic
          for F <em>per sample</em>?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Nó có thể giảm theo 1/N, vì một lần nạp F có thể phục vụ mọi mẫu.">
            It can fall as 1/N, because one fetch of F can serve every sample.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không thay đổi gì: mỗi mẫu cần tất cả trọng số, nên mỗi mẫu tốn cùng một lưu lượng F.">
            Nothing: every sample needs all the weights, so each sample costs the same F traffic.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi mẫu đúng là cần mọi trọng số. Nhưng nó không cần bản sao riêng nạp từ bộ nhớ: một kernel giữ một trọng số trên chip trong khi phục vụ nhiều mẫu chỉ trả giá cho trọng số đó một lần.">
            Every sample does need every weight. It does not need its own copy fetched from memory: a kernel that keeps a weight on chip while it serves several samples pays for that weight once.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó tăng, vì một lưới lớn hơn có nhiều khối nạp F hơn.">
            It grows, because a larger grid has more blocks fetching F.
          </Translated>
        ),
        why: (
          <Translated vi="Nhiều khối hơn có thể nạp nhiều hơn, nhưng không gì bắt chúng phải làm vậy; lý tưởng vẫn là một lần nạp cho cả batch.">
            More blocks can fetch more, but nothing forces them to; the ideal is still one fetch for the whole batch.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="F phải được sao chép một lần cho mỗi mẫu trước khi lớp chạy.">
            F must be copied once per sample before the layer runs.
          </Translated>
        ),
        why: (
          <Translated vi="F được dùng chung, không nhân bản (p. 459): chỉ có một mảng cho cả batch.">
            F is shared, not replicated (p. 459): there is one array for the whole batch.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Các trọng số được dùng chung trong cả batch (p. 459), nên lưu lượng lý tưởng cho F cố định trong khi khối lượng
            công việc tăng theo N. Trên mỗi mẫu, nó giảm theo 1/N, và tỉ lệ phép toán trên byte của lớp tăng lên. Trình khám
            phá LeNet ở trên cho thấy điều đó: ở batch 1, trọng số của C5 chiếm 99% lưu lượng của nó; ở 256, khoảng một
            phần tư. Đó cũng là lý do batching làm tăng tỉ lệ của một lớp tuyến tính ở mục 5.1.
          </p>
        }
      >
        <p>
          The weights are shared across the batch (p. 459), so the ideal traffic for F is fixed while the work grows with
          N. Per sample it falls as 1/N, and the layer&rsquo;s ratio of operations to bytes rises. The LeNet explorer above
          shows it: at batch 1, C5&rsquo;s weights are 99% of its traffic; at 256, about a quarter. It is the same reason
          batching raises a linear layer&rsquo;s ratio in section 5.1.
        </p>
      </Translated>
    ),
    source: 'ch. 19 §19.1, p. 459; ch. 5 §5.1 (the linear-layer explorer)',
    vi: { source: 'ch. 19 §19.1, p. 459; ch. 5 §5.1 (trình khám phá lớp tuyến tính)' },
  },
};

// ---------------------------------------------------------------- one output, term by term

const cellKeys = (fn: () => void) => (e: KeyboardEvent) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fn();
  }
};

function OneOutput() {
  const [sel, setSel] = useState({ m: 0, h: 0, w: 0 });
  const [k, setK] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [flip, setFlip] = useState(false);

  const Ho = hOut(S);
  const Wo = wOut(S);
  const tr = useMemo(() => terms(S, BOOK_X, BOOK_F, sel.m, sel.h, sel.w, flip), [sel, flip]);
  const Y = useMemo(() => forward(S, BOOK_X, BOOK_F, flip), [flip]);
  const Yplain = useMemo(() => forward(S, BOOK_X, BOOK_F), []);
  const changed = useMemo(() => new Set(changedByFlip(S, BOOK_X, BOOK_F)), []);
  const parts = perInputMap(S, BOOK_X, BOOK_F, sel.m, sel.h, sel.w, flip);
  const total = tr.length;
  const at = Math.min(k, total);
  const cur = at > 0 ? tr[at - 1] : undefined;
  const sum = cur ? cur.sum : 0;

  useEffect(() => setK(0), [sel, flip]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setK((x) => x + 1), 520);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);

  // Patch cells of the selected output, and the filters it uses.
  const inPatch = (i: number) => {
    const h = Math.floor(i / S.W);
    const w = i % S.W;
    return h >= sel.h && h < sel.h + S.K && w >= sel.w && w < sel.w + S.K;
  };
  // A filter as displayed: rotated when the switch is on, so position (p, q) on screen is what term (p, q) uses.
  const shown = (m: number, c: number) =>
    Array.from({ length: S.K * S.K }, (_, i) => {
      const p = Math.floor(i / S.K);
      const q = i % S.K;
      return flip ? BOOK_F[fAt(S, m, c, S.K - 1 - p, S.K - 1 - q)] : BOOK_F[fAt(S, m, c, p, q)];
    });
  const subtotals = parts.map((v, c) => (cur && cur.c > c ? v : cur && cur.c === c ? tr.filter((t) => t.c === c && t.k < at).reduce((a, t) => a + t.product, 0) : null));

  const say = cur
    ? trText(
        `c = ${cur.c}, p = ${cur.p}, q = ${cur.q}: X[${cur.c}, ${sel.h + cur.p}, ${sel.w + cur.q}] × F[${sel.m}, ${cur.c}, ${flip ? `${S.K - 1 - cur.p}, ${S.K - 1 - cur.q}` : `${cur.p}, ${cur.q}`}] = ${cur.x} × ${cur.f} = ${cur.product}. Running sum ${cur.sum}.`,
        `c = ${cur.c}, p = ${cur.p}, q = ${cur.q}: X[${cur.c}, ${sel.h + cur.p}, ${sel.w + cur.q}] × F[${sel.m}, ${cur.c}, ${flip ? `${S.K - 1 - cur.p}, ${S.K - 1 - cur.q}` : `${cur.p}, ${cur.q}`}] = ${cur.x} × ${cur.f} = ${cur.product}. Tổng tích lũy ${cur.sum}.`,
      ) +
      (at === total
        ? trText(` That is Y[${sel.m}, ${sel.h}, ${sel.w}] = ${parts.join(' + ')} = ${sum}.`, ` Đó là Y[${sel.m}, ${sel.h}, ${sel.w}] = ${parts.join(' + ')} = ${sum}.`)
        : '')
    : trText(
        `Y[${sel.m}, ${sel.h}, ${sel.w}] adds ${total} products: ${S.K * S.K} from each of the ${S.C} input maps, each through the filter F[${sel.m}, c]. Press Step.`,
        `Y[${sel.m}, ${sel.h}, ${sel.w}] cộng ${total} tích: ${S.K * S.K} tích từ mỗi map trong ${S.C} map đầu vào, mỗi map qua bộ lọc F[${sel.m}, c]. Nhấn Bước.`,
      );

  return (
    <Bench
      title={trText('One output, term by term', 'Một đầu ra, từng số hạng')}
      task={trText('Click any output, then step through the loops over c, p and q.', 'Nhấn vào một đầu ra bất kỳ, rồi đi từng bước qua các vòng lặp trên c, p và q.')}
      foot={
        <Translated
          vi={
            <>
              Mọi ô ở đây đều nằm trong bộ nhớ toàn cục (global memory); tổng tích lũy là giá trị duy nhất mà một luồng
              (thread) sẽ giữ trong thanh ghi (register). Các con số là của sách (Fig. 19.2), được tính bằng cùng tổ vòng lặp
              như Fig. 19.3.
            </>
          }
        >
          <>
            Every cell here is in global memory; the running sum is the one value a thread would keep in a register. The
            numbers are the book&rsquo;s (Fig. 19.2), computed by the same loop nest as Fig. 19.3.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setK(0); }} disabled={at === 0}><Translated vi="Làm lại">Restart</Translated></button>
        <button onClick={() => { setPlaying(false); setK(at - 1); }} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => { setPlaying(false); setK(at + 1); }} disabled={at >= total}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? trText('Pause', 'Tạm dừng') : trText('Play', 'Chạy')}</button>
        <button onClick={() => { setPlaying(false); setK(total); }} disabled={at >= total}><Translated vi="Đến cuối">Finish</Translated></button>
        <span className="muted" aria-live="polite"><Translated vi={<>số hạng {at} trên {total}</>}>term {at} of {total}</Translated></span>
        <label className="check">
          <input type="checkbox" checked={flip} onChange={(e) => setFlip(e.target.checked)} />{' '}
          <Translated vi="xoay bộ lọc 180° (tích chập thực sự; không có trong sách)">
            rotate filters 180° (true
            convolution; not in the book)
          </Translated>
        </label>
      </div>
      <div className="narrate" role="status" data-out="term" data-k={at} data-sum={sum} style={{ marginBottom: '0.9rem' }}>
        {say}
      </div>
      <div className="row" style={{ alignItems: 'flex-start', gap: '1.6rem' }}>
        <div>
          <div className="mx-label t-global"><Translated vi="Map đầu vào X (C = 3, mỗi map 3 × 3)">Input maps X (C = 3, each 3 × 3)</Translated></div>
          <div className="row" style={{ gap: '0.7rem', alignItems: 'flex-start' }}>
            {Array.from({ length: S.C }, (_, c) => (
              <div key={c}>
                <div className="muted" style={{ fontSize: '0.74rem' }}>X{c}{subtotals[c] != null ? <Translated vi={<> · cho <b data-part={c}>{subtotals[c]}</b></>}><> · gives <b data-part={c}>{subtotals[c]}</b></></Translated> : null}</div>
                <div className="mx global" data-grid={`X${c}`} style={{ gridTemplateColumns: `repeat(${S.W}, auto)` }}>
                  {Array.from({ length: S.H * S.W }, (_, i) => {
                    const xi = c * S.H * S.W + i;
                    const hot = cur?.xi === xi;
                    return <span key={i} className={hot ? 'hot' : inPatch(i) ? 'on' : ''}>{BOOK_X[xi]}</span>;
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
        <div>
          <div className="mx-label t-global"><Translated vi="Bộ lọc">Filters</Translated> F[m, c] ({flip ? trText('rotated 180°', 'đã xoay 180°') : trText('as stored', 'như khi lưu')})</div>
          {Array.from({ length: S.M }, (_, m) => (
            <div key={m} className="row" style={{ gap: '0.7rem', marginBottom: '0.4rem', alignItems: 'flex-end' }}>
              <span className="muted" style={{ fontSize: '0.74rem', width: '3.6rem' }}><Translated vi={<>cho Y{m}</>}>for Y{m}</Translated></span>
              {Array.from({ length: S.C }, (_, c) => (
                <div key={c} className="mx global" data-grid={`F${m}${c}`} style={{ gridTemplateColumns: `repeat(${S.K}, auto)` }}>
                  {shown(m, c).map((v, i) => {
                    const hot = cur && m === sel.m && c === cur.c && i === cur.p * S.K + cur.q;
                    return <span key={i} className={hot ? 'hot' : m === sel.m ? 'on' : ''}>{v}</span>;
                  })}
                </div>
              ))}
            </div>
          ))}
        </div>
        <div>
          <div className="mx-label t-global"><Translated vi="Map đầu ra Y (nhấn chọn một)">Output maps Y (click one)</Translated></div>
          <div className="row" style={{ gap: '0.7rem', alignItems: 'flex-start' }}>
            {Array.from({ length: S.M }, (_, m) => (
              <div key={m}>
                <div className="muted" style={{ fontSize: '0.74rem' }}>Y{m}</div>
                <div className="mx global" data-grid={`Y${m}`} style={{ gridTemplateColumns: `repeat(${Wo}, auto)` }}>
                  {Array.from({ length: Ho * Wo }, (_, i) => {
                    const h = Math.floor(i / Wo);
                    const w = i % Wo;
                    const yi = m * Ho * Wo + i;
                    const isSel = sel.m === m && sel.h === h && sel.w === w;
                    const moved = flip && changed.has(yi);
                    const pick = () => setSel({ m, h, w });
                    return (
                      <span
                        key={i}
                        role="button"
                        tabIndex={0}
                        aria-pressed={isSel}
                        aria-label={`Y[${m}, ${h}, ${w}]`}
                        data-y={`${m},${h},${w}`}
                        data-value={Y[yi]}
                        data-changed={moved ? 'yes' : 'no'}
                        className={isSel ? 'hot' : ''}
                        title={moved ? trText(`the book's formula gives ${Yplain[yi]}`, `công thức của sách cho ${Yplain[yi]}`) : undefined}
                        style={{ cursor: 'pointer', outline: moved ? '2px solid var(--ink)' : undefined, outlineOffset: -3 }}
                        onClick={pick}
                        onKeyDown={cellKeys(pick)}
                      >
                        {Y[yi]}
                      </span>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
          <p style={{ margin: '0.8rem 0 0' }}>
            <span className="muted"><Translated vi="tổng trong thanh ghi ">register sum </Translated></span>
            <b className="t-reg" style={{ fontSize: '1.3rem' }}>{sum}</b>
          </p>
          {flip ? (
            <p className="muted" style={{ margin: '0.4rem 0 0', maxWidth: '12rem' }} data-out="changes">
              <Translated
                vi={
                  <>
                    Có viền: {changed.size} trên {Y.length} đầu ra khác với của sách (
                    {[...changed].map((i) => `${Yplain[i]} → ${Y[i]}`).join(', ')}).
                  </>
                }
              >
                Outlined: {changed.size} of {Y.length} outputs differ from the book&rsquo;s (
                {[...changed].map((i) => `${Yplain[i]} → ${Y[i]}`).join(', ')}).
              </Translated>
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- LeNet-5

const STAGE_TEXT: Record<string, string> = {
  get INPUT() {
    return trText('A 32 × 32 grey image: one input map.', 'Một ảnh xám 32 × 32: một map đầu vào.');
  },
  get S2() {
    return trText(
      'Subsampling: each output pixel summarises a 2 × 2 neighbourhood of one map, so the maps halve in size. Named in the book, described in Appendix B; not computed here.',
      'Lấy mẫu con: mỗi pixel đầu ra tóm tắt một lân cận 2 × 2 của một map, nên các map giảm một nửa kích thước. Được nêu tên trong sách, mô tả ở Phụ lục B; không tính ở đây.',
    );
  },
  get S4() {
    return trText(
      'Subsampling again: sixteen 10 × 10 maps become sixteen 5 × 5 maps. Not computed here.',
      'Lại lấy mẫu con: mười sáu map 10 × 10 thành mười sáu map 5 × 5. Không tính ở đây.',
    );
  },
  get F6() {
    return trText(
      'Fully connected: 84 outputs, each reading all 120 values from C5 (p. 458). Not computed here.',
      'Kết nối đầy đủ: 84 đầu ra, mỗi đầu ra đọc cả 120 giá trị từ C5 (p. 458). Không tính ở đây.',
    );
  },
  get OUTPUT() {
    return trText(
      'Gaussian connections producing 10 values, one per digit class (p. 459). Not computed here.',
      'Các kết nối Gauss tạo ra 10 giá trị, mỗi giá trị cho một lớp chữ số (p. 459). Không tính ở đây.',
    );
  },
};

function Bar({ t }: { t: ReturnType<typeof batchTraffic> }) {
  const seg = (v: number) => `${(100 * v) / t.total}%`;
  return (
    <div style={{ display: 'flex', height: '0.8rem', width: '100%', minWidth: '6rem', border: '1px solid var(--global)', borderRadius: 2, overflow: 'hidden' }} aria-hidden>
      <div style={{ width: seg(t.xBytes), background: 'var(--global-wash)' }} />
      <div style={{ width: seg(t.fBytes), background: 'var(--global)' }} />
      <div style={{ width: seg(t.yBytes), background: 'var(--global-wash)', borderLeft: '1px solid var(--global)' }} />
    </div>
  );
}

function LeNet() {
  const [id, setId] = useState('C1');
  const [lgN, setLgN] = useState(0);
  const N = 2 ** lgN;
  const stage = LENET.find((s) => s.id === id) as Stage;
  const conv = LENET.filter((s) => s.layer) as (Stage & { layer: LayerShape })[];

  return (
    <Bench
      title={trText("LeNet-5's convolution layers", 'Các lớp tích chập của LeNet-5')}
      task={trText("Compare C1, C3 and C5, then move the batch size and watch F's share of the traffic.", 'So sánh C1, C3 và C5, rồi thay đổi kích thước batch và xem phần lưu lượng của F.')}
      foot={
        <Translated
          vi={
            <>
              Thanh: X, <b className="t-global">F</b>, Y, tính theo byte. Lưu lượng lý tưởng: mỗi phần tử của X, F và Y được
              chuyển đúng một lần cho cả batch, dưới dạng float 4 byte. Kernel thực tế chuyển nhiều hơn; mục 19.2 đếm những gì
              kernel trực tiếp yêu cầu.
            </>
          }
        >
          <>
            Bar: X, <b className="t-global">F</b>, Y, in bytes. Ideal traffic: each element of X, F and Y is moved exactly
            once for the whole batch, as 4-byte floats. Real kernels move more; section 19.2 counts what the direct kernel
            asks for.
          </>
        </Translated>
      }
    >
      <div className="seg" role="group" aria-label={trText('LeNet-5 stages', 'Các tầng của LeNet-5')} style={{ flexWrap: 'wrap', rowGap: 4, marginBottom: '0.9rem' }}>
        {LENET.map((s) => (
          <button key={s.id} aria-pressed={s.id === id} onClick={() => setId(s.id)}>
            {s.id}
            <span style={{ fontWeight: 400, opacity: 0.72 }}> {s.out}</span>
          </button>
        ))}
      </div>
      <div className="cols">
        <div className="panel" data-out="stage" data-id={id}>
          {stage.layer ? (
            (() => {
              const l = stage.layer;
              const sh = shapes(l, N);
              const t = batchTraffic(l, N);
              return (
                <>
                  <p style={{ margin: '0 0 0.5rem' }}>
                    <b>{id}</b>:{' '}
                    <Translated vi={<>tích chập với bộ lọc {l.K} × {l.K}, không có ô ma.</>}>
                      convolution with {l.K} × {l.K} filters, no ghost cells.
                    </Translated>
                  </p>
                  <table className="data" style={{ margin: 0 }}>
                    <tbody>
                      <tr><td>X <span className="muted">(N × C × H × W)</span></td><td data-out="X">{dims(sh.X)}</td></tr>
                      <tr><td>F <span className="muted">(M × C × K × K)</span></td><td data-out="F">{dims(sh.F)}</td></tr>
                      <tr><td>Y <span className="muted">(N × M × H_out × W_out){/* i18n-ignore: shape notation */}</span></td><td data-out="Y">{dims(sh.Y)}</td></tr>
                      <tr><td><Translated vi="bộ lọc">filters</Translated></td><td className="num" data-out="filters">{big(sh.filters)}</td></tr>
                      <tr><td><Translated vi="trọng số">weights</Translated></td><td className="num" data-out="weights">{big(sh.weights)}</td></tr>
                      <tr><td><Translated vi="phép nhân-cộng mỗi mẫu">multiply-adds per sample</Translated></td><td className="num" data-out="macs">{big(sh.macsPerSample)}</td></tr>
                    </tbody>
                  </table>
                  <label className="field" style={{ marginTop: '0.8rem', width: '100%' }}>
                    <Translated vi={<>Kích thước batch N = {N}</>}>Batch size N = {N}</Translated>
                    <input type="range" min="0" max="8" value={lgN} aria-label={trText('batch size', 'kích thước batch')} onChange={(e) => setLgN(+e.target.value)} />
                  </label>
                  <table className="data" style={{ margin: '0.4rem 0 0' }}>
                    <tbody>
                      <tr><td><Translated vi="phép nhân-cộng cho cả batch">multiply-adds for the batch</Translated></td><td className="num" data-out="batch-macs">{big(t.macs)}</td></tr>
                      <tr><td><Translated vi="số byte lý tưởng cho cả batch">ideal bytes for the batch</Translated></td><td className="num" data-out="total">{bytes(t.total)}</td></tr>
                      <tr><td><Translated vi="trong đó F">of which F</Translated></td><td className="num" data-out="fshare">{pct(t.fShare)}</td></tr>
                      <tr><td><Translated vi="byte của F mỗi mẫu">F bytes per sample</Translated></td><td className="num" data-out="fpersample">{big(Math.round(t.fBytesPerSample * 100) / 100)}</td></tr>
                    </tbody>
                  </table>
                </>
              );
            })()
          ) : (
            <p style={{ margin: 0 }} data-out="not-computed">
              <b>{id}</b> ({stage.out}). {STAGE_TEXT[id]}
            </p>
          )}
        </div>
        <div className="panel">
          <p style={{ margin: '0 0 0.5rem' }}><Translated vi={<>Ba lớp tích chập ở kích thước batch {N}</>}>The three convolution layers at batch size {N}</Translated></p>
          <table className="data" style={{ margin: 0, width: '100%' }} aria-label={trText('LeNet-5 convolution layers compared', 'So sánh các lớp tích chập của LeNet-5')}>
            <thead>
              <tr>
                <th><Translated vi="lớp">layer</Translated></th>
                <th className="num">C</th>
                <th className="num"><Translated vi="map">map</Translated></th>
                <th className="num">M</th>
                <th className="num nowrap"><Translated vi="phần của F">F share</Translated></th>
                <th style={{ width: '30%' }}><Translated vi="lưu lượng">traffic</Translated></th>
              </tr>
            </thead>
            <tbody>
              {conv.map((s) => {
                const t = batchTraffic(s.layer, N);
                return (
                  <tr key={s.id} data-row={s.id} data-fshare={t.fShare} style={s.id === id ? { background: 'var(--rule-soft)' } : undefined}>
                    <td>{s.id}</td>
                    <td className="num">{s.layer.C}</td>
                    <td className="num nowrap">{s.layer.H} × {s.layer.W}</td>
                    <td className="num">{s.layer.M}</td>
                    <td className="num">{pct(t.fShare)}</td>
                    <td><Bar t={t} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the sequential reference

function Reference() {
  const text = useLabFile(LAB_MAIN);
  if (text == null) return null;
  const region = lessonRegion(text, 'cpu-reference');
  return (
    <Code
      tabs={[{ label: trText('CPU reference', 'Tham chiếu CPU'), code: region.code, tiers: {} }]}
      caption={
        <Translated
          vi={
            <>
              {region.found ? 'Bản tham chiếu CPU' : 'Chương trình điều khiển (driver) của lab 19.1 (không tìm thấy vùng tham chiếu, nên cả tệp được hiển thị)'}, nạp
              từ <code>labs/{LAB_MAIN}</code>. Mỗi lần chạy lab 19.1 đều kiểm tra kernel của bạn với nó theo một dung sai.
            </>
          }
        >
          <>
            {region.found ? 'The CPU reference' : 'Lab 19.1’s driver (the reference region was not found, so the whole file is shown)'}, loaded
            from <code>labs/{LAB_MAIN}</code>. Every run of lab 19.1 checks your kernels against it with a tolerance.
          </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- the page

export function S191() {
  return (
    <>
      <Translated
        vi={
          <p>
            Mạng tích chập còn có trước cả những GPU đã giúp chúng trở nên thực dụng. Theo lược sử ngắn của sách, chúng được
            phát minh vào cuối thập niên 1980, và mạng truyền thẳng quay trở lại vào khoảng năm 2006.
          </p>
        }
      >
        <p>
          Convolutional networks are older than the GPUs that made them practical. In the book&rsquo;s short history they
          were invented in the late 1980s, and feed-forward networks came back around 2006.
        </p>
      </Translated>
      <Source section="19.1" pages="453–455" />
      <Translated
        vi={
          <p>
            Năm 2012, một bài dự thi của Đại học Toronto thắng cuộc thi ImageNet với một mạng khoảng 60 triệu tham số, huấn
            luyện trên hai GPU bằng một thư viện CUDA tên là cuda-convnet; tỉ lệ lỗi của nó là 15.3% so với 26.2% của đội
            về nhì. Sách xác định năm đó là lúc học sâu cất cánh trong công nghiệp.
          </p>
        }
      >
        <p>
          In 2012 a University of Toronto entry won the ImageNet challenge with a network of about 60 million parameters,
          trained on two GPUs with a CUDA library called cuda-convnet; its error rate was 15.3% against 26.2% for the
          runner-up. The book dates deep learning&rsquo;s take-off in industry to that year.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch19-facts A9)" /* i18n-ignore: citation */>
        <Translated
          vi={
            <p>
              Bài báo NIPS 2012 nói thời gian huấn luyện là năm đến sáu ngày, không phải một tuần, và ghi con số 15.3% cho
              &ldquo;một biến thể của mô hình này&rdquo;.
            </p>
          }
        >
          <p>
            The NIPS 2012 paper says five to six days of training, not a week, and credits the 15.3% to &ldquo;a variant of
            this model&rdquo;.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Ví dụ xuyên suốt của chương là LeNet-5, một bộ phân loại chữ số nhỏ với ba loại lớp: tích chập, lấy mẫu con
            (subsampling) và kết nối đầy đủ. Sách khẳng định các lớp tích chập chiếm phần lớn thời gian của một CNN, nên
            chương chỉ đề cập đến chúng, và chỉ lượt xuôi của chúng. Lượt đó là thứ cần tăng tốc trước tiên: suy luận chỉ
            chạy lượt xuôi, và huấn luyện cũng chạy nó.
          </p>
        }
      >
        <p>
          The chapter&rsquo;s running example is LeNet-5, a small digit classifier with three kinds of layer:
          convolution, subsampling and fully connected. The book&rsquo;s claim is that convolution layers take most of a
          CNN&rsquo;s time, so the chapter covers only them, and only their forward path. That path is the one to speed up
          first: inference runs only the forward path, and training runs it too.
        </p>
      </Translated>
      <Source section="19.1" pages="455" figs="Fig. 19.1" />

      <h2>
        <Translated vi="Một lớp là một dãy các phép tích chập 2D">A layer is a bank of 2D convolutions</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Đầu vào và đầu ra của một lớp là các chồng ảnh 2D gọi là feature map (gọi tắt là map). Một bộ lọc là mặt nạ
            (mask) của chương 7: một mảng trọng số K × K nhỏ. Cái mới là việc sổ sách. Mỗi map đầu ra lấy từ mọi map đầu vào,
            mỗi map qua bộ lọc riêng của nó, nên một lớp có C map đầu vào và M map đầu ra có M · C bộ lọc. Map đầu ra m là
            tổng của C phép tích chập 2D thông thường, mỗi phép cho một map đầu vào c, với bộ lọc của cặp (m, c).
          </p>
        }
      >
        <p>
          A layer&rsquo;s inputs and outputs are stacks of 2D images called feature maps. A filter is chapter 7&rsquo;s
          mask: a small K × K array of weights. What is new is the bookkeeping. Every output map draws on every input
          map, each through its own filter, so a layer with C input maps and M output maps has M · C filters. Output map m
          is the sum of C ordinary 2D convolutions, one per input map c, each with the filter for the pair (m, c).
        </p>
      </Translated>
      <Source section="19.1" pages="456" figs="Fig. 19.2(a)" />
      <Translated
        vi={
          <p>
            Sách bỏ qua hàm kích hoạt. Nó có thể được áp dụng sau đó, bằng một kernel riêng hoặc ở cuối chính kernel đó, và
            nó không thay đổi gì cách các tổng được tính.
          </p>
        }
      >
        <p>
          The book leaves out the activation function. It can be applied afterwards, as a separate kernel or at the end of
          the same one, and it changes nothing about how the sums are computed.
        </p>
      </Translated>
      <Source section="19.1" pages="456" />

      <h2>
        <Translated vi="Tính tay một đầu ra">One output by hand</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Fig. 19.2 cụ thể hóa điều này: ba map đầu vào 3 × 3, hai map đầu ra, và do đó sáu bộ lọc 2 × 2. Pixel (0, 0) của
            map đầu ra 0 lấy vùng 2 × 2 ở góc trên bên trái của mỗi map đầu vào, nhân từng phần tử với bộ lọc của map đó cho
            đầu ra 0, rồi cộng tất cả lại: 7 từ map 0, 5 từ map 1, 2 từ map 2, vậy là 14. Sách dùng số nguyên nhỏ để bạn có
            thể kiểm tra bằng tay; một lớp thực tế làm việc với số dấu phẩy động.
          </p>
        }
      >
        <p>
          Fig. 19.2 makes this concrete: three 3 × 3 input maps, two output maps, and so six 2 × 2 filters. Pixel (0, 0)
          of output map 0 takes the top-left 2 × 2 patch of each input map, multiplies it element by element with that
          map&rsquo;s filter for output 0, and adds everything up: 7 from map 0, 5 from map 1, 2 from map 2, so 14. The
          book uses small integers so you can check by hand; a real layer works in floating point.
        </p>
      </Translated>
      <Source section="19.1" pages="457" figs="Eq. 19.1" />
      <Predict
        id="ch19/19.1/p-pixel"
        prompt={
          <Translated
            vi={
              <>
                Trước khi dùng trình khám phá, hãy tính tay pixel (0, 1) của map đầu ra 0. Mỗi vùng dịch sang phải một cột;
                các bộ lọc giữ nguyên.
              </>
            }
          >
            <>
              Before using the explorer, compute pixel (0, 1) of output map 0 by hand. Every patch moves one column to the
              right; the filters stay the same.
            </>
          </Translated>
        }
      >
        {(v) => {
          const parts = perInputMap(S, BOOK_X, BOOK_F, 0, 0, 1);
          const y = parts.reduce((a, b) => a + b, 0);
          return (
            <p>
              {parts.join(' + ')} = <b>{y}</b>.{' '}
              {v === y
                ? trText('That is what you had.', 'Đúng như bạn đã tính.')
                : trText(`Click Y0's top-right cell in the explorer and step through the twelve products to find where yours went another way. The usual slip is to move the filter instead of the patch.`, `Nhấn vào ô trên cùng bên phải của Y0 trong trình khám phá và đi qua từng bước mười hai tích để tìm chỗ bạn tính khác đi. Lỗi thường gặp là dịch bộ lọc thay vì dịch vùng.`)}
            </p>
          );
        }}
      </Predict>
      <Translated
        vi={
          <p>
            Trình khám phá vẽ Fig. 19.2. Nhấn vào bất kỳ đầu ra nào trong tám đầu ra để làm sáng ba vùng và ba bộ lọc của
            nó, rồi đi qua từng bước mười hai phép nhân-cộng theo thứ tự các vòng lặp trên c, p và q của Fig. 19.3.
          </p>
        }
      >
        <p>
          The explorer draws Fig. 19.2. Click any of the eight outputs to light up its three patches and three filters,
          then step through the twelve multiply-adds in the order of Fig. 19.3&rsquo;s loops over c, p and q.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Số nguyên, như trong sách, và không có hàm kích hoạt. Thứ tự các bước là thứ tự vòng lặp của Fig. 19.3; không
              có gì ở đây nói GPU lập lịch chúng ra sao.
            </p>
          }
        >
          <p>
            Integers, as in the book, and no activation. The order of the steps is Fig. 19.3&rsquo;s loop order; nothing
            here says how a GPU schedules them.
          </p>
        </Translated>
      </Simplification>
      <OneOutput />
      <Translated
        vi={
          <p>
            Hai điều rút ra từ đây. Mỗi map đầu ra lấy tổng trên mọi map đầu vào, nên một pixel đầu vào ảnh hưởng tới mọi
            map đầu ra. Và công thức trượt mỗi bộ lọc trên đầu vào đúng như nó được lưu, không lật nó.
          </p>
        }
      >
        <p>
          Two things to take from it. Each output map sums over every input map, so one input pixel influences every
          output map. And the formula slides each filter over the input as it is stored, without flipping it.
        </p>
      </Translated>

      <h2>
        <Translated vi="Không có ô ma">No ghost cells</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương 7 cho mỗi pixel đầu vào một đầu ra bằng cách giả vờ ảnh được bao quanh bởi các{' '}
            <a href={href('ch07', '7.1')}>ô ma</a> bằng 0. LeNet thì không. Một đầu ra chỉ tồn tại ở nơi toàn bộ vùng K × K
            nằm gọn trong đầu vào, nên mỗi cạnh mất K − 1 pixel: với bộ lọc 5 × 5, một ảnh 32 × 32 cho 28 × 28. Trong mã,
            đó là <code>H_out = H - K + 1</code> và <code>W_out = W - K + 1</code>.
          </p>
        }
      >
        <p>
          Chapter 7 gave every input pixel an output by pretending the image is surrounded by{' '}
          <a href={href('ch07', '7.1')}>ghost cells</a> of zeros. LeNet does not. An output exists only where the whole K
          × K patch fits inside the input, so each side loses K − 1 pixels: with 5 × 5 filters, a 32 × 32 image gives 28
          × 28. In code that is <code>H_out = H - K + 1</code> and <code>W_out = W - K + 1</code>.
        </p>
      </Translated>
      <Source section="19.1" pages="457" />
      <Translated
        vi={
          <p>
            Đây chính xác là <code>nn.Conv2d</code> với mặc định <code>padding=0</code>. PyTorch cũng ghi trong tài liệu rằng{' '}
            <code>Conv2d</code> là tương quan chéo (cross-correlation), tức công thức của sách: không lật. Vậy lớp của sách
            và của PyTorch cho cùng các con số với cùng trọng số.
          </p>
        }
      >
        <p>
          This is <code>nn.Conv2d</code> with its default <code>padding=0</code>, exactly. PyTorch also documents{' '}
          <code>Conv2d</code> as cross-correlation, which is the book&rsquo;s formula: no flip. So the book&rsquo;s layer
          and PyTorch&rsquo;s give the same numbers for the same weights.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch07-facts.md, D1)" /* i18n-ignore: citation */>
        <Translated
          vi={
            <p>
              PyTorch 2.15 gọi <code>Conv2d</code> là &ldquo;the valid 2D cross-correlation operator&rdquo;, mặc định{' '}
              <code>padding=0</code>. cuDNN cung cấp cả hai chế độ và PyTorch yêu cầu tương quan chéo; MIOpen tính tương quan
              chéo (ch07-facts D1, ch19-facts B3); MPSGraph của Apple không ghi rõ bên nào (C3).
            </p>
          }
        >
          <p>
            PyTorch 2.15 calls <code>Conv2d</code> &ldquo;the valid 2D cross-correlation operator&rdquo;, default{' '}
            <code>padding=0</code>. cuDNN offers both modes and PyTorch asks for cross-correlation; MIOpen computes cross-correlation
            (ch07-facts D1, ch19-facts B3); Apple&rsquo;s MPSGraph documents neither (C3).
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS191['ch19/19.1/q-outsize']} label={trText('Work it out', 'Tự tính')} />
      <Ask q={questionsS191['ch19/19.1/q-flip']} label={trText('Work it out', 'Tự tính')} />

      <h2>
        <Translated vi="Ba tensor và kích thước của chúng">Three tensors and their shapes</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Với một mẫu, lớp đọc X, kích thước C × H × W, và F, kích thước M × C × K × K, rồi ghi Y, kích thước M × H_out ×
            W_out.
          </p>
        }
      >
        <p>
          For one sample, the layer reads X, of shape C × H × W, and F, of shape M × C × K × K, and writes Y, of shape M ×
          H_out × W_out.
        </p>
      </Translated>
      <Source section="19.1" pages="458" />
      <Translated
        vi={
          <p>
            Cả ba đều là mảng phẳng theo thứ tự hàng (row-major) với chiều cao nhất đứng trước. Vậy <code>F[m, c, _, _]</code>{' '}
            là bộ lọc K × K đưa map đầu vào c vào map đầu ra m, và các bộ lọc của một map đầu ra nằm cạnh nhau trong bộ nhớ.
          </p>
        }
      >
        <p>
          All three are flat arrays in row-major order with the highest dimension first. So <code>F[m, c, _, _]</code> is
          the K × K filter that carries input map c into output map m, and the filters for one output map sit next to each
          other in memory.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Hai chỗ sơ suất ở p. 458: phần chữ gọi mảng bộ lọc là <code>W</code>, còn mã gọi là <code>F</code>; và không
              bài tập nào xử lý lớp kết nối đầy đủ mà nó để lại &ldquo;làm bài tập&rdquo;.
            </p>
          }
        >
          <p>
            Two slips on p. 458: the text names the filter array <code>W</code>, the code <code>F</code>; and no exercise
            takes up the fully connected layer it leaves &ldquo;as an exercise&rdquo;.
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            PyTorch lưu <code>nn.Conv2d.weight</code> với kích thước (out_channels, in_channels, kH, kW): đúng F của sách,
            từng phần tử một. Mục 19.3 tận dụng điều đó.
          </p>
        }
      >
        <p>
          PyTorch stores <code>nn.Conv2d.weight</code> with shape (out_channels, in_channels, kH, kW): the book&rsquo;s F,
          element for element. Section 19.3 makes use of that.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch19-facts.md, D4)" /* i18n-ignore: citation */>
        <Translated
          vi={
            <p>
              PyTorch 2.15 ghi trọng số trong tài liệu là &ldquo;(out_channels, in_channels/groups, kernel_size[0],
              kernel_size[1])&rdquo;; với một group, đó là M × C × K × K.
            </p>
          }
        >
          <p>
            PyTorch 2.15 documents the weight as &ldquo;(out_channels, in_channels/groups, kernel_size[0],
            kernel_size[1])&rdquo;; with one group that is M × C × K × K.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Trong LeNet, C1 biến một ảnh 32 × 32 thành sáu map 28 × 28. C3 biến sáu map 14 × 14 thành mười sáu map 10 × 10
            và cần 96 bộ lọc. C5 biến mười sáu map 5 × 5 thành 120 map mỗi map một pixel, với 1920 bộ lọc: một bộ lọc 5 × 5
            trên một map 5 × 5 chỉ có đúng một chỗ để đặt.
          </p>
        }
      >
        <p>
          In LeNet, C1 turns one 32 × 32 image into six 28 × 28 maps. C3 turns six 14 × 14 maps into sixteen 10 × 10 maps
          and needs 96 filters. C5 turns sixteen 5 × 5 maps into 120 maps of one pixel each, with 1920 filters: a 5 × 5
          filter on a 5 × 5 map has exactly one place to sit.
        </p>
      </Translated>
      <Source section="19.1" pages="458" figs="Fig. 19.1" />
      <Simplification>
        <Translated
          vi={
            <p>
              Chỉ có kích thước và số đếm. Lưu lượng là lý tưởng: mỗi phần tử của X, F và Y được chuyển một lần mỗi batch
              dưới dạng float 4 byte, không có bộ nhớ đệm (cache) và không gắn với kernel cụ thể nào.
            </p>
          }
        >
          <p>
            Shapes and counts only. Traffic is the ideal: each element of X, F and Y moved once per batch as a 4-byte
            float, with no caches and no particular kernel.
          </p>
        </Translated>
      </Simplification>
      <LeNet />
      <Translated
        vi={
          <p>
            Đọc bảng so sánh từ trên xuống dưới. Các lớp đầu có ít kênh và map lớn; các lớp sau có nhiều kênh và map nhỏ.
            Mục 19.3 quay lại điều này khi nó hỏi liệu các ma trận của một phép tích chập có đủ lớn để dùng tốt GPU ở mọi
            lớp hay không.
          </p>
        }
      >
        <p>
          Read the comparison table from top to bottom. Early layers have few channels and large maps; late layers have
          many channels and small maps. Section 19.3 comes back to this when it asks whether the matrices of a convolution
          stay large enough to use a GPU well in every layer.
        </p>
      </Translated>
      <Ask q={questionsS191['ch19/19.1/q-weights']} label={trText('Work it out', 'Tự tính')} />

      <h2>
        <Translated vi="Một batch các mẫu">A batch of samples</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Cho nhiều đầu vào đi qua lớp cùng lúc tạo ra một lớp tích chập có batch; các map đầu vào và đầu ra thuộc về một
            đầu vào là một <b>mẫu</b>. X và Y có thêm một chiều N đứng đầu. F thì không: cùng các trọng số phục vụ mọi mẫu. Lý
            do sách đưa ra cho việc dùng batch là một lưới (grid) lớn hơn, với nhiều khối luồng hơn để giữ GPU bận rộn; điều
            này đặc biệt phổ biến khi huấn luyện.
          </p>
        }
      >
        <p>
          Running several inputs through the layer together gives a batched convolution layer; the input and output maps
          that belong to one input are a <b>sample</b>. X and Y gain a leading dimension N. F does not: the same weights
          serve every sample. The book&rsquo;s reason to batch is a bigger grid, with more thread blocks to keep the GPU
          busy; it is especially common in training.
        </p>
      </Translated>
      <Source section="19.1" pages="459" figs="Fig. 19.4" />
      <Translated
        vi={
          <p>
            Có một lý do thứ hai, và bạn đã gặp nó. Kéo thanh batch trong trình khám phá LeNet và xem phần byte của F: một
            trọng số nạp một lần có thể phục vụ mọi mẫu. Trong{' '}
            <a href={href('ch05', '5.1')}>trình khám phá lớp tuyến tính</a> của mục 5.1, cùng hiệu ứng này đã nâng tỉ lệ phép
            toán trên byte của một lớp tuyến tính. C5 là họ hàng gần: ở batch 1, 48,000 trọng số của nó gần như là toàn bộ
            lưu lượng, mỗi trọng số dùng một lần, và Fig. 19.1 thậm chí gọi tầng đó là kết nối đầy đủ.
          </p>
        }
      >
        <p>
          There is a second reason, and you have met it before. Move the batch slider in the LeNet explorer and watch
          F&rsquo;s share of the bytes: a weight fetched once can serve every sample. In section 5.1&rsquo;s{' '}
          <a href={href('ch05', '5.1')}>linear-layer explorer</a> the same effect raised a linear layer&rsquo;s ratio of
          operations to bytes. C5 is close kin: at batch 1 its 48,000 weights are almost all of its traffic, each used
          once, and Fig. 19.1 even labels that stage a full connection.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Các số đếm lưu lượng trong trình khám phá là của khóa học này, từ các kích thước ở p. 458 và phương pháp lưu
              lượng lý tưởng của mục 5.1. Sách không đưa ra con số nào cho batching.
            </p>
          }
        >
          <p>
            The traffic counts in the explorer are ours, from the shapes on p. 458 and the ideal-traffic method of section
            5.1. The book gives no numbers for batching.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS191['ch19/19.1/q-batch']} />

      <h2>
        <Translated vi="Bản tham chiếu tuần tự">The sequential reference</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Fig. 19.4 là toàn bộ lớp có batch dưới dạng bảy vòng lặp lồng nhau: n, m, h và w chọn một phần tử đầu ra, còn c,
            p và q cộng dồn các tích của nó. Chương trình điều khiển (driver) của lab 19.1 giữ một phiên bản C++ của nó làm
            bản tham chiếu để kiểm tra mọi kernel bạn viết trong chương này.
          </p>
        }
      >
        <p>
          Fig. 19.4 is the whole batched layer as seven nested loops: n, m, h and w pick an output element, and c, p and q
          add up its products. Lab 19.1&rsquo;s driver keeps a C++ version of it as the reference that every kernel you
          write in this chapter is checked against.
        </p>
      </Translated>
      <Source section="19.1" pages="457, 459" figs="Figs. 19.3, 19.4" />
      <Reference />
      <p>
        <Translated vi="Những chỗ nó khác với trang sách:">Where it differs from the page:</Translated>
      </p>
      <ul>
        <Translated
          vi={
            <li>
              Sách viết <code>X[n, c, h + p, w + q]</code> dưới dạng mã giả nhiều chiều; bản tham chiếu tự tính chỉ số phẳng
              theo thứ tự hàng, như chương 3 đã dạy.
            </li>
          }
        >
          <li>
            The book writes <code>X[n, c, h + p, w + q]</code> as multi-dimensional pseudo-code; the reference computes
            the flat row-major index itself, as chapter 3 taught.
          </li>
        </Translated>
        <li>
          <Translated vi="Nó tính H_out và W_out từ H, W và K thay vì giả định chúng.">It computes H_out and W_out from H, W and K instead of assuming them.</Translated>
        </li>
        <Translated
          vi={
            <li>
              Nó cộng dồn bằng <code>double</code>, để bản tham chiếu chính xác hơn các kernel float mà nó đánh giá. Bài kiểm
              tra vẫn so sánh theo một dung sai.
            </li>
          }
        >
          <li>
            It accumulates in <code>double</code>, so that the reference is more accurate than the float kernels it
            judges. The test still compares with a tolerance.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Nó cũng ghi lại, cho mỗi đầu ra, tổng giá trị tuyệt đối của các tích; các bài kiểm tra nhân dung sai của chúng
              theo giá trị này, nên một đầu ra lớn được phép có sai số làm tròn lớn hơn tương ứng.
            </li>
          }
        >
          <li>
            It also records, for each output, the sum of the products&rsquo; absolute values; the tests scale their
            tolerance by it, so a large output is allowed a proportionally larger rounding error.
          </li>
        </Translated>
        <li>
          <Translated vi={<>Các vòng lặp giữ đúng thứ tự của sách, nên bạn có thể đọc nó từng dòng đối chiếu với Fig. 19.4.</>}>
            The loops stay in the book&rsquo;s order, so you can read it line for line against Fig. 19.4.
          </Translated>
        </li>
      </ul>
    </>
  );
}
