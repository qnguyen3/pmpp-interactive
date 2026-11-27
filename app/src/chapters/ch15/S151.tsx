import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  BLAS_EDITS, OPERATIONS, agreement, blasCall, editBlas, editMps, exampleBuffers, mapOperation, mpsCall, reference, runBlas, runMps,
  type BlasCall, type BlasEdit, type Mapping, type MpsCall, type OpId, type RunResult,
} from '../../sim/gemmlayout';

// ---------------------------------------------------------------- questions
export const questionsS151: Record<string, Question> = {
  'ch15/15.1/q-linear': {
    kind: 'choice',
    id: 'ch15/15.1/q-linear',
    prompt: (
      <Translated
        vi={
          <>
            <code>nn.Linear</code> tính y = x Wᵀ + b, với W được lưu dạng (out_features, in_features). Cách ánh xạ nào vào
            D = αA × B + βC của sách là đúng?
          </>
        }
      >
        <>
          <code>nn.Linear</code> computes y = x Wᵀ + b, with W stored as (out_features, in_features). Which mapping onto the
          book's D = αA × B + βC is right?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"A = x, thứ tự hàng (row-major); B = Wᵀ, chính là vùng nhớ của W đọc theo thứ tự cột (column-major); C = b trên mọi hàng; α = β = 1."}>{'A = x, row-major; B = Wᵀ, which is W\'s memory read column-major; C = b on every row; α = β = 1.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"A = x; B = W đúng như được lưu, thứ tự hàng, out × in; C = b lặp lại trên mọi hàng; α = β = 1."}>{'A = x; B = W exactly as it is stored, row-major, out × in; C = b repeated on every row; α = β = 1.'}</Translated>,
        why: <Translated vi={"W là out × in, còn B của sách phải là K × N, tức in × out. Dùng nguyên như được lưu, bạn sẽ tính x W, phép nhân mà kích thước chỉ khớp khi in = out."}>{'W is out × in, and the book\'s B has to be K × N, in × out. Used as stored you would compute x W, which only even has matching sizes when in = out.'}</Translated>,
      },
      {
        label: <Translated vi={"A = x; B = Wᵀ lưu theo thứ tự cột; C = y, đầu ra của lớp; α = β = 1."}>{'A = x; B = Wᵀ stored column-major; C = y, the layer\'s output; α = β = 1.'}</Translated>,
        why: <Translated vi={"C là ma trận được cộng vào tích. Kết quả là D."}>{'C is the matrix added to the product. The result is D.'}</Translated>,
      },
      {
        label: <Translated vi={"A = x; B = Wᵀ; C = b lặp lại trên mọi hàng; α = 1/batch để lấy trung bình trên batch, β = 1."}>{'A = x; B = Wᵀ; C = b repeated on every row; α = 1/batch to average over the batch, β = 1.'}</Translated>,
        why: <Translated vi={"Không có gì trong một lớp tuyến tính lấy trung bình trên batch. Mỗi hàng được biến đổi riêng, với α = 1."}>{'Nothing in a linear layer averages over the batch. Each row is transformed on its own, with α = 1.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Sách xếp các vector cần biến đổi thành các hàng của A, nên A = x. Cột j của B phải chứa các hệ số cho đầu ra j, và đó
            là hàng j của W: B = Wᵀ. Một W theo thứ tự hàng, đọc theo từng cột, chính là B đó. Bias giống nhau cho mọi đầu vào,
            nên C là b được chép vào từng hàng của batch.
          </p>
        }
      >
        <p>
          The book stacks the vectors to transform as the rows of A, so A = x. Column j of B must hold the coefficients
          for output j, and those are row j of W: B = Wᵀ. A row-major W, read column by column, is exactly that B. The bias
          is the same for every input, so C is b copied onto each of the batch rows.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.1, pp. 349–350; the shapes of nn.Linear from ch03-facts D6; the mapping is ours',
    vi: { source: 'ch. 15 §15.1, pp. 349–350; hình dạng của nn.Linear từ ch03-facts D6; cách ánh xạ là của khóa học này' },
  },
  'ch15/15.1/q-rowmajor-trick': {
    kind: 'choice',
    id: 'ch15/15.1/q-rowmajor-trick',
    prompt: (
      <Translated
        vi={
          <>
            A (M × K), B (K × N) và C (M × N) đều theo thứ tự hàng. Vì sao{' '}
            <code>cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &amp;one, B, N, A, K, &amp;zero, C, N)</code> để lại
            C = A B trong C?
          </>
        }
      >
        <>
          A (M × K), B (K × N) and C (M × N) are row-major. Why does{' '}
          <code>cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &amp;one, B, N, A, K, &amp;zero, C, N)</code> leave
          C = A B in C?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Mỗi mảng thứ tự hàng, khi đọc theo thứ tự cột, là chuyển vị của nó, nên cuBLAS tính Cᵀ = Bᵀ Aᵀ, và Cᵀ theo thứ tự cột chính là C theo thứ tự hàng."}>{'Each row-major array, read column-major, is its transpose, so cuBLAS computes Cᵀ = Bᵀ Aᵀ, and Cᵀ column-major is C row-major.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"cuBLAS thấy các chiều đầu (leading dimension) bằng độ dài hàng, nhận ra đầu vào thứ tự hàng, và trước tiên chuyển vị nó vào một bản sao ẩn."}>{'cuBLAS sees leading dimensions equal to the row lengths, recognises row-major input, and transposes it into a hidden copy first.'}</Translated>,
        why: <Translated vi={"Không có gì bị sao chép. Các đối số chỉ mô tả cách đọc vùng nhớ đã có sẵn."}>{'Nothing is copied. The arguments only describe how to read memory that is already there.'}</Translated>,
      },
      {
        label: <Translated vi={"Nó chỉ đúng với ma trận vuông; các hình dạng khác phải được đệm thành vuông trước lời gọi."}>{'It works only for square matrices; other shapes must be padded to square before the call.'}</Translated>,
        why: <Translated vi={"Chạy một ví dụ 3 × 4 trong trình khám phá ở trên: nó khớp từng ô một."}>{'Run a 3 × 4 example in the explorer above: it agrees cell for cell.'}</Translated>,
      },
      {
        label: <Translated vi={"Truyền B trước khiến cuBLAS tính B A, và B A bằng A B với các ma trận có kích thước khớp nhau."}>{'Passing B first makes cuBLAS compute B A, and B A equals A B for matrices of matching sizes.'}</Translated>,
        why: <Translated vi={"Phép nhân ma trận không giao hoán. Thứ tự đổi chỗ vì chuyển vị một tích sẽ đảo ngược nó: (A B)ᵀ = Bᵀ Aᵀ."}>{'Matrix products do not commute. The order swaps because transposing a product reverses it: (A B)ᵀ = Bᵀ Aᵀ.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một mảng M × K theo thứ tự hàng chiếm bộ nhớ y hệt một mảng K × M theo thứ tự cột, tức chuyển vị của nó. Vì vậy thư
            viện thấy Bᵀ (N × K) và Aᵀ (K × M), nhân chúng thành một kết quả N × M theo thứ tự cột, và vùng nhớ đó đọc theo thứ tự
            hàng chính là ma trận M × N C. Các chiều đầu là độ dài hàng của các mảng thứ tự hàng: N, K và N.
          </p>
        }
      >
        <p>
          A row-major M × K array occupies memory exactly like a column-major K × M array, its transpose. So the library
          sees Bᵀ (N × K) and Aᵀ (K × M), multiplies them into an N × M column-major result, and that memory read
          row-major is the M × N matrix C. The leading dimensions are the row lengths of the row-major arrays: N, K and N.
        </p>
      </Translated>
    ),
    source: 'ch15-facts A10 (our derivation of the standard idiom); llama.cpp calls cuBLAS the same way (ch03-facts D2)',
    vi: { source: 'ch15-facts A10 (suy luận của khóa học này cho cách làm tiêu chuẩn); llama.cpp gọi cuBLAS theo cùng cách (ch03-facts D2)' },
  },
  'ch15/15.1/q-qkt': {
    kind: 'choice',
    id: 'ch15/15.1/q-qkt',
    prompt: (
      <Translated
        vi={
          <>
            Điểm attention S = Q Kᵀ, với Q (n_q × d) và K (n_kv × d), cả hai theo thứ tự hàng. Toán hạng nào đóng vai trò trọng số
            của <code>nn.Linear</code>?
          </>
        }
      >
        <>
          Attention scores S = Q Kᵀ, with Q (n_q × d) and K (n_kv × d), both row-major. Which operand plays the role of{' '}
          <code>nn.Linear</code>'s weight?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"K: các hàng của nó chạy dọc theo d, và nó được dùng ở dạng chuyển vị, như W (out, in)."}>{'K: its rows run along d, and it is used transposed, like W (out, in).'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Q, vì toán hạng bên trái của một tích là cái mà thư viện coi là trọng số."}>{'Q, because the left operand of a product is the one a library treats as the weight.'}</Translated>,
        why: <Translated vi={"Q là A của sách, được lưu như sách lưu A: mỗi hàng một vector."}>{'Q is the book\'s A, stored as the book stores A: one vector per row.'}</Translated>,
      },
      {
        label: <Translated vi={"Cả hai: trong Q Kᵀ mỗi toán hạng đều được đọc ở dạng chuyển vị, như W trong x Wᵀ."}>{'Both: in Q Kᵀ each operand is read transposed, as W is in x Wᵀ.'}</Translated>,
        why: <Translated vi={"Toán hạng bên trái được lưu đúng như sách mong đợi. Chỉ toán hạng bên phải bị xoay lại."}>{'The left operand is stored as the book expects. Only the right one is turned around.'}</Translated>,
      },
      {
        label: <Translated vi={"Không cái nào: attention cần một cách bố trí riêng, với K được lưu theo từng cột."}>{'Neither: attention needs a layout of its own, with K stored column by column.'}</Translated>,
        why: <Translated vi={"Chọn Q Kᵀ trong trình khám phá. Lời gọi của nó có cùng dạng với của nn.Linear, với transa = T."}>{'Pick Q Kᵀ in the explorer. Its call has the same shape as nn.Linear\'s, with transa = T.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Theo cách gọi của sách, M = n_q, N = n_kv và K = d. B của sách là d × n_kv, và vùng nhớ của K, mỗi hàng một token,
            chính là B đó lưu theo thứ tự cột: cùng tình huống với một trọng số lưu dạng (out, in). Vì vậy cách bố trí mà mốc dự
            án M3 xử lý cũng bao luôn Q Kᵀ, và chương 20 xây attention trên nó.
          </p>
        }
      >
        <p>
          In the book's terms M = n_q, N = n_kv and K = d. The book's B is d × n_kv, and K's memory, one token per row, is
          that B stored column-major: the same situation as a weight stored (out, in). The layout milestone M3 handles
          therefore covers Q Kᵀ as well, and chapter 20 builds attention on it.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.1, p. 350 (attention as large products, ch. 20); the mapping is ours',
    vi: { source: 'ch. 15 §15.1, p. 350 (attention như các tích lớn, ch. 20); cách ánh xạ là của khóa học này' },
  },
};

// ---------------------------------------------------------------- the explorer
function Matrix({ title, rows, cols, ld, data, ok, tier, testid }: {
  title: string;
  rows: number;
  cols: number;
  ld: number;
  data: number[];
  ok?: boolean[];
  tier: 'global' | 'reg';
  testid: string;
}) {
  return (
    <div data-matrix={testid} style={{ minWidth: 0 }}>
      <div className="ui muted" style={{ fontSize: '0.8rem', marginBottom: '0.25rem' }}>{title}</div>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, 2.1rem)`, gap: 2 }}>
        {Array.from({ length: rows * cols }, (_, i) => {
          const r = Math.floor(i / cols);
          const c = i % cols;
          const v = data[r * ld + c];
          const good = ok ? ok[i] : undefined;
          const bg = good === false ? 'var(--fail-wash)' : good ? 'var(--ok-wash)' : `var(--${tier}-wash)`;
          const bd = good === false ? 'var(--fail)' : good ? 'var(--ok)' : `var(--${tier})`;
          return (
            <span key={i} data-cell={`${r},${c}`} data-ok={good == null ? undefined : good ? 1 : 0}
              className="mono"
              style={{ background: bg, border: `1px solid ${bd}`, borderRadius: 2, textAlign: 'center', fontSize: '0.78rem', padding: '0.15rem 0', color: good === false ? 'var(--fail)' : 'var(--ink)' }}>
              {Number.isFinite(v) ? v : '?'}
            </span>
          );
        })}
      </div>
    </div>
  );
}

const opName = (t: 'N' | 'T') => (t === 'N' ? 'CUBLAS_OP_N' : 'CUBLAS_OP_T');

function blasText(c: BlasCall, names: Mapping['op']['names'], numeric: boolean) {
  const v = (n: number, s: string) => (numeric ? String(n) : s);
  const nm = (role: 'A' | 'B' | 'C') => names[role];
  const lines: [string, string][] = [
    [`cublasSgemm(handle,`, ''],
    [`    ${opName(c.transa)}, ${opName(c.transb)},`, '// transa, transb'],
    [`    ${v(c.m, c.symbolic.m)}, ${v(c.n, c.symbolic.n)}, ${v(c.k, c.symbolic.k)},`, '// m, n, k'],
    [`    &alpha, ${nm(c.a)}, ${v(c.lda, c.symbolic.lda)},`, '// A, lda'], // i18n-ignore: code listing
    [`    ${nm(c.b)}, ${v(c.ldb, c.symbolic.ldb)},`, '// B, ldb'],
    [`    &beta, ${nm('C')}, ${v(c.ldc, c.symbolic.ldc)});`, '// C, ldc'], // i18n-ignore: code listing
  ];
  const w = Math.max(...lines.slice(1).map(([a]) => a.length)) + 2;
  return lines.map(([a, b]) => (b ? a.padEnd(w) + b : a)).join('\n');
}

function mpsText(c: MpsCall, numeric: boolean) {
  const v = (n: number, s: string) => (numeric ? String(n) : s);
  return (
    `[[MPSMatrixMultiplication alloc]\n` + // i18n-ignore: Objective-C listing
    `    initWithDevice: device\n` + // i18n-ignore: Objective-C listing
    `    transposeLeft: ${c.transposeLeft ? 'YES' : 'NO'}\n` + // i18n-ignore: Objective-C listing
    `    transposeRight: ${c.transposeRight ? 'YES' : 'NO'}\n` + // i18n-ignore: Objective-C listing
    `    resultRows: ${v(c.resultRows, c.symbolic.resultRows)}\n` + // i18n-ignore: Objective-C listing
    `    resultColumns: ${v(c.resultColumns, c.symbolic.resultColumns)}\n` + // i18n-ignore: Objective-C listing
    `    interiorColumns: ${v(c.interiorColumns, c.symbolic.interiorColumns)}\n` + // i18n-ignore: Objective-C listing
    `    alpha: 1.0 beta: 0.0]` // i18n-ignore: Objective-C listing
  );
}

function verdict(map: Mapping, r: RunResult, ok: boolean[]) {
  const good = ok.filter(Boolean).length;
  if (r.readOutside || r.writeOutside)
    return (
      <p data-verdict="outside" style={{ margin: '0.4rem 0 0' }}>
        <span className="chip fail">
          {r.readOutside ? tr('reads past the end of a buffer', 'đọc quá cuối một buffer') : tr('writes past the end of C', 'ghi quá cuối C')}
        </span>{' '}
        <Translated vi="Lời gọi biên dịch và chạy được; trên GPU nó sẽ đọc hoặc ghi vào vùng nhớ không thuộc ma trận này.">
          The call compiles and runs; on a GPU it would read or write memory that is not this matrix.
        </Translated>
      </p>
    );
  return (
    <p data-verdict={good === ok.length ? 'agree' : 'wrong'} style={{ margin: '0.4rem 0 0' }}>
      {good === ok.length ? (
        <span className="chip ok">
          <Translated vi="khớp với tích theo thứ tự hàng">agrees with the row-major product</Translated>
        </span>
      ) : (
        <span className="chip fail">
          <Translated vi="giá trị sai">wrong values</Translated>
        </span>
      )}{' '}
      <Translated vi={<>{good} trên {ok.length} ô của {map.op.names.C} khớp.</>}>
        {good} of {ok.length} cells of {map.op.names.C} agree.
      </Translated>
    </p>
  );
}

const DIM_MAX = 6;

function SpellingsBench() {
  const [id, setId] = useState<OpId>('linear');
  const [shape, setShape] = useState({ M: 3, N: 4, K: 5 });
  const [edits, setEdits] = useState<Partial<Record<BlasEdit, boolean>>>({});
  const [flipRight, setFlipRight] = useState(false);
  const [numeric, setNumeric] = useState(false);

  const map = useMemo(() => mapOperation(id, shape), [id, shape]);
  const bufs = useMemo(() => exampleBuffers(map), [map]);
  const ref = useMemo(() => reference(map, bufs), [map, bufs]);
  const call = editBlas(blasCall(map), edits);
  const blas = runBlas(call, bufs);
  const blasOk = agreement(map, blas.C, ref);
  const mps = editMps(mpsCall(map), flipRight);
  const mpsRun = runMps(mps, bufs);
  const mpsOk = agreement(map, mpsRun.C, ref);
  const { op } = map;
  const d = op.dims;
  const n = op.names;

  const slider = (key: 'M' | 'N' | 'K') => (
    <label className="field" key={key}>
      <Translated vi={<>{d[key]} ({key} của sách) = {shape[key]}</>}>
        {d[key]} (the book's {key}) = {shape[key]}
      </Translated>
      <input type="range" min="1" max={DIM_MAX} value={shape[key]} aria-label={tr(`book ${key}`, `${key} của sách`)}
        onChange={(e) => setShape({ ...shape, [key]: +e.target.value })} />
    </label>
  );

  return (
    <Bench
      title={tr('One product, three spellings', 'Một tích, ba cách viết')}
      task={tr('Pick an operation. Find where its transposes went, then break the call on purpose.', 'Chọn một phép toán. Tìm xem các phép chuyển vị của nó đã đi đâu, rồi cố ý làm hỏng lời gọi.')}
      foot={
        <Translated
          vi={
            <>
              Đầu vào là các số nguyên nhỏ cố định theo vị trí, nên mọi tích đều chính xác. Các lời gọi được thực thi bởi một mô
              hình về cách mỗi thư viện đọc bộ nhớ (src/sim/gemmlayout.ts): thứ tự cột với lời gọi BLAS, thứ tự hàng qua rowBytes
              với MPS. Không có gì ở đây chạy trên GPU.
            </>
          }
        >
          <>
            The inputs are small integers fixed by position, so every product is exact. The calls are executed by a model of
            how each library reads memory (src/sim/gemmlayout.ts): column-major for the BLAS call, row-major through rowBytes
            for MPS. Nothing here ran on a GPU.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi="Phép toán">Operation</Translated>
          <select value={id} aria-label={tr('operation', 'phép toán')} onChange={(e) => setId(e.target.value as OpId)}>
            {OPERATIONS.map((o) => (
              <option key={o.id} value={o.id}>{tr(
                  o.label,
                  {
                    book: 'C = A × B của sách',
                    linear: 'nn.Linear, W lưu dạng (out, in)',
                    xw: 'x @ W, W lưu dạng (in, out)',
                    qkt: 'Điểm attention Q Kᵀ',
                    pv: 'Đầu ra attention P V',
                  }[o.id],
                )}</option>
            ))}
          </select>
        </label>
        {slider('M')}
        {slider('N')}
        {slider('K')}
        <label className="check">
          <input type="checkbox" checked={numeric} onChange={(e) => setNumeric(e.target.checked)} />{' '}
          <Translated vi="hiện các lời gọi bằng số">show the calls with numbers</Translated>
        </label>
      </div>
      <div className="cols">
        <div className="panel" data-panel="book">
          <p style={{ margin: '0 0 0.4rem' }}>
            <Translated vi={<>
              <b>{op.formula}</b> theo cách gọi của sách
            </>}>
              <b>{op.formula}</b> in the book's terms
            </Translated>
          </p>
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <tr><td className="nowrap">M, N, K</td><td data-dims>{d.M} = {shape.M}, {d.N} = {shape.N}, {d.K} = {shape.K}</td></tr>
              <tr><td>A</td><td>
                <Translated vi={<>{n.A}: {d.M} × {d.K}, thứ tự hàng, như sách lưu A</>}>
                  {n.A}: {d.M} × {d.K}, row-major, as the book stores A
                </Translated>
              </td></tr>
              <tr>
                <td>B</td>
                <td data-bstore={op.bStoredNK ? 'column' : 'row'}>
                  {op.bStoredNK ? (
                    <Translated vi={<>{n.B} được lưu {d.N} × {d.K}: B của sách ({d.K} × {d.N}) <b>lưu theo thứ tự cột</b></>}>
                      <>{n.B} is stored {d.N} × {d.K}: the book's B ({d.K} × {d.N}) <b>stored column-major</b></>
                    </Translated>
                  ) : (
                    <Translated vi={<>{n.B}: {d.K} × {d.N}, thứ tự hàng, như sách lưu B</>}>
                      <>{n.B}: {d.K} × {d.N}, row-major, as the book stores B</>
                    </Translated>
                  )}
                </td>
              </tr>
              <tr><td>C</td><td>
                <Translated vi={<>{n.C}: {d.M} × {d.N}, thứ tự hàng</>}>
                  {n.C}: {d.M} × {d.N}, row-major
                </Translated>
              </td></tr>
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.5rem 0 0', fontSize: '0.85rem' }}>{tr(
              op.note,
              {
                book: 'Cả hai đầu vào theo thứ tự hàng, như chương này giả định (p. 350).',
                linear: 'Trọng số của PyTorch có hình dạng (out_features, in_features); ggml lưu nó theo cùng cách (ch03-facts D2, D6).',
                xw: 'Một trọng số giữ mỗi hàng một đặc trưng đầu vào: đúng cách bố trí của sách.',
                qkt: 'Q và K đều giữ mỗi hàng một token, dài d giá trị.',
                pv: 'P có mỗi truy vấn một hàng trọng số; V mỗi key một hàng.',
              }[op.id],
            )}</p>
        </div>
        <div className="panel" data-panel="ref">
          <p style={{ margin: '0 0 0.5rem' }}>
            <Translated vi={<>
              <b>Dữ liệu</b>, và kết quả tham chiếu theo thứ tự hàng tính thẳng từ {op.formula}
            </>}>
              <b>The data</b>, and the row-major reference computed straight from {op.formula}
            </Translated>
          </p>
          <div className="row" style={{ alignItems: 'flex-start' }}>
            <Matrix title={tr(`${n.A} as stored`, `${n.A} như được lưu`)} rows={map.A.rows} cols={map.A.cols} ld={map.A.ld} data={bufs.A} tier="global" testid="A" />
            <Matrix title={tr(`${n.B} as stored`, `${n.B} như được lưu`)} rows={map.B.rows} cols={map.B.cols} ld={map.B.ld} data={bufs.B} tier="global" testid="B" />
            <Matrix title={`${n.C} = ${op.formula.split(' = ')[1]}`} rows={shape.M} cols={shape.N} ld={shape.N} data={ref} tier="reg" testid="ref" /* i18n-ignore: test id */ />
          </div>
        </div>
      </div>

      <div className="cols" style={{ marginTop: '1rem' }}>
        <div className="panel" data-panel="blas">
          <p style={{ margin: '0 0 0.4rem' }}>
            <Translated
              vi={
                <>
                  <b>BLAS thứ tự cột:</b> cuBLAS, và hipBLAS với cùng các đối số (<code>hipblasSgemm</code>,{' '}
                  <code>HIPBLAS_OP_N</code>, <code>HIPBLAS_OP_T</code>)
                </>
              }
            >
              <b>Column-major BLAS:</b> cuBLAS, and hipBLAS with the same arguments (<code>hipblasSgemm</code>,{' '}
              <code>HIPBLAS_OP_N</code>, <code>HIPBLAS_OP_T</code>)
            </Translated>
          </p>
          <pre className="mono" data-call="blas" style={{ fontSize: '0.78rem', overflowX: 'auto', margin: '0 0 0.6rem', padding: '0.5rem', background: 'var(--cell)', border: '1px solid var(--rule)', borderRadius: 3 }}>
            {blasText(call, n, numeric)}
          </pre>
          <div className="row" style={{ gap: '0.3rem 1rem', marginBottom: '0.6rem' }}>
            {(Object.keys(BLAS_EDITS) as BlasEdit[]).map((k) => (
              <label key={k} className="check">
                <input type="checkbox" checked={!!edits[k]} data-edit={k}
                  onChange={(e) => setEdits({ ...edits, [k]: e.target.checked })} /> {tr(BLAS_EDITS[k], { swap: 'đổi chỗ hai con trỏ đầu vào', flipTransa: 'lật transa', mnBook: 'truyền m, n theo thứ tự của sách (M, N)' }[k])}
              </label>
            ))}
          </div>
          <Matrix title={tr(`${n.C} after the call, read row-major`, `${n.C} sau lời gọi, đọc theo thứ tự hàng`)} rows={shape.M} cols={shape.N} ld={shape.N} data={blas.C} ok={blasOk} tier="reg" testid="blas" /* i18n-ignore: test id */ />
          {verdict(map, blas, blasOk)}
        </div>
        <div className="panel" data-panel="mps">
          <p style={{ margin: '0 0 0.4rem' }}>
            <Translated vi={<>
              <b>MPS thứ tự hàng:</b> <code>MPSMatrixMultiplication</code> với các cờ chuyển vị
            </>}>
              <b>Row-major MPS:</b> <code>MPSMatrixMultiplication</code> with transpose flags
            </Translated>
          </p>
          <pre className="mono" data-call="mps" style={{ fontSize: '0.78rem', overflowX: 'auto', margin: '0 0 0.6rem', padding: '0.5rem', background: 'var(--cell)', border: '1px solid var(--rule)', borderRadius: 3 }}>
            {mpsText(mps, numeric)}
          </pre>
          <table className="data" style={{ margin: '0 0 0.6rem' }}>
            <thead>
              <tr>
                <th>
                  <Translated vi="Ma trận">Matrix</Translated>
                </th>
                <th className="num">
                  <Translated vi="hàng">rows</Translated>
                </th>
                <th className="num">
                  <Translated vi="cột">columns</Translated>
                </th>
                <th className="num">rowBytes</th>
              </tr>
            </thead>
            <tbody>
              {([['left', mps.left], ['right', mps.right], ['result', mps.result]] as const).map(([k, m]) => (
                <tr key={k}>
                  <td>{tr(k, { left: 'trái', right: 'phải', result: 'kết quả' }[k])}: {n[m.role]}</td>
                  <td className="num">{m.rows}</td>
                  <td className="num">{m.columns}</td>
                  <td className="num">{m.rowBytes}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <label className="check" style={{ marginBottom: '0.6rem' }}>
            <input type="checkbox" checked={flipRight} data-edit="flipRight" onChange={(e) => setFlipRight(e.target.checked)} />{' '}
            <Translated vi="lật transposeRight">flip transposeRight</Translated>
          </label>
          <Matrix title={tr(`${n.C} after the call`, `${n.C} sau lời gọi`)} rows={shape.M} cols={shape.N} ld={shape.N} data={mpsRun.C} ok={mpsOk} tier="reg" testid="mps" /* i18n-ignore: test id */ />
          {verdict(map, mpsRun, mpsOk)}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the page
export function S151() {
  return (
    <>
      <Translated
        vi={
          <p>
            Mục này nói GEMM là gì và các tích trong một transformer ánh xạ vào nó như thế nào.
          </p>
        }
      >
        <p>
          This section says what GEMM is and how the products in a transformer map onto it.
        </p>
      </Translated>

      <h2>
        <Translated vi="Từ một vector đến GEMM">From one vector to GEMM</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bắt đầu với một vector hàng v nhân một ma trận B: u = v × B. Mỗi phần tử của u trộn tất cả các phần tử của v, và công
            thức trộn cho phần tử j là cột j của B. Vậy cột j chứa các hệ số tạo ra đầu ra j. Ví dụ của sách là đồ họa: xoay một
            điểm bằng cách nhân tọa độ 3D của nó với một ma trận 3 × 3, rồi cộng một vector để dịch nó và nhân với một hệ số để
            co giãn nó.
          </p>
        }
      >
        <p>
          Start with a single row vector v times a matrix B: u = v × B. Every element of u mixes all the elements of v,
          and the mix for element j is column j of B. So column j holds the coefficients that produce output j. The book's
          example is graphics: rotate a point by multiplying its 3D coordinates with a 3 × 3 matrix, then add a vector to
          shift it and multiply by a factor to scale it.
        </p>
      </Translated>
      <Source section="15.1" pages="349–350" />
      <Translated
        vi={
          <p>
            Để biến đổi nhiều vector với cùng một ma trận, hãy xếp chúng thành các hàng của A. Khi đó D = A × B, và hàng i của D là
            vector i đã được biến đổi. Thêm một hệ số co giãn và một phép dịch, bạn có dạng tổng quát D = αA × B + βC, mà các thư
            viện đại số tuyến tính gọi là GEMM, phép nhân ma trận tổng quát (general matrix multiply). Các hàng của C là những gì
            được cộng vào các hàng tương ứng của tích.
          </p>
        }
      >
        <p>
          To transform many vectors with the same matrix, stack them as the rows of A. Then D = A × B, and row i of D is
          vector i transformed. Add a scale and a shift and you have the general form D = αA × B + βC, which linear algebra
          libraries call GEMM, general matrix multiply. The rows of C are what gets added to the matching rows of the
          product.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Chỉ A × B là tốn kém. Đó là phần duy nhất sách tối ưu hóa, và là phần duy nhất các lab của chương này tính.
          </p>
        }
      >
        <p>
          Only A × B is expensive. It is the only part the book optimises, and the only part this chapter's labs compute.
        </p>
      </Translated>
      <Source section="15.1" pages="350" />

      <h2>
        <Translated vi="Mỗi ma trận được lưu theo chiều nào">Which way each matrix is stored</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một ma trận trong bộ nhớ có thể theo thứ tự hàng (row-major) hoặc thứ tự cột (column-major) tùy vào nguồn gốc của nó,
            chẳng hạn mã làm việc trên ma trận hay trên chuyển vị của nó (chương 6). Chương này giả định cả hai đầu vào đều theo thứ
            tự hàng. Các kỹ thuật của nó áp dụng được cho các tổ hợp khác, và các thư viện GEMM chấp nhận tất cả.
          </p>
        }
      >
        <p>
          A matrix in memory can be row-major or column-major depending on where it came from, for instance whether the
          code works on the matrix or on its transpose (chapter 6). The chapter assumes both inputs are row-major. Its
          techniques carry over to the other combinations, and GEMM libraries accept all of them.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Kích thước quyết định tối ưu hóa nào có lợi. Trước thời học sâu, phần lớn các tích trong ứng dụng đều khiêm tốn, vì các
            vector cần biến đổi thì ngắn. Chương 19 và 20 viết các lớp tích chập (convolution) và các lớp attention thành tích của
            những ma trận lớn, và phần lớn chương này nhắm vào trường hợp lớn đó.
          </p>
        }
      >
        <p>
          Size decides which optimisations pay. Before deep learning, most products in applications were modest, because
          the vectors being transformed were short. Chapters 19 and 20 write convolution layers and attention layers as
          products of large matrices, and much of this chapter is aimed at that large case.
        </p>
      </Translated>
      <Source section="15.1" pages="350" />

      <h2>
        <Translated vi="Bên dưới nn.Linear">Underneath nn.Linear</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn đã biết GEMM phổ biến nhất trong một mô hình. <code>nn.Linear</code> tính y = x Wᵀ + b, với x có hình dạng (batch,
            in) và trọng số W được lưu dạng (out_features, in_features). Theo cách gọi của sách: A = x, batch × in, thứ tự hàng,
            nên M = batch và K = in. B = Wᵀ, in × out, nên N = out; trong bộ nhớ W là out hàng, mỗi hàng in float, chính là B của
            sách lưu theo thứ tự cột. C là bias b lặp lại trên mọi hàng, và α = β = 1.
          </p>
        }
      >
        <p>
          You already know the most common GEMM in a model. <code>nn.Linear</code> computes y = x Wᵀ + b, with x of shape
          (batch, in) and the weight W stored as (out_features, in_features). In the book's terms: A = x, batch × in,
          row-major, so M = batch and K = in. B = Wᵀ, in × out, so N = out; in memory W is out rows of in floats, which is
          the book's B stored column-major. C is the bias b repeated on every row, and α = β = 1.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cách ánh xạ là của khóa học này. Các hình dạng là của PyTorch; ggml lưu trọng số theo cùng cách (ch03-facts D2, D6).
            </p>
          }
        >
          <p>
            The mapping is ours. The shapes are PyTorch's; ggml stores weights the same way (ch03-facts D2, D6).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Lab 6.4 (mốc dự án M2) xử lý cách bố trí này bằng cách bẻ góc trong bộ nhớ chia sẻ (shared memory). Lab 15.6 (mốc dự án
            M3) xử lý nó cho kernel nhanh mà chương này xây (<a href={href('ch15', 'labs')}>lab</a>).
          </p>
        }
      >
        <p>
          Lab 6.4 (milestone M2) took this layout by turning the corner in shared memory. Lab 15.6 (milestone M3) takes it
          for the fast kernel this chapter builds (<a href={href('ch15', 'labs')}>labs</a>).
        </p>
      </Translated>
      <Ask q={questionsS151['ch15/15.1/q-linear']} />

      <h2>
        <Translated vi="Một tích, ba cách viết">One product, three spellings</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Các thư viện mà bạn sẽ đo để so sánh không dùng chung một cách bố trí. cuBLAS và hipBLAS theo thứ tự cột; MPSMatrix của
            Apple theo thứ tự hàng. Một chương trình thứ tự hàng gọi một thư viện thứ tự cột không sao chép gì cả. Nó mô tả các
            mảng của chính nó theo cách thư viện sẽ đọc: một mảng M × N theo thứ tự hàng, đọc theo thứ tự cột, là chuyển vị N × M.
            Vì vậy thay vì C = A B, nó yêu cầu Cᵀ = Bᵀ Aᵀ, với B được truyền trước. MPS không cần mẹo nào, chỉ cần một cờ cho toán
            hạng được lưu theo chiều kia.
          </p>
        }
      >
        <p>
          The libraries you will measure against do not share a layout. cuBLAS and hipBLAS are column-major; Apple's
          MPSMatrix is row-major. A row-major program calling a column-major library copies nothing. It describes its own
          arrays the way the library will read them: a row-major M × N array, read column-major, is the N × M transpose. So
          instead of C = A B it asks for Cᵀ = Bᵀ Aᵀ, with B passed first. MPS needs no trick, only a flag for an operand
          stored the other way.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 against the cuBLAS 13.5 and rocBLAS docs, hipblas.h at ROCm 10.2 and the macOS 27.0 SDK headers (ch03-facts A7, B6, C7; ch15-facts A10, B6, C6)',
          '2026-11-20, đối chiếu với tài liệu cuBLAS 13.5 và rocBLAS, hipblas.h ở ROCm 10.2 và các header của SDK macOS 27.0 (ch03-facts A7, B6, C7; ch15-facts A10, B6, C6)',
        )}
      >
        <Translated
          vi={
            <p>
              cuBLAS theo thứ tự cột "for maximum compatibility with existing Fortran environments" (để tương thích tối đa với các
              môi trường Fortran hiện có); rocBLAS "the same as legacy BLAS and cuBLAS" (giống BLAS cũ và cuBLAS);{' '}
              <code>hipblasSgemm</code> có cùng các đối số với <code>cublasSgemm</code>. Dữ liệu MPSMatrix "is assumed to be stored
              in row-major order" (được giả định lưu theo thứ tự hàng), và <code>rowBytes</code> phải là bội số của kích thước phần
              tử.
            </p>
          }
        >
          <p>
            cuBLAS is column-major "for maximum compatibility with existing Fortran environments"; rocBLAS is "the same as
            legacy BLAS and cuBLAS"; <code>hipblasSgemm</code> has the same arguments as <code>cublasSgemm</code>.
            MPSMatrix data "is assumed to be stored in row-major order", and <code>rowBytes</code> must be a multiple of the
            element size.
          </p>
        </Translated>
      </Today>
      <SpellingsBench />
      <Translated
        vi={
          <p>
            Ba điều cần tìm. So sánh <code>nn.Linear</code> với x @ W: khác biệt duy nhất là transa, T thay vì N, và transposeRight
            ở phía MPS. So sánh Q Kᵀ với <code>nn.Linear</code>: cùng một lời gọi, vì K được lưu như một trọng số. Rồi bật một chỉnh
            sửa bất kỳ. Một đối số sai không làm sập gì cả: nó cho ra những con số, những con số sai, hoặc đọc quá cuối một buffer. Chỉ việc so sánh với một kết quả tham chiếu mới bắt được nó, đó là lý do mọi lab đều kiểm tra với một kết quả
            tham chiếu trên CPU.
          </p>
        }
      >
        <p>
          Three things to find. Compare <code>nn.Linear</code> with x @ W: the only difference is transa, T instead of N, and
          transposeRight on the MPS side. Compare Q Kᵀ with <code>nn.Linear</code>: the same call, because K is stored like
          a weight. Then switch on any edit. A wrong argument does not crash anything: it gives numbers, wrong ones, or
          reads past the end of a buffer. Only a comparison with a reference catches it, which is why every lab checks
          against one on the CPU.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Trình khám phá thực thi mỗi lời gọi đúng như thư viện định nghĩa, với α = 1 và β = 0, trên các số nguyên nhỏ. Nó không
              tái tạo các phép kiểm tra đối số hay mã lỗi của thư viện, và nó tính chính xác thay vì bằng FP32.
            </p>
          }
        >
          <p>
            The explorer executes each call as the library defines it, with α = 1 and β = 0, on small integers. It does not
            reproduce a library's checks on its arguments or its error codes, and it computes exactly rather than in FP32.
          </p>
        </Translated>
      </Simplification>
      <Ask q={questionsS151['ch15/15.1/q-rowmajor-trick']} />
      <Ask q={questionsS151['ch15/15.1/q-qkt']} />

      <h2>
        <Translated vi="Mốc so sánh tính gì">What the baseline computes</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Các lab 15.1 đến 15.5 đo kernel của bạn bên cạnh kernel của nhà cung cấp, với lời gọi từ hàng đầu tiên của trình khám
            phá, tức tích của sách (nó nằm trong <code>labs/ch15/shared/vendor_blas.cu</code>, và được kiểm tra với kết quả tham
            chiếu trên CPU trước khi đo bất cứ thứ gì). Lab 15.6 làm điều tương tự cho lớp tuyến tính. Phép so sánh chỉ công bằng
            nếu cả hai bên làm cùng một phép tính, và mặc định thì đúng vậy. Một lời gọi{' '}
            <code>cublasSgemm</code> thông thường giữ độ chính xác FP32: tensor core TF32 và chế độ giả lập BF16x9 là những chế độ
            bạn phải yêu cầu. hipBLAS có chế độ XF32 mà bạn phải yêu cầu. MPS tính với độ chính xác nội bộ cao hơn trừ khi bạn cho
            phép nó giảm xuống. Vì vậy khoảng cách bạn sẽ đo nằm ở kỹ thuật, không phải độ chính xác.
          </p>
        }
      >
        <p>
          Labs 15.1 to 15.5 time your kernel beside the vendor's, with the call from the explorer's first row, the book's
          product (it is in <code>labs/ch15/shared/vendor_blas.cu</code>, and checked against the CPU reference before
          anything is timed). Lab 15.6 does the same for the linear layer. The comparison is fair only if both sides do
          the same arithmetic, and by default they do. A plain{' '}
          <code>cublasSgemm</code> keeps FP32 precision: TF32 tensor cores and the BF16x9 emulation are modes you have to
          ask for. hipBLAS has an XF32 mode you have to ask for. MPS computes with higher internal precision unless you allow
          it to reduce it. So the gap you will measure is technique, not precision.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch15-facts A10, B6, C6, D4)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Chế độ toán mặc định của cuBLAS dùng "precisions with at least the same number of mantissa and exponent bits as
              requested" (độ chính xác có ít nhất cùng số bit phần định trị và số mũ như được yêu cầu);{' '}
              <code>NVIDIA_TF32_OVERRIDE=0</code> tắt TF32 ở mọi nơi; BF16x9 chỉ có trên compute capability 10.0 và 10.3. hipBLAS:{' '}
              <code>HIPBLAS_XF32_XDL_MATH</code> phải chủ động bật. MPS:{' '}
              <code>MPSKernelOptionsAllowReducedPrecision</code> phải chủ động bật. PyTorch gửi một phép matmul float đến cuBLASLt
              hoặc cuBLAS, và trên ROCm có thể gửi nó đến Composable Kernel.
            </p>
          }
        >
          <p>
            cuBLAS's default math mode uses "precisions with at least the same number of mantissa and exponent bits as
            requested"; <code>NVIDIA_TF32_OVERRIDE=0</code> turns TF32 off everywhere; BF16x9 exists on compute capability
            10.0 and 10.3 only. hipBLAS: <code>HIPBLAS_XF32_XDL_MATH</code> is opt-in. MPS:{' '}
            <code>MPSKernelOptionsAllowReducedPrecision</code> is opt-in. PyTorch sends a float matmul to cuBLASLt or
            cuBLAS, and on ROCm can send it to Composable Kernel.
          </p>
        </Translated>
      </Today>
    </>
  );
}
