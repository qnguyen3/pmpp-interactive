import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { BOOK_THRESHOLD, ELEMENT_BYTES, gridFetches, regime, tilePair } from '../../sim/gemmreuse';
import { ReuseScene } from './ReuseScene';

// ---------------------------------------------------------------- questions
export const questionsS152: Record<string, Question> = {
  'ch15/15.2/q-bk': {
    kind: 'choice',
    id: 'ch15/15.2/q-bk',
    prompt: (
      <Translated vi={<>Trong kernel 128 × 128, độ sâu ô bK tăng gấp đôi từ 8 lên 16. Điều gì thay đổi?</>}>
        <>In the 128 × 128 kernel the tile depth bK doubles from 8 to 16. What changes?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Cường độ tính toán không đổi; bộ nhớ chia sẻ cho một cặp ô tăng gấp đôi lên 16 KB; chỉ cần một nửa số cặp ô, và số rào chắn, để phủ K."}>{'Intensity is unchanged; one tile pair\'s shared memory doubles to 16 KB; half as many tile pairs, and barriers, cover K.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Cường độ tính toán tăng gấp đôi: mỗi phần tử được nạp giờ nuôi gấp đôi số phép nhân-cộng, trong khi bộ nhớ chia sẻ mỗi cặp ô vẫn ở 8 KB."}>{'Intensity doubles: each loaded element now feeds twice as many multiply-adds, while shared memory per tile pair stays at 8 KB.'}</Translated>,
        why: <Translated vi={"Số phép toán và số byte mỗi cặp ô đều tăng gấp đôi, nên tỉ số của chúng không đổi, và cặp ô giờ chiếm 16 KB."}>{'Operations and bytes per tile pair both double, so their ratio does not move, and the pair now takes 16 KB.'}</Translated>,
      },
      {
        label: <Translated vi={"Cường độ tính toán giảm một nửa, vì mỗi cặp ô giờ nạp gấp đôi số byte vào bộ nhớ chia sẻ."}>{'Intensity halves, because each tile pair now loads twice as many bytes into shared memory.'}</Translated>,
        why: <Translated vi={"Số phép toán và số byte mỗi cặp ô đều tăng gấp đôi, nên tỉ số của chúng không đổi."}>{'Operations and bytes per tile pair both double, so their ratio does not move.'}</Translated>,
      },
      {
        label: <Translated vi={"Không có gì đáng kể: cường độ tính toán, bộ nhớ chia sẻ mỗi cặp và số rào chắn đều giữ nguyên."}>{'Nothing that matters: intensity, shared memory per pair and the number of barriers all stay the same.'}</Translated>,
        why: <Translated vi={"Tỉ số thì không đổi, nhưng mỗi cặp giờ chiếm gấp đôi bộ nhớ chia sẻ, và chỉ cần một nửa số cặp để phủ K."}>{'The ratio does not, but each pair now takes twice the shared memory, and half as many pairs cover K.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một cặp ô là 4 × 8 × (128 + 128) = 8,192 byte ở bK = 8 và 16,384 ở bK = 16, và bộ nhớ chia sẻ phải chứa được nó. K được
            phủ bằng K / bK cặp, mỗi cặp có hai rào chắn của nó. Mục 15.7 dùng đúng điều này: vì bK không mua thêm được cường độ
            tính toán nào, sách giữ nó nhỏ để tiết kiệm bộ nhớ chia sẻ.
          </p>
        }
      >
        <p>
          One tile pair is 4 × 8 × (128 + 128) = 8,192 bytes at bK = 8 and 16,384 at bK = 16, and the shared memory has to
          hold it. K is covered in K / bK pairs, each with its two barriers. Section 15.7 uses exactly this: since bK buys no
          intensity, the book keeps it small to save shared memory.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.2, p. 351; §15.7, p. 363', // i18n-ignore: citation only
  },
  'ch15/15.2/q-rect': {
    kind: 'numeric',
    id: 'ch15/15.2/q-rect',
    prompt: (
      <Translated vi={<>Cường độ tính toán của một ô đầu ra 128 × 64 với phần tử 4 byte là bao nhiêu?</>}>
        <>What is the intensity of a 128 × 64 output tile with 4-byte elements?</>
      </Translated>
    ),
    answer: 21.33,
    tol: 0.02,
    unit: 'FLOP/B',
    explain: (
      <Translated
        vi={
          <p>
            0.5 × 128 × 64 / (128 + 64) = 4096 / 192 ≈ 21.3 FLOP/B: vừa trên ngưỡng 20 trong ví dụ của sách. Một hình chữ nhật nằm
            giữa hai hình vuông lân cận, 64 × 64 (16) và 128 × 128 (32), gần phía nhỏ hơn.
          </p>
        }
      >
        <p>
          0.5 × 128 × 64 / (128 + 64) = 4096 / 192 ≈ 21.3 FLOP/B: just above the book's example threshold of 20. A
          rectangle sits between its square neighbours, 64 × 64 (16) and 128 × 128 (32), closer to the smaller side.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.2, p. 351',
  },
  'ch15/15.2/q-decode': {
    kind: 'choice',
    id: 'ch15/15.2/q-decode',
    prompt: (
      <Translated
        vi={
          <>
            Ở batch kích thước 1, tích của một lớp tuyến tính có m = 1: mỗi ô một hàng đầu ra. Thay đổi nào nâng cường độ tính toán
            của nó?
          </>
        }
      >
        <>
          At batch size 1, a linear layer's product has m = 1: one row of output per tile. Which change raises its
          intensity?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi={"Gộp thêm nhiều hàng vào batch, hoặc ít byte hơn cho mỗi trọng số."}>{'Batching more rows, or fewer bytes per weight.'}</Translated>, correct: true },
      {
        label: <Translated vi={"Một ô đầu ra rộng hơn, để một khối phủ được nhiều đầu ra của lớp hơn."}>{'A wider output tile, so that one block covers more of the layer\'s outputs.'}</Translated>,
        why: <Translated vi={"Với m = 1, cường độ tính toán là 0.5 n / (n + 1), dưới 0.5 với mọi n. Mở rộng chỉ tiến gần 0.5 và không bao giờ vượt qua."}>{'With m = 1 the intensity is 0.5 n / (n + 1), below 0.5 for every n. Widening approaches 0.5 and never passes it.'}</Translated>,
      },
      {
        label: <Translated vi={"Một ô sâu hơn, bK lớn hơn, để mỗi lần nạp mang về nhiều phần hơn của chiều trong."}>{'A deeper tile, larger bK, so that each load brings in more of the inner dimension.'}</Translated>,
        why: <Translated vi={"k triệt tiêu: nó thay đổi số phép toán và số byte cùng một hệ số."}>{'k cancels: it changes operations and bytes by the same factor.'}</Translated>,
      },
      {
        label: <Translated vi={"Nhiều luồng hơn mỗi khối, để có nhiều phép nhân-cộng đang chạy cùng lúc hơn."}>{'More threads per block, so that more multiply-adds are in flight at once.'}</Translated>,
        why: <Translated vi={"Luồng thay đổi ai làm việc, không thay đổi mỗi phép toán cần bao nhiêu byte."}>{'Threads change who does the work, not how many bytes each operation needs.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một hàng đầu ra dùng mỗi trọng số đúng một lần: hai phép toán cho mỗi trọng số được nạp, 0.5 FLOP/B với float32, bất
            kể kernel nào. Batching khiến một trọng số đã nạp phục vụ nhiều hàng hơn (m tăng); lượng tử hóa khiến mỗi trọng số tốn
            ít byte hơn. llama.cpp chuyển sang kernel ma trận-vector cho batch đến 8.
          </p>
        }
      >
        <p>
          One row of output uses each weight once: two operations per weight fetched, 0.5 FLOP/B in float32, whatever the
          kernel. Batching makes a fetched weight serve more rows (m grows); quantisation makes each weight fewer bytes.
          llama.cpp switches to matrix-vector kernels for batches up to 8.
        </p>
      </Translated>
    ),
    source: 'our derivation from the formula on p. 351; llama.cpp from ch15-facts D1',
    vi: { source: 'suy luận của khóa học này từ công thức ở p. 351; llama.cpp theo ch15-facts D1' },
  },
  'ch15/15.2/q-batch59': {
    kind: 'numeric',
    id: 'ch15/15.2/q-batch59',
    prompt: (
      <Translated
        vi={
          <>
            Các ô đầu ra rộng 128 cột, và chiều cao của chúng là kích thước batch b (b ≤ 128), với float32. Giá trị b nhỏ nhất đạt
            ngưỡng 20 FLOP/B trong ví dụ của sách là bao nhiêu?
          </>
        }
      >
        <>
          Output tiles are 128 columns wide, and their height is the batch size b (b ≤ 128), in float32. What is the
          smallest b that reaches the book's example threshold of 20 FLOP/B?
        </>
      </Translated>
    ),
    answer: 59,
    unit: 'rows',
    vi: { unit: 'hàng', source: 'phép tính của khóa học này với công thức ở p. 351' },
    explain: (
      <Translated
        vi={
          <p>
            0.5 × 128 b / (b + 128) = 64 b / (b + 128) ≥ 20 cho 44 b ≥ 2560, nên b ≥ 58.2: 59. Kiểm tra trong trình khám phá: 58 ×
            128 cho 19.96 và 59 × 128 cho 20.19.
          </p>
        }
      >
        <p>
          0.5 × 128 b / (b + 128) = 64 b / (b + 128) ≥ 20 gives 44 b ≥ 2560, so b ≥ 58.2: 59. Check it in the explorer:
          58 × 128 gives 19.96 and 59 × 128 gives 20.19.
        </p>
      </Translated>
    ),
    source: 'our arithmetic with the formula on p. 351',
  },
  'ch15/15.2/q-fetches': {
    kind: 'numeric',
    id: 'ch15/15.2/q-fetches',
    prompt: (
      <Translated
        vi={
          <>
            M = N = K = 4096, tính với các ô đầu ra 128 × 128. Trong mô hình (không có cache), mỗi phần tử của A được nạp từ bộ nhớ
            toàn cục bao nhiêu lần?
          </>
        }
      >
        <>
          M = N = K = 4096, computed with 128 × 128 output tiles. In the model (no caches), how many times is each element
          of A fetched from global memory?
        </>
      </Translated>
    ),
    answer: 32,
    unit: 'times',
    vi: { unit: 'lần', source: 'ch. 15 §15.2, pp. 351–352; phép đếm là của khóa học này' },
    explain: (
      <Translated
        vi={
          <p>
            Mọi khối trong một hàng của C đều đọc dải hàng đó của A, và một hàng của C chứa ⌈N / bN⌉ = 4096 / 128 = 32 khối. Với ô
            rộng 32, con số sẽ là 128 lần: lưu lượng gấp bốn cho cùng số phép toán.
          </p>
        }
      >
        <p>
          Every block in a row of C reads that row band of A, and a row of C holds ⌈N / bN⌉ = 4096 / 128 = 32 blocks.
          With 32-wide tiles it would be 128 times: four times the traffic for the same operations.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.2, pp. 351–352; the count is ours',
  },
};

// ---------------------------------------------------------------- thresholds
interface Threshold {
  key: string;
  label: string;
  value: number;
  note: string;
}
const BOOK: Threshold = {
  key: 'book',
  get label() {
    return tr("The book's example (H100, from chapter 5)", 'Ví dụ của sách (H100, từ chương 5)');
  },
  value: BOOK_THRESHOLD,
  get note() {
    return tr("The book's example device: about 20 FLOP/B (chapter 5).", 'Thiết bị ví dụ của sách: khoảng 20 FLOP/B (chương 5).');
  },
};

function measuredThresholds(reports: Reports): Threshold[] {
  return realBackends(reports).flatMap((b) => {
    const r = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
    const peak = num(r?.compute_gflops);
    const bw = num(r?.bandwidth_gb_per_s);
    if (!peak || !bw) return [];
    return [{
      key: b,
      // Getters: this list is memoised on the reports alone, so the text is read when it renders.
      get label() {
        return tr(`Yours: ${String(reports[b].device.name ?? b)} (${b})`, `Của bạn: ${String(reports[b].device.name ?? b)} (${b})`);
      },
      value: peak / bw,
      get note() {
        return tr(
          `Measured by lab 5.1 on ${reports[b].generated.slice(0, 10)}: ${fmt(peak)} GFLOP/s ÷ ${fmt(bw)} GB/s, two probe kernels, not vendor peaks.`,
          `Đo bởi lab 5.1 ngày ${reports[b].generated.slice(0, 10)}: ${fmt(peak)} GFLOP/s ÷ ${fmt(bw)} GB/s, hai kernel thăm dò, không phải số đỉnh của nhà cung cấp.`,
        );
      },
    }];
  });
}

function useThreshold() {
  const { reports } = useReports();
  const options = useMemo(() => [BOOK, ...measuredThresholds(reports)], [reports]);
  const [key, setKey] = useState('book');
  const [custom, setCustom] = useState(10);
  const th: Threshold = key === 'custom'
    ? { key, label: tr('Custom', 'Tùy chọn'), value: custom, note: tr('A number you typed.', 'Một con số bạn đã nhập.') }
    : (options.find((o) => o.key === key) ?? BOOK);
  const picker = (
    <label className="field">
      <Translated vi="Ngưỡng">Threshold</Translated>
      <select value={key} aria-label={tr('threshold', 'ngưỡng')} onChange={(e) => setKey(e.target.value)}>
        {options.map((o) => (
          <option key={o.key} value={o.key}>{o.label}</option>
        ))}
        <option value="custom">{tr('Custom number', 'Số tùy chọn')}</option>
      </select>
      {key === 'custom' ? (
        <input type="number" min="0.1" step="any" value={custom} aria-label={tr('custom threshold', 'ngưỡng tùy chọn')}
          onChange={(e) => setCustom(Math.max(0.1, +e.target.value || 0.1))} />
      ) : null}
    </label>
  );
  return { th, picker, measuredCount: options.length - 1 };
}

const RegimeChip = ({ fpb, threshold }: { fpb: number; threshold: number }) =>
  regime(fpb, threshold) === 'memory-bound' ? (
    <span className="chip global" data-regime="memory-bound">
      <Translated vi="bị giới hạn bởi bộ nhớ">memory-bound</Translated>
    </span>
  ) : (
    <span className="chip reg" data-regime="compute-bound">
      <Translated vi="bị giới hạn bởi tính toán">compute-bound</Translated>
    </span>
  );

const bytesLabel = (b: number) =>
  b === 4 ? tr('4 (float32)', '4 (float32)') : b === 2 ? tr('2 (16-bit)', '2 (16 bit)') : b === 1 ? tr('1 (8-bit)', '1 (8 bit)') : tr('0.5 (4-bit)', '0.5 (4 bit)');

// ---------------------------------------------------------------- explorer 1: one output tile
function TileDrawing({ m, n, k }: { m: number; n: number; k: number }) {
  // Fig. 15.1's L layout, to scale: A's tile left of C, B's tile above it.
  const s = 0.62; // px per element
  const pad = 8;
  const kw = Math.max(2, k * s);
  const W = pad + 64 * s + 6 + 256 * s + pad;
  const H = pad + 64 * s + 6 + 256 * s + pad;
  const cx = pad + 64 * s + 6;
  const cy = pad + 64 * s + 6;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('The tile pair and the output tile, to scale', 'Cặp ô và ô đầu ra, theo đúng tỉ lệ')} data-layout="narrow"
      style={{ width: '100%', maxWidth: 260, display: 'block' }}>
      <rect x={cx - 6 - kw} y={cy} width={kw} height={Math.max(2, m * s)} fill="var(--shared-wash)" stroke="var(--shared)" data-tile="A" />
      <rect x={cx} y={cy - 6 - kw} width={Math.max(2, n * s)} height={kw} fill="var(--shared-wash)" stroke="var(--shared)" data-tile="B" />
      <rect x={cx} y={cy} width={Math.max(2, n * s)} height={Math.max(2, m * s)} fill="var(--reg-wash)" stroke="var(--reg)" data-tile="C" />
      <text x={cx + 4} y={cy + Math.max(2, m * s) + 14} fontSize="11" style={{ fill: 'var(--reg)' }}><Translated vi={<>ô C {m} × {n}</>}>C tile {m} × {n}</Translated></text>
    </svg>
  );
}

const KCOVER = 4096;

function IntensityBench() {
  const { th, picker } = useThreshold();
  // Starts between the book's two examples, so reaching each of them is the reader's move.
  const [m, setM] = useState(64);
  const [n, setN] = useState(64);
  const [k, setK] = useState(8);
  const [bytes, setBytes] = useState<number>(4);
  // Checkpoints: which of the section's discoveries the reader has reached.
  const [kSeen, setKSeen] = useState<{ key: string; ks: number[] }>({ key: '64,64,4', ks: [8] });
  const [hit, setHit] = useState<Record<string, boolean>>({});
  const t = tilePair(m, n, k, bytes);
  const shapeKey = `${m},${n},${bytes}`;

  const note = (nm: number, nn: number, nk: number, nb: number) => {
    const key = `${nm},${nn},${nb}`;
    const ks = key === kSeen.key ? (kSeen.ks.includes(nk) ? kSeen.ks : [...kSeen.ks, nk]) : [nk];
    setKSeen({ key, ks });
    setHit((h) => ({
      ...h,
      eight: h.eight || (nm === 32 && nn === 32 && nb === 4),
      thirtytwo: h.thirtytwo || (nm === 128 && nn === 128 && nb === 4),
      kfree: h.kfree || ks.length >= 3,
      one: h.one || nm === 1,
    }));
  };
  const set = (what: 'm' | 'n' | 'k', v: number) => {
    const nm = what === 'm' ? v : m;
    const nn = what === 'n' ? v : n;
    const nk = what === 'k' ? v : k;
    if (what === 'm') setM(v);
    if (what === 'n') setN(v);
    if (what === 'k') setK(v);
    note(nm, nn, nk, bytes);
  };
  const setB = (b: number) => {
    setBytes(b);
    note(m, n, k, b);
  };
  const checks: [string, string][] = [
    ['eight', tr('Reproduce the book\'s 8 FLOP/B (32 × 32, float32).', 'Tái tạo con số 8 FLOP/B của sách (32 × 32, float32).')],
    ['thirtytwo', tr('Reproduce the book\'s 32 FLOP/B (128 × 128, float32).', 'Tái tạo con số 32 FLOP/B của sách (128 × 128, float32).')],
    ['kfree', tr('Try three depths k for the same tile, and watch the intensity.', 'Thử ba độ sâu k cho cùng một ô, và quan sát cường độ tính toán.')],
    ['one', tr('Set m = 1: one row of output, as in decoding one token.', 'Đặt m = 1: một hàng đầu ra, như khi decode một token.')],
  ];
  const range = (label: string, what: 'm' | 'n' | 'k', v: number, max: number) => (
    <label className="field">
      {label} = {v}
      <input type="range" min="1" max={max} value={v} aria-label={what} onChange={(e) => set(what, +e.target.value)} />
    </label>
  );
  return (
    <Bench
      title={tr('Intensity of an output tile', 'Cường độ tính toán của một ô đầu ra')}
      task={tr("Reproduce the book's two numbers, then find what k does.", 'Tái tạo hai con số của sách, rồi tìm xem k làm gì.')}
      foot={
        <Translated
          vi={
            <>
              Mô hình của §15.2: chỉ các lần nạp ô từ bộ nhớ toàn cục được tính, lần ghi duy nhất của ô đầu ra thì không, và không có
              cache. Thiết lập số byte thu nhỏ cả hai ô đầu vào; lượng tử hóa chỉ trọng số thì chỉ thu nhỏ ô của trọng số, mà trong
              một lớp tuyến tính là phía n. Hình vẽ đúng tỉ lệ, trừ các ô mỏng hơn 2 pixel được vẽ dày 2. {th.note}
            </>
          }
        >
          <>
            The model of §15.2: only the tile loads from global memory are counted, the output tile's single write is not,
            and there are no caches. The bytes setting shrinks both input tiles; weight-only quantisation shrinks only the
            weights' tile, which in a linear layer is the n side. Drawing to scale except that tiles thinner than 2 pixels
            are drawn at 2. {th.note}
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        {range(tr('m, output tile height', 'm, chiều cao ô đầu ra'), 'm', m, 256)}
        {range(tr('n, output tile width', 'n, chiều rộng ô đầu ra'), 'n', n, 256)}
        {range(tr('k, tile depth', 'k, độ sâu ô'), 'k', k, 64)}
        <label className="field">
          <Translated vi="Số byte mỗi phần tử">Bytes per element</Translated>
          <span className="seg">
            {ELEMENT_BYTES.map((b) => (
              <button key={b} aria-pressed={bytes === b} onClick={() => setB(b)}>{bytesLabel(b)}</button>
            ))}
          </span>
        </label>
        {picker}
      </div>
      <div className="cols">
        <div className="panel">
          <TileDrawing m={m} n={n} k={k} />
          <ul style={{ margin: '0.6rem 0 0', paddingLeft: '1.1rem' }} data-checkpoints>
            {checks.map(([id, text]) => (
              <li key={id} data-check={id} data-done={hit[id] ? 1 : 0} style={{ color: hit[id] ? 'var(--ok)' : undefined }}>
                {hit[id] ? '✓ ' : ''}{text}
              </li>
            ))}
          </ul>
        </div>
        <div className="panel">
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <tr><td>{tr('Operations per tile pair, 2·m·n·k', 'Số phép toán mỗi cặp ô, 2·m·n·k')}</td><td className="num" data-out="ops">{fmt(t.ops, 6)}</td></tr>
              <tr><td><Translated vi={<>Số byte mỗi cặp ô, {bytes}·k·(m + n)</>}>Bytes per tile pair, {bytes}·k·(m + n)</Translated></td><td className="num" data-out="bytes">{fmt(t.traffic, 6)}</td></tr>
              <tr><td>{tr('Shared memory to hold the pair', 'Bộ nhớ chia sẻ để chứa cặp ô')}</td><td className="num" data-out="smem">{fmt(t.traffic / 1024, 4)} KB</td></tr>
              <tr><td><Translated vi={<>Số cặp ô để phủ K = {KCOVER}</>}>Tile pairs to cover K = {KCOVER}</Translated></td><td className="num" data-out="pairs">{Math.ceil(KCOVER / k)}</td></tr>
            </tbody>
          </table>
          <p style={{ fontSize: '1.6rem', margin: '0.7rem 0 0.2rem', fontWeight: 600 }} data-out="intensity" data-value={t.intensity}>
            {fmt(t.intensity, 4)} FLOP/B
          </p>
          <p className="muted" style={{ margin: 0 }} data-shape={shapeKey}>
            <Translated vi={<><RegimeChip fpb={t.intensity} threshold={th.value} /> so với ngưỡng {fmt(th.value)} FLOP/B.</>}>
              <RegimeChip fpb={t.intensity} threshold={th.value} /> against a threshold of {fmt(th.value)} FLOP/B.
            </Translated>
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer 2: the whole grid
const GRID_PRESETS: { label: string; M: number; N: number; K: number }[] = [
  // Getters: the labels are read in the selected language when the buttons render.
  { get label() { return tr('square, 4096', 'vuông, 4096'); }, M: 4096, N: 4096, K: 4096 },
  { get label() { return tr('1024 → 4096 layer, batch 1', 'lớp 1024 → 4096, batch 1'); }, M: 1, N: 4096, K: 1024 },
  { get label() { return tr('4096 → 4096 layer, batch 8', 'lớp 4096 → 4096, batch 8'); }, M: 8, N: 4096, K: 4096 },
  { get label() { return tr('1024 → 4096 layer, batch 256', 'lớp 1024 → 4096, batch 256'); }, M: 256, N: 4096, K: 1024 },
];
const TILES = [32, 64, 128, 256];

function GridBench() {
  const [M, setM] = useState(4096);
  const [N, setN] = useState(4096);
  const [K, setK] = useState(4096);
  const [bM, setBM] = useState(128);
  const [bN, setBN] = useState(128);
  const g = gridFetches(M, N, K, bM, bN);
  const numIn = (label: string, v: number, set: (x: number) => void, key: string) => (
    <label className="field">
      {label}
      <input type="number" min="1" max="65536" value={v} aria-label={key}
        onChange={(e) => set(Math.max(1, Math.min(65536, Math.round(+e.target.value) || 1)))} />
    </label>
  );
  const tileSel = (label: string, v: number, set: (x: number) => void, key: string) => (
    <label className="field">
      {label}
      <select value={v} aria-label={key} onChange={(e) => set(+e.target.value)}>
        {TILES.map((t) => <option key={t} value={t}>{t}</option>)}
      </select>
    </label>
  );
  const W = 260;
  return (
    <Bench
      title={tr('Fetches across the grid', 'Số lần nạp trên toàn lưới')}
      task={tr('Find how many times each input element is fetched, then shrink the batch.', 'Tìm xem mỗi phần tử đầu vào được nạp bao nhiêu lần, rồi thu nhỏ batch.')}
      foot={tr(
        'Counts follow from the tile layout alone: every block reads its whole row band of A and column band of B. No caches, output writes not counted. 1 MB = 10⁶ bytes.',
        'Các con số chỉ suy ra từ cách bố trí ô: mọi khối đọc toàn bộ dải hàng của A và dải cột của B. Không có cache, không tính các lần ghi đầu ra. 1 MB = 10⁶ byte.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        {numIn(tr('M (rows of A and C)', 'M (số hàng của A và C)'), M, setM, tr('grid M', 'M của lưới'))}
        {numIn(tr('N (columns of B and C)', 'N (số cột của B và C)'), N, setN, tr('grid N', 'N của lưới'))}
        {numIn(tr('K (inner)', 'K (chiều trong)'), K, setK, tr('grid K', 'K của lưới'))}
        {tileSel('bM', bM, setBM, 'bM')}
        {tileSel('bN', bN, setBN, 'bN')}
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', gap: '0.4rem' }}>
        <span className="ui muted">
          <Translated vi="Thiết lập sẵn:">Presets:</Translated>
        </span>
        {GRID_PRESETS.map((p) => (
          <button key={p.label} className="quiet" onClick={() => { setM(p.M); setN(p.N); setK(p.K); }}>{p.label}</button>
        ))}
      </div>
      <div className="cols">
        <div className="panel">
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <tr><td>{tr('Blocks', 'Số khối')}</td><td className="num" data-out="blocks">{g.blocksM} × {g.blocksN}</td></tr>
              <tr><td>{tr('Fetches of each element of A, ⌈N / bN⌉', 'Số lần nạp mỗi phần tử của A, ⌈N / bN⌉')}</td><td className="num" data-out="afetch">{g.aFetches}</td></tr>
              <tr><td>{tr('Fetches of each element of B, ⌈M / bM⌉', 'Số lần nạp mỗi phần tử của B, ⌈M / bM⌉')}</td><td className="num" data-out="bfetch">{g.bFetches}</td></tr>
              <tr><td>{tr('Bytes requested (float32)', 'Số byte được yêu cầu (float32)')}</td><td className="num" data-out="requested">{fmt(g.requested / 1e6, 4)} MB</td></tr>
              <tr><td>{tr('Bytes if each element came once', 'Số byte nếu mỗi phần tử chỉ đến một lần')}</td><td className="num" data-out="ideal">{fmt(g.ideal / 1e6, 4)} MB</td></tr>
            </tbody>
          </table>
        </div>
        <div className="panel">
          <svg viewBox={`0 0 ${W + 70} 70`} role="img" aria-label={tr('Requested against needed bytes', 'Số byte được yêu cầu so với số byte cần')} style={{ width: '100%', maxWidth: 360, display: 'block' }}>
            <text x="0" y="12" fontSize="11">{tr('requested', 'được yêu cầu')}</text>
            <rect x="0" y="16" width={W} height="14" fill="var(--global)" />
            <text x="0" y="48" fontSize="11">{tr('needed', 'cần')}</text>
            <rect x="0" y="52" width={Math.max(1, (W * g.ideal) / g.requested)} height="14" fill="var(--shared)" />
            <text x={W + 6} y="28" fontSize="11">× {fmt(g.requested / g.ideal)}</text>
          </svg>
          <p style={{ fontSize: '1.4rem', margin: '0.6rem 0 0.2rem', fontWeight: 600 }} data-out="grid-intensity" data-value={g.intensity}>
            {fmt(g.intensity, 4)} FLOP/B
          </p>
          <p className="muted" style={{ margin: 0 }}>
            <Translated vi="2·M·N·K phép toán chia cho số byte được yêu cầu: cường độ tính toán của cả kernel trong mô hình này.">
              2·M·N·K operations over the bytes requested: the whole kernel's intensity in this model.
            </Translated>
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the page
function GatedExplorer() {
  const { progress } = useProgress();
  const committed = typeof progress.answers['ch15/15.2/p-intensity128']?.value === 'number';
  return committed ? (
    <IntensityBench />
  ) : (
    <Bench
      title={tr('Intensity of an output tile', 'Cường độ tính toán của một ô đầu ra')}
      task={tr('Opens once you have committed a prediction above.', 'Mở ra khi bạn đã chốt một dự đoán ở trên.')}
    >
      <p style={{ margin: 0 }} data-gated>
        <Translated vi="Hãy chốt dự đoán của bạn cho ô 128 × 128 trước; trình khám phá sẽ để lộ đáp án.">
          Commit your prediction for the 128 × 128 tile first; the explorer would give it away.
        </Translated>
      </p>
    </Bench>
  );
}

export function S152() {
  return (
    <>
      <Translated
        vi={
          <p>
            Các ô của chương 5 là hình vuông, mỗi luồng một đầu ra, và phép đếm tái sử dụng của nó giả định ô vuông. Để đi xa hơn,
            bạn cần phép đếm cho ô có hình dạng bất kỳ. Hóa ra chỉ có một điều về ô là quan trọng.
          </p>
        }
      >
        <p>
          Chapter 5's tiles were square, with one output per thread, and its count of reuse assumed square tiles. To go
          further you need the count for any tile shape. It turns out that only one thing about the tile matters.
        </p>
      </Translated>

      <h2>
        <Translated vi="Bối cảnh">The setting</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một khối luồng (thread block) sở hữu một ô m × n của C. Ô đó nằm trong thanh ghi (register) của khối, trải trên các
            luồng của nó, từ bước đầu tiên đến bước cuối cùng. Để lấp đầy nó, khối đi dọc chiều trong K theo từng bước k. Ở mỗi bước,
            nó nạp một ô m × k của A và ô k × n tương ứng của B từ bộ nhớ toàn cục (global memory) vào bộ nhớ chia sẻ (shared
            memory), và mỗi luồng cộng phần đóng góp của cặp đó vào các đầu ra của mình. Ô đầu ra được ghi ra bộ nhớ toàn cục một
            lần, ở cuối.
          </p>
        }
      >
        <p>
          One thread block owns an m × n tile of C. That tile lives in the block's registers, spread over its threads, from
          the first step to the last. To fill it, the block walks along the inner dimension K in steps of k. At each step it
          loads an m × k tile of A and the matching k × n tile of B from global memory into shared memory, and every thread
          adds that pair's contribution to its outputs. The output tile goes to global memory once, at the end.
        </p>
      </Translated>
      <Source section="15.2" pages="350–351" figs={tr('Fig. 15.1', 'Hình 15.1')} />

      <h2>
        <Translated vi="Đếm nó">Count it</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Với một cặp ô: mỗi đầu ra trong m·n đầu ra nhận k phép nhân và k phép cộng, nên có 2mnk phép toán. Khối nạp m·k + k·n
            float, tức 4k(m + n) byte. Chia cái này cho cái kia và k xuất hiện ở cả hai vế:
          </p>
        }
      >
        <p>
          For one tile pair: each of the m·n outputs gets k multiplications and k additions, so 2mnk operations. The block
          loads m·k + k·n floats, which is 4k(m + n) bytes. Divide one by the other and k appears on both sides:
        </p>
      </Translated>
      <p style={{ textAlign: 'center', fontSize: '1.1rem' }}>
        2mnk / 4k(m + n) = <b>0.5 · m · n / (m + n)</b> FLOP/B
      </p>
      <Translated
        vi={
          <p>
            Độ sâu của một bước triệt tiêu. Chỉ còn lại chiều cao và chiều rộng của ô đầu ra, và tỉ số tăng theo cả hai.
          </p>
        }
      >
        <p>
          The depth of a step cancels. Only the output tile's height and width are left, and the ratio grows with both.
        </p>
      </Translated>
      <Source section="15.2" pages="351" />
      <Predict
        id="ch15/15.2/p-intensity128"
        prompt={tr('What is the intensity of a 128 × 128 output tile with 4-byte elements?', 'Cường độ tính toán của một ô đầu ra 128 × 128 với phần tử 4 byte là bao nhiêu?')}
        unit="FLOP/B"
      >
        {(v) => (
          <p>
            0.5 × 16,384 / 256 = <b>32 FLOP/B</b>.{' '}{/* i18n-ignore: number and unit */}
            {Math.abs(v - 32) < 0.5
              ? tr('That is what you had.', 'Đó đúng là con số bạn đã có.')
              : v > 32
                ? tr('If you were higher, check the denominator: it is m + n, not m or n alone.', 'Nếu bạn đoán cao hơn, hãy kiểm tra mẫu số: nó là m + n, không phải chỉ m hay n.')
                : tr('If you were lower, check that the 4 bytes appear once: 2mnk over 4k(m + n).', 'Nếu bạn đoán thấp hơn, hãy kiểm tra rằng 4 byte chỉ xuất hiện một lần: 2mnk chia cho 4k(m + n).')}{' '}
            <Translated vi="Gấp bốn lần 8 FLOP/B của ô 32 × 32 ở chương 5: ô có số đầu ra gấp mười sáu lần, trong khi mỗi bước chỉ nạp gấp bốn lần số byte.">
              Four times the 8 FLOP/B of chapter 5's 32 × 32 tiles: the tile has sixteen times the outputs, while each step
              loads only four times the bytes.
            </Translated>
          </p>
        )}
      </Predict>
      <GatedExplorer />

      <h2>
        <Translated vi="So với một ngưỡng">Against a threshold</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương 5 vạch ranh giới giữa một kernel đói dữ liệu và một kernel giữ cho phần tính toán luôn bận: số phép toán đỉnh mỗi
            giây của thiết bị chia cho số byte đỉnh mỗi giây của nó. Trên thiết bị ví dụ của sách, ranh giới đó nằm ở 20 FLOP/B. Ô
            32 × 32 của chương 5 cho 8 và vẫn bị giới hạn bởi bộ nhớ ở đó; ô 128 × 128 cho 32 và bị giới hạn bởi tính toán.
          </p>
        }
      >
        <p>
          Chapter 5 drew the line between a kernel that starves for data and one that keeps its arithmetic busy: the
          device's peak operations per second divided by its peak bytes per second. On the book's example device that line
          is at 20 FLOP/B. Chapter 5's 32 × 32 tiles give 8 and stay memory-bound there; 128 × 128 tiles give 32 and are
          compute-bound.
        </p>
      </Translated>
      <Source section="15.2" pages="351" />
      <Translated
        vi={
          <p>
            Trên thiết bị của bạn, ranh giới nằm ở chỗ khác. Nếu bạn đã chạy lab 5.1, hãy chọn thiết bị của bạn trong menu ngưỡng của
            trình khám phá (<a href={href('ch05', 'labs')}>lab chương 5</a>).
          </p>
        }
      >
        <p>
          On your own device the line is somewhere else. If you have run lab 5.1, pick your device in the explorer's
          threshold menu (<a href={href('ch05', 'labs')}>chapter 5 labs</a>).
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch15-facts A1; ch05-facts A8)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Thiết bị của sách là H100: NVIDIA công bố 67 TFLOPS FP32 và 3.35 TB/s (SXM), 20.0 FLOP/B. Con số 66.9 của sách không có
              trong nguồn nào của NVIDIA.
            </p>
          }
        >
          <p>
            The book's device is an H100: NVIDIA prints 67 TFLOPS FP32 and 3.35 TB/s (SXM), 20.0 FLOP/B. The book's 66.9
            is in no NVIDIA source.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS152['ch15/15.2/q-rect']} />
      <Ask q={questionsS152['ch15/15.2/q-bk']} />

      <h2>
        <Translated vi="Sự tái sử dụng đến từ đâu">Where the reuse comes from</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một ô đầu ra lớn hơn nâng cường độ tính toán vì các khối thôi nạp cùng một dữ liệu. Cắt C thành các ô rộng 32 và bốn
            khối lân cận dọc một hàng của C đều cần cùng một dải của A: mỗi khối tự nạp nó. Làm ô rộng 128 thì một khối làm việc của
            bốn khối kia, và chỉ nạp dải đó một lần. Làm ô cao hơn cũng có tác dụng tương tự với các dải của B.
          </p>
        }
      >
        <p>
          A bigger output tile raises intensity because blocks stop fetching the same data. Cut C into 32-wide tiles and
          the four neighbouring blocks along a row of C all need the same band of A: each fetches it on its own. Make the
          tile 128 wide and one block does what those four did, and fetches the band once. Making the tile taller does the
          same for the bands of B.
        </p>
      </Translated>
      <Source section="15.2" pages="351–352" />
      <ReuseScene />
      <Translated
        vi={
          <p>
            Trên cả ma trận, phép đếm rất đơn giản. Mỗi phần tử của A được nạp một lần bởi mỗi khối trong hàng của nó trong C, tức
            ⌈N / bN⌉ lần; mỗi phần tử của B được nạp một lần bởi mỗi khối trong cột của nó, tức ⌈M / bM⌉ lần.
          </p>
        }
      >
        <p>
          Over a whole matrix the count is simple. Every element of A is fetched once by each block in its row of C,
          ⌈N / bN⌉ times; every element of B once by each block in its column, ⌈M / bM⌉ times.
        </p>
      </Translated>
      <GridBench />
      <Translated
        vi={
          <p>
            Các lớp thiết lập sẵn cho thấy cùng điều đó từ phía bên kia: khi M là batch và batch nhỏ, các khối hết hàng, và mỗi trọng
            số được nạp chỉ cho rất ít việc.
          </p>
        }
      >
        <p>
          The preset layers show the same thing from the other side: when M is the batch and the batch is small, the
          blocks run out of rows, and every weight is fetched for very little work.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Chỉ các ô đầu vào nạp từ bộ nhớ toàn cục được tính; lần ghi duy nhất của đầu ra bị bỏ qua, như trong tỉ số của sách.
              Không có cache: một GPU thật phục vụ một số lần nạp lặp lại từ bộ nhớ đệm (cache), nên ô lớn hơn có thể được lợi ít hơn
              nhiều so với tỉ số của mô hình, như lab của chương 5 đã thấy trên chiếc máy dùng để xây khóa học này.
            </p>
          }
        >
          <p>
            Only input tiles loaded from global memory count; the output's single write is left out, as in the book's
            ratio. No caches: a real GPU serves some repeated fetches from cache, so bigger tiles can gain far less than the
            model's ratio, as chapter 5's lab found on the machine this course was built on.
          </p>
        </Translated>
      </Simplification>
      <Ask q={questionsS152['ch15/15.2/q-fetches']} />

      <h2>
        <Translated vi="Batch một">Batch one</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Giờ hãy đưa một lớp tuyến tính qua công thức. Trong một kernel phân ô cho y = x Wᵀ, một ô đầu ra không thể cao hơn
            batch. Khi sinh từng token một, m = 1, và cường độ tính toán là 0.5 n / (n + 1): dưới 0.5 FLOP/B dù ô rộng đến đâu.
            Không hình dạng ô nào cứu được nó. Hai thứ thì cứu được: nhiều hàng hơn cho mỗi trọng số được nạp, tức batching, và ít
            byte hơn cho mỗi trọng số, tức lượng tử hóa. Thiết lập số byte của trình khám phá thu nhỏ cả hai đầu vào, nên phần tử 2
            byte nhân đôi mọi cường độ tính toán. Lượng tử hóa chỉ trọng số chỉ thu nhỏ số hạng k·n của số byte. Ở m = 1, số hạng
            đó gần như là tất cả, nên mức lợi gần như như nhau; khi m tăng, số hạng k·m của activation, vẫn 4 byte mỗi phần tử,
            lấy lại một phần ngày càng lớn.
          </p>
        }
      >
        <p>
          Now put a linear layer through the formula. In a tiled kernel for y = x Wᵀ, an output tile can be no taller than
          the batch. Generating one token at a time, m = 1, and the intensity is 0.5 n / (n + 1): below 0.5 FLOP/B however
          wide the tile. No tile shape rescues it. Two things do: more rows per fetched weight, which is batching, and
          fewer bytes per weight, which is quantisation. The explorer's bytes setting shrinks both inputs, so 2-byte
          elements double every intensity. Quantising only the weights shrinks only the k·n term of the bytes. At m = 1
          that term is nearly all of them, so the gain is nearly the same; as m grows, the activations' k·m term, still
          4 bytes an element, takes back a growing share.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cận m = 1 là suy luận của khóa học này từ công thức của sách. llama.cpp dùng kernel ma trận-vector cho batch đến 8 (
              <code>MMVQ_MAX_BATCH_SIZE</code>, <code>MMVF_MAX_BATCH_SIZE</code>), và mã của PyTorch nói gemv "is bw bound" (bị
              giới hạn bởi băng thông) (ch15-facts D1, D4). Các kernel lượng tử hóa là chủ đề của <a href={href('x2')}>module X2</a>.
            </p>
          }
        >
          <p>
            The m = 1 bound is our derivation from the book's formula. llama.cpp uses matrix-vector kernels up to batch 8
            (<code>MMVQ_MAX_BATCH_SIZE</code>, <code>MMVF_MAX_BATCH_SIZE</code>), and PyTorch's code says gemv "is bw bound"
            (ch15-facts D1, D4). Quantised kernels are the subject of <a href={href('x2')}>module X2</a>.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS152['ch15/15.2/q-decode']} />
      <Ask q={questionsS152['ch15/15.2/q-batch59']} />
    </>
  );
}
