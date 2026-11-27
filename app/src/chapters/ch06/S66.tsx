import { useId, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { DEFAULTS, FACTORS, WHERE_SNIPPETS, bodyText, schedule, whereLives, type Cell, type Place } from '../../sim/unroll';
import { Translated, tr } from '../../lib/i18n';

// labs/ch06/snippets/s66_unroll.cu, which ./verify.sh front-end-checks.
// snippets-s66.test.ts fails if this copy drifts from the file.
const UNROLL = `// Section 6.6, pp. 144-145: the book's 16-iteration loop, made compilable. A and B
// stand for the book's two statements: B(i) uses what A(i) produced, and A(i + 1)
// does not depend on A(i) or B(i). Each thread works on its own 16 elements.
__device__ float A(const float* in, unsigned int i) { return in[i]*in[i]; }
__device__ void B(float* out, unsigned int i, float a) { out[i] = a + 1.0f; }

__global__ void original(const float* in, float* out) {
    unsigned int base = 16*(blockIdx.x*blockDim.x + threadIdx.x);
    for (unsigned int i = 0; i < 16; ++i) {
        float a = A(in, base + i);
        B(out, base + i, a);
    }
}

// Unrolled by 4 and reordered by hand: four A's, then four B's, four times.
__global__ void unrolledReordered(const float* in, float* out) {
    unsigned int base = 16*(blockIdx.x*blockDim.x + threadIdx.x);
    for (unsigned int i = 0; i < 16; i += 4) {
        float a0 = A(in, base + i);
        float a1 = A(in, base + i + 1);
        float a2 = A(in, base + i + 2);
        float a3 = A(in, base + i + 3);
        B(out, base + i, a0);
        B(out, base + i + 1, a1);
        B(out, base + i + 2, a2);
        B(out, base + i + 3, a3);
    }
}

// The same request made to the compiler. #pragma unroll 1 would forbid unrolling.
__global__ void askedToUnroll(const float* in, float* out) {
    unsigned int base = 16*(blockIdx.x*blockDim.x + threadIdx.x);
#pragma unroll 4
    for (unsigned int i = 0; i < 16; ++i) {
        float a = A(in, base + i);
        B(out, base + i, a);
    }
}`;

// ---------------------------------------------------------------- issue timeline
const PER_ROW = 32;
const CW = 20;

function CellBox({ c, x, y, hatch }: { c: Cell; x: number; y: number; hatch: string }) {
  const w = CW - 2;
  const h = 22;
  let body;
  if (c.kind === 'issueA' || c.kind === 'issueB') {
    const a = c.kind === 'issueA';
    body = (
      <>
        <rect x={x} y={y} width={w} height={h} rx="2" fill={a ? 'var(--ink)' : 'var(--ink-2)'} />
        <text x={x + w / 2} y={y + 15} fontSize="9" textAnchor="middle" style={{ fill: 'var(--page)' }}>{(a ? 'A' : 'B') + c.i}</text>
      </>
    );
  } else if (c.kind === 'branch')
    body = (
      <>
        <rect x={x} y={y} width={w} height={h} rx="2" fill="var(--cell)" stroke="var(--ink)" strokeWidth="1.2" />
        <text x={x + w / 2} y={y + 15} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink)' }}>br</text>
      </>
    );
  else if (c.kind === 'stallDep') body = <rect x={x} y={y} width={w} height={h} rx="2" fill={`url(#${hatch})`} stroke="var(--ink-3)" strokeWidth="0.6" />;
  else body = <rect x={x} y={y} width={w} height={h} rx="2" fill="var(--rule)" stroke="var(--ink-3)" strokeWidth="0.6" strokeDasharray="2 2" />;
  return <g data-cell={c.kind}>{body}</g>;
}

function Legend({ hatch }: { hatch: string }) {
  const items: [Cell, string][] = [
    [{ kind: 'issueA', i: 0 }, tr('issue A(i)', 'phát lệnh A(i)')],
    [{ kind: 'issueB', i: 0 }, tr('issue B(i)', 'phát lệnh B(i)')],
    [{ kind: 'branch' }, tr('loop branch', 'nhánh vòng lặp')],
    [{ kind: 'stallDep' }, tr('wait for A(i)', 'chờ A(i)')],
    [{ kind: 'stallBranch' }, tr('wait at the branch', 'chờ tại nhánh')],
  ];
  return (
    <svg viewBox="0 0 640 28" role="img" aria-label={tr('legend', 'chú giải')} style={{ width: '100%', maxWidth: 640, display: 'block', marginBottom: '0.4rem' }}>
      {items.map(([c, label], k) => (
        <g key={k}>
          <CellBox c={c} x={k * 128} y={2} hatch={hatch} />
          <text x={k * 128 + CW + 4} y={17} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{label}</text>
        </g>
      ))}
    </svg>
  );
}

function IssueTimeline() {
  const [unroll, setUnroll] = useState(1);
  const [reorder, setReorder] = useState(false);
  const [latencyAB, setLatency] = useState(DEFAULTS.latencyAB);
  const [branchStall, setBranch] = useState(DEFAULTS.branchStall);
  const hatch = 'hatch' + useId().replace(/:/g, '');
  const cfg = { ...DEFAULTS, unroll, reorder, latencyAB, branchStall };
  const t = schedule(cfg);
  const base = schedule({ ...cfg, unroll: 1, reorder: false });
  const rows = Math.ceil(t.total / PER_ROW);
  const rowH = 30;

  return (
    <Bench
      title={tr('Issue timeline', 'Dòng thời gian phát lệnh')}
      task={tr('Unroll by 4 and watch the branch stalls; then reorder. What does full unrolling leave?', 'Trải vòng lặp với hệ số 4 và quan sát các lần chờ tại nhánh; rồi sắp xếp lại. Trải vòng lặp hoàn toàn để lại gì?')}
      foot={tr('One thread, one instruction per cycle, issued in order. Each square is one cycle; rows wrap every 32 cycles.', 'Một luồng, một lệnh mỗi chu kỳ, phát theo thứ tự. Mỗi ô vuông là một chu kỳ; các hàng xuống dòng sau mỗi 32 chu kỳ.')}
    >
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <span className="seg" role="group" aria-label={tr('unroll factor', 'hệ số trải vòng lặp')}>
          {FACTORS.map((u) => (
            <button key={u} aria-pressed={u === unroll} onClick={() => setUnroll(u)}>{u === 16 ? tr('unroll 16 (full)', 'trải 16 (hoàn toàn)') : tr(`unroll ${u}`, `trải ${u}`)}</button>
          ))}
        </span>
        <label className="check">
          <input type="checkbox" checked={reorder} onChange={(e) => setReorder(e.target.checked)} /> <Translated vi="sắp xếp lại: các A trước, rồi các B">reorder: A's first, then B's</Translated>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi={<>Độ trễ từ A đến B: {latencyAB} chu kỳ</>}>A-to-B latency: {latencyAB} cycle{latencyAB === 1 ? '' : 's'}</Translated>
          <input type="range" min={0} max={8} value={latencyAB} aria-label={tr('A-to-B latency', 'độ trễ từ A đến B')} onChange={(e) => setLatency(+e.target.value)} />
        </label>
        <label className="field">
          <Translated vi={<>Thời gian chờ ở mỗi nhánh: {branchStall} chu kỳ</>}>Stall at each branch: {branchStall} cycle{branchStall === 1 ? '' : 's'}</Translated>
          <input type="range" min={0} max={8} value={branchStall} aria-label={tr('branch stall', 'thời gian chờ tại nhánh')} onChange={(e) => setBranch(+e.target.value)} />
        </label>
      </div>
      <p className="ui" style={{ margin: '0 0 0.7rem' }}>
        {unroll === 16 ? tr('No loop left. Body: ', 'Không còn vòng lặp. Thân: ') : `for (i = 0; i < 16; i += ${unroll}): `}
        <code data-k="body">{bodyText(unroll, reorder)}</code>
      </p>
      <Legend hatch={hatch} />
      <svg viewBox={`0 0 ${PER_ROW * CW + 50} ${rows * rowH + 4}`} role="img" aria-label={tr('cycle-by-cycle issue of one thread', 'việc phát lệnh theo từng chu kỳ của một luồng')}
        style={{ width: '100%', maxWidth: PER_ROW * CW + 50, display: 'block' }}>
        <defs>
          <pattern id={hatch} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="5" height="5" fill="var(--cell)" />
            <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1.6" />
          </pattern>
        </defs>
        {Array.from({ length: rows }, (_, r) => (
          <text key={r} x={0} y={r * rowH + 17} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{r * PER_ROW}</text>
        ))}
        {t.cells.map((c, k) => (
          <CellBox key={k} c={c} x={34 + (k % PER_ROW) * CW} y={Math.floor(k / PER_ROW) * rowH + 2} hatch={hatch} />
        ))}
      </svg>
      <table className="data" style={{ background: 'var(--page)', margin: '0.8rem 0 0' }} aria-label={tr('counters', 'bộ đếm')}>
        <thead><tr><th></th><th><Translated vi="thiết lập này">this setting</Translated></th><th><Translated vi="không trải vòng lặp, cùng độ trễ">no unrolling, same latencies</Translated></th></tr></thead>
        <tbody>
          <tr><td><Translated vi="nhánh vòng lặp">loop branches</Translated></td><td data-k="branches">{t.branches}</td><td>{base.branches}</td></tr>
          <tr><td><Translated vi="chu kỳ chờ A">cycles waiting for A</Translated></td><td data-k="stallDep">{t.stallDep}</td><td>{base.stallDep}</td></tr>
          <tr><td><Translated vi="chu kỳ chờ tại nhánh">cycles waiting at branches</Translated></td><td data-k="stallBranch">{t.stallBranch}</td><td>{base.stallBranch}</td></tr>
          <tr><td><Translated vi="tổng số chu kỳ">total cycles</Translated></td><td data-k="total"><b>{t.total}</b></td><td>{base.total}</td></tr>
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- where does x live?
function WhereXLives() {
  const [pick, setPick] = useState<Record<string, Place>>({});
  const right = WHERE_SNIPPETS.filter((s) => pick[s.id] === whereLives(s).place).length;
  const answered = Object.keys(pick).length;
  return (
    <Bench
      title={tr('Where does x live?', 'x nằm ở đâu?')}
      task={tr('Sort each kernel: does x end up in registers, or in local memory off the chip?', 'Phân loại từng kernel: x kết thúc trong thanh ghi, hay trong bộ nhớ cục bộ ngoài chip?')}
      foot={
        <Translated vi={<>Từ <code>labs/ch06/snippets/s66_where_x_lives.cu</code>, nơi <code>COARSEN</code> bằng 4. Đáp án là điều trình biên dịch nhiều khả năng sẽ làm (quy tắc của sách và danh sách của NVIDIA); <code>--ptxas-options=-v</code> cho thấy nó đã làm gì.</>}>
          From <code>labs/ch06/snippets/s66_where_x_lives.cu</code>, where <code>COARSEN</code> is 4. The answers are
          what the compiler is likely to do (the book's rule and NVIDIA's list); <code>--ptxas-options=-v</code> shows
          what it did.
        </Translated>
      }
    >
      {WHERE_SNIPPETS.map((s, k) => {
        const mine = pick[s.id];
        const ans = whereLives(s);
        return (
          <div key={s.id} data-snippet={s.id} style={{ marginBottom: '1.2rem', paddingBottom: '0.9rem', borderBottom: '1px solid var(--rule-soft)' }}>
            <p className="ui" style={{ margin: '0 0 0.3rem' }}><b>{k + 1}.</b> {s.label}</p>
            <Code tabs={[{ label: s.id, code: s.code }]} />
            <div className="row" role="group" aria-label={tr(`where x lives in kernel ${k + 1}`, `x nằm ở đâu trong kernel ${k + 1}`)}>
              {(['registers', 'local memory'] as Place[]).map((p) => ( // i18n-ignore: Place keys
                <button key={p} aria-pressed={mine === p} onClick={() => setPick({ ...pick, [s.id]: p })}>
                  {p === 'registers' ? tr('registers', 'thanh ghi') : tr('local memory (off-chip)', 'bộ nhớ cục bộ (ngoài chip)')}
                </button>
              ))}
              {mine ? (
                <span data-k="verdict" className={'chip ' + (mine === ans.place ? 'ok' : 'fail')}>
                  {mine === ans.place ? tr('right', 'đúng') : tr('not this', 'không phải')}: {ans.place === 'registers' ? tr('registers', 'thanh ghi') : tr('local memory', 'bộ nhớ cục bộ')}
                </span>
              ) : null}
            </div>
            {mine ? <p className="muted" style={{ margin: '0.3rem 0 0' }}>{ans.rule}</p> : null}
          </div>
        );
      })}
      <p className="ui" style={{ margin: 0 }} data-k="score">{answered ? tr(`${right} of ${answered} right so far.`, `Đến giờ đúng ${right} trên ${answered}.`) : tr('Nothing sorted yet.', 'Chưa phân loại gì.')}</p>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS66: Record<string, Question> = {
  'ch06/6.6/q-branches': {
    kind: 'numeric',
    id: 'ch06/6.6/q-branches',
    prompt: <Translated vi={<>Một vòng lặp 16 lần lặp được trải với hệ số 4. Một luồng thực thi bao nhiêu nhánh vòng lặp?</>}>A 16-iteration loop is unrolled by a factor of 4. How many loop branches does a thread execute?</Translated>,
    answer: 4,
    explain: (
      <Translated
        vi={
          <p>
            Mỗi lần lặp còn lại một nhánh: 16 ÷ 4 = 4, giảm từ 16. Trải với hệ số 16 thì vòng lặp biến mất và nhánh cuối cùng của nó cũng vậy.
          </p>
        }
      >
        <p>
          One per remaining iteration: 16 ÷ 4 = 4, down from 16. Unrolled by 16, the loop disappears and so does its last
          branch.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.6, p. 144',
  },
  'ch06/6.6/q-reorder': {
    kind: 'choice',
    id: 'ch06/6.6/q-reorder',
    prompt: (
      <Translated
        vi={
          <>
            Khi nào thân đã trải A(i) B(i) A(i+1) B(i+1) … được phép sắp xếp lại thành A(i) A(i+1) … B(i) B(i+1) …?
          </>
        }
      >
        <>
          When may the unrolled body A(i) B(i) A(i+1) B(i+1) … be reordered to A(i) A(i+1) … B(i) B(i+1) …?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Khi A(i+1) không phụ thuộc vào A(i) hay B(i).">When A(i+1) does not depend on A(i) or B(i).</Translated>, correct: true },
      { label: <Translated vi="Bất cứ lúc nào: GPU giải quyết phụ thuộc lúc chạy.">At any time: the GPU resolves dependences at run time.</Translated>, why: <Translated vi="Một luồng GPU phát lệnh theo thứ tự và không thực thi suy đoán. Nếu A(i+1) cần kết quả của B(i), đưa nó lên trước sẽ tính ra sai.">A GPU thread issues in order and does not speculate. If A(i+1) needed B(i)'s result, moving it in front would compute the wrong thing.</Translated> },
      { label: <Translated vi="Khi B(i) không phụ thuộc vào A(i).">When B(i) does not depend on A(i).</Translated>, why: <Translated vi="Chưa đủ: A(i+1) vẫn có thể đọc kết quả của B(i), và khi đó nó không thể đi trước B(i). Và nếu B(i) không chờ A(i), sẽ chẳng có lần chờ nào cần che giấu.">Not enough: A(i+1) may still read B(i)'s result, and then it cannot go ahead of B(i). And if B(i) did not wait on A(i), there would be no stall to hide.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đưa A(i+1) lên trước B(i) làm thay đổi thứ tự chúng xảy ra, nên chỉ được phép nếu không thứ gì A(i+1) đọc lại được ghi bởi A(i) hay B(i). Khi đó luồng có việc hữu ích để làm trong khi kết quả của A(i) đang trên đường tới. Đây là một phần của cái mà trình biên dịch gọi là lập lịch lệnh (instruction scheduling).
          </p>
        }
      >
        <p>
          Moving A(i+1) in front of B(i) changes the order in which they happen, so it is allowed only if nothing A(i+1)
          reads is written by A(i) or B(i). Then the thread has useful work while A(i)'s result is on its way. This is part
          of what a compiler calls instruction scheduling.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.6, pp. 144–145',
  },
  'ch06/6.6/q-promotion': {
    kind: 'choice',
    id: 'ch06/6.6/q-promotion',
    prompt: <Translated vi={<>Vì sao trải vòng lặp quan trọng hơn sau khi gộp việc?</>}>Why does unrolling matter more after coarsening?</Translated>,
    options: [
      {
        label: <Translated vi="Luồng đã gộp việc giữ các giá trị của nó trong một mảng x[c], mảng này chỉ có thể nằm trong thanh ghi khi mọi chỉ số đều là hằng lúc biên dịch; nếu không nó nằm trong bộ nhớ cục bộ ngoài chip.">The coarsened thread keeps its values in an array x[c], which can live in registers only once every index is a compile-time constant; otherwise it sits in off-chip local memory.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Các vòng lặp đã gộp việc có nhiều nhánh hơn số thanh ghi luồng có để theo dõi chúng.">Coarsened loops have more branches than the thread has registers to track them.</Translated>, why: <Translated vi="Nhánh và thanh ghi là những tài nguyên không liên quan đến nhau.">Branches and registers are unrelated resources.</Translated> },
      { label: <Translated vi="Trải vòng lặp làm tăng occupancy, thứ mà gộp việc vừa hạ xuống.">Unrolling raises occupancy, which coarsening has just lowered.</Translated>, why: <Translated vi="Nếu có gì thì trải vòng lặp giữ nhiều giá trị còn sống cùng lúc hơn và có thể làm tăng mức dùng thanh ghi.">If anything, unrolling keeps more values live at once and can raise register use.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Thanh ghi không có địa chỉ, nên một phần tử của mảng chỉ có thể nằm trong thanh ghi nếu trình biên dịch biết lúc biên dịch mỗi lệnh ám chỉ phần tử nào. Trải hoàn toàn vòng lặp gộp việc biến x[c] thành x[0], x[1], x[2], x[3], và mỗi cái có thể trở thành một thanh ghi riêng. Nếu để trong bộ nhớ cục bộ, lưu lượng của mảng có thể triệt tiêu lợi ích của gộp việc.
          </p>
        }
      >
        <p>
          A register has no address, so an element of an array can be in a register only if the compiler knows at compile
          time which element each instruction means. Fully unrolling the coarsening loop turns x[c] into x[0], x[1], x[2],
          x[3], and each can become its own register. Left in local memory, the array&rsquo;s traffic can cancel the
          coarsening gain.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.6, pp. 145–146',
  },
  'ch06/6.6/q-pragma1': {
    kind: 'choice',
    id: 'ch06/6.6/q-pragma1',
    prompt: (
      <Translated
        vi={
          <>
            Ai đó thêm <code>#pragma unroll 1</code> phía trên một vòng lặp gộp việc đánh chỉ số <code>float v[4]</code> bằng biến đếm vòng lặp. Bạn kỳ vọng gì trên CUDA, và bạn kiểm tra thế nào?
          </>
        }
      >
        <>
          Someone adds <code>#pragma unroll 1</code> above a coarsening loop that indexes <code>float v[4]</code> with the
          loop counter. What do you expect on CUDA, and how do you check?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="v nhiều khả năng sẽ vào bộ nhớ cục bộ; nvcc --ptxas-options=-v báo cáo số byte lmem của nó.">v likely goes to local memory; nvcc --ptxas-options=-v reports its lmem bytes.</Translated>, correct: true },
      { label: <Translated vi="Không thay đổi: trên CUDA pragma chỉ là gợi ý, và trình biên dịch dù sao cũng trải vòng lặp.">No change: on CUDA the pragma is only a hint, and the compiler unrolls the loop anyway.</Translated>, why: <Translated vi="NVIDIA ghi trong tài liệu rằng 1 nghĩa là “the loop will not be unrolled” (vòng lặp sẽ không được trải) (infra A7). Chính AMD mới gọi dạng có số là một gợi ý (B7).">NVIDIA documents 1 as "the loop will not be unrolled" (infra A7). It is AMD that calls the numeric form a hint (B7).</Translated> },
      { label: <Translated vi="Một lỗi biên dịch: vòng lặp đánh chỉ số một mảng cục bộ bắt buộc phải được trải.">A compile error: a loop that indexes a local array must be unrolled.</Translated>, why: <Translated vi="Nó vẫn biên dịch được. Mảng chỉ cần nằm ở một nơi có địa chỉ.">It compiles. The array just has to live somewhere addressable.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Khi trải vòng lặp bị cấm, mọi truy cập đều là v[c] với c là biến, và NVIDIA liệt kê “arrays for which it cannot determine that they are indexed with constant quantities” (các mảng mà nó không xác định được là được đánh chỉ số bằng hằng) đầu tiên trong những thứ bị đưa vào bộ nhớ cục bộ. Trong PTX các truy cập đó xuất hiện dưới dạng <code>ld.local</code> và <code>st.local</code>.
          </p>
        }
      >
        <p>
          With unrolling forbidden, every access is v[c] with a variable c, and NVIDIA lists "arrays for which it cannot
          determine that they are indexed with constant quantities" first among what goes to local memory. In PTX those
          accesses appear as <code>ld.local</code> and <code>st.local</code>.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.6, p. 145; notes/today/ch06-facts.md A13; infra-facts.md A7', // i18n-ignore: citation only
  },
  'ch06/6.6/q-metal': {
    kind: 'choice',
    id: 'ch06/6.6/q-metal',
    prompt: (
      <Translated
        vi={
          <>
            Bạn chuyển một kernel đã gộp việc sang Metal. Vòng lặp gộp việc của nó đánh chỉ số một mảng cục bộ của luồng <code>float v[4]</code> bằng biến đếm vòng lặp, và bạn đặt <code>_Pragma("clang loop unroll(full)")</code> của MLX phía trước nó. Nó biên dịch không có cảnh báo, và AIR mà front end phát ra có ghi lại yêu cầu. <code>v</code> có nằm trong thanh ghi không, và bạn làm gì tiếp theo?
          </>
        }
      >
        <>
          You port a coarsened kernel to Metal. Its coarsening loop indexes a thread-local <code>float v[4]</code> with
          the loop counter, and you put MLX&rsquo;s <code>_Pragma("clang loop unroll(full)")</code> in front of it. It
          compiles without a warning, and the AIR the front end emits records the request. Is <code>v</code> in
          registers, and what do you do next?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Bạn không thể biết từ mã nguồn hay AIR. Hãy viết một phiên bản với chỉ số hằng (v[0] đến v[3], hoặc bốn biến có tên) và đo cả hai.">You cannot tell from the source or the AIR. Write a version with constant indices (v[0] to v[3], or four named variables) and measure the two.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Có: AIR ghi lại việc trải hoàn toàn, nên mọi chỉ số đều là hằng trong mã GPU.">Yes: the AIR records full unrolling, so every index is a constant in the GPU code.</Translated>,
        why: <Translated vi="AIR được driver biên dịch lại, và Apple không ghi tài liệu gì về việc driver làm gì với yêu cầu đó. AIR cho thấy điều đã được yêu cầu, không phải điều thực sự chạy.">AIR is compiled again by the driver, and Apple documents nothing about what it does with the request. The AIR shows what was asked for, not what runs.</Translated>,
      },
      {
        label: <Translated vi="Không: Metal bỏ qua pragma, nên hãy xóa nó và chấp nhận bộ nhớ cục bộ.">No: Metal ignores pragmas, so delete it and accept local memory.</Translated>,
        why: <Translated vi="Đặc tả có ghi một vài pragma về dấu phẩy động, và front end ghi lại pragma này. Driver có tôn trọng nó hay không là điều chưa biết, chứ không phải đã biết là không.">The specification documents several floating-point pragmas, and the front end records this one. Whether the driver honours it is unknown, not known to be no.</Translated>,
      },
      {
        label: <Translated vi="Có: Apple ghi trong tài liệu pragma này là cách viết của Metal cho #pragma unroll.">Yes: Apple documents this pragma as the Metal spelling of #pragma unroll.</Translated>,
        why: <Translated vi="Đặc tả Metal Shading Language hoàn toàn không có pragma vòng lặp nào (C4). MLX và llama.cpp dùng cách viết của Clang, và đo hiệu năng.">The Metal Shading Language specification has no loop pragma at all (C4). MLX and llama.cpp use Clang's spelling, and benchmark.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Front end trên máy chấp nhận cả <code>_Pragma("clang loop unroll(full)")</code> lẫn <code>#pragma unroll 4</code> và ghi chúng vào AIR nó phát ra; không tài liệu nào Apple công bố nói driver làm gì với chúng. Điều Apple có nói là quy tắc đằng sau câu hỏi: tránh &ldquo;implementations that randomly index into thread-scoped arrays&rdquo; (các cách cài đặt đánh chỉ số ngẫu nhiên vào mảng phạm vi luồng), vì chúng có thể đẩy dữ liệu ra bộ nhớ device. Các chỉ số hằng do bạn tự viết không phụ thuộc vào pragma, nên phiên bản viết tay là mốc để đo so sánh.
          </p>
        }
      >
        <p>
          The local front end accepts both <code>_Pragma("clang loop unroll(full)")</code> and <code>#pragma unroll 4</code>{' '}
          and records them in the AIR it emits; nothing Apple publishes says what the driver does with them. What Apple
          does say is the rule behind the question: avoid &ldquo;implementations that randomly index into thread-scoped
          arrays&rdquo;, because they can push data out to device memory. Constant indices that you write yourself do not
          depend on the pragma, so the hand-written version is the baseline to measure against.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch06-facts.md C4, D2, D3 (compile-only observation; not in the book)',
    vi: { source: 'notes/today/ch06-facts.md C4, D2, D3 (quan sát chỉ ở mức biên dịch; không có trong sách)' },
  },
};

export function S66() {
  return (
    <>
      <Translated
        vi={
          <p>
            GPU che giấu độ trễ chủ yếu bằng cách luân phiên: khi một warp chờ một kết quả chậm, bộ lập lịch phát lệnh của warp khác thay vào (<a href={href('ch04', '4.6')}>chương 4</a>). Trải vòng lặp là công cụ thứ hai, và nó hoạt động bên trong một luồng duy nhất. Nhân bản thân vòng lặp lên một số lần và chia số lần lặp cho cùng con số đó.
          </p>
        }
      >
        <p>
          A GPU hides latency mostly by switching: when one warp waits for a slow result, the scheduler issues another
          warp's instruction instead (<a href={href('ch04', '4.6')}>chapter 4</a>). Loop unrolling is a second tool, and it
          works inside a single thread. Replicate the loop body some number of times and divide the trip count by the same
          number.
        </p>
      </Translated>
      <Source section="6.6" pages="144" />
      <Code
        tabs={[{ label: 's66_unroll.cu', code: UNROLL, status: tr('front-end-checked; unverified on CUDA and HIP hardware', 'front-end-checked; unverified trên phần cứng CUDA và HIP') }]}
        highlight={[18, 19, 20, 21, 22, 23, 24, 25, 26]}
        caption={
          <Translated vi={<>Vòng lặp 16 lần lặp của sách (p. 144) dưới dạng một kernel, từ <code>labs/ch06/snippets/</code>, rồi được trải với hệ số 4 và sắp xếp lại bằng tay, rồi cùng yêu cầu đó được đưa ra bằng một pragma. A và B là của khóa học này; sách để chúng ở dạng trừu tượng.</>}>
            The book's 16-iteration loop (p. 144) as a kernel, from <code>labs/ch06/snippets/</code>, then unrolled by 4
            and reordered by hand, then the same request made with a pragma. A and B are ours; the book leaves them
            abstract.
          </Translated>
        }
      />

      <h2>
        <Translated vi="Nó mang lại gì">What it gains</Translated>
      </h2>
      <Translated
        vi={
          <p>
            <b>Ít nhánh hơn.</b> CPU đoán nhánh sẽ đi hướng nào và tiếp tục chạy. Một luồng GPU thì không: nó dừng chờ tại nhánh cho đến khi biết kết quả.
          </p>
        }
      >
        <p>
          <b>Fewer branches.</b> A CPU guesses which way a branch will go and carries on. A GPU thread does not: it stalls at
          the branch until the outcome is known.
        </p>
      </Translated>
      <Source section="6.6" pages="144" />
      <Translated
        vi={
          <p>
            Một vòng lặp rẽ nhánh một lần mỗi lần lặp, nên một vòng lặp được trải với hệ số 4 chỉ rẽ nhánh bằng một phần tư số lần.
          </p>
        }
      >
        <p>
          A loop branches once per iteration, so a loop unrolled by 4 branches a quarter as often.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md, A11; ch06-facts.md, A14)">{/* i18n-ignore: date and notes reference */}
        <Translated
          vi={
            <p>
              NVIDIA: các SM “issue instructions in order and do not perform branch prediction or speculative execution” (phát lệnh theo thứ tự và không dự đoán nhánh hay thực thi suy đoán). Tài liệu hướng dẫn của họ nói thêm rằng trình biên dịch có thể thay một nhánh ngắn bằng predication, khi đó hoàn toàn không có nhánh nào.
            </p>
          }
        >
          <p>
            NVIDIA: SMs "issue instructions in order and do not perform branch prediction or speculative execution". Its
            guide adds that the compiler may replace a short branch with predication, in which case there is no branch at
            all.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            <b>Các lệnh độc lập lấp khoảng trống.</b> Nếu B(i) cần kết quả của A(i), luồng phải chờ giữa chúng. Trong thân đã trải có bốn A, và nếu A(i+1) không phụ thuộc vào A(i) hay B(i), bốn A có thể đi trước: đến lúc B(i) được phát, A(i) đã có ba chu kỳ để hoàn tất. Chọn thứ tự đó là một phần của lập lịch lệnh.
          </p>
        }
      >
        <p>
          <b>Independent instructions to fill the gap.</b> If B(i) needs A(i)'s result, the thread waits between them. In
          the unrolled body there are four A's, and if A(i+1) does not depend on A(i) or B(i), the four can go first: by the
          time B(i) issues, A(i) has had three cycles to finish. Choosing that order is part of instruction scheduling.
        </p>
      </Translated>
      <Source section="6.6" pages="144–145" />
      <Translated
        vi={
          <p>
            Dòng thời gian bên dưới phát lệnh cho ba phiên bản của vòng lặp, bản gốc, bản đã trải và bản đã sắp xếp lại, để bạn đếm số chu kỳ mỗi bản mất vào nhánh và vào việc chờ A.
          </p>
        }
      >
        <p>
          The timeline below issues the three versions of the loop, original, unrolled and reordered, so you can count
          the cycles each one loses to branches and to waiting for A.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một luồng đơn độc, không có warp nào khác để luân phiên; một lệnh mỗi chu kỳ, theo thứ tự; độ trễ từ A đến B cố định và thời gian chờ cố định sau mỗi nhánh, với việc cập nhật biến đếm của vòng lặp gộp vào nhánh. Trình biên dịch thật lập lịch tự do hơn.
            </p>
          }
        >
          <p>
            One thread alone, with no other warps to switch to; one instruction per cycle, in order; a fixed A-to-B latency
            and a fixed stall after every branch, with the loop's counter update folded into the branch. Real compilers
            schedule more freely.
          </p>
        </Translated>
      </Simplification>
      <IssueTimeline />
      <Translated
        vi={
          <p>
            Với độ trễ mặc định, chỉ riêng việc trải với hệ số 4 đưa luồng từ 128 chu kỳ xuống 92: thời gian chờ tại nhánh giảm từ 32 chu kỳ xuống 8, trong khi 48 chu kỳ chờ A vẫn còn. Sắp xếp lại thân đã trải loại bỏ luôn 48 chu kỳ đó, còn 44 chu kỳ. Trải hoàn toàn không để lại vòng lặp và nhánh nào.
          </p>
        }
      >
        <p>
          With the default latencies, unrolling by 4 alone takes the thread from 128 cycles to 92: the branch stalls shrink
          from 32 cycles to 8, while the 48 cycles spent waiting for A stay. Reordering the unrolled body removes those 48 as
          well, for 44 cycles. Full unrolling leaves no loop and no branch.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Theo ngôn ngữ C++, điều kiện của sách cũng liên quan đến aliasing: một lần ghi qua <code>out</code> trong B(i) có thể thay đổi thứ mà một lần nạp sau đó qua <code>in</code> nhìn thấy, trừ khi trình biên dịch biết chắc hai vùng không bao giờ chồng lên nhau. Điều này là của khóa học này, không phải của sách.
            </p>
          }
        >
          <p>
            In C++ terms the book's condition is about aliasing too: a store through <code>out</code> in B(i) could change
            what a later load through <code>in</code> sees, unless the compiler can tell the two never overlap. That is ours,
            not the book's.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS66['ch06/6.6/q-reorder']} />

      <h2>
        <Translated vi="Ai trải vòng lặp">Who unrolls</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Thường là trình biên dịch, tự nó và một cách quyết liệt, cùng với việc lập lịch ở trên. Một vòng lặp có thân nhỏ và số lần lặp là hằng nhỏ nhiều khả năng sẽ được trải hoàn toàn, nên không còn nhánh nào. Bạn có thể yêu cầu một hệ số bằng <code>#pragma unroll N</code> ở dòng trước vòng lặp, và <code>#pragma unroll 1</code> cấm trải vòng lặp. Sách chỉ viết pragma ở những chỗ muốn bạn để ý rằng việc trải vòng lặp là điều được kỳ vọng.
          </p>
        }
      >
        <p>
          Usually the compiler, on its own and aggressively, together with the scheduling above. A loop with a small body and
          a small constant trip count will most likely be unrolled completely, so it has no branches left. You can ask for a
          factor with <code>#pragma unroll N</code> on the line before the loop, and <code>#pragma unroll 1</code> forbids
          unrolling. The book writes the pragma only where it wants you to notice that unrolling is expected.
        </p>
      </Translated>
      <Source section="6.6" pages="145" />
      <Ask q={questionsS66['ch06/6.6/q-branches']} label={tr('Work it out', 'Tự tính ra')} />
      <Translated vi={<p>Cách yêu cầu trải vòng lặp, hoặc cấm nó, trên từng backend:</p>}>
        <p>How to ask for unrolling, or forbid it, on each backend:</p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ đề cập CUDA. Hai cột còn lại lấy từ tài liệu của nhà sản xuất và, với Metal, từ việc biên dịch một kernel thử bằng front end trên máy (notes/today/ch06-facts.md, C4).
            </p>
          }
        >
          <p>
            The book covers CUDA only. The other two columns come from vendor documentation and, for Metal, from compiling a
            test kernel with the local front end (notes/today/ch06-facts.md, C4).
          </p>
        </Translated>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th></th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="Yêu cầu trải vòng lặp">Ask for unrolling</Translated></td>
            <td>
              <Translated vi={<><code>#pragma unroll</code> (hoàn toàn, nếu số lần lặp là hằng) hoặc <code>#pragma unroll N</code></>}>
                <code>#pragma unroll</code> (complete, if the trip count is constant) or <code>#pragma unroll N</code>
              </Translated>
            </td>
            <td><Translated vi="cùng cách viết; AMD gọi dạng có số là một gợi ý">same spelling; AMD calls the numeric form a hint</Translated></td>
            <td><Translated vi="không có gì trong đặc tả ngôn ngữ">nothing in the language specification</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Cấm nó">Forbid it</Translated></td>
            <td><code>#pragma unroll 1</code></td>
            <td><code>#pragma unroll 1</code></td>
            <td><Translated vi="không có tài liệu">nothing documented</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Mã thực tế viết gì">What real code writes</Translated></td>
            <td colSpan={2}><Translated vi="pragma ở trên">the pragma above</Translated></td>
            <td>
              <Translated vi={<><code>_Pragma("clang loop unroll(full)")</code> (MLX, llama.cpp); front end chấp nhận nó và <code>#pragma unroll 4</code>, chỉ ở mức biên dịch</>}>
                <code>_Pragma("clang loop unroll(full)")</code> (MLX, llama.cpp); the front end accepts it and <code>#pragma unroll 4</code>, compile-only
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="Nguồn">Source</Translated></td>
            <td>infra-facts A7{/* i18n-ignore: facts-file reference */}</td>
            <td>infra-facts B7{/* i18n-ignore: facts-file reference */}</td>
            <td>ch06-facts C4, D2, D3{/* i18n-ignore: facts-file reference */}</td>
          </tr>
        </tbody>
      </table>
      <Ask q={questionsS66['ch06/6.6/q-metal']} />
      <Translated
        vi={
          <p>
            Khi đọc các kernel Metal của MLX bạn sẽ thấy <code>STEEL_PRAGMA_UNROLL</code> trước gần như mọi vòng lặp trong, và trong llama.cpp là <code>FOR_UNROLL(...)</code>. Cả hai đều là macro cho pragma Clang đó: tác giả yêu cầu trải hoàn toàn, và tài liệu của Apple không nói trình biên dịch GPU làm gì với yêu cầu đó.
          </p>
        }
      >
        <p>
          When you read MLX's Metal kernels you will see <code>STEEL_PRAGMA_UNROLL</code> in front of nearly every inner
          loop, and in llama.cpp's <code>FOR_UNROLL(...)</code>. Both are macros for that Clang pragma: the authors ask for
          full unrolling, and Apple's documentation does not say what the GPU compiler does with the request.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md, D2, D3)">{/* i18n-ignore: date and notes reference */}
        <p>
          MLX: <code>#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")</code>. llama.cpp:{' '}
          <code>#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)</code>.
        </p>
      </Today>

      <h2>
        <Translated vi="Mảng trong thanh ghi">Arrays in registers</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lợi ích thứ ba là thứ mà gộp việc cần. Một luồng đã gộp việc giữ một giá trị cho mỗi đơn vị công việc, nên một biến vô hướng <code>x</code> trở thành mảng <code>x[4]</code>, được dùng dưới dạng <code>foo(x[c])</code> trong vòng lặp gộp việc. Một mảng được đánh chỉ số bằng biến phải nằm trong bộ nhớ, vì thanh ghi không có địa chỉ; bộ nhớ đó nằm ngoài chip, và lưu lượng của nó có thể ăn mất lợi ích của gộp việc. Trải hoàn toàn vòng lặp thì mọi chỉ số đều thành hằng: <code>foo(x[0]); foo(x[1]); foo(x[2]); foo(x[3]);</code>. Khi đó trình biên dịch có thể cấp cho mỗi phần tử một thanh ghi riêng.
          </p>
        }
      >
        <p>
          The third gain is the one coarsening needs. A coarsened thread keeps one value per unit of work, so a scalar{' '}
          <code>x</code> becomes an array <code>x[4]</code>, used as <code>foo(x[c])</code> inside the coarsening loop. An
          array indexed with a variable has to live in memory, because registers have no addresses; that memory is off the
          chip, and its traffic can eat the coarsening gain. Unroll the loop fully and every index becomes a constant:{' '}
          <code>foo(x[0]); foo(x[1]); foo(x[2]); foo(x[3]);</code>. Then the compiler can give each element its own register.
        </p>
      </Translated>
      <Source section="6.6" pages="145–146" />
      <Translated
        vi={
          <p>
            Có một từ cần cẩn thận. Chương 6 nói những mảng như vậy đi vào “global memory” (bộ nhớ toàn cục); chương 5 gọi cùng chỗ đó là bộ nhớ cục bộ (local memory). NVIDIA gọi nó là bộ nhớ cục bộ, có vị trí vật lý là bộ nhớ device, cùng DRAM với bộ nhớ toàn cục, với cùng độ trễ. Tài liệu của AMD nói vùng lưu trữ riêng của một luồng cho biến cục bộ và thanh ghi bị tràn nằm trong bộ nhớ toàn cục. Tài liệu của Apple nói việc đánh chỉ số động vào mảng phạm vi luồng có thể buộc dữ liệu bị đẩy ra bộ nhớ device. Cả ba mô tả cùng một thứ: nằm ngoài chip, riêng cho luồng.
          </p>
        }
      >
        <p>
          One word needs care. Chapter 6 says such arrays go to "global memory"; chapter 5 called the same place local
          memory. NVIDIA calls it local memory, whose physical location is device memory, the same DRAM as global memory, with
          the same latency. AMD's documentation says a thread's private storage for local variables and spilled registers
          resides in global memory. Apple's says that dynamically indexing thread-scoped arrays can force data out to device
          memory. All three describe one thing: off the chip, private to the
          thread.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md, A13, B6, C4)">{/* i18n-ignore: date and notes reference */}
        <Translated
          vi={
            <p>
              Để thấy điều này trên CUDA, biên dịch với <code>nvcc --ptxas-options=-v</code>: nó báo cáo số byte bộ nhớ cục bộ (lmem) của từng kernel. Trong PTX các truy cập hiện ra dưới dạng <code>ld.local</code> và <code>st.local</code>.
            </p>
          }
        >
          <p>
            To see it on CUDA, compile with <code>nvcc --ptxas-options=-v</code>: it reports each kernel's local-memory
            bytes (lmem). In PTX the accesses show up as <code>ld.local</code> and <code>st.local</code>.
          </p>
        </Translated>
      </Today>
      <WhereXLives />
      <Ask q={questionsS66['ch06/6.6/q-promotion']} />
      <Ask q={questionsS66['ch06/6.6/q-pragma1']} />

      <h2>
        <Translated vi="Điều này dẫn tới đâu">Where this goes</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Ít nhánh hơn, nhiều lệnh hơn để lập lịch, các chỉ số hằng mà trình biên dịch có thể rút gọn, và mảng được đưa lên thanh ghi: đó là bốn lợi ích sách liệt kê. Lợi ích cuối là lý do trải vòng lặp trở nên thiết yếu một khi các luồng được gộp việc. Chương 11 (tổng tiền tố) và chương 15 (phép nhân ma trận) dựa vào nó cho phân ô thanh ghi, nơi mỗi luồng giữ vài kết quả trong thanh ghi; chương 15 cũng dựa vào nó để bù cho occupancy thấp.
          </p>
        }
      >
        <p>
          Fewer branches, more instructions to schedule, constant indices the compiler can fold, and arrays promoted to
          registers: those are the four benefits the book lists. The last is why unrolling becomes essential once threads are
          coarsened. Chapter 11 (prefix sum) and chapter 15 (matrix multiplication) depend on it for register tiling, where
          each thread keeps several results in registers; chapter 15 also leans on it to make up for low occupancy.
        </p>
      </Translated>
      <Source section="6.6" pages="146" figs={tr('§6.8, p. 149 for chapters 11 and 15', '§6.8, p. 149 cho chương 11 và 15')} />
    </>
  );
}
