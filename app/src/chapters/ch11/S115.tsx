import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Beyond, Bench, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  CHARGE_LABEL,
  CLOSED_FORM_TEXT,
  VOCAB_POW2,
  VOCAB_QWEN3_8B,
  bookKoggeStoneSteps,
  breakEven,
  closedForm,
  lg,
  slotsTotal,
  span,
  stepsWithUnits,
  type Charge,
  type NetId,
} from '../../sim/prefixnet';
import { widthChoices } from '../../sim/warps';

/** A caption above a group of buttons (a <label> would forward clicks to the first button). */
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const WRAP: CSSProperties = { flexWrap: 'wrap' };
const int = (v: number) => Math.round(v).toLocaleString('en-US');
const dec = (v: number, d = 1) => Number(v.toFixed(d)).toLocaleString('en-US');
const times = (v: number) => (v >= 100 ? int(v) : v >= 10 ? dec(v, 1) : dec(v, 2));

// ---------------------------------------------------------------- the explorer
const ROWS: NetId[] = ['sequential', 'kogge-stone', 'brent-kung'];
const ROW_NAME: Record<NetId, string> = {
  get sequential() {
    return tr('Loop (Fig. 11.1)', 'Vòng lặp (Fig. 11.1)');
  },
  get 'kogge-stone'() {
    return tr('Kogge-Stone', 'Kogge-Stone');
  },
  get 'brent-kung'() {
    return tr('Brent-Kung', 'Brent-Kung');
  },
};
const CHARGES: Charge[] = ['threads', 'warps', 'adds'];
const CHARGE_NOTE: Record<Charge, string> = {
  get threads() {
    return tr(
      'Every thread of the block is charged in every step, whether it adds or not: the book\'s estimate, (N·log₂N)/P for Kogge-Stone (p. 266). Brent-Kung\'s block has N/2 threads (our kernel in section 11.10).',
      'Mọi luồng của khối đều bị tính ở mọi bước, dù có cộng hay không: ước lượng của sách, (N·log₂N)/P cho Kogge-Stone (p. 266). Khối của Brent-Kung có N/2 luồng (kernel của chúng tôi ở mục 11.10).',
    );
  },
  get warps() {
    return tr(
      'A warp with at least one adding lane is charged all its lanes; a warp with none is free. This is the reason the book gives on p. 266 for resource use above the formula.',
      'Một warp có ít nhất một lane đang cộng bị tính toàn bộ lane của nó; warp không có lane nào cộng thì miễn phí. Đây là lý do sách đưa ra ở p. 266 cho việc tài nguyên dùng nhiều hơn công thức.',
    );
  },
  get adds() {
    return tr('Only the threads that add are charged: the cost the closed forms count.', 'Chỉ các luồng có cộng mới bị tính: đúng chi phí mà các công thức dạng đóng đếm.');
  },
};
/** Slider position (0..120) to execution units, on a log scale: 2^(v/10). */
const unitsOf = (v: number) => Math.max(1, Math.round(2 ** (v / 10)));
const sliderOf = (p: number) => Math.round(10 * Math.log2(p));
const PRESETS = [
  {
    key: 'b512',
    get label() {
      return tr('Book: 512 elements', 'Sách: 512 phần tử');
    },
    logN: 9,
    p: 4096,
    charge: 'threads' as Charge,
  },
  {
    key: 'b1024',
    get label() {
      return tr('Book: 1,024 elements, 32 units', 'Sách: 1,024 phần tử, 32 đơn vị');
    },
    logN: 10,
    p: 32,
    charge: 'threads' as Charge,
  },
  {
    key: 'vocab',
    get label() {
      return tr('A vocabulary row: 2¹⁷ elements, 1,024 units', 'Một hàng từ vựng: 2¹⁷ phần tử, 1,024 đơn vị');
    },
    logN: 17,
    p: 1024,
    charge: 'threads' as Charge,
  },
];

function Bar({ value, max }: { value: number; max: number }) {
  const w = max > 0 ? Math.max(1, (value / max) * 100) : 0;
  return (
    <span aria-hidden="true" style={{ display: 'block', height: 6, marginTop: 3, background: 'var(--rule-soft)', borderRadius: 2 }}>
      <span style={{ display: 'block', height: 6, width: `${w}%`, background: 'var(--ink-2)', borderRadius: 2 }} />
    </span>
  );
}

function WorkExplorer() {
  const { reports } = useReports();
  const widths = useMemo(() => widthChoices(reports as Reports), [reports]);
  const [logN, setLogN] = useState(10);
  const [p, setP] = useState(32);
  const [pText, setPText] = useState('32');
  const [charge, setCharge] = useState<Charge>('threads');
  const [wKey, setWKey] = useState('book');
  const W = (widths.find((c) => c.key === wKey) ?? widths[0]).width;
  const n = 2 ** logN;
  const setUnits = (v: number) => {
    const c = Math.max(1, Math.min(4096, Math.round(v)));
    setP(c);
    setPText(String(c));
  };

  const rows = ROWS.map((id) => {
    const adds = closedForm(id, n);
    const steps = stepsWithUnits(id, n, p, charge, W);
    return { id, adds, steps, unlimited: span(id, n) };
  });
  const loop = rows[0];
  const maxAdds = Math.max(...rows.map((r) => r.adds));
  const maxSteps = Math.max(...rows.map((r) => r.steps));
  const beKS = breakEven('kogge-stone', n, charge, W);
  const beBK = breakEven('brent-kung', n, charge, W);
  const naive = closedForm('naive', n);
  const ks = rows[1];

  return (
    <Bench
      title={tr('Work and steps as N grows', 'Khối lượng việc và số bước khi N tăng')}
      task={tr("Start at the book's 1,024 elements on 32 units. Then shrink P until Kogge-Stone loses to the loop, and grow N.", 'Bắt đầu với 1,024 phần tử trên 32 đơn vị của sách. Rồi giảm P cho đến khi Kogge-Stone thua vòng lặp, và tăng N.')}
      foot={
        <Translated
          vi={
            <>
              Số phép cộng được đếm từ các mạng của mục 11.2 (Kogge-Stone) và mục 11.10 (Brent-Kung) và khớp với các công thức dạng
              đóng bên cạnh. Một bước có nhiều lane bị tính hơn số đơn vị thì mất vài vòng: nó tốn ⌈charge / P⌉. Vòng lặp là tham
              chiếu tuần tự: khoảng N bước, theo cách sách đếm, bất kể P là bao nhiêu.
            </>
          }
        >
          <>
            Additions are counted from the networks of section 11.2 (Kogge-Stone) and section 11.10 (Brent-Kung) and agree
            with the closed forms beside them. A step with more charged lanes than units takes several rounds: it costs
            ⌈charge / P⌉. The loop is the sequential reference: about N steps, as the book counts it, whatever P is.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi={<span>N = <b data-value="n">{int(n)}</b> phần tử</span>}>
            <span>
              N = <b data-value="n">{int(n)}</b> elements
            </span>
          </Translated>
          <input type="range" min={3} max={20} value={logN} aria-label={tr('Elements, as a power of two', 'Số phần tử, dạng lũy thừa của hai')} onChange={(e) => setLogN(Number(e.target.value))} style={{ width: '13rem' }} />
        </label>
        <label className="field">
          <Translated vi={<span>P = <b data-value="p">{int(p)}</b> đơn vị thực thi</span>}>
            <span>
              P = <b data-value="p">{int(p)}</b> execution units
            </span>
          </Translated>
          <input type="range" min={0} max={120} value={sliderOf(p)} aria-label={tr('Execution units', 'Đơn vị thực thi')} onChange={(e) => setUnits(unitsOf(Number(e.target.value)))} style={{ width: '13rem' }} />
        </label>
        <label className="field">
          <Translated vi="chính xác">exactly</Translated>
          <input type="number" min={1} max={4096} step={1} value={pText} aria-label={tr('Execution units, exactly', 'Đơn vị thực thi, chính xác')} style={{ width: '6rem' }}
            onChange={(e) => {
              setPText(e.target.value);
              const v = Number(e.target.value);
              if (e.target.value.trim() !== '' && Number.isFinite(v) && v >= 1) setP(Math.min(4096, Math.round(v)));
            }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Mỗi bước tính phí cho">Charge each step for</Translated>
          <span className="seg" role="group" aria-label={tr('Charge each step for', 'Mỗi bước tính phí cho')} style={WRAP}>
            {CHARGES.map((c) => (
              <button key={c} aria-pressed={charge === c} onClick={() => setCharge(c)}>{CHARGE_LABEL[c]}</button>
            ))}
          </span>
        </span>
        <label className="field">
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <select value={wKey} aria-label={tr('warp width', 'độ rộng warp')} disabled={charge !== 'warps'} onChange={(e) => setWKey(e.target.value)}>
            {widths.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span style={FIELD}>
          <Translated vi="Bộ giá trị có sẵn">Presets</Translated>
          <span className="seg" role="group" aria-label={tr('Presets', 'Bộ giá trị có sẵn')} style={WRAP}>
            {PRESETS.map((q) => (
              <button key={q.key} onClick={() => { setLogN(q.logN); setUnits(q.p); setCharge(q.charge); }}>{q.label}</button>
            ))}
          </span>
        </span>
      </div>
      <p className="muted" style={{ margin: '0 0 0.7rem' }} data-readout="charge">{CHARGE_NOTE[charge]}</p>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0, minWidth: '30rem' }} data-part="work-table">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}><Translated vi="Scan">Scan</Translated></th>
              <th className="num"><Translated vi="phép cộng">additions</Translated></th>
              <th className="num"><Translated vi="× so với vòng lặp">× the loop's</Translated></th>
              <th className="num"><Translated vi="số bước với P đơn vị">steps with P units</Translated></th>
              <th className="num"><Translated vi="so với vòng lặp">against the loop</Translated></th>
              <th className="num"><Translated vi="đơn vị không giới hạn">unlimited units</Translated></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} data-row={r.id} data-adds={r.adds} data-steps={r.steps} data-span={r.unlimited}>
                <td style={{ paddingLeft: '0.6rem' }}>{ROW_NAME[r.id]}</td>
                <td className="num">
                  {int(r.adds)}
                  <Bar value={r.adds} max={maxAdds} />
                  <span className="muted" style={{ display: 'block', fontSize: '0.8em', whiteSpace: 'nowrap' }}>{CLOSED_FORM_TEXT[r.id]}</span>
                </td>
                <td className="num" data-cell="work-ratio">{times(r.adds / loop.adds)}</td>
                <td className="num">
                  {int(r.steps)}
                  <Bar value={r.steps} max={maxSteps} />
                </td>
                <td className="num" data-cell="vs-loop">
                  {r.id === 'sequential' ? '' : r.steps < loop.steps ? (
                    <span className="chip ok"><Translated vi={<>ít hơn {times(loop.steps / r.steps)}×</>}>{times(loop.steps / r.steps)}× fewer</Translated></span>
                  ) : r.steps === loop.steps ? (
                    <span className="chip"><Translated vi="bằng nhau">as many</Translated></span>
                  ) : (
                    <span className="chip fail"><Translated vi={<>nhiều hơn {times(r.steps / loop.steps)}×</>}>{times(r.steps / loop.steps)}× more</Translated></span>
                  )}
                </td>
                <td className="num">{int(r.unlimited)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="cols" style={{ marginTop: '0.9rem' }}>
        <div className="panel" aria-live="polite" data-readout="breakeven" data-ks={beKS ?? ''} data-bk={beBK ?? ''}>
          <Translated
            vi={
              <p style={{ margin: 0 }}>
                Kogge-Stone cần ít bước hơn vòng lặp từ <b data-value="be-ks">{beKS == null ? 'không có' : int(beKS)}</b>{' '}
                đơn vị thực thi; Brent-Kung từ <b data-value="be-bk">{beBK == null ? 'không có' : int(beBK)}</b>.
              </p>
            }
          >
            <p style={{ margin: 0 }}>
              Kogge-Stone needs fewer steps than the loop from <b data-value="be-ks">{beKS == null ? 'no' : int(beKS)}</b>{' '}
              execution units; Brent-Kung from <b data-value="be-bk">{beBK == null ? 'no' : int(beBK)}</b>.
            </p>
          </Translated>
          <Translated
            vi={
              <p className="muted" style={{ margin: '0.4rem 0 0' }}>
                Ước lượng của sách cho Kogge-Stone, (N·log₂N)/P = <span data-value="book-ks">{dec(bookKoggeStoneSteps(n, p), 1)}</span>{' '}
                bước, thắng vòng lặp khi P &gt; log₂N = {lg(n)}. Làm tròn mỗi bước lên số vòng nguyên có thể đẩy điểm hòa vốn cao
                thêm một đơn vị.
              </p>
            }
          >
            <p className="muted" style={{ margin: '0.4rem 0 0' }}>
              The book's estimate for Kogge-Stone, (N·log₂N)/P = <span data-value="book-ks">{dec(bookKoggeStoneSteps(n, p), 1)}</span>{' '}
              steps, beats the loop once P &gt; log₂N = {lg(n)}. Rounding each step up to whole rounds can push the break-even
              a unit higher.
            </p>
          </Translated>
        </div>
        <div className="panel" data-readout="extremes">
          <Translated
            vi={
              <p style={{ margin: 0 }}>
                Với số đơn vị không giới hạn, Kogge-Stone cần ít bước hơn vòng lặp N/log₂N ={' '}
                <b data-value="unlimited-ratio">{dec(loop.unlimited / ks.unlimited, 1)}</b> lần.
              </p>
            }
          >
            <p style={{ margin: 0 }}>
              With unlimited units Kogge-Stone needs N/log₂N = <b data-value="unlimited-ratio">{dec(loop.unlimited / ks.unlimited, 1)}</b>{' '}
              times fewer steps than the loop.
            </p>
          </Translated>
          <Translated
            vi={
              <p className="muted" style={{ margin: '0.4rem 0 0' }}>
                Tổng số ô lane bị tính: Kogge-Stone {int(slotsTotal('kogge-stone', n, charge, W))}, Brent-Kung{' '}
                {int(slotsTotal('brent-kung', n, charge, W))}. Scan ngây thơ của mục 11.2 sẽ làm N(N − 1)/2 ={' '}
                <span data-value="naive">{int(naive)}</span> phép cộng, gấp {times(naive / loop.adds)} lần vòng lặp.
              </p>
            }
          >
            <p className="muted" style={{ margin: '0.4rem 0 0' }}>
              Lane slots charged in all: Kogge-Stone {int(slotsTotal('kogge-stone', n, charge, W))}, Brent-Kung{' '}
              {int(slotsTotal('brent-kung', n, charge, W))}. The naive scan of section 11.2 would do N(N − 1)/2 ={' '}
              <span data-value="naive">{int(naive)}</span> additions, {times(naive / loop.adds)} times the loop's.
            </p>
          </Translated>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const KS512 = closedForm('kogge-stone', 512);
const LOOP512 = closedForm('sequential', 512);
const STEPS2048 = stepsWithUnits('kogge-stone', 2048, 64, 'threads');
const BE4096 = breakEven('kogge-stone', 4096, 'threads')!;
const STEPS1024 = stepsWithUnits('kogge-stone', 1024, 32, 'threads');

export const questionsS115: Record<string, Question> = {
  'ch11/11.5/q-work512': {
    kind: 'numeric',
    id: 'ch11/11.5/q-work512',
    prompt: (
      <Translated vi={<>Kogge-Stone thực hiện bao nhiêu phép cộng trên một đoạn 512 phần tử?</>}>
        <>How many additions does Kogge-Stone perform on a segment of 512 elements?</>
      </Translated>
    ),
    answer: KS512,
    unit: 'additions',
    vi: { unit: 'phép cộng' },
    explain: (
      <Translated
        vi={
          <p>
            log₂512 = 9 bước. Cộng N − stride qua các stride 1, 2, …, 256 được 512·9 − (1 + 2 + … + 256) = 4608 − 511 ={' '}
            {int(KS512)}. Vòng lặp làm {LOOP512}.
          </p>
        }
      >
        <p>
          log₂512 = 9 steps. Summing N − stride over the strides 1, 2, …, 256 gives 512·9 − (1 + 2 + … + 256) = 4608 − 511 ={' '}
          {int(KS512)}. The loop does {LOOP512}.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.5, p. 266',
  },
  'ch11/11.5/q-ratio': {
    kind: 'numeric',
    id: 'ch11/11.5/q-ratio',
    prompt: (
      <Translated vi={<>Con số đó gấp bao nhiêu lần khối lượng việc của vòng lặp tuần tự? Cho hai chữ số thập phân.</>}>
        <>How many times the sequential loop's work is that? Give two decimals.</>
      </Translated>
    ),
    answer: Number((KS512 / LOOP512).toFixed(2)),
    tol: 0.01,
    unit: 'times',
    vi: { unit: 'lần' },
    explain: (
      <Translated
        vi={
          <p>
            {KS512} / {LOOP512} = {(KS512 / LOOP512).toFixed(3)}: đúng câu "between eight and nine times" (gấp tám đến chín lần)
            của sách. Tỉ lệ này khoảng log₂N − 1, nên nó tiếp tục tăng theo kích thước đoạn.
          </p>
        }
      >
        <p>
          {KS512} / {LOOP512} = {(KS512 / LOOP512).toFixed(3)}: the book's "between eight and nine times". The ratio is about
          log₂N − 1, so it keeps growing with the segment.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.5, p. 266',
  },
  'ch11/11.5/q-steps': {
    kind: 'numeric',
    id: 'ch11/11.5/q-steps',
    prompt: (
      <Translated vi={<>Theo mô hình của sách, Kogge-Stone cần bao nhiêu bước cho 2,048 phần tử trên 64 đơn vị thực thi?</>}>
        <>By the book's model, how many steps does Kogge-Stone need for 2,048 elements on 64 execution units?</>
      </Translated>
    ),
    answer: STEPS2048,
    unit: 'steps',
    vi: { unit: 'bước', source: 'ch. 11 §11.5, p. 266 (các con số là của khóa học này)' },
    explain: (
      <Translated
        vi={
          <p>
            (N·log₂N)/P = 2048·11/64 = {STEPS2048}, so với 2,048 của vòng lặp: giảm {(2048 / STEPS2048).toFixed(1)} lần. Gấp đôi
            số đơn vị của ví dụ của sách, gấp đôi số phần tử, mà mức lợi chỉ tăng từ 3.2 lên {(2048 / STEPS2048).toFixed(1)}.
          </p>
        }
      >
        <p>
          (N·log₂N)/P = 2048·11/64 = {STEPS2048}, against the loop's 2,048: a {(2048 / STEPS2048).toFixed(1)}-fold reduction.
          Twice the units of the book's example, twice the elements, and the gain grows only from 3.2 to {(2048 / STEPS2048).toFixed(1)}.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.5, p. 266 (the numbers are ours)',
  },
  'ch11/11.5/q-breakeven': {
    kind: 'numeric',
    id: 'ch11/11.5/q-breakeven',
    prompt: (
      <Translated
        vi={
          <>
            Với 4,096 phần tử, theo mô hình của sách, số đơn vị thực thi P nhỏ nhất để Kogge-Stone cần ít bước hơn vòng lặp là
            bao nhiêu?
          </>
        }
      >
        <>
          For 4,096 elements, what is the smallest number of execution units P with which Kogge-Stone needs fewer steps than
          the loop, by the book's model?
        </>
      </Translated>
    ),
    answer: BE4096,
    unit: 'units',
    vi: { unit: 'đơn vị' },
    explain: (
      <Translated
        vi={
          <p>
            (4096·12)/P &lt; 4096 đúng khi P &gt; 12, nên đáp án là {BE4096}. Với 12 đơn vị thì hai bên hòa; ít hơn thì scan
            song song chậm hơn một luồng chạy vòng lặp. Ngưỡng là log₂N, nên nó tăng theo kích thước dữ liệu, đó là lý do phần
            tóm tắt của sách nói Kogge-Stone hợp với các khối cỡ vừa trên bộ xử lý có nhiều đơn vị thực thi.
          </p>
        }
      >
        <p>
          (4096·12)/P &lt; 4096 holds when P &gt; 12, so {BE4096}. With 12 units the two tie; with fewer the parallel scan is
          slower than one thread looping. The threshold is log₂N, so it rises with the data size, which is why the book's
          summary says Kogge-Stone suits modest-sized blocks on processors with plenty of execution units.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.5, pp. 266–267; §11.11, p. 285', // i18n-ignore: citation
  },
  'ch11/11.5/q-energy': {
    kind: 'choice',
    id: 'ch11/11.5/q-energy',
    prompt: (
      <Translated vi={<>Trên một thiết bị nào đó, scan Kogge-Stone xong sớm hơn vòng lặp. Vậy phần việc thêm của nó có miễn phí không?</>}>
        <>On some device a Kogge-Stone scan finishes sooner than the loop. Is its extra work then free?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Có: một khi nó nhanh hơn, chỉ thời gian là đáng kể.">Yes: once it is faster, only the time counts.</Translated>,
        why: <Translated vi="Thời gian là một trong hai chi phí mà sách nêu.">Time is one of two costs the book names.</Translated>,
      },
      {
        label: <Translated vi="Không: mỗi phép cộng thêm vẫn tốn năng lượng, điều quan trọng trên điện thoại.">No: each extra addition still costs energy, which matters on phones.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Có, miễn là các phép cộng thêm chạy trên những đơn vị lẽ ra sẽ rảnh.">Yes, as long as the extra additions run on units that would otherwise idle.</Translated>,
        why: (
          <Translated vi="Thực thi một phép cộng tốn năng lượng dù đơn vị nào chạy nó; đơn vị rảnh không làm cho nó miễn phí.">
            Executing an addition costs energy whichever unit runs it; idle units are not what makes it free.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Sách nêu hai cái giá. Khi P nhỏ, scan song song cần nhiều bước hơn vòng lặp. Và ngay cả khi nó nhanh hơn, phần việc
            thêm tiêu tốn thêm năng lượng, khiến kernel kém phù hợp với các môi trường hạn chế năng lượng như ứng dụng di động.
          </p>
        }
      >
        <p>
          The book names two prices. When P is small the parallel scan needs more steps than the loop. And even when it is
          faster, the extra work consumes additional energy, which makes the kernel less suitable for power-constrained
          settings such as mobile applications.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.5, p. 267',
  },
  'ch11/11.5/q-slots': {
    kind: 'choice',
    id: 'ch11/11.5/q-slots',
    prompt: (
      <Translated vi={<>Vì sao sách nói mức dùng tài nguyên thực của Kogge-Stone gần N·log₂N hơn là N·log₂N − (N − 1)?</>}>
        <>Why does the book say Kogge-Stone's real resource use is closer to N·log₂N than to N·log₂N − (N − 1)?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Công thức quên các phép cộng của bước đầu tiên.">The formula forgets the additions of the first step.</Translated>,
        why: <Translated vi="Bước đầu tiên, stride 1, có trong tổng: N − 1 phép cộng.">The first step, stride 1, is in the sum: N − 1 additions.</Translated>,
      },
      {
        label: <Translated vi="Các luồng rảnh vẫn chiếm lane của những warp vẫn đang thực thi.">Idle threads still occupy lanes of warps that keep executing.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Mỗi phép cộng cần hai lần đọc bộ nhớ chia sẻ, làm khối lượng việc tăng gấp đôi.">Each addition needs two shared-memory reads, which doubles the work.</Translated>,
        why: <Translated vi="Thứ được đếm là phép cộng, không phải lượt truy cập bộ nhớ.">The count is of additions, not of memory accesses.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            N − 1 bị trừ đi là các luồng dưới stride, đứng ngoài bước của chúng. Chúng không rời khỏi warp: một warp thực thi cho
            mọi lane của nó, với các lane không hoạt động bị che đi, cho đến khi cả warp xong. Các ô của chúng đằng nào cũng bị
            tiêu tốn. Mục 11.10 dùng chính sự thật này để chống lại Brent-Kung.
          </p>
        }
      >
        <p>
          The N − 1 subtracted are the threads below the stride, which sit out their step. They do not leave the warp: a warp
          executes for all its lanes, with the inactive ones masked off, until the whole warp is done. Their slots are spent
          either way. Section 11.10 turns the same fact against Brent-Kung.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.5, p. 266; NVIDIA and AMD on masked lanes: notes/today/ch04-facts.md A9, B7',
    vi: { source: 'ch. 11 §11.5, p. 266; NVIDIA và AMD về các lane bị che: notes/today/ch04-facts.md A9, B7' },
  },
};

// ---------------------------------------------------------------- the section
export function S115() {
  const q = questionsS115;
  return (
    <>
      <Translated
        vi={
          <p>
            Các mục 11.2 đến 11.4 làm Kogge-Stone nhanh bằng cách cho nó một span ngắn rồi cắt bớt đồng bộ hóa. Mục này đếm cái
            giá tính bằng phép cộng, và tìm các điều kiện khiến cái giá đó làm scan song song chậm hơn vòng lặp sáu dòng mà nó
            thay thế.
          </p>
        }
      >
        <p>
          Sections 11.2 to 11.4 made Kogge-Stone fast by giving it a short span and then cutting its synchronisation. This
          section counts what it costs in additions, and finds the conditions under which that cost makes the parallel scan
          slower than the six-line loop it replaces.
        </p>
      </Translated>

      <h2><Translated vi="Hiệu quả công việc">Work efficiency</Translated></h2>
      <Translated
        vi={
          <p>
            <b>Hiệu quả công việc</b> (work efficiency) của một thuật toán là khối lượng việc nó làm gần đến mức nào so với mức
            tối thiểu mà phép tính cần, trong đó "tối thiểu" nghĩa là khối lượng việc ít nhất của mọi thuật toán đã biết. Với scan
            bao gồm, mức tối thiểu là N − 1 phép cộng, đúng như vòng lặp của Fig. 11.1. Scan song song ngây thơ của mục 11.2, mỗi
            đầu ra một luồng cộng tiền tố của mình từ đầu, làm N(N − 1)/2: O(N²). Như vậy là không hiệu quả công việc. Với 1,024
            phần tử, đó là 523,776 phép cộng trong khi 1,023 là đủ.
          </p>
        }
      >
        <p>
          An algorithm's <b>work efficiency</b> is how close the work it does comes to the minimum the computation needs,
          where "minimum" means the least work of any known algorithm. For an inclusive scan the minimum is N − 1 additions,
          what the loop of Fig. 11.1 does. The naive parallel scan of section 11.2, one thread per output summing its prefix
          from scratch, does N(N − 1)/2: O(N²). That is not work efficient. For 1,024 elements it is 523,776 additions where
          1,023 would do.
        </p>
      </Translated>
      <Source section="11.5" pages="265" />

      <h2><Translated vi="Đếm số phép cộng của Kogge-Stone">Counting Kogge-Stone's additions</Translated></h2>
      <Translated
        vi={
          <p>
            Kogge-Stone chạy log₂N bước. Ở bước có stride s, s luồng dưới stride đứng ngoài và N − s luồng còn lại mỗi luồng cộng
            một lần. Cộng lại qua s = 1, 2, 4, …, N/2. Các số hạng N cho N·log₂N; các stride là chuỗi 1 + 2 + … + N/2 = N − 1.
            Kogge-Stone làm N·log₂N − (N − 1) phép cộng, tức O(N·log₂N).
          </p>
        }
      >
        <p>
          Kogge-Stone runs log₂N steps. In the step with stride s, the s threads below the stride sit out and the other N − s
          add once each. Sum that over s = 1, 2, 4, …, N/2. The N terms give N·log₂N; the strides are the series 1 + 2 + … +
          N/2 = N − 1. Kogge-Stone does N·log₂N − (N − 1) additions, which is O(N·log₂N).
        </p>
      </Translated>
      <Source section="11.5" pages="266" figs="Fig. 11.2" />
      <Ask q={q['ch11/11.5/q-work512']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={q['ch11/11.5/q-ratio']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Như vậy tốt hơn nhiều so với O(N²) của scan ngây thơ nhưng vẫn kém O(N) của vòng lặp. Khoảng cách không nhỏ ngay cả
            với các đoạn cỡ vừa: ở 512 phần tử, ví dụ của sách, Kogge-Stone làm gấp tám đến chín lần khối lượng việc của vòng lặp,
            và tỉ lệ tăng khoảng một mỗi lần N gấp đôi.
          </p>
        }
      >
        <p>
          That is far better than the naive scan's O(N²) and still worse than the loop's O(N). The gap is not small even for
          modest segments: at 512 elements, the book's example, Kogge-Stone does between eight and nine times the loop's work,
          and the ratio grows by about one with every doubling of N.
        </p>
      </Translated>

      <h2><Translated vi="Trên phần cứng, gần N·log₂N hơn">On hardware, closer to N·log₂N</Translated></h2>
      <Translated
        vi={
          <p>
            Công thức chỉ tính các luồng có cộng. Trên một thiết bị thật, sách chỉ ra, các luồng đã ngừng cộng vẫn dùng tài nguyên
            thực thi cho đến khi warp chứa chúng hoàn tất. Vì vậy tài nguyên Kogge-Stone tiêu tốn gần N·log₂N hơn là N·log₂N − (N
            − 1). Dù thế nào, bậc vẫn là O(N·log₂N).
          </p>
        }
      >
        <p>
          The formula charges only the threads that add. On a real device, the book points out, threads that have stopped
          adding still use execution resources until the warp they belong to completes. So the resources Kogge-Stone consumes
          are nearer N·log₂N than N·log₂N − (N − 1). Either way the order is O(N·log₂N).
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch04-facts.md A9, B7, C4)', '2026-11-20 (notes/today/ch04-facts.md A9, B7 và C4)')}>
        <Translated
          vi={
            <p>
              NVIDIA: các lane không ở nhánh hiện tại "will be masked off while the threads which follow the branch are executed"
              (sẽ bị che đi trong khi các luồng đi theo nhánh được thực thi). AMD: "ALUs still process the full warp, but the result
              for divergent threads is masked out" (ALU vẫn xử lý toàn bộ warp, nhưng kết quả của các luồng phân kỳ bị che đi).
              Apple không công bố mô hình thực thi nào cho các luồng không hoạt động trong SIMD group.
            </p>
          }
        >
          <p>
            NVIDIA: lanes not on the current path "will be masked off while the threads which follow the branch are executed".
            AMD: "ALUs still process the full warp, but the result for divergent threads is masked out". Apple documents no
            execution model for inactive SIMD-group threads.
          </p>
        </Translated>
      </Today>
      <Ask q={q['ch11/11.5/q-slots']} />

      <h2><Translated vi="Ít bước hơn, nếu bạn có đủ đơn vị">Fewer steps, if you have the units</Translated></h2>
      <Translated
        vi={
          <p>
            Phần việc thêm mua được một span ngắn. Vòng lặp chạy N lần lặp; vòng lặp của kernel chạy tối đa log₂N. Với số đơn vị
            thực thi không giới hạn, Kogge-Stone cần ít bước hơn vòng lặp khoảng N/log₂N lần: 512/9 = 56.9 ở 512 phần tử. Số đơn
            vị không bao giờ là vô hạn. Với P đơn vị thực thi, sách ước lượng Kogge-Stone cần (N·log₂N)/P bước so với N của vòng
            lặp. Nếu P = N thì lại là log₂N; với ít đơn vị hơn, mỗi bước mất vài vòng.
          </p>
        }
      >
        <p>
          The extra work buys a short span. The loop runs N iterations; the kernel's loop runs at most log₂N. With unlimited
          execution units Kogge-Stone needs about N/log₂N times fewer steps than the loop: 512/9 = 56.9 at 512 elements.
          Units are never unlimited. With P execution units the book estimates Kogge-Stone at (N·log₂N)/P steps against the
          loop's N. If P = N that is log₂N again; with fewer units every step takes several rounds.
        </p>
      </Translated>
      <Source section="11.5" pages="266" />
      <Predict
        id="ch11/11.5/p-steps"
        prompt={
          <Translated vi={<>Với 1,024 phần tử và 32 đơn vị thực thi, Kogge-Stone cần ít bước hơn vòng lặp bao nhiêu lần?</>}>
            <>With 1,024 elements and 32 execution units, how many times fewer steps does Kogge-Stone take than the loop?</>
          </Translated>
        }
        unit={tr('times fewer', 'lần ít hơn')}
      >
        {(v) => (
          <Translated
            vi={
              <p>
                Theo mô hình của sách, (1024·10)/32 = <b>{STEPS1024}</b> bước so với 1,024 của vòng lặp: ít hơn{' '}
                <b>{1024 / STEPS1024}</b> lần.{' '}
                {Math.abs(v - 1024 / STEPS1024) < 0.3
                  ? 'Đúng như bạn dự đoán.'
                  : v > 1024 / STEPS1024
                    ? 'Ít hơn bạn nghĩ: 32 đơn vị cho 1,024 luồng nghĩa là mỗi bước trong 10 bước mất 32 vòng.'
                    : 'Nhiều hơn bạn nghĩ: dù mỗi bước mất 32 vòng, 10 bước vẫn ít hơn 1,024.'}{' '}
                32 đơn vị không cho số bước ít hơn 32 lần; phần việc thêm ăn mất một hệ số mười.
              </p>
            }
          >
            <p>
              By the book's model, (1024·10)/32 = <b>{STEPS1024}</b> steps against the loop's 1,024:{' '}
              <b>{1024 / STEPS1024}</b> times fewer.{' '}
              {Math.abs(v - 1024 / STEPS1024) < 0.3
                ? 'That is what you predicted.'
                : v > 1024 / STEPS1024
                  ? 'Less than you expected: 32 units against 1,024 threads means every one of the 10 steps takes 32 rounds.'
                  : 'More than you expected: even at 32 rounds a step, 10 steps are fewer than 1,024.'}{' '}
              The 32 units do not give 32 times fewer steps; the extra work eats a factor of ten of them.
            </p>
          </Translated>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Giờ hãy khám phá. Trình khám phá đếm số phép cộng và số bước cho vòng lặp, Kogge-Stone và Brent-Kung (mục 11.10) với
            kích thước từ 8 đến một triệu phần tử. Cách tính mặc định là của sách: mọi luồng của khối ở mọi bước. Hai cách tính còn
            lại cho thấy bao nhiêu phần trong đó là các luồng rảnh.
          </p>
        }
      >
        <p>
          Now explore it. The explorer counts additions and steps for the loop, Kogge-Stone and Brent-Kung (section 11.10)
          across sizes from 8 to a million elements. The default charge is the book's: every thread of the block in every
          step. The other two charges show how much of that is the idle threads.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi đơn vị một phép cộng mỗi vòng, không gì khác: không bộ nhớ, không rào chắn, không độ trễ, và một bước chỉ bắt đầu
              khi bước trước đã xong. Một "đơn vị" là đơn vị thực thi trừu tượng của sách, không phải con số của thiết bị nào.
            </p>
          }
        >
          <p>
            One addition per unit per round, nothing else: no memory, no barriers, no latency, and a step starts only when
            the previous one has finished. A "unit" is the book's abstract execution unit, not a figure for any device.
          </p>
        </Translated>
      </Simplification>
      <WorkExplorer />

      <h2><Translated vi="Hai cái giá của phần việc thêm">Two prices for the extra work</Translated></h2>
      <Translated
        vi={
          <p>
            Sách nêu hai. Thứ nhất, phần cứng làm nhiều hơn cho cùng kết quả, và nếu P nhỏ, scan song song cần nhiều bước hơn vòng
            lặp và chậm hơn. Thứ hai, ngay cả khi nó nhanh hơn, các phép cộng thêm tiêu tốn thêm năng lượng, khiến nó không hợp
            với nơi năng lượng bị giới hạn, như trên điện thoại. Phần tóm tắt của chương biến điều thứ nhất thành một quy tắc: số
            đơn vị Kogge-Stone cần để hòa vốn với vòng lặp tăng theo kích thước dữ liệu, nên nó hợp với các khối cỡ vừa trên bộ xử
            lý có nhiều đơn vị thực thi.
          </p>
        }
      >
        <p>
          The book names two. First, the hardware does more for the same result, and if P is small the parallel scan needs
          more steps than the loop and is slower. Second, even when it is faster, the extra additions consume additional
          energy, which makes it a poor fit where power is limited, as on a phone. The chapter's summary turns the first into
          a rule: the units Kogge-Stone needs to break even with the loop grow with the data size, so it suits modest-sized
          blocks on processors with plenty of execution units.
        </p>
      </Translated>
      <Source section="11.5" pages="267" figs="§11.11, p. 285" />
      <Ask q={q['ch11/11.5/q-steps']} />
      <Ask q={q['ch11/11.5/q-breakeven']} />
      <Ask q={q['ch11/11.5/q-energy']} />

      <h2><Translated vi="Tiếp theo là gì">What comes next</Translated></h2>
      <Translated
        vi={
          <p>
            Chương này đưa ra hai câu trả lời. <a href={href('ch11', '11.10')}>Mục 11.10</a> trình bày Brent-Kung, làm 2N − 2 −
            log₂N phép cộng, gần 2N, trong khoảng gấp đôi số bước; bạn đã có thể so sánh nó trong trình khám phá. Câu trả lời thực
            sự của sách đến trước, ở <a href={href('ch11', '11.6')}>mục 11.6</a>: cho mỗi luồng scan tuần tự một dãy phần tử, vốn
            hiệu quả công việc, và chỉ dùng scan song song cho số ít giá trị còn lại.
          </p>
        }
      >
        <p>
          The chapter gives two answers. <a href={href('ch11', '11.10')}>Section 11.10</a> presents Brent-Kung, which does 2N
          − 2 − log₂N additions, close to 2N, in about twice the steps; you can already compare it in the explorer. The book's
          real answer comes first, in <a href={href('ch11', '11.6')}>section 11.6</a>: let each thread scan a run of elements
          sequentially, which is work efficient, and keep the parallel scan for the few values left over.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Tổng tích lũy của một bộ lấy mẫu chạy trên một hàng từ vựng, và hàng của Qwen3-8B có {int(VOCAB_QWEN3_8B)} mục. Hãy
            chọn bộ giá trị 2¹⁷ = {int(VOCAB_POW2)} phần tử trên 1,024 đơn vị. Kogge-Stone vẫn cần ít bước hơn nhiều so với một
            luồng chạy vòng lặp, nhưng mỗi bước trong 17 bước của nó chạm vào cả hàng: khoảng {lg(VOCAB_POW2) - 1} lần số phép
            cộng của vòng lặp. Đó là chế độ có nhiều phần tử hơn hẳn số đơn vị, nơi gộp việc cho luồng (thread coarsening) của mục
            11.6 đáng giá. Kernel CUDA và HIP của llama.cpp lẫn kernel Metal của MLX đều giao cho mỗi luồng một dãy 4 giá trị (với
            kiểu 4 byte, trong MLX) để tự scan trước mọi bước giữa các lane, và đi qua một hàng dài theo từng khúc với một tổng
            mang sang.
          </p>
        }
      >
        <p>
          A sampler's cumulative sum runs over one vocabulary row, and Qwen3-8B's has {int(VOCAB_QWEN3_8B)} entries. Pick the
          preset of 2¹⁷ = {int(VOCAB_POW2)} elements on 1,024 units. Kogge-Stone still needs far fewer steps than one thread
          looping, but each of its 17 steps touches the whole row: {lg(VOCAB_POW2) - 1} or so times the loop's additions. That
          is the regime of many more elements than units where section 11.6's coarsening pays. llama.cpp's CUDA and HIP
          kernel and MLX's Metal kernel both give each thread a run of 4 values (for 4-byte types, in MLX) to scan on its own
          before any step between lanes, and walk a long row in chunks with a carried total.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cầu nối sang lấy mẫu là của chúng tôi. Kích thước từ vựng: notes/today/ch09-facts.md D7. <code>cumsum_kernel</code>{' '}
              của llama.cpp và <code>contiguous_scan</code> của MLX, đọc từ mã nguồn ngày 2026-11-20: ch11-facts D1, D3.
            </p>
          }
        >
          <p>
            The sampling bridge is ours. Vocabulary size: notes/today/ch09-facts.md D7. llama.cpp's <code>cumsum_kernel</code>{' '}
            and MLX's <code>contiguous_scan</code>, read from source on 2026-11-20: ch11-facts D1, D3.
          </p>
        </Translated>
      </Beyond>
    </>
  );
}
