import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  FIG12_1,
  arrivalOrder,
  crossings,
  inPlaceOverwrites,
  keepFrom,
  keptOf,
  sameKeys,
  stableFilter,
  unstableFilter,
} from '../../sim/filter';

// Section 12.1, Background (pp. 289-290). The explorer re-draws Fig. 12.1 from
// src/sim/filter.ts; the classification is static data. Vendor and library facts:
// notes/today/ch12-facts.md (A7, A8, B5, C5) and ch09-facts.md (A3, B1).

const N = FIG12_1.n;
const CELL = 34;
const PITCH = 38;
const LEFT = 70;
const SVG_W = LEFT + N * PITCH + 8;
const IN_Y = 22;
const OUT_Y = 100;

const isBook = (keep: boolean[]) => keptOf(keep).join(',') === FIG12_1.kept.join(',');

/** One panel of Fig. 12.1: the input row, the output row and the arrows between them. */
function Panel({ name, keep, output, inPlace, onToggle }: {
  name: 'stable' | 'unstable';
  keep: boolean[];
  output: number[];
  inPlace: boolean;
  onToggle: (k: number) => void;
}) {
  const over =inPlace && name === 'stable' ? inPlaceOverwrites(keep) : [];
  const h = OUT_Y + CELL + (over.length ? 46 : 18);
  const cx = (i: number) => LEFT + i * PITCH + CELL / 2;
  return (
    <svg viewBox={`0 0 ${SVG_W} ${h}`} role="img" aria-label={tr(`${name} filter`, name === 'stable' ? 'bộ lọc ổn định' : 'bộ lọc không ổn định')} data-panel={name}
      data-output={output.map((k) => 'k' + k).join(' ')} data-crossings={crossings(output)}
      style={{ width: '100%', minWidth: 540, display: 'block' }}>
      <text x={0} y={IN_Y + 21} fontSize="12"><Translated vi="đầu vào">input</Translated></text>
      <text x={0} y={OUT_Y + 21} fontSize="12"><Translated vi="đầu ra">output</Translated></text>
      {output.map((k, s) => (
        <line key={'a' + k} x1={cx(k)} y1={IN_Y + CELL + 2} x2={cx(s)} y2={OUT_Y - 3}
          stroke="var(--ink-2)" strokeWidth="1.3" opacity="0.8" />
      ))}
      {keep.map((kept, i) => (
        <g key={i} data-key={i} data-kept={kept ? 1 : 0} style={{ cursor: 'pointer' }} onClick={() => onToggle(i)}>
          <rect x={LEFT + i * PITCH} y={IN_Y} width={CELL} height={CELL} rx="3"
            fill={kept ? 'var(--global-wash)' : 'var(--cell)'} stroke={kept ? 'var(--global)' : 'var(--rule)'} />
          <text x={cx(i)} y={IN_Y + 21} fontSize="12" textAnchor="middle" style={{ fill: kept ? 'var(--ink)' : 'var(--ink-3)' }}>k{i}</text>
        </g>
      ))}
      {output.map((k, s) => (
        <g key={'o' + s}>
          <rect x={LEFT + s * PITCH} y={OUT_Y} width={CELL} height={CELL} rx="3" fill="var(--global-wash)" stroke="var(--global)" />
          <text x={cx(s)} y={OUT_Y + 21} fontSize="12" textAnchor="middle">k{k}</text>
        </g>
      ))}
      {over.length ? (
        <>
          <text x={0} y={OUT_Y + CELL + 16} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="ghi đè">overwrites</Translated></text>
          <text x={0} y={OUT_Y + CELL + 30} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="đọc trước">read first</Translated></text>
        </>
      ) : null}
      {over.map((o) => (
        <g key={'h' + o.slot} data-held={o.held} data-moves={o.heldMovesTo ?? ''} data-kind={o.kind}>
          <text x={cx(o.slot)} y={OUT_Y + CELL + 16} fontSize="11" textAnchor="middle"
            style={{ fill: o.kind === 'own' ? 'var(--ink-3)' : 'var(--fail)' }}>{o.kind === 'own' ? tr('itself', 'chính nó') : `k${o.held}`}</text>
          {o.kind !== 'own' ? (
            <text x={cx(o.slot)} y={OUT_Y + CELL + 30} fontSize="10" textAnchor="middle" style={{ fill: 'var(--fail)' }}>
              {o.kind === 'moved' ? tr(`to ${o.heldMovesTo}`, `tới ${o.heldMovesTo}`) : tr('test', 'kiểm tra')}
            </text>
          ) : null}
        </g>
      ))}
    </svg>
  );
}

function StableOrUnstable() {
  const [keep, setKeep] = useState<boolean[]>(() => keepFrom(N, FIG12_1.kept));
  const [run, setRun] = useState(0);
  const [inPlace, setInPlace] = useState(false);
  const book = isBook(keep);
  // Run 0 on the book's keys is the book's own drawing; every other run is a seeded draw.
  const unstable = useMemo(
    () => unstableFilter(keep, run === 0 && book ? [...FIG12_1.bookUnstable] : arrivalOrder(keep, 1000 + run)),
    [keep, run, book],
  );
  const stable = useMemo(() => stableFilter(keep), [keep]);
  const toggle = (k: number) => {
    setKeep((v) => v.map((x, i) => (i === k ? !x : x)));
    setRun(0);
  };
  const set = (kept: number[]) => {
    setKeep(keepFrom(N, kept));
    setRun(0);
  };
  const writes = inPlace ? inPlaceOverwrites(keep) : [];
  const moved = writes.filter((o) => o.kind === 'moved').length;
  const tested = writes.filter((o) => o.kind === 'tested').length;
  const same = sameKeys(stable, unstable);
  const ordered = unstable.every((k, i) => k === stable[i]);
  return (
    <Bench
      title={tr('Stable or unstable?', 'Ổn định hay không ổn định?')}
      task={tr(
        'Click keys to keep or drop them, then press Run again a few times. What never changes?',
        'Bấm vào các khóa để giữ hoặc bỏ chúng, rồi bấm Chạy lại vài lần. Điều gì không bao giờ thay đổi?',
      )}
      foot={
        <Translated
          vi={
            <>
              Hình 12.1 được vẽ lại: các khóa tô đậm là khóa được giữ. Lần chạy không ổn định đầu tiên trên các khóa của
              sách là thứ tự mà sách vẽ; mỗi lần chạy sau là một thứ tự hoàn thành mới.
            </>
          }
        >
          <>
            Fig. 12.1 drawn again: the shaded keys are kept. The first unstable run on the book's keys is the order the
            book draws; each later run is a new completion order.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button className="primary" onClick={() => setRun((r) => r + 1)}><Translated vi="Chạy lại">Run again</Translated></button>
        <button onClick={() => set([...FIG12_1.kept])} disabled={book}><Translated vi="Các khóa của sách">The book's keys</Translated></button>
        <button onClick={() => set([])}><Translated vi="Bỏ tất cả">Drop all</Translated></button>
        <button onClick={() => set(Array.from({ length: N }, (_, i) => i))}><Translated vi="Giữ tất cả">Keep all</Translated></button>
        <label className="check">
          <input type="checkbox" checked={inPlace} onChange={(e) => setInPlace(e.target.checked)} /> <Translated vi="hiện những gì một bộ lọc ổn định tại chỗ ghi đè">show what an in-place stable filter overwrites</Translated>
        </label>
      </div>
      <div className="mx-label"><Translated vi="Ổn định: các khóa được giữ theo thứ tự đầu vào">Stable: kept keys in input order</Translated></div>
      <Panel name="stable" /* i18n-ignore: panel key */ keep={keep} output={stable} inPlace={inPlace} onToggle={toggle} />
      <p className="status-line" style={{ margin: '0.3rem 0 0.9rem' }} data-line="stable">
        <Translated vi={<>{stable.length} khóa được giữ, {crossings(stable)} mũi tên cắt nhau.</>}>
          {stable.length} kept, {crossings(stable)} crossing arrows.
        </Translated>
        {inPlace ? (
          moved + tested ? (
            <Translated
              vi={
                <>
                  {' '}Khi làm tại chỗ, {moved + tested} trong {stable.length} lần ghi rơi vào một khóa khác, khóa đó phải
                  được đọc trước: {moved} khóa được giữ phải được đọc và di chuyển ("tới"), và {tested} khóa bị bỏ vẫn phải
                  được đọc để kiểm tra ("kiểm tra").
                </>
              }
            >
              <>
                {' '}In place, {moved + tested} of the {stable.length} writes land on another key, which must be read first:{' '}
                {moved} kept {moved === 1 ? 'key' : 'keys'} that must be read and moved ("to"), and {tested} dropped{' '}
                {tested === 1 ? 'key' : 'keys'} that must still be read, to be tested ("test").
              </>
            </Translated>
          ) : stable.length ? (
            <Translated vi={<> Khi làm tại chỗ, mọi khóa được giữ đều được ghi lên chính nó: không có gì phải bảo vệ.</>}>
              <> In place, every kept key is written over itself: nothing to protect.</>
            </Translated>
          ) : null
        ) : null}
      </p>
      <div className="mx-label"><Translated vi={<>Không ổn định: lần chạy {run + 1}</>}>Unstable: run {run + 1}</Translated></div>
      <Panel name="unstable" /* i18n-ignore: panel key */ keep={keep} output={unstable} inPlace={false} onToggle={toggle} />
      <p className="status-line" style={{ margin: '0.3rem 0 0' }} role="status" data-line="unstable">
        <Translated vi={<>{unstable.length} khóa được giữ, {crossings(unstable)} mũi tên cắt nhau.</>}>
          {unstable.length} kept, {crossings(unstable)} crossing arrows.
        </Translated>{' '}
        <span className={'chip ' + (same ? 'ok' : 'fail')} data-check={same ? 'same' : 'different'}>
          {same ? tr('same keys, same count', 'cùng khóa, cùng số lượng') : tr('different keys', 'khác khóa')}
        </span>{' '}
        {unstable.length > 1 ? (ordered ? tr('This run happens to be in input order.', 'Lần chạy này tình cờ theo đúng thứ tự đầu vào.') : tr('Order differs from the stable output.', 'Thứ tự khác với đầu ra ổn định.')) : null}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- does it need order?
type Verdict = 'stable' | 'unstable' | 'none';
const VERDICT_LABEL: Record<Verdict, string> = {
  get stable() {
    return tr('stable needed', 'cần ổn định');
  },
  get unstable() {
    return tr('unstable is enough', 'không ổn định là đủ');
  },
  get none() {
    return tr('no compaction needed', 'không cần nén gọn');
  },
};
const JOBS: { id: string; job: string; answer: Verdict; why: string }[] = [
  {
    id: 'sort-next',
    get job() {
      return tr(
        'Collect the (token id, logit) pairs above a threshold; the next step sorts them by logit, breaking ties by token id.',
        'Thu thập các cặp (token id, logit) trên một ngưỡng; bước tiếp theo sắp xếp chúng theo logit, phá thế hòa bằng token id.',
      );
    },
    answer: 'unstable',
    get why() {
      return tr(
        'The sort decides the final order, ties included, and each id travels with its logit, so the order the filter produced is thrown away. Without the tie rule, a stable sort would keep tied logits in whatever order the filter left them, and that order changes from run to run.',
        'Phép sắp xếp quyết định thứ tự cuối cùng, kể cả các thế hòa, và mỗi id đi kèm logit của nó, nên thứ tự mà bộ lọc tạo ra bị bỏ đi. Không có quy tắc phá thế hòa, một phép sắp xếp ổn định sẽ giữ các logit bằng nhau theo bất kỳ thứ tự nào bộ lọc để lại, và thứ tự đó thay đổi giữa các lần chạy.',
      );
    },
  },
  {
    id: 'dedup',
    get job() {
      return tr(
        'Remove the duplicates from a sorted list of token ids.',
        'Loại bỏ các phần tử trùng lặp khỏi một danh sách token id đã sắp xếp.',
      );
    },
    answer: 'stable',
    get why() {
      return tr(
        'The result must still be sorted. Section 12.8 meets this job again: it is a filter whose test is "differs from the key before me".',
        'Kết quả vẫn phải được sắp xếp. Mục 12.8 gặp lại việc này: đó là một bộ lọc có phép kiểm tra là "khác với khóa đứng trước tôi".',
      );
    },
  },
  {
    id: 'gather',
    get job() {
      return tr(
        'Find the indices of the nonzero entries for a later gather that expects them in ascending order.',
        'Tìm chỉ số của các phần tử khác không cho một phép gather sau đó, vốn cần chúng theo thứ tự tăng dần.',
      );
    },
    answer: 'stable',
    get why() {
      return tr(
        'The consumer depends on the order, so the filter must keep it.',
        'Bên dùng kết quả phụ thuộc vào thứ tự, nên bộ lọc phải giữ nó.',
      );
    },
  },
  {
    id: 'seeded',
    get job() {
      return tr(
        'Draw a token by walking the compacted candidates until a running sum passes a seeded random number; the same seed must give the same token.',
        'Rút một token bằng cách duyệt các ứng viên đã nén gọn cho đến khi tổng tích lũy vượt qua một số ngẫu nhiên có seed; cùng seed phải cho cùng token.',
      );
    },
    answer: 'stable',
    get why() {
      return tr(
        'The walk depends on the array order. Any order that is the same every run would do; a stable filter is the simplest way to get one.',
        'Việc duyệt phụ thuộc vào thứ tự của mảng. Bất kỳ thứ tự nào giống nhau ở mọi lần chạy đều được; bộ lọc ổn định là cách đơn giản nhất để có một thứ tự như vậy.',
      );
    },
  },
  {
    id: 'count',
    get job() {
      return tr(
        'Count how many logits pass a threshold.',
        'Đếm xem có bao nhiêu logit vượt qua một ngưỡng.',
      );
    },
    answer: 'none',
    get why() {
      return tr(
        'Nothing has to move. A count is a reduction (chapter 10).',
        'Không có gì phải di chuyển. Phép đếm là một phép rút gọn (reduction) (chương 10).',
      );
    },
  },
  {
    id: 'kv-free',
    get job() {
      return tr(
        'Collect the ids of free KV-cache blocks, where any free block is as good as another.',
        'Thu thập id của các khối KV cache còn trống, khi khối trống nào cũng tốt như nhau.',
      );
    },
    answer: 'unstable',
    get why() {
      return tr(
        'No later step reads the order. (Our example, not the book\'s.)',
        'Không bước nào sau đó đọc thứ tự. (Ví dụ của khóa học này, không phải của sách.)',
      );
    },
  },
];

function NeedsOrder() {
  const [picks, setPicks] = useState<Record<string, Verdict>>({});
  const right = JOBS.filter((j) => picks[j.id] === j.answer).length;
  const done = Object.keys(picks).length;
  return (
    <Bench
      title={tr('Does it need order?', 'Có cần thứ tự không?')}
      task={tr('Place each job, then read the reason.', 'Xếp từng việc, rồi đọc lý do.')}
      foot={done ? (
        <span data-score={right}>
          <Translated vi={<>{right} trên {JOBS.length} việc được xếp như chúng tôi sẽ xếp ({done} đã trả lời).</>}>
            {right} of {JOBS.length} placed as we would place them ({done} answered).
          </Translated>
        </span>
      ) : tr('Six jobs from an inference server.', 'Sáu việc từ một máy chủ suy luận.')}
    >
      <div style={{ display: 'grid', gap: '0.7rem' }}>
        {JOBS.map((j) => {
          const pick = picks[j.id];
          return (
            <div key={j.id} className="panel" data-job={j.id} data-pick={pick ?? ''} data-right={pick ? (pick === j.answer ? 1 : 0) : ''}>
              <p style={{ margin: '0 0 0.5rem' }}>{j.job}</p>
              <div className="seg" role="group" aria-label={tr('Place: ' + j.id, 'Xếp: ' + j.id)}>
                {(Object.keys(VERDICT_LABEL) as Verdict[]).map((v) => (
                  <button key={v} aria-pressed={pick === v} onClick={() => setPicks((p) => ({ ...p, [j.id]: v }))}>
                    {VERDICT_LABEL[v]}
                  </button>
                ))}
              </div>
              {pick ? (
                <p style={{ margin: '0.5rem 0 0' }}>
                  <span className={'chip ' + (pick === j.answer ? 'ok' : 'fail')}>{pick === j.answer ? tr('yes', 'đúng') : tr(`we would say: ${VERDICT_LABEL[j.answer]}`, `chúng tôi sẽ chọn: ${VERDICT_LABEL[j.answer]}`)}</span>{' '}
                  {j.why}
                </p>
              ) : null}
            </div>
          );
        })}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS121: Record<string, Question> = {
  'ch12/12.1/q-reproduce': {
    kind: 'choice',
    id: 'ch12/12.1/q-reproduce',
    prompt: (
      <Translated
        vi={
          <>
            Một bộ lấy mẫu nén gọn các cặp (token id, xác suất) trên một ngưỡng bằng một bộ lọc không ổn định, rồi rút một
            token bằng cách duyệt mảng đã nén gọn cho đến khi tổng tích lũy của các xác suất vượt qua một số ngẫu nhiên có
            seed u. Cùng logits, cùng seed, hai lần chạy. Điều nào đúng?
          </>
        }
      >
        <>
          A sampler compacts the (token id, probability) pairs above a threshold with an unstable filter, then draws a token
          by walking the compacted array until the running sum of probabilities passes a seeded random number u. Same
          logits, same seed, two runs. Which holds?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Lần nào cũng cùng một token: seed cố định u, và tập ứng viên cùng xác suất của chúng giống nhau ở mọi lần chạy.">
            The same token every time: the seed fixes u, and the set of candidates and their probabilities is the same in every run.
          </Translated>
        ),
        why: (
          <Translated vi="u cố định, nhưng việc duyệt đi qua các ứng viên theo thứ tự của mảng. Một thứ tự khác đặt một token khác vào chỗ tổng tích lũy vượt qua u.">
            u is fixed, but the walk visits the candidates in array order. Another order puts a different token where the running sum crosses u.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Phân phối bị sai, vì các xác suất không theo thứ tự.">
            The distribution is wrong, because the probabilities are out of order.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi id đi kèm xác suất của nó, và tập các cặp giống nhau ở mọi lần chạy. Duyệt một danh sách bị xáo trộn vẫn chọn mỗi token với đúng xác suất của nó.">
            Each id travels with its probability, and the set of pairs is the same in every run. Walking a shuffled list still picks each token with its own probability.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Phân phối của các lần rút là đúng, nhưng token được chọn cho một seed cho trước có thể khác nhau giữa các lần chạy.">
            The distribution of draws is right, but the token chosen for a given seed can differ between runs.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Bộ lọc cố định tập ứng viên, không cố định thứ tự của chúng. Việc lấy mẫu chỉ cần tập đó, nên nó vẫn đúng về
            phân phối. Tái tạo một lần rút từ một seed thì cần cả thứ tự. Hoặc giữ thứ tự đầu vào (một bộ lọc ổn định), hoặc
            bảo đảm thứ tự giống nhau ở mọi lần chạy.
          </p>
        }
      >
        <p>
          The filter fixes the set of candidates, not their order. Sampling only needs the set, so it stays correct in
          distribution. Reproducing a draw from a seed needs the order too. Either keep the input order (a stable filter)
          or make sure the order is the same every run.
        </p>
      </Translated>
    ),
    source: 'our analysis of §12.1, p. 290',
    vi: { source: 'phân tích của khóa học này về §12.1, p. 290' },
  },
  'ch12/12.1/q-inplace-slot': {
    kind: 'numeric',
    id: 'ch12/12.1/q-inplace-slot',
    prompt: (
      <Translated
        vi={
          <>
            Các khóa k0 đến k7. Giữ k1, k3, k4 và k6, nén gọn tại chỗ, theo thứ tự ổn định. k6 được chuyển vào vị trí nào?
          </>
        }
      >
        <>
          Keys k0 to k7. Keep k1, k3, k4 and k6, compacted in place, in stable order. Into which position does k6 move?
        </>
      </Translated>
    ),
    answer: 3,
    unit: 'position',
    explain: (
      <Translated
        vi={
          <p>
            k1 về 0, k3 về 1, k4 về 2, k6 về 3. Vị trí 3 đang chứa k3, mà chính k3 cũng chuyển về vị trí 1, nên k3 phải được
            đọc trước khi k6 được ghi đè lên nó. Đó là mối nguy mà sách nêu ở p. 290; mục 12.7 sắp thứ tự các lần đọc và ghi
            để điều đó không thể xảy ra.
          </p>
        }
      >
        <p>
          k1 goes to 0, k3 to 1, k4 to 2, k6 to 3. Position 3 held k3, which itself moves to position 1, so k3 must be read
          before k6 is written over it. That is the hazard the book names on p. 290; section 12.7 orders the reads and the
          writes so it cannot happen.
        </p>
      </Translated>
    ),
    source: '§12.1, p. 290; §12.7, p. 300',
    vi: { unit: 'vị trí' },
  },
  'ch12/12.1/q-which-place': {
    kind: 'choice',
    id: 'ch12/12.1/q-which-place',
    prompt: (
      <Translated
        vi={
          <>
            Một danh sách chiếm gần hết bộ nhớ của thiết bị, và công việc chỉ cần các khóa còn lại sau đó. Không tại chỗ hay
            tại chỗ, và làm tại chỗ khiến bạn tốn gì?
          </>
        }
      >
        <>
          A list fills most of a device's memory, and the job only needs the surviving keys afterwards. Out of place or
          in place, and what does in place cost you?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không tại chỗ: làm tại chỗ là không thể trên GPU, vì các luồng chạy không theo thứ tự cố định và không thể chờ nhau.">
            Out of place: in place is impossible on a GPU, because threads run in no fixed order and cannot wait for one another.
          </Translated>
        ),
        why: (
          <Translated vi="Mục 12.7 làm được điều đó, bằng cách xếp các lần đọc trước các lần ghi có thể đè lên chúng.">
            Section 12.7 does it, by ordering the reads before the writes they could clobber.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Tại chỗ, vì danh sách thứ hai sẽ không vừa; nhưng không vị trí nào được ghi trước khi khóa cũ của nó được đọc.">
            In place, since a second list will not fit; but no position may be written before its old key is read.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Tại chỗ, và không tốn thêm gì: các khóa chỉ chuyển về vị trí thấp hơn, nên danh sách không bao giờ cần nhiều chỗ hơn chỗ nó đang có.">
            In place, and it costs nothing extra: keys only move to lower positions, so the list never needs more room than it has.
          </Translated>
        ),
        why: (
          <Translated vi="Nó thêm các ràng buộc thứ tự giữa các luồng, và giữa các khối khi một khóa chuyển vào phạm vi của khối khác.">
            It adds ordering constraints between threads, and between blocks when a key moves into another block's range.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Không tại chỗ cần chỗ cho cả một danh sách thứ hai, đúng thứ mà công việc này không có. Tại chỗ giữ một danh sách
            nhưng thêm một sự phụ thuộc: một khóa có thể chuyển vào một vị trí mà khóa cũ ở đó chưa được đọc.
          </p>
        }
      >
        <p>
          Out of place needs room for a whole second list, which is exactly what this job does not have. In place keeps
          one list but adds a dependence: a key may move into a position whose old key has not been read yet.
        </p>
      </Translated>
    ),
    source: '§12.1, p. 290; §12.7, p. 300',
  },
};

export function S121() {
  return (
    <>
      <Translated
        vi={
            <p>
              Một bộ lọc nhận một danh sách, loại bỏ các phần tử không qua một phép kiểm tra, và lấp các khoảng trống, để các
              phần tử còn lại nằm sát nhau, không có lỗ hổng nào. Sách gọi các phần tử của danh sách là <b>khóa</b> (keys). Nếu
              xem như hai danh sách, các khóa được giữ và các khóa bị bỏ, thì bộ lọc là một phép trộn (merge) chạy ngược: chương
              13 gộp hai danh sách thành một, còn bộ lọc tách một danh sách và giữ lại một phần.
            </p>
        }
      >
        <p>
          A filter takes a list, removes the items that fail a test, and closes the gaps, so the survivors sit next to
          each other with no holes between them. The book calls the items of the list <b>keys</b>. Seen as two lists, the
          keys kept and the keys dropped, a filter is a merge run backwards: chapter 13 combines two lists into one, a
          filter splits one list and keeps a part.
        </p>
      </Translated>
      <Source section="12.1" pages="289–290" />
      <Translated
        vi={
            <p>
              Ví dụ của sách là quản lý bộ nhớ. Sau khi một chương trình giải phóng nhiều đối tượng, heap của nó là một mảng chắp
              vá các khoảng trống nhỏ; gộp lại chúng có thể chứa rất nhiều chỗ, nhưng có thể không khoảng nào đủ lớn cho một lần
              cấp phát lớn. Dồn các đối tượng còn sống lại với nhau để lại một vùng trống duy nhất ở cuối. Với một heap lớn, nhiều
              luồng cùng di chuyển đối tượng một lúc sẽ rút ngắn thời gian.
            </p>
        }
      >
        <p>
          The book's example is memory management. After a program frees many objects, its heap is a patchwork of small
          free gaps; together they may hold plenty of space, but none may be big enough for one large allocation. Moving
          the live objects together leaves a single free region at the end. For a large heap, many threads moving objects
          at once cut the time.
        </p>
      </Translated>

      <h2><Translated vi="Không tại chỗ hay tại chỗ">Out of place or in place</Translated></h2>
      <Translated
        vi={
            <p>
              Một bộ lọc <b>không tại chỗ</b> (out-of-place) ghi các phần tử còn lại vào một danh sách mới. Đây là loại đơn giản
              hơn và là loại thường gặp trong tìm kiếm và truy xuất, nơi các phần tử không liên quan bị bỏ để dựng một danh sách
              các phần tử liên quan. Cái giá của nó là chỗ cho danh sách thứ hai đó; với heap, là chỗ cho một heap thứ hai.
            </p>
        }
      >
        <p>
          An <b>out-of-place</b> filter writes the survivors to a new list. It is the simpler kind and the usual one in
          search and retrieval, where irrelevant items are dropped to build a list of relevant ones. Its price is room
          for that second list; for the heap, room for a second heap.
        </p>
      </Translated>
      <Source section="12.1" pages="290" />
      <Translated
        vi={
            <p>
              Một bộ lọc <b>tại chỗ</b> (in-place) nén gọn danh sách ngay tại nơi nó nằm, giúp tiết kiệm chỗ đó nhưng thêm một
              mối nguy. Một khóa chuyển về một vị trí sớm hơn, và vị trí đó có thể vẫn chứa một khóa chưa được đọc. Với các khóa
              của Hình 12.1 bên dưới, k3 rơi vào chỗ k1 từng nằm: ghi k3 trước khi k1 được đọc, và k1 bị mất. Một khóa bị bỏ cũng
              cần cẩn thận như vậy, vì luồng của nó phải đọc nó để kiểm tra trước khi có gì đó ghi đè lên. Mục 12.7 chỉ ra cách
              sắp thứ tự các lần đọc và ghi; sách xây các bộ lọc tại chỗ của mình trên phép scan của chương 11.
            </p>
        }
      >
        <p>
          An <b>in-place</b> filter compacts the list where it lies, which saves that room and adds a hazard. A key moves
          to an earlier position, and that position may still hold a key that has not been read yet. With the keys of
          Fig. 12.1 below, k3 lands where k1 lived: write k3 before k1 has been read, and k1 is lost. A dropped key needs
          the same care, because its thread must read it to test it before anything lands on top. Section 12.7 shows
          how to order the reads and the writes; the book builds its in-place filters on the scan of chapter 11.
        </p>
      </Translated>
      <Ask q={questionsS121['ch12/12.1/q-which-place']} />
      <Ask q={questionsS121['ch12/12.1/q-inplace-slot']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Ổn định hay không ổn định">Stable or unstable</Translated></h2>
      <Translated
        vi={
            <p>
              Một bộ lọc <b>ổn định</b> (stable) giữ các phần tử còn lại theo thứ tự đầu vào. Một bộ lọc <b>không ổn định</b>{' '}
              (unstable) giữ cùng các phần tử đó nhưng theo một thứ tự nào đó, điều này ổn khi không bước nào phía sau nhìn vào
              thứ tự. Chỉ một tính chất này định hình phần còn lại của chương. Bộ lọc không ổn định cho mỗi luồng chiếm bất kỳ ô
              trống nào (mục 12.2 đến 12.4). Bộ lọc ổn định phải tính chính xác mỗi khóa đi đâu, và việc đó cần một phép scan
              (mục 12.5 đến 12.7).
            </p>
        }
      >
        <p>
          A <b>stable</b> filter keeps the survivors in their input order. An <b>unstable</b> filter keeps the same
          survivors in some order, which is fine whenever nothing downstream looks at the order. This one property shapes
          the rest of the chapter. An unstable filter lets each thread grab any free slot (sections 12.2 to 12.4). A
          stable filter must work out exactly where each key goes, and that takes a scan (sections 12.5 to 12.7).
        </p>
      </Translated>
      <Source section="12.1" pages="290" figs={tr('Fig. 12.1', 'Hình 12.1')} />
      <Translated
        vi={
            <p>
              Trình khám phá vẽ Hình 12.1: mười sáu khóa, các khóa tô đậm được giữ, một bộ lọc ổn định ở trên và một bộ lọc không
              ổn định ở dưới. Bấm vào các khóa để đổi khóa nào được giữ, bấm <i>Chạy lại</i> vài lần, và quan sát các mũi tên cùng
              dòng chữ dưới mỗi bảng. Sau đó đánh dấu ô tại chỗ và tìm những lần ghi rơi vào một khóa chưa ai đọc.
            </p>
        }
      >
        <p>
          The explorer draws Fig. 12.1: sixteen keys, the shaded ones kept, a stable filter above and an unstable one
          below. Click keys to change which are kept, press <i>Run again</i> a few times, and watch the arrows and the
          line under each panel. Then tick the in-place box and find the writes that land on a key nobody has read yet.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Thứ tự không ổn định là một lần rút của một phép xáo trộn có seed, đại diện cho thứ tự mà các phép toán nguyên
              tử (atomic) của các luồng tình cờ hoàn thành. Không nhà sản xuất nào cố định thứ tự đó (ch09-facts A3, B1), và
              đây không phải là mô hình về cách bất kỳ phần cứng nào chọn nó.
            </p>
          }
        >
          <p>
            The unstable order is one draw of a seeded shuffle, standing for the order in which the threads' atomic
            operations happen to complete. No vendor fixes that order (ch09-facts A3, B1), and this is not a model of how
            any hardware picks it.
          </p>
        </Translated>
      </Simplification>
      <StableOrUnstable />
      <Translated
        vi={
            <p>
              Hai điều không bao giờ thay đổi giữa các lần chạy: khóa nào đi ra, và bao nhiêu khóa. Chỉ thứ tự của chúng thay đổi.
              Ở bảng ổn định không có hai mũi tên nào cắt nhau, đó chỉ là một cách khác để nói thứ tự được giữ; ở bảng không ổn
              định chúng thường cắt nhau.
            </p>
        }
      >
        <p>
          Two things never change from run to run: which keys come out, and how many. Only their order moves. In the
          stable panel no two arrows ever cross, which is just another way to say the order is kept; in the unstable
          panel they usually do.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Bản PDF ghi chú thích cả hai bảng của Hình 12.1 là "(a)". Bảng thứ hai, bộ lọc ổn định, lẽ ra phải là (b).
            </p>
          }
        >
          <p>
            The PDF captions both panels of Fig. 12.1 "(a)". The second one, the stable filter, should be (b).
          </p>
        </Translated>
      </Aside>

      <h2><Translated vi="Một công việc có cần thứ tự không?">Does a job need order?</Translated></h2>
      <Translated
        vi={
            <p>
              Không ổn định là đủ khi bước tiếp theo không đọc thứ tự. Cần ổn định khi danh sách đã được sắp xếp và phải giữ nguyên
              như vậy, và cả khi một kết quả phải ra giống nhau ở mọi lần chạy. Hãy xếp sáu việc từ một máy chủ suy luận.
            </p>
        }
      >
        <p>
          Unstable is enough when the next step does not read the order. Stable is needed when the list was sorted and
          must stay sorted, and also when a result has to come out the same in every run. Place six jobs from an
          inference server.
        </p>
      </Translated>
      <Source section="12.1" pages="290" />
      <NeedsOrder />
      <Translated
        vi={
            <p>
              Việc lấy mẫu cho thấy vì sao thứ tự có thể quan trọng ngay cả khi không có gì được sắp xếp. Nén gọn các cặp ứng viên
              (token id, xác suất) trên một ngưỡng, rồi chọn một token bằng cách duyệt mảng đã nén gọn cho đến khi tổng tích lũy
              vượt qua một số ngẫu nhiên có seed. Vì mỗi id đi kèm xác suất của nó, phân phối của các lần rút là đúng với bất kỳ
              thứ tự nào. Nhưng việc duyệt đi qua các ứng viên theo thứ tự của mảng, nên sau một bộ lọc không ổn định, cùng một
              seed có thể rơi vào một token khác ở lần chạy sau.
            </p>
        }
      >
        <p>
          Sampling shows why order can matter even when nothing is sorted. Compact the candidate (token id, probability)
          pairs above a threshold, then pick a token by walking the compacted array until a running sum passes a seeded
          random number. Because each id travels with its probability, the distribution of draws is right in any order.
          But the walk visits candidates in array order, so after an unstable filter the same seed can land on a
          different token in the next run.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách không bàn về lấy mẫu; lập luận này là của khóa học này. Chỗ PyTorch giữ thứ tự vì cùng lý do đó nằm ở{' '}
              <a href={href('ch12', 'wrap')}>trang tổng kết</a>.
            </p>
          }
        >
          <p>
            The book does not discuss sampling; this argument is ours. Where PyTorch keeps order for the same reason is
            on the <a href={href('ch12', 'wrap')}>summary page</a>.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS121['ch12/12.1/q-reproduce']} />
      <Translated
        vi={
            <p>
              Vậy tại sao lại học bộ lọc không ổn định? Vì khi thứ tự không quan trọng, chúng làm ít việc hơn. Chính bài viết của
              NVIDIA về kỹ thuật warp của chương này nói thẳng: bộ lọc ổn định được xây trên tổng tiền tố (prefix sum) và "is more
              expensive as a result", trong khi "a purely atomic approach is simpler and performs less work". Ba mục tiếp theo xây
              cách tiếp cận nguyên tử đó và làm nó nhanh; các <a href={href('ch12', 'labs')}>bài thực hành</a> đo nó.
            </p>
        }
      >
        <p>
          Why learn unstable filters at all, then? Because when order does not matter they do less work. NVIDIA's own
          write-up of this chapter's warp technique puts it plainly: a stable filter is built on a prefix sum and "is
          more expensive as a result", while "a purely atomic approach is simpler and performs less work". The next
          three sections build that atomic approach and make it fast; the <a href={href('ch12', 'labs')}>labs</a> measure
          it.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch12-facts.md A7, A8, B5, C5)', '2026-11-20 (notes/today/ch12-facts.md, mục A7, A8, B5, C5)')}>
        <Translated
          vi={
            <p>
              Mọi bộ lọc trong thư viện tìm được đều ổn định. <code>copy_if</code> của Thrust "is stable"; các bản sao của{' '}
              <code>DeviceSelect</code> trong CUB "maintain their original relative ordering", và nó có các overload tại
              chỗ. Không tìm thấy phép select không ổn định nào ở cả hai. Trên AMD, hipCUB chuyển tiếp sang rocPRIM, mà tài
              liệu chỉ cho thấy thứ tự qua ví dụ. Trên GPU của Apple, phép nén gọn duy nhất tìm thấy trong thư viện là{' '}
              <code>nonZeroIndices</code> của MPSGraph (macOS 14 trở lên, chỉ số <code>int32</code>).
            </p>
          }
        >
          <p>
            Every library filter found is stable. Thrust's <code>copy_if</code> "is stable"; CUB's{' '}
            <code>DeviceSelect</code> copies "maintain their original relative ordering", and it has in-place overloads.
            No unstable select was found in either. On AMD, hipCUB forwards to rocPRIM, whose documentation shows the
            order only by example. On Apple GPUs the one library compaction found is MPSGraph's{' '}
            <code>nonZeroIndices</code> (macOS 14 and later, <code>int32</code> indices).
          </p>
        </Translated>
      </Today>
    </>
  );
}
