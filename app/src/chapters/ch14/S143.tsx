import { useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  BLOCK_CHOICES,
  N_CHOICES,
  PANEL_KEYS,
  PANEL_SEG,
  blocksOf,
  log2i,
  mergePlan,
  mergeSortStages,
  type StageRow,
  type TieRule,
} from '../../sim/mergestages';

const fmt = (v: number) => v.toLocaleString('en-US');
/** A key with a letter for its place in the input: a came before b. */
const K = (k: number, t?: string) => <>{k}{t ? <sub>{t}</sub> : null}</>;
const keys = (ks: [number, string?][]) => ks.map(([k, t], j) => <span key={j}>{j ? ', ' : ''}{K(k, t)}</span>);

// q-morework: the last stage of a sort of 2^20 keys, with the explorer's steps per search.
const MW_N = 2 ** 20;
const MW_PER_THREAD = 8;
const MW_STEPS = mergePlan({ n: MW_N, segLen: 1024, blocks: 64 }).rows.at(-1)!.stepsPerSearch;
const MW_SEARCH_STEPS = (MW_N / MW_PER_THREAD) * MW_STEPS;
const pow = (n: number) => `2^${log2i(n)} = ${fmt(n)}`;
const range = (bs: number[]) => (bs.length === 1 ? tr(`block ${bs[0]}`, `khối ${bs[0]}`) : tr(`blocks ${bs[0]}–${bs[bs.length - 1]}`, `các khối ${bs[0]}–${bs[bs.length - 1]}`));

/** How the grid is shared in one stage, in words. */
function share(r: StageRow): string {
  const unit = r.stage === 0 ? 'segment' : 'merge'; // i18n-ignore: English unit word; the Vietnamese strings below name the unit themselves
  return r.blocksPerMerge > 0
    ? tr(`${r.blocksPerMerge} block${r.blocksPerMerge > 1 ? 's' : ''} per ${unit}`, `${r.blocksPerMerge} khối mỗi ${r.stage === 0 ? 'đoạn' : 'phép trộn'}`)
    : tr(`${r.mergesPerBlock} ${unit}s per block`, `${r.mergesPerBlock} ${r.stage === 0 ? 'đoạn' : 'phép trộn'} mỗi khối`);
}

const TW = 720;
const MAX_ROWS_DRAWN = 64; // segments drawn one by one up to this many

function Tree({ rows, stage, n }: { rows: StageRow[]; stage: number; n: number }) {
  const rh = rows.length > 12 ? 22 : 30;
  const x0 = 70;
  const w = TW - x0 - 10;
  const cur = rows[stage];
  const merges = Math.max(cur.segments, 1);
  return (
    <svg viewBox={`0 0 ${TW} ${rows.length * rh + 54}`} role="img" aria-label={tr('Merge tree', 'Cây trộn')} style={{ width: '100%', maxWidth: TW, minWidth: 560, display: 'block' }}
      data-stage={stage} data-segments={cur.segments}>
      {rows.map((r, s) => {
        const y = 8 + s * rh;
        const on = s === stage;
        const many = r.segments > MAX_ROWS_DRAWN;
        return (
          <g key={s} opacity={s > stage ? 0.35 : 1}>
            <text x={x0 - 8} y={y + 16} fontSize="11.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
              {s === 0 ? tr('sorted', 'đã sắp') : tr(`stage ${s}`, `giai đoạn ${s}`)}
            </text>
            {many ? (
              <g>
                <rect x={x0} y={y} width={w} height={rh - 8} fill="var(--global-wash)" stroke="var(--global)" strokeWidth={on ? 2.2 : 1} />
                <text x={x0 + w / 2} y={y + 15} fontSize="11.5" textAnchor="middle">
                  <Translated vi={<>{fmt(r.segments)} đoạn, mỗi đoạn {fmt(r.elementsPerMerge)}</>}>{fmt(r.segments)} segments of {fmt(r.elementsPerMerge)}</Translated>
                </text>
              </g>
            ) : (
              Array.from({ length: r.segments }, (_, k) => (
                <rect key={k} x={x0 + (k * w) / r.segments + 1} y={y} width={w / r.segments - 2} height={rh - 8} rx="2"
                  fill="var(--global-wash)" stroke="var(--global)" strokeWidth={on ? 2.2 : 1} />
              ))
            )}
          </g>
        );
      })}
      {/* who works on what, under the current stage */}
      <g data-assign="">
        {merges <= 16
          ? Array.from({ length: merges }, (_, m) => (
              <text key={m} x={x0 + ((m + 0.5) * w) / merges} y={rows.length * rh + 26} fontSize="11" textAnchor="middle" data-merge={m}>
                {range(blocksOf(cur, m))}
              </text>
            ))
          : (
            <text x={x0 + w / 2} y={rows.length * rh + 26} fontSize="11.5" textAnchor="middle">
              <Translated vi={<>{fmt(merges)} {stage === 0 ? 'đoạn' : 'phép trộn'}: {share(cur)}</>}>
                {fmt(merges)} {stage === 0 ? 'segments' : 'merges'}: {share(cur)}
              </Translated>
            </text>
          )}
        <text x={x0} y={rows.length * rh + 46} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
          <Translated
            vi={<>{n.toLocaleString('en-US')} khóa; giai đoạn hiện tại được vẽ đậm, {stage === 0 ? 'mỗi đoạn được sắp riêng' : `mỗi ô là đầu ra của một phép trộn hai đầu vào ${fmt(cur.elementsPerMerge / 2)} khóa`}</>}
          >
            {n.toLocaleString('en-US')} keys; the stage drawn in bold, {stage === 0 ? 'each segment sorted on its own' : `each box is the output of one merge of two ${fmt(cur.elementsPerMerge / 2)}-key inputs`}
          </Translated>
        </text>
      </g>
    </svg>
  );
}

function MergeTree() {
  const [n, setN] = useState(2 ** 20);
  const [seg, setSeg] = useState(2 ** 17);
  const [blocks, setBlocks] = useState(8);
  const [stage, setStage] = useState(0);
  const segOptions = Array.from({ length: log2i(n) - 3 }, (_, k) => 2 ** (k + 3)).filter((s) => s < n);
  const segLen = Math.min(seg, n / 2);
  const plan = mergePlan({ n, segLen, blocks });
  const st = Math.min(stage, plan.stages);
  return (
    <Bench
      title={tr('Merge tree', 'Cây trộn')}
      task={
        <Translated vi="Bắt đầu với tám khối của sách, kéo thanh giai đoạn, rồi cho lưới ít khối hơn số đoạn.">
          Start at the book's eight blocks, move the stage slider, then give the grid fewer blocks than segments.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Một lần tìm là một lần tìm co-rank, xác định phần đầu ra của một khối trong một phép trộn bắt đầu ở đâu trong
              hai đầu vào: một phép trộn được b khối chia nhau cần b − 1 lần tìm. Các khóa nằm trong bộ nhớ toàn cục (global
              memory); mỗi giai đoạn đọc và ghi tất cả chúng một lần.
            </>
          }
        >
          <>
            A search is a co-rank search that finds where one block's share of a merge's output starts in the two inputs:
            a merge shared by b blocks needs b − 1 of them. The keys live in global memory; each stage reads and writes all of
            them once.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Số khóa">Keys</Translated>
          <select value={n} onChange={(e) => { setN(+e.target.value); setStage(0); }}>
            {N_CHOICES.map((v) => <option key={v} value={v}>{pow(v)}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Độ dài đoạn ban đầu">Initial segment length</Translated>
          <select value={segLen} onChange={(e) => { setSeg(+e.target.value); setStage(0); }}>
            {segOptions.map((v) => <option key={v} value={v}>{pow(v)}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Số khối trong lưới">Blocks in the grid</Translated>
          <select value={blocks} onChange={(e) => setBlocks(+e.target.value)}>
            {BLOCK_CHOICES.map((v) => <option key={v} value={v}>{fmt(v)}</option>)}
          </select>
        </label>
        <button onClick={() => { setN(2 ** 20); setSeg(2 ** 17); setBlocks(8); setStage(0); }}>
          <Translated vi="Tám khối của sách">The book's eight blocks</Translated>
        </button>
      </div>
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <label className="field" style={{ flex: '1 1 16rem' }}>
          <Translated vi={<>Giai đoạn: {st === 0 ? 'các đoạn đã sắp' : `giai đoạn trộn ${st} trên ${plan.stages}`}</>}>
            Stage: {st === 0 ? 'segments sorted' : `merge stage ${st} of ${plan.stages}`}
          </Translated>
          <input type="range" min={0} max={plan.stages} value={st} aria-label={tr('stage', 'giai đoạn')} onChange={(e) => setStage(+e.target.value)} />
        </label>
        <span className="ui" data-summary="">
          <Translated vi={<>{fmt(plan.segments)} đoạn, nên có {plan.stages} giai đoạn trộn</>}>
            {fmt(plan.segments)} segments, so {plan.stages} merge stage{plan.stages === 1 ? '' : 's'}
          </Translated>
        </span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <Tree rows={plan.rows} stage={st} n={n} />
      </div>
      <table className="threads" aria-label={tr('Stages', 'Các giai đoạn')} style={{ marginTop: '0.8rem' }}>
        <thead>
          <tr>
            <th><Translated vi="giai đoạn">stage</Translated></th>
            <th><Translated vi="số đoạn sau đó">segments after</Translated></th>
            <th><Translated vi="số phép trộn">merges</Translated></th>
            <th><Translated vi="cách chia lưới">how the grid is shared</Translated></th>
            <th><Translated vi="khóa mỗi phép trộn">keys per merge</Translated></th>
            <th><Translated vi="số lần tìm điểm chia">split searches</Translated></th>
            <th><Translated vi="số bước mỗi lần tìm, xấp xỉ">steps per search, about</Translated></th>
          </tr>
        </thead>
        <tbody>
          {plan.rows.map((r) => (
            <tr key={r.stage} className={r.stage === st ? 'cur' : ''} data-stage-row={r.stage} data-share={share(r)} data-searches={r.splitSearches}>
              <td>{r.stage === 0 ? tr('sort', 'sắp xếp') : r.stage}</td>
              <td>{fmt(r.segments)}</td>
              <td>{r.stage === 0 ? '' : fmt(r.merges)}</td>
              <td>{share(r)}</td>
              <td>{r.stage === 0 ? tr(`${fmt(r.elementsPerMerge)} per segment`, `${fmt(r.elementsPerMerge)} mỗi đoạn`) : fmt(r.elementsPerMerge)}</td>
              <td>{r.stage === 0 ? '' : fmt(r.splitSearches)}</td>
              <td>{r.stage === 0 ? '' : r.stepsPerSearch}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Bench>
  );
}

function SixteenKeys() {
  const [stage, setStage] = useState(0);
  const [tie, setTie] = useState<TieRule>('left');
  const snaps = mergeSortStages(PANEL_KEYS, PANEL_SEG, tie);
  const s = snaps[Math.min(stage, snaps.length - 1)];
  const final = stage === snaps.length - 1;
  // Tie groups by key, in output order, once the sort is complete.
  const groups = new Map<number, number[]>();
  for (const it of s.items) groups.set(it.key, [...(groups.get(it.key) ?? []), it.id]);
  const tied = [...groups].filter(([, ids]) => ids.length > 1);
  const okKey = (k: number) => {
    const ids = groups.get(k)!;
    return ids.every((v, j) => j === 0 || ids[j - 1] < v);
  };
  return (
    <Bench
      title={tr('Sixteen keys, merged for real', 'Mười sáu khóa, trộn thật')}
      task={
        <Translated vi="Đi qua từng giai đoạn, rồi đổi quy tắc xử lý giá trị trùng và nhìn các khóa lặp lại.">
          Step through the stages, then switch the tie rule and look at the repeated keys.
        </Translated>
      }
      foot={
        <Translated vi={<>Số nhỏ dưới mỗi khóa là vị trí của nó trong đầu vào, để bạn thấy các khóa bằng nhau có giữ thứ tự hay không. Tám đoạn, mỗi đoạn hai khóa, như tám khối của sách sẽ có.</>}>
          <>Small numbers under each key are its position in the input, so you can see whether equal keys kept their order. Eight segments of two, as the book's eight blocks would have.</>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Stage', 'Giai đoạn')}>
          {snaps.map((x) => (
            <button key={x.stage} aria-pressed={x.stage === stage} onClick={() => setStage(x.stage)}>
              {x.stage === 0 ? tr('segments sorted', 'các đoạn đã sắp') : tr(`merge ${x.stage}`, `trộn ${x.stage}`)}
            </button>
          ))}
        </span>
        <span className="seg" role="group" aria-label={tr('On equal keys', 'Khi khóa bằng nhau')}>
          <button aria-pressed={tie === 'left'} onClick={() => setTie('left')}>
            <Translated vi="trùng: lấy dải bên trái trước">ties: take the left run first</Translated>
          </button>
          <button aria-pressed={tie === 'right'} onClick={() => setTie('right')}>
            <Translated vi="trùng: lấy dải bên phải trước">ties: take the right run first</Translated>
          </button>
        </span>
      </div>
      <div className="mx-label"><Translated vi="Đầu vào">Input</Translated></div>
      <div className="mono muted" style={{ marginBottom: '0.6rem' }}>{PANEL_KEYS.join(' ')}</div>
      <div className="mx-label">
        <Translated vi={<>Sau {s.stage === 0 ? 'khi sắp từng đoạn' : `giai đoạn trộn ${s.stage}`}: các dải dài {s.segLen}</>}>
          After {s.stage === 0 ? 'sorting each segment' : `merge stage ${s.stage}`}: runs of {s.segLen}
        </Translated>
      </div>
      <div className="mx global" data-row="panel" data-values={s.items.map((x) => `${x.key}:${x.id}`).join(' ')}
        style={{ display: 'flex', flexWrap: 'wrap', rowGap: 4, columnGap: 2 }}>
        {s.items.map((it, k) => (
          <span key={it.id} className={final && groups.get(it.key)!.length > 1 ? (okKey(it.key) ? 'good' : 'bad') : ''}
            style={{ marginLeft: k > 0 && k % s.segLen === 0 ? '0.55rem' : 0, minWidth: '1.9rem', height: 'auto', padding: '0.15rem 0', lineHeight: 1.2 }}>
            <b>{it.key}</b>
            <small className="muted">{it.id}</small>
          </span>
        ))}
      </div>
      {final ? (
        <p style={{ margin: '0.8rem 0 0' }} data-ties={tied.every(([k]) => okKey(k)) ? 'stable' : 'reordered'}>
          {tied.every(([k]) => okKey(k)) ? (
            <span className="chip ok">
              <Translated vi="mọi khóa lặp lại đều theo thứ tự đầu vào: ổn định">every repeated key in input order: stable</Translated>
            </span>
          ) : (
            <span className="chip fail">
              <Translated vi={<>các khóa lặp lại bị đảo thứ tự: {tied.filter(([k]) => !okKey(k)).map(([k]) => k).join(', ')}</>}>
                repeated keys reordered: {tied.filter(([k]) => !okKey(k)).map(([k]) => k).join(', ')}
              </Translated>
            </span>
          )}
        </p>
      ) : (
        <p className="muted" style={{ margin: '0.8rem 0 0' }}>
          <Translated vi="Đi tới phép trộn cuối để kiểm tra các khóa lặp lại.">Step to the last merge to check the repeated keys.</Translated>
        </p>
      )}
    </Bench>
  );
}

export const questionsS143: Record<string, Question> = {
  'ch14/14.3/q-blocks': {
    kind: 'numeric',
    id: 'ch14/14.3/q-blocks',
    prompt: (
      <Translated
        vi={
          <>
          Một lưới 64 khối sắp 64 đoạn, mỗi khối một đoạn, rồi trộn. Ở giai đoạn trộn 4 (đếm từ 1), có bao nhiêu khối làm việc
          trên mỗi phép trộn?
          </>
        }
      >
        <>
          A grid of 64 blocks sorts 64 segments, one per block, and then merges. In merge stage 4 (counting from 1), how many
          blocks work on each merge?
        </>
      </Translated>
    ),
    answer: 16,
    explain: (
      <Translated
        vi={
          <p>
          Mỗi giai đoạn giảm một nửa số đoạn, nên giai đoạn s có 64 / 2^s phép trộn: 32, 16, 8, 4. Sáu mươi tư khối chia cho 4
          phép trộn là 16 khối mỗi phép trộn, tức 2^s: số khối mỗi phép trộn tăng gấp đôi sau mỗi giai đoạn.
          </p>
        }
      >
        <p>
          Each stage halves the number of segments, so stage s has 64 / 2^s merges: 32, 16, 8, 4. Sixty-four blocks shared
          among 4 merges is 16 per merge, which is 2^s: blocks per merge double every stage.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.3, pp. 333–334 (the book\'s eight-block example)',
    vi: { source: 'ch. 14 §14.3, pp. 333–334 (ví dụ tám khối của sách)' },
  },
  'ch14/14.3/q-stages': {
    kind: 'numeric',
    id: 'ch14/14.3/q-stages',
    prompt: (
      <Translated vi={<>2^20 khóa trước hết được sắp theo các đoạn 1,024 khóa. Sau đó có bao nhiêu giai đoạn trộn?</>}>
        <>2^20 keys are first sorted in segments of 1,024. How many merge stages follow?</>
      </Translated>
    ),
    answer: 10,
    explain: (
      <Translated vi={<p>2^20 / 2^10 = 1,024 đoạn. Mỗi giai đoạn giảm một nửa số đó: 512, 256, ..., 1. Tức là log2(1,024) = 10 giai đoạn.</p>}>
        <p>2^20 / 2^10 = 1,024 segments. Each stage halves the count: 512, 256, ..., 1. That is log2(1,024) = 10 stages.</p>
      </Translated>
    ),
    source: 'ch. 14 §14.3, p. 334',
  },
  'ch14/14.3/q-morework': {
    kind: 'numeric',
    id: 'ch14/14.3/q-morework',
    prompt: (
      <Translated
        vi={
          <>
          Giai đoạn cuối của một phép sắp xếp trộn song song trộn hai dải 2^19 khóa thành {fmt(MW_N)} đầu ra. Mỗi luồng ghi{' '}
          {MW_PER_THREAD} đầu ra liên tiếp, và trước đó tìm xem chúng bắt đầu ở đâu trong hai dải bằng một lần tìm co-rank khoảng{' '}
          {MW_STEPS} bước. Cả giai đoạn thực hiện tổng cộng bao nhiêu bước tìm?
          </>
        }
      >
        <>
          The last stage of a parallel merge sort merges two runs of 2^19 keys into {fmt(MW_N)} outputs. Each thread writes{' '}
          {MW_PER_THREAD} consecutive outputs, and first finds where they start in the two runs with one co-rank search of
          about {MW_STEPS} steps. How many search steps does the stage make in all?
        </>
      </Translated>
    ),
    answer: MW_SEARCH_STEPS,
    unit: 'steps',
    vi: {
      unit: 'bước',
      source:
        'cách đọc của khóa học này cho ch. 14 §14.3, p. 334 (notes/ch14.md, mục 9 của "Unclear in the PDF"); số bước mỗi lần tìm: ch. 13 §13.4 và src/sim/mergestages.ts',
    },
    explain: (
      <Translated
        vi={
          <p>
          {fmt(MW_N)} / {MW_PER_THREAD} = {fmt(MW_N / MW_PER_THREAD)} luồng, mỗi luồng một lần tìm: {fmt(MW_N / MW_PER_THREAD)} ×{' '}
          {MW_STEPS} = {fmt(MW_SEARCH_STEPS)} bước, {MW_SEARCH_STEPS / MW_N} bước mỗi đầu ra, cộng thêm vào một lần di chuyển mỗi
          đầu ra mà phép trộn tuần tự thực hiện. Tính trên mỗi đầu ra, đó là log2 N / {MW_PER_THREAD}, tăng theo N: ở 2^30 khóa sẽ
          là 30 / 8 = 3.75. Vậy một giai đoạn tốn khoảng N + (N / {MW_PER_THREAD}) log2 N, và qua log N giai đoạn, số hạng tìm kiếm
          chính là thừa số thêm vào: O(N log² N) công việc so với O(N log N) của phép sắp xếp tuần tự.
          </p>
        }
      >
        <p>
          {fmt(MW_N)} / {MW_PER_THREAD} = {fmt(MW_N / MW_PER_THREAD)} threads, one search each: {fmt(MW_N / MW_PER_THREAD)} ×{' '}
          {MW_STEPS} = {fmt(MW_SEARCH_STEPS)} steps, {MW_SEARCH_STEPS / MW_N} per output, on top of the one move per output a
          sequential merge makes. Per output that is log2 N / {MW_PER_THREAD}, which grows with N: at 2^30 keys it would be
          30 / 8 = 3.75. So a stage costs about N + (N / {MW_PER_THREAD}) log2 N, and over the log N stages the search term
          is the extra factor: O(N log² N) work against O(N log N) for the sequential sort.
        </p>
      </Translated>
    ),
    source: 'our reading of ch. 14 §14.3, p. 334 (notes/ch14.md, "Unclear in the PDF" item 9); steps per search: ch. 13 §13.4 and src/sim/mergestages.ts',
  },
  'ch14/14.3/q-stable': {
    kind: 'choice',
    id: 'ch14/14.3/q-stable',
    prompt: (
      <Translated
        vi={
          <>
          Một phép trộn lấy từ dải bên trái khi <code>A[i] &lt; B[j]</code> và từ dải bên phải trong các trường hợp còn lại. Nó trộn
          dải trái {keys([[2, 'a'], [5, 'a'], [7]])} với dải phải {keys([[2, 'b'], [5, 'b'], [9]])}, trong đó a đánh dấu một khóa
          đến trước khóa b bằng nó trong đầu vào. Đầu ra là gì?
          </>
        }
      >
        <>
          A merge takes from the left run when <code>A[i] &lt; B[j]</code> and from the right run otherwise. It merges the
          left run {keys([[2, 'a'], [5, 'a'], [7]])} with the right run {keys([[2, 'b'], [5, 'b'], [9]])}, where a marks a
          key that came earlier in the input than its equal b. What does it output?
        </>
      </Translated>
    ),
    options: [
      { label: keys([[2, 'b'], [2, 'a'], [5, 'b'], [5, 'a'], [7], [9]]), correct: true },
      {
        label: keys([[2, 'a'], [2, 'b'], [5, 'a'], [5, 'b'], [7], [9]]),
        why: (
          <Translated vi='Đó là kết quả của A[i] <= B[j]: khi trùng, nó lấy dải trái. Với <, một cặp trùng không phải "nhỏ hơn", nên dải phải đi trước.'>
            {'That is what A[i] <= B[j] gives: on a tie it takes the left run. With <, a tie is not "less", so the right run goes first.'}
          </Translated>
        ),
      },
      {
        label: keys([[2, 'a'], [2, 'b'], [5, 'b'], [5, 'a'], [7], [9]]),
        why: (
          <Translated vi="Cả hai cặp trùng gặp cùng một phép thử, nên chúng ra theo cùng một cách.">
            Both ties meet the same test, so they come out the same way.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
          Ở mỗi cặp trùng, <code>A[i] &lt; B[j]</code> sai, nên phép trộn lấy khóa của dải phải, khóa đến sau trong đầu vào: mọi b
          vượt lên trước a của nó. Đầu ra được sắp xếp nhưng không ổn định. Phép trộn của chương 13 viết{' '}
          <code>A[i] &lt;= B[j]</code> chính vì lý do này; trong bảng ở trên, "trùng: lấy dải bên phải trước" là phép trộn này.
          Lấy dải trái khi trùng giữ các khóa bằng nhau theo thứ tự đầu vào ở mọi phép trộn, và do đó qua cả cây, miễn là phép
          sắp xếp đoạn ở giai đoạn 0 cũng ổn định.
          </p>
        }
      >
        <p>
          At each tie <code>A[i] &lt; B[j]</code> is false, so the merge takes the right run's key, which came later in the
          input: every b overtakes its a. The output is sorted and not stable. Chapter 13's merge writes{' '}
          <code>A[i] &lt;= B[j]</code> for exactly this reason; in the panel above, "ties: take the right run first" is this
          merge. Taking the left run on ties keeps equal keys in input order at every merge, and so through the whole tree,
          provided stage 0's segment sort is stable too.
        </p>
      </Translated>
    ),
    source: 'ch. 13 §13.1, pp. 304–305; src/sim/mergestages.ts (mergeRuns); notes/today/ch14-facts.md D4',
    vi: { source: 'ch. 13 §13.1, pp. 304–305; src/sim/mergestages.ts (mergeRuns); notes/today/ch14-facts.md mục D4' },
  },
};

export function S143() {
  return (
    <>
      <Translated
        vi={
          <p>
          Sắp xếp chẵn-lẻ đơn giản nhưng làm quá nhiều việc. Sắp xếp trộn (merge) là một phép sắp xếp so sánh song song hóa tốt:
          chia danh sách thành các đoạn, sắp mỗi đoạn bằng bất kỳ phép sắp xếp nào bạn muốn, rồi trộn các đoạn đã sắp theo từng
          cặp, giai đoạn này đến giai đoạn khác, cho đến khi chỉ còn một đoạn.
          </p>
        }
      >
        <p>
          Odd-even sort is simple and does far too much work. Merge sort is a comparison sort that parallelises well: divide
          the list into segments, sort each segment with any sort you like, then merge sorted segments in pairs, stage after
          stage, until one segment is left.
        </p>
      </Translated>
      <Source section="14.3" pages="333" figs={tr('Fig. 14.3', 'Hình 14.3')} />
      <Translated
        vi={
          <p>
          Trong hình của sách, một lưới (grid) tám khối sắp tám đoạn, mỗi khối một đoạn. Giai đoạn trộn đầu có bốn phép trộn, nên
          hai khối chia nhau mỗi phép trộn; giai đoạn kế có hai phép trộn, mỗi phép bốn khối; giai đoạn cuối có một phép trộn với
          cả tám khối. Các giai đoạn đầu có nhiều phép trộn nhỏ, nên các khối chỉ cần làm những phép trộn khác nhau. Các giai đoạn
          cuối có vài phép trộn lớn, nên nhiều khối phải chia nhau một phép, và điều đó cần{' '}
          <a href={href('ch13', '13.3')}>phép tìm co-rank của chương 13</a>: mỗi khối tìm xem phần đầu ra của nó bắt đầu ở đâu
          trong hai đầu vào, rồi tự trộn phần của mình.
          </p>
        }
      >
        <p>
          In the book's figure a grid of eight blocks sorts eight segments, one each. The first merge stage has four merges,
          so two blocks share each merge; the next has two merges, four blocks each; the last has one merge and all eight
          blocks. Early stages have many small merges, so blocks can simply work on different merges. Late stages have a few
          large merges, so several blocks must share one, and that needs{' '}
          <a href={href('ch13', '13.3')}>chapter 13's co-rank search</a>: each block finds where its share of the output
          starts in the two inputs, then merges its share on its own.
        </p>
      </Translated>
      <Source section="14.3" pages="333–334" />
      <Translated
        vi={
          <p>
          Dựng tám khối của sách trong trình khám phá và kéo qua các giai đoạn. Rồi giữ nguyên số khóa, cho lưới ít khối hơn số
          đoạn, và quan sát ở chỗ nào trong cây các khối bắt đầu chia nhau phép trộn.
          </p>
        }
      >
        <p>
          Set up the book's eight blocks in the explorer and slide through the stages. Then keep the keys and give the grid
          fewer blocks than there are segments, and watch where in the tree blocks start to share merges.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
            Trình khám phá đếm phép trộn, khối và các lần tìm chia phép trộn giữa các khối. Nó không đo thời gian, coi mọi khối
            nhanh như nhau, và không đếm các lần tìm thêm mà kernel của chương 13 thực hiện bên trong một khối.
            </p>
          }
        >
          <p>
            The explorer counts merges, blocks and the searches that split merges between blocks. It has no timing, treats
            every block as equally fast, and does not count the further searches chapter 13's kernels make inside a block.
          </p>
        </Translated>
      </Simplification>
      <MergeTree />
      <Translated
        vi={
          <p>
          Tính song song chuyển từ giữa các phép trộn vào bên trong một phép trộn: số phép trộn giảm một nửa mỗi giai đoạn trong
          khi số khối mỗi phép trộn tăng gấp đôi, và số giai đoạn là log2 của số đoạn.
          </p>
        }
      >
        <p>
          The parallelism moves from across merges to within a merge: the number of merges halves each stage while the
          blocks per merge double, and the number of stages is log2 of the number of segments.
        </p>
      </Translated>
      <Ask q={questionsS143['ch14/14.3/q-blocks']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS143['ch14/14.3/q-stages']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Vì sao nó làm nhiều việc hơn phép sắp xếp tuần tự">Why it does more work than the sequential sort</Translated></h2>
      <Translated
        vi={
          <p>
          Có O(log N) giai đoạn trộn. Sách cho mỗi giai đoạn O(log N) thời gian và O(N log N) công việc khi đủ tài nguyên, "as seen
          in Chapter 13" (như đã thấy ở chương 13), nên cả phép sắp xếp mất O(log² N) thời gian và O(N log² N) công việc. Sắp xếp
          trộn tuần tự làm O(N log N) công việc; thừa số log thêm vào đến từ các lần tìm co-rank mà phép trộn song song thực hiện
          còn phép trộn tuần tự thì không.
          </p>
        }
      >
        <p>
          There are O(log N) merge stages. The book gives each stage O(log N) time and O(N log N) work with enough resources,
          "as seen in Chapter 13", so the whole sort takes O(log² N) time and O(N log² N) work. Sequential merge sort does
          O(N log N) work; the extra log factor comes from the co-rank searches a parallel merge makes and a sequential merge
          does not.
        </p>
      </Translated>
      <Source section="14.3" pages="334" />
      <Translated
        vi={
          <p>
          Bản PDF viết lỏng ở đây. Chương 13 nêu O(m + n) cho phép trộn tuần tự và khoảng log2 N bước cho một lần tìm co-rank,
          nhưng không bao giờ đưa ra cận cho cả một phép trộn song song. Cách chúng tôi đọc hình của sách: nếu mỗi luồng tạo ra
          một số đầu ra cố định và tìm một lần xem chúng bắt đầu ở đâu, thì một giai đoạn N khóa thực hiện cỡ N lần tìm, mỗi lần
          log N bước, tức O(N log N) mỗi giai đoạn. Thuật toán của Cole đạt O(log N) thời gian và O(N log N) công việc; sách nhắc
          tên nó nhưng không trình bày, và chúng tôi cũng vậy.
          </p>
        }
      >
        <p>
          The PDF is loose here. Chapter 13 states O(m + n) for the sequential merge and about log2 N steps for one co-rank
          search, but never a bound for a whole parallel merge. Our reading of the book's figure: if each thread produces a
          fixed number of outputs and searches once for where they start, a stage of N keys makes on the order of N searches
          of log N steps each, which is the O(N log N) per stage. Cole's algorithm reaches O(log N) time and O(N log N) work;
          the book names it and does not cover it, and neither do we.
        </p>
      </Translated>
      <Beyond title={tr('Our reading', 'Cách đọc của chúng tôi')}>
        <Translated
          vi={
            <p>
            Phép suy ra cho từng giai đoạn là của chúng tôi (notes/ch14.md, mục 9 của "Unclear in the PDF"). Số bước mỗi lần tìm
            của trình khám phá là ceil(log2(input length + 1)), tức "khoảng log2 N" của chương 13; mô hình phép tìm đó của khóa học
            (src/sim/corank.ts) chỉ nhiều hơn tối đa một bước.
            </p>
          }
        >
          <p>
            The per-stage derivation is ours (notes/ch14.md, "Unclear in the PDF" item 9). The explorer's steps per search
            are ceil(log2(input length + 1)), chapter 13's "about log2 N"; the course's model of that search
            (src/sim/corank.ts) stays within one step more.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS143['ch14/14.3/q-morework']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Tính ổn định">Stability</Translated></h2>
      <Translated
        vi={
          <p>
          Một phép sắp xếp trộn chỉ ổn định khi mọi phép trộn đều ổn định. Khi hai dải chứa các khóa bằng nhau, phép trộn phải lấy
          từ dải trái trước, vì dải trái chứa những phần tử đến sớm hơn trong đầu vào. Phép trộn của chương 13 làm đúng điều đó
          bằng phép thử <code>A[i] &lt;= B[j]</code> của nó.
          </p>
        }
      >
        <p>
          A merge sort is stable only if every merge is. When the two runs hold equal keys, the merge must take from the left
          run first, because the left run holds elements that came earlier in the input. Chapter 13's merge does exactly that
          with its <code>A[i] &lt;= B[j]</code> test.
        </p>
      </Translated>
      <Source section="13.1" pages="304–305" />
      <Simplification>
        <Translated
          vi={
            <p>
            Bảng này chạy mỗi phép trộn như một phép trộn tuần tự, lần lượt từng cái. Một phép trộn song song chia theo co-rank cho
            cùng đầu ra với cùng quy tắc xử lý trùng; bảng cho thấy kết quả, không phải lịch thực thi.
            </p>
          }
        >
          <p>
            The panel runs each merge as a sequential merge, one at a time. A parallel merge that splits by co-rank gives
            the same output for the same tie rule; the panel shows the result, not the schedule.
          </p>
        </Translated>
      </Simplification>
      <SixteenKeys />
      <Ask q={questionsS143['ch14/14.3/q-stable']} />

      <h2><Translated vi="Bài tập 4, phác thảo">Exercise 4, in outline</Translated></h2>
      <Translated
        vi={
          <p>
          Sách để phần cài đặt cho bạn. Một cách bố trí: mỗi khối sắp một đoạn trong bộ nhớ chia sẻ (shared memory); sau đó chạy
          log2(segments) giai đoạn trộn, mỗi giai đoạn đọc từ một buffer và ghi vào buffer kia, đổi vai hai buffer giữa các giai
          đoạn. Ở mỗi giai đoạn, mỗi phép trộn được grid / merges khối, các khối này dùng phép tìm co-rank để chia phép trộn; khi
          số khối ít hơn số phép trộn, mỗi khối lặp qua vài phép. Một đoạn không có cặp (khi số đoạn không phải lũy thừa của hai)
          được chép sang nguyên vẹn. Đây là một phác thảo, không phải một lab của chương này.
          </p>
        }
      >
        <p>
          The book leaves the implementation to you. One way to lay it out: one block sorts each segment in shared memory;
          then log2(segments) merge stages run, each reading from one buffer and writing to the other, swapping the two between
          stages. In each stage every merge gets grid / merges blocks, which use the co-rank search to split the merge; when
          there are fewer blocks than merges, each block loops over several. A segment left without a partner (when the count
          is not a power of two) is copied across unchanged. This is a sketch, not one of this chapter's labs.
        </p>
      </Translated>
      <Source section="14.11" pages="347" />

      <h2><Translated vi="Bạn sẽ gặp nó ở đâu">Where you will meet it</Translated></h2>
      <Translated
        vi={
          <p>
          Phép sắp xếp trên GPU của MLX là một phép sắp xếp trộn được dựng theo cách này. Mỗi luồng trước hết sắp vài giá trị
          trong thanh ghi (register) của nó bằng vòng lặp chẵn-lẻ của <a href={href('ch14', '14.2')}>mục 14.2</a>; sau đó một
          khối trộn các dải đó, tìm điểm chia của mỗi luồng bằng tìm kiếm nhị phân dọc theo đường chéo, chính là co-rank của
          chương 13; các hàng dài hơn đi qua nhiều khối, chuyền qua lại giữa các buffer. Phép trộn của nó chỉ lấy từ dải phải khi
          phần tử đó nhỏ hơn hẳn, nên các khóa bằng nhau giữ thứ tự đầu vào và phép sắp xếp ổn định, như tài liệu MLX ghi.
          </p>
        }
      >
        <p>
          MLX's GPU sort is a merge sort built this way. Each thread first sorts the few values in its registers with the
          odd-even loop of <a href={href('ch14', '14.2')}>section 14.2</a>; a block then merges those runs, finding each
          thread's split with a binary search along the diagonal, which is chapter 13's co-rank; longer rows go through
          several blocks, ping-ponging between buffers. Its merge takes from the right run only when that element is
          strictly smaller, so equal keys stay in input order and the sort is stable, as MLX documents.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch14-facts.md, D2 and D4)', '2026-11-20 (notes/today/ch14-facts.md, mục D2 và D4)')}>
        <Translated
          vi={
            <p>
            Trên Metal, <code>topk</code>, <code>partition</code> và <code>argpartition</code> của MLX hiện chạy toàn bộ phép sắp
            xếp trộn này ("We direct arg partition to sort for now"). Argsort trên Metal của llama.cpp trộn các dải đã sắp qua các
            threadgroup bằng cùng phép phân hoạch co-rank, ưu tiên dải trái khi giá trị bằng nhau.
            </p>
          }
        >
          <p>
            On Metal, MLX's <code>topk</code>, <code>partition</code> and <code>argpartition</code> run this full merge sort
            today ("We direct arg partition to sort for now"). llama.cpp's Metal argsort merges sorted runs across
            threadgroups with the same co-rank partition, preferring the left run on equal values.
          </p>
        </Translated>
      </Today>
    </>
  );
}
