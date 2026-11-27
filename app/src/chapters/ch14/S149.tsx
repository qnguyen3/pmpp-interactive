import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  TIES16,
  TIES8,
  applyNetwork,
  bitonicCount,
  bitonicStages,
  byValue,
  tieGroups,
  transpositionCount,
  type Item,
} from '../../sim/sortnet';

const fmtP = (p: number) => (Number.isInteger(p) ? String(p) : p.toFixed(3).replace(/0+$/, '').replace(/\.$/, ''));

/** Reads exactly n non-negative numbers; null if malformed. */
function parseKeys(text: string, n: number): number[] | null {
  const parts = text.split(/[\s,]+/).filter(Boolean);
  if (parts.length !== n) return null;
  const xs = parts.map(Number);
  return xs.every((v) => Number.isFinite(v) && v >= 0) ? xs : null;
}

/** A deterministic shuffle, so a check can predict it: the k-th shuffle of the preset. */
function shuffled<T>(xs: T[], seed: number): T[] {
  const out = xs.slice();
  let s = seed * 2654435761 >>> 0;
  for (let i = out.length - 1; i > 0; i--) {
    s = (s * 1103515245 + 12345) >>> 0;
    const j = (s >>> 8) % (i + 1);
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

function Network() {
  const [n, setN] = useState<8 | 16>(8);
  const [keys, setKeys] = useState<number[]>(TIES8.map((x) => x.key));
  const [text, setText] = useState(TIES8.map((x) => fmtP(x.key)).join(' '));
  const [byIndex, setByIndex] = useState(false);
  const [done, setDone] = useState(0); // stages applied
  const [shuffles, setShuffles] = useState(0);
  const stages = useMemo(() => bitonicStages(n), [n]);
  const items: Item[] = keys.map((key, id) => ({ key, id }));
  const run = useMemo(() => applyNetwork(items, stages, byValue(byIndex)), [keys, stages, byIndex]); // eslint-disable-line react-hooks/exhaustive-deps
  const at = Math.min(done, stages.length);
  const cur = at === 0 ? items : run.stages[at - 1].after;
  const finished = at === stages.length;
  const groups = finished ? tieGroups(cur) : [];
  const bad = parseKeys(text, n) === null;

  const load = (m: 8 | 16) => {
    const preset = m === 8 ? TIES8 : TIES16;
    setN(m);
    setKeys(preset.map((x) => x.key));
    setText(preset.map((x) => fmtP(x.key)).join(' '));
    setDone(0);
    setShuffles(0);
  };
  const shuffle = () => {
    const k = shuffles + 1;
    const next = shuffled(keys, k);
    setShuffles(k);
    setKeys(next);
    setText(next.map(fmtP).join(' '));
    setDone(0);
  };

  // Geometry.
  const gap = n === 8 ? 34 : 25;
  const top = 30;
  const left = 96;
  const colW = n === 8 ? 82 : 52;
  const width = left + stages.length * colW + 100;
  const height = top + n * gap + 10;
  const wy = (k: number) => top + k * gap;
  const sx = (s: number) => left + s * colW + colW / 2;
  const swappedNow = at > 0 ? run.stages[at - 1].swapped : [];

  return (
    <Bench
      title={tr('Bitonic network', 'Mạng bitonic')}
      task={
        <Translated vi="Đi qua từng giai đoạn, xáo các giá trị rồi đi lại lần nữa, sau đó đánh dấu so sánh theo chỉ số và chạy thiết lập có giá trị trùng.">
          Step through the stages, shuffle the values and step again, then tick the index comparison and run the ties preset.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Mỗi đường dọc là một bộ so sánh: nó nhìn hai vị trí và đưa giá trị lớn hơn về đầu mũi tên, chỉ hoán đổi khi
              cặp đó sai thứ tự hẳn. Sắp lớn nhất trước. Số nhỏ cạnh mỗi giá trị là chỉ số token của nó, tức vị trí của nó
              trong đầu vào.
            </>
          }
        >
          <>
            Each vertical line is a comparator: it looks at two positions and sends the larger value to the arrow's end,
            swapping only when the pair is strictly out of order. Sorted largest first. The small number beside each value
            is its token index, its position in the input.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label="N">
          <button aria-pressed={n === 8} onClick={() => load(8)}>N = 8</button>
          <button aria-pressed={n === 16} onClick={() => load(16)}>N = 16</button>
        </span>
        <button onClick={() => load(n)}><Translated vi="thiết lập có giá trị trùng">ties preset</Translated></button>
        <button onClick={shuffle}><Translated vi="xáo các giá trị">shuffle values</Translated></button>
        <label className="check">
          <input type="checkbox" checked={byIndex} onChange={(e) => { setByIndex(e.target.checked); setDone(0); }} />{' '}
          <Translated vi="so sánh (value, index)">compare (value, index)</Translated>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field" style={{ flex: '1 1 20rem' }}>
          <Translated vi={<>Các giá trị ({n}, cách nhau bằng dấu cách)</>}>Values ({n}, separated by spaces)</Translated>
          <input type="text" value={text} aria-label={tr('values', 'các giá trị')} style={{ width: '100%' }}
            onChange={(e) => { setText(e.target.value); const v = parseKeys(e.target.value, n); if (v) { setKeys(v); setDone(0); } }} />
        </label>
        {bad ? <span className="chip fail"><Translated vi={<>đúng {n} số, không số nào âm</>}>exactly {n} numbers, none negative</Translated></span> : null}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setDone(0)} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => setDone(at - 1)} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => setDone(at + 1)} disabled={finished}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => setDone(stages.length)} disabled={finished}><Translated vi="Chạy">Run</Translated></button>
        <span className="muted" aria-live="polite" data-stage={at}>
          <Translated vi={<>giai đoạn {at} trên {stages.length}</>}>stage {at} of {stages.length}</Translated>
        </span>
      </div>
      <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={tr('Bitonic sorting network', 'Mạng sắp xếp bitonic')} style={{ width: '100%', maxWidth: width, minWidth: 600, display: 'block' }}
        data-values={cur.map((x) => `${fmtP(x.key)}:${x.id}`).join(' ')}>
        {stages.map((_, s) => (
          <rect key={'col' + s} x={left + s * colW + 2} y={top - 18} width={colW - 4} height={n * gap + 4} rx="3"
            fill={s === at - 1 ? 'var(--rule-soft)' : 'none'} />
        ))}
        {stages.map((_, s) => (
          <text key={'h' + s} x={sx(s)} y={top - 22} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{s + 1}</text>
        ))}
        {Array.from({ length: n }, (_, k) => (
          <g key={'w' + k}>
            <line x1={left - 4} y1={wy(k)} x2={left + stages.length * colW + 4} y2={wy(k)} stroke="var(--rule)" />
            <text x={left - 10} y={wy(k) + 4} fontSize="12" textAnchor="end" data-in={k}>
              {fmtP(items[k].key)}<tspan fontSize="9" style={{ fill: 'var(--ink-3)' }}> t{items[k].id}</tspan>
            </text>
            <text x={left + stages.length * colW + 12} y={wy(k) + 4} fontSize="12" data-out={k}
              style={{ fill: finished && groups.some((g) => g.key === cur[k].key) ? (groups.find((g) => g.key === cur[k].key)!.increasing ? 'var(--ok)' : 'var(--fail)') : 'var(--ink)' }}>
              {fmtP(cur[k].key)}<tspan fontSize="9" style={{ fill: 'var(--ink-3)' }}> t{cur[k].id}</tspan>
            </text>
          </g>
        ))}
        {stages.map((stage, s) =>
          stage.map((c, j) => {
            // Comparators of one stage that overlap vertically get their own slots across the column.
            const d = c.hi - c.lo;
            const slot = c.lo % d;
            const x = d === 1 ? sx(s) : left + s * colW + 6 + ((slot + 0.5) * (colW - 12)) / d;
            const y1 = wy(c.lo);
            const y2 = wy(c.hi);
            const live = s === at - 1;
            const swapped = live && swappedNow[j];
            const tipY = c.up ? y1 : y2; // the larger value goes to the tip
            const dir = c.up ? 1 : -1;
            return (
              <g key={`${s}-${j}`} opacity={s < at ? 1 : 0.45} data-comp={`${s}:${c.lo}-${c.hi}`} data-swapped={swapped ? 'yes' : 'no'}>
                <line x1={x} y1={y1} x2={x} y2={y2} stroke="var(--ink)" strokeWidth={swapped ? 3 : 1.3} />
                <circle cx={x} cy={c.up ? y2 : y1} r={2.6} fill="var(--ink)" />
                <path d={`M${x - 4} ${tipY + dir * 7} L${x} ${tipY} L${x + 4} ${tipY + dir * 7} Z`} fill="var(--ink)" />
              </g>
            );
          }),
        )}
      </svg>
      </div>
      <table className="threads wrap" aria-label={tr('Counters', 'Bộ đếm')} style={{ marginTop: '0.8rem' }}>
        <tbody>
          <tr>
            <td><Translated vi="số bộ so sánh đã áp dụng">comparators applied</Translated></td>
            <td>
              <Translated vi={<><b data-counter="applied">{(at * n) / 2}</b> trên {bitonicCount(n)} ({stages.length} giai đoạn, mỗi giai đoạn {n / 2})</>}>
                <b data-counter="applied">{(at * n) / 2}</b> of {bitonicCount(n)} ({stages.length} stages of {n / 2})
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="số lần hoán đổi ở giai đoạn vừa rồi">swaps in the last stage</Translated></td>
            <td><b data-counter="swaps">{swappedNow.filter(Boolean).length}</b></td>
          </tr>
          <tr>
            <td><Translated vi="chuyển vị chẵn-lẻ, cùng N">odd-even transposition, same N</Translated></td>
            <td>
              <Translated vi={<><b data-counter="transposition">{transpositionCount(n)}</b> bộ so sánh ({n} bước)</>}>
                <b data-counter="transposition">{transpositionCount(n)}</b> comparators ({n} steps)
              </Translated>
            </td>
          </tr>
        </tbody>
      </table>
      {finished ? (
        <div style={{ marginTop: '0.8rem' }} data-ties={groups.length === 0 ? 'none' : groups.every((g) => g.increasing) ? 'stable' : 'reordered'}>
          {groups.length === 0 ? (
            <span className="muted"><Translated vi="Các giá trị này không có giá trị trùng.">No ties in these values.</Translated></span>
          ) : (
            groups.map((g) => (
              <span key={g.key} className={'chip ' + (g.increasing ? 'ok' : 'fail')} style={{ marginRight: '0.4rem' }} data-group={fmtP(g.key)} data-ok={g.increasing ? 'yes' : 'no'}>
                {fmtP(g.key)}: {g.ids.map((i) => 't' + i).join(' ')}
              </span>
            ))
          )}
        </div>
      ) : null}
    </Bench>
  );
}

export const questionsS149: Record<string, Question> = {
  'ch14/14.9/q-comparators': {
    kind: 'numeric',
    id: 'ch14/14.9/q-comparators',
    prompt: (
      <Translated vi={<>Một mạng sắp xếp bitonic với N = 16 có bao nhiêu bộ so sánh?</>}>
        <>A bitonic sorting network for N = 16 has how many comparators?</>
      </Translated>
    ),
    answer: 80,
    explain: (
      <Translated
        vi={
          <p>
          log2 16 = 4, nên mạng có 4 × 5 / 2 = 10 giai đoạn, mỗi giai đoạn N/2 = 8 bộ so sánh: 80. Sắp xếp chuyển vị chẵn-lẻ, chạy
          như một mạng cố định trong 16 bước, cần 8 bước 8 bộ và 8 bước 7 bộ: 120. Với N = 1,024, khoảng cách là 28,160 so với
          khoảng nửa triệu.
          </p>
        }
      >
        <p>
          log2 16 = 4, so the network has 4 × 5 / 2 = 10 stages, each with N/2 = 8 comparators: 80. Odd-even transposition
          sort, run as a fixed network for 16 steps, needs 8 steps of 8 and 8 of 7: 120. For N = 1,024 the gap is 28,160
          against about half a million.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.9, p. 345; src/sim/sortnet.ts', // i18n-ignore: citation and file path only
  },
  'ch14/14.9/q-network-ties': {
    kind: 'choice',
    id: 'ch14/14.9/q-network-ties',
    prompt: (
      <Translated
        vi={
          <>
          Một bộ lấy mẫu sắp 64 ứng viên mỗi hàng bên trong một khối và phải liệt kê các xác suất bằng nhau theo chỉ số token
          tăng dần. Bạn viết gì?
          </>
        }
      >
        <>
          A sampler sorts 64 candidates per row inside one block and must list equal probabilities by increasing token
          index. What do you write?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một radix sort một bit trên bộ nhớ toàn cục, vốn ổn định theo cách xây dựng.">
            A one-bit radix sort over global memory, which is stable by construction.
          </Translated>
        ),
        why: (
          <Translated vi="Nó sẽ ổn định, nhưng 32 lượt qua bộ nhớ toàn cục cho 64 giá trị là quá nhiều lưu lượng cho một việc mà một khối làm được ngay trên chip.">
            It would be stable, but 32 passes through global memory for 64 values is a great deal of traffic for a job one block can do on chip.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Sắp xếp chẵn-lẻ với một vòng lặp phía host, vì chỉ hoán đổi các phần tử kề nhau sai thứ tự thì giữ được giá trị trùng.">
            Odd-even sort with a host loop, since swapping only out-of-order neighbours keeps ties.
          </Translated>
        ),
        why: (
          <Translated vi="Ở đây đủ ổn định, nhưng mỗi bước một lần khởi chạy và một lần sao chép về, tới 64 bước mỗi hàng.">
            Stable enough here, but one launch and one copy back per step, for up to 64 steps per row.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một mạng bitonic trong bộ nhớ chia sẻ có các bộ so sánh không bao giờ hoán đổi các giá trị bằng nhau.">
            A bitonic network in shared memory whose comparators never swap equal values.
          </Translated>
        ),
        why: (
          <Translated vi="Các bộ so sánh vươn qua cả mảng và mang các giá trị khác vượt qua những giá trị bằng nhau; không quy tắc nào bên trong một bộ so sánh làm cho mạng ổn định.">
            Comparators reach across the array and carry other values past equal ones; no rule inside one comparator makes the network stable.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một mạng bitonic trong bộ nhớ chia sẻ hoặc thanh ghi so sánh các cặp (value, index).">
            A bitonic network in shared memory or registers that compares (value, index) pairs.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
          Với (value, index) làm khóa, không có hai khóa nào bằng nhau, nên tính không ổn định của mạng không có gì để tác động:
          đầu ra là thứ tự ổn định, bất kể mỗi giá trị đi theo đường nào. 64 giá trị dễ dàng vừa trong một khối, và mẫu cố định
          không cần cờ hay vòng lặp phía host.
          </p>
        }
      >
        <p>
          With (value, index) as the key, no two keys are equal, so the network's instability has nothing to act on: the output
          is the stable order, whatever path each value takes. 64 values fit easily in one block, and the fixed pattern
          needs no flag and no host loop.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.9, pp. 345–346; notes/today/ch14-facts.md D1',
    vi: { source: 'ch. 14 §14.9, pp. 345–346; notes/today/ch14-facts.md mục D1' },
  },
  'ch14/14.9/q-msd': {
    kind: 'choice',
    id: 'ch14/14.9/q-msd',
    prompt: (
      <Translated vi={<>Vì sao radix MSD hợp với top-k còn radix LSD thì không?</>}>
        <>Why does MSD radix suit top-k when LSD radix does not?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="LSD quyết định chữ số cao nhất sau cùng, nên các lượt đầu của nó không nói gì về các khóa lớn nhất; lượt đầu của MSD tách riêng bucket của khóa lớn thứ k.">
            {'LSD settles the top digit last, so its early passes say nothing about the largest keys; MSD\'s first pass isolates the k-th largest key\'s bucket.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="MSD chạy tổng cộng ít lượt hơn: đọc chữ số từ trên xuống, nó chỉ cần khoảng log k lượt.">
            MSD makes fewer passes in total: reading digits from the top, it needs only about log k of them.
          </Translated>
        ),
        why: (
          <Translated vi="Số lượt phụ thuộc vào độ rộng khóa và độ rộng chữ số, không phụ thuộc k. Một phép chọn có thể dừng sớm, nhưng đó là hệ quả của điều mỗi lượt MSD quyết định, và đó mới là lý do MSD phù hợp.">
            The pass count depends on the key width and the digit width, not on k. A select may stop early, but that follows from what each MSD pass decides, which is the reason MSD fits.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="LSD không ổn định, nên k khóa lớn nhất có thể ra theo thứ tự khác nhau giữa các lần chạy.">
            LSD is not stable, so the k largest keys could come out in a different order from run to run.
          </Translated>
        ),
        why: (
          <Translated vi="Radix sort LSD ổn định; nó phải như vậy thì các lượt của nó mới ghép được với nhau (mục 14.4).">
            LSD radix sort is stable; it has to be, for its passes to compose (section 14.4).
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
          Sau một lượt MSD trên chữ số cao nhất, mọi bucket phía trên bucket chứa khóa lớn thứ k đều nằm trọn trong top k và mọi
          bucket phía dưới đều nằm trọn ngoài; chỉ còn một bucket chưa được quyết định. Đó là lý do top-k của llama.cpp và top-k
          nhiều khối của PyTorch (<code>mbtopk</code>) đếm histogram chữ số từ chữ số có nghĩa lớn nhất trở xuống.
          </p>
        }
      >
        <p>
          After an MSD pass on the top digit, every bucket above the one holding the k-th largest key is wholly in the top k
          and every bucket below it is wholly out; only one bucket remains undecided. That is why llama.cpp's top-k and
          PyTorch's multi-block top-k (<code>mbtopk</code>) count digit histograms from the most significant digit down.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.9, p. 346; notes/today/ch14-facts.md D2, D5',
    vi: { source: 'ch. 14 §14.9, p. 346; notes/today/ch14-facts.md mục D2, D5' },
  },
  'ch14/14.9/q-oversample': {
    kind: 'choice',
    id: 'ch14/14.9/q-oversample',
    prompt: (
      <Translated
        vi={
          <>
          Sample sort cắt một triệu khóa ngẫu nhiên thành 4 bucket bằng 3 khóa chia, rồi mỗi bucket được sắp riêng. Phiên bản A
          dùng 3 khóa rút ngẫu nhiên làm khóa chia. Phiên bản B rút 30 khóa, sắp chúng, và lấy các khóa cách đều nhau (khóa thứ 8,
          15 và 23). Qua nhiều lần chạy, bạn kỳ vọng gì ở bucket lớn nhất?
          </>
        }
      >
        <>
          Sample sort cuts a million random keys into 4 buckets with 3 splitters, and each bucket is then sorted on its own.
          Version A uses 3 keys drawn at random as the splitters. Version B draws 30 keys, sorts them, and takes evenly spaced
          ones (the 8th, 15th and 23rd). Over many runs, what do you expect of the largest bucket?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Khoảng một phần tư đầu vào ở cả hai: các khóa ngẫu nhiên, nên các khóa chia trải đều.">
            About a quarter of the input in both: the keys are random, so the splitters are spread evenly.
          </Translated>
        ),
        why: (
          <Translated vi="Khóa chia ngẫu nhiên trải ra một cách ngẫu nhiên, không đều. Ba khóa chia thường dồn cụm, để lại một khoảng trống rộng.">
            Random splitters are spread at random, not evenly. Three of them are often bunched, which leaves one wide gap.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Lớn hơn ở B: ba mươi mẫu cho ba mươi cơ hội rút xui, và bất kỳ mẫu nào cũng có thể làm lệch một khóa chia.">
            Larger in B: thirty samples give thirty chances of an unlucky draw, and any one can skew a splitter.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi khóa chia của B là mẫu thứ 8, 15 hoặc 23 trong 30 mẫu đã sắp, và một hạng như thế dao động ít hơn nhiều so với một khóa ngẫu nhiên.">
            Each splitter of B is the 8th, 15th or 23rd of 30 sorted samples, and such a rank varies far less than one random key.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Ở A nó thường gần một nửa đầu vào; ở B nó gần một phần tư hơn nhiều.">
            In A it is often near half the input; in B it stays much nearer a quarter.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Như nhau ở A và B: tính trung bình, mỗi bucket đều là một phần tư dù theo cách nào.">
            The same in A and B: on average every bucket is a quarter either way.
          </Translated>
        ),
        why: (
          <Translated vi="Bucket trung bình là một phần tư ở cả hai. Thời gian phải chờ bucket lớn nhất, và đó mới là chỗ khác nhau.">
            The average bucket is a quarter in both. The time waits for the largest bucket, and that is what differs.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
          Ba điểm ngẫu nhiên cắt miền khóa thành bốn khoảng, và khoảng lớn nhất trung bình bằng (1 + 1/2 + 1/3 + 1/4) / 4 = 25/48
          miền: ở phiên bản A, một bucket thường chứa khoảng một nửa số khóa, và việc sắp nó là đường găng. Các khóa cách đều của
          một mẫu đã sắp lớn hơn dao động ít hơn nhiều, nên bucket lớn nhất co dần về một phần tư khi mẫu lớn lên. Đó là câu
          "modest oversampling gives buckets of nearly equal size with high probability" của sách (lấy mẫu dư vừa phải cho các
          bucket gần bằng nhau với xác suất cao).
          </p>
        }
      >
        <p>
          Three random points cut the key range into four gaps, and the largest of them averages (1 + 1/2 + 1/3 + 1/4) / 4 =
          25/48 of the range: in version A one bucket typically holds about half the keys, and its sort is the critical path.
          Evenly spaced keys of a larger sorted sample vary much less, so the largest bucket shrinks towards a quarter as the
          sample grows. That is the book's "modest oversampling gives buckets of nearly equal size with high probability".
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.9, p. 346 (the book cites Blelloch et al.; not checked by us); the 25/48 is our arithmetic',
    vi: { source: 'ch. 14 §14.9, p. 346 (sách dẫn Blelloch et al.; chúng tôi chưa kiểm tra); con số 25/48 là phép tính của chúng tôi' },
  },
};

export function S149() {
  return (
    <>
      <Translated
        vi={
          <p>
          Chương kết thúc bằng một phần điểm qua. Ý đầu tiên nhìn lại sắp xếp chẵn-lẻ: các phép so sánh nó thực hiện theo một mẫu
          cố định, như nhau bất kể dữ liệu, và nó song song hóa dễ dàng vì mỗi bước so sánh các cặp không chồng lên nhau. Cả một họ
          phương pháp hoạt động theo cách này. Chúng được gọi là <b>mạng sắp xếp</b> (sorting network), và nổi tiếng nhất cho máy
          song song là bitonic sort và odd-even merge sort của Batcher.
          </p>
        }
      >
        <p>
          The chapter ends with a survey. Its first point looks back at odd-even sort: the comparisons it makes follow a fixed
          pattern, the same whatever the data, and it parallelises easily because each step compares pairs that do not
          overlap. A whole family of methods works this way. They are called <b>sorting networks</b>, and the best known for
          parallel machines are Batcher's bitonic sort and odd-even merge sort.
        </p>
      </Translated>
      <Source section="14.9" pages="345–346" />
      <Translated
        vi={
          <p>
          Chúng sắp một số lượng giá trị cố định bằng O(N log² N) bộ so sánh: ít hơn nhiều so với O(N²) của chuyển vị chẵn-lẻ,
          nhiều hơn O(N log N) của sắp xếp trộn. Vậy là kém hơn về tiệm cận, nhưng thường lại là lựa chọn nhanh nhất cho các dãy
          ngắn, vì chúng đơn giản. Trên GPU, sự đơn giản đó có hình thức cụ thể: mẫu không phụ thuộc dữ liệu, nên không có cờ,
          không có vòng lặp phía host và không có tìm kiếm, và mỗi luồng biết bạn cặp của mình ở mọi giai đoạn.
          </p>
        }
      >
        <p>
          They sort a fixed number of values with O(N log² N) comparators: far fewer than odd-even transposition's O(N²), more
          than the O(N log N) of merge sort. Asymptotically worse, then, yet often the fastest choice for short sequences,
          because they are simple. On a GPU that simplicity has a concrete form: the pattern does not depend on the data, so
          there is no flag, no host loop and no search, and every thread knows its partner at every stage.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
            Sách nói "simplicity" (sự đơn giản); danh sách những gì nó đem lại trên GPU là của khóa học này.
            </p>
          }
        >
          <p>
            The book says "simplicity"; the list of what that buys on a GPU is ours.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
          Bitonic sort không ổn định. Các bộ so sánh của nó vươn qua cả mảng, nên một giá trị có thể bị mang vượt qua một giá trị
          bằng nó qua các phép so sánh với giá trị khác. So sánh các cặp (value, index) loại bỏ vấn đề: khi đó không có hai khóa
          nào bằng nhau, và kết quả là thứ tự ổn định.
          </p>
        }
      >
        <p>
          Bitonic sort is not stable. Its comparators reach across the array, so a value can be carried past an equal value
          by comparisons with others. Comparing (value, index) pairs removes the problem: no two keys are then equal, and the
          result is the stable order.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
            Mạng chỉ được vẽ cho N là lũy thừa của hai; các độ dài khác được đệm thêm. Một giai đoạn là một bước song song không gắn
            chi phí: trên GPU nó sẽ là một khoảng giữa hai rào chắn (barrier) trong bộ nhớ chia sẻ, hoặc các lần shuffle giữa thanh
            ghi.
            </p>
          }
        >
          <p>
            The network is drawn for N a power of two only; other lengths are padded. A stage is one parallel step with no
            cost attached: on a GPU it would be a barrier interval in shared memory, or shuffles between registers.
          </p>
        </Translated>
      </Simplification>
      <Network />
      <Translated
        vi={
          <p>
          Mạng sắp đúng thiết lập có giá trị trùng nhưng đảo thứ tự các token trùng. Cùng các bộ so sánh đó hoạt động dù bạn xáo
          đầu vào thế nào; chỉ có việc bộ nào hoán đổi là thay đổi. Khi đánh dấu so sánh theo chỉ số, mọi nhóm trùng ra theo thứ
          tự token tăng dần.
          </p>
        }
      >
        <p>
          The network sorts the ties preset correctly and reorders tied tokens. The same comparators fire however you shuffle
          the input; only which of them swap changes. With the index comparison ticked, every tie group comes out in increasing
          token order.
        </p>
      </Translated>
      <Ask q={questionsS149['ch14/14.9/q-comparators']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
          Đây là chỗ các mạng xuất hiện trong mã bạn sẽ đọc. Argsort CUDA của llama.cpp sắp một hàng bằng mạng bitonic trong bộ
          nhớ chia sẻ của một khối khi hàng có tối đa 1,024 cột và độ dài đã đệm vừa; các hàng dài hơn chuyển sang CUB. PyTorch
          trên CUDA sắp các lát cắt từ 32 phần tử trở xuống bằng mạng bitonic. Cả hai đều là "small sequences" (dãy ngắn) của sách.
          </p>
        }
      >
        <p>
          This is where the networks show up in the code you will read. llama.cpp's CUDA argsort sorts a row with a bitonic
          network in one block's shared memory when the row has at most 1,024 columns and its padded length fits; longer rows
          go to CUB. PyTorch on CUDA sorts slices of 32 or fewer with a bitonic network. Both are the book's "small sequences".
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch14-facts.md, D1 and D5)', '2026-11-20 (notes/today/ch14-facts.md, mục D1 và D5)')}>
        <Translated
          vi={
            <p>
            <code>ggml/src/ggml-cuda/argsort.cu</code>: "Bitonic sort of one row per block". PyTorch,{' '}
            <code>Sort.cpp</code>: <code>bitonicSortKVInPlace</code> "For very small unstable sorts (n &lt;= 32)". Trên HIP,
            llama.cpp không có đường đi qua CUB, nên các hàng dài đi theo lối khác.
            </p>
          }
        >
          <p>
            <code>ggml/src/ggml-cuda/argsort.cu</code>: "Bitonic sort of one row per block". PyTorch,{' '}
            <code>Sort.cpp</code>: <code>bitonicSortKVInPlace</code> "For very small unstable sorts (n &lt;= 32)". On HIP,
            llama.cpp has no CUB path, so long rows take another route.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS149['ch14/14.9/q-network-ties']} />

      <h2><Translated vi="Sắp xếp bằng phân hoạch: sample sort">Sorting by partitioning: sample sort</Translated></h2>
      <Translated
        vi={
          <p>
          Các phép sắp xếp so sánh không có mẫu cố định chia làm hai loại lớn. Một số cắt đầu vào thành các ô (tile), sắp từng ô và
          dành phần lớn công sức để ghép các ô: sắp xếp trộn, mục 14.3. Số khác dành phần lớn công sức để phân hoạch, sao cho việc
          ghép trở nên tầm thường. Sample sort là ví dụ điển hình: chọn p − 1 khóa, sắp chúng, và dùng chúng làm khóa chia để cắt
          đầu vào thành p bucket, mỗi bucket chỉ chứa các khóa nằm giữa hai khóa chia kề nhau. Sắp riêng từng bucket, rồi nối lại.
          Nó tổng quát hóa phép chia đôi của quicksort thành phép chia p phần.
          </p>
        }
      >
        <p>
          Comparison sorts without a fixed pattern come in two broad kinds. Some cut the input into tiles, sort each tile and
          do most of their work combining the tiles: merge sort, section 14.3. Others do most of their work partitioning, so
          that combining is trivial. Sample sort is the typical one: choose p − 1 keys, sort them, and use them as splitters to
          cut the input into p buckets, each holding only keys between two neighbouring splitters. Sort every bucket on its
          own, then concatenate. It generalises quicksort's split in two to a split in p.
        </p>
      </Translated>
      <Source section="14.9" pages="346" />
      <Translated
        vi={
          <p>
          Các khóa chia thường được chọn từ một mẫu lớn hơn p − 1 khóa, vì lấy mẫu dư vừa phải cho các bucket có kích thước gần bằng
          nhau với xác suất cao. Sách khuyên dùng sample sort cho các đầu vào rất lớn trải trên nhiều bộ nhớ vật lý, kể cả bộ nhớ
          của nhiều GPU trong một node; chúng tôi chưa kiểm tra lời khuyên đó.
          </p>
        }
      >
        <p>
          The splitters are usually chosen from a larger sample than p − 1 keys, because modest oversampling gives buckets of
          nearly equal size with high probability. The book recommends sample sort for very large inputs spread over several
          physical memories, including the memories of several GPUs in one node; we have not checked that recommendation.
        </p>
      </Translated>
      <Ask q={questionsS149['ch14/14.9/q-oversample']} />

      <h2><Translated vi="Radix sort LSD và MSD">LSD and MSD radix sort</Translated></h2>
      <Translated
        vi={
          <p>
          Radix sort cũng có sự phân chia giữa từ dưới lên và từ trên xuống như vậy. Radix sort của chương này đi từ chữ số có
          nghĩa nhỏ nhất trước (LSD): mỗi lượt sắp lại cả mảng, nên mỗi lượt xáo dữ liệu trên toàn cục. Một radix sort đi từ chữ số
          có nghĩa lớn nhất (MSD) phân hoạch theo chữ số cao nhất trước, rồi sắp riêng từng bucket theo chữ số kế tiếp, và cứ thế;
          mỗi bước làm việc trên một vùng nhỏ hơn, cục bộ hơn. Giống sample sort, nó thường là lựa chọn tốt hơn cho các đầu vào rất
          lớn.
          </p>
        }
      >
        <p>
          Radix sort has the same split between bottom-up and top-down. The chapter's radix sort is least-significant-digit
          first (LSD): every pass reorders the whole array, so every pass shuffles data globally. A most-significant-digit
          (MSD) radix sort partitions by the top digit first, then sorts each bucket on its own by the next digit, and so on;
          each step works on a smaller, more local region. Like sample sort, it is often the better choice for very large
          inputs.
        </p>
      </Translated>
      <Source section="14.9" pages="346" />
      <Ask q={questionsS149['ch14/14.9/q-msd']} />

      <h2><Translated vi="Top-k là một phép chọn, không phải một phép sắp xếp">Top-k is a selection, not a sort</Translated></h2>
      <Translated
        vi={
          <p>
          Top-k của một bộ lấy mẫu cần k xác suất lớn nhất, không cần thứ tự của trăm nghìn xác suất còn lại. Lối nghĩ MSD áp vào
          phép chọn làm đúng điều đó: đếm histogram của chữ số cao nhất, đi qua các bin từ trên xuống cho đến khi phủ đủ k khóa, giữ
          mọi bucket phía trên bin đó, và chỉ tinh chỉnh tiếp bin chứa khóa lớn thứ k. Đó là điều{' '}
          <code>top_k_radix_cuda</code> của llama.cpp làm (chữ số 8 bit, histogram theo từng khối trong bộ nhớ chia sẻ, một phép chọn
          đi qua các bin từ trên xuống), và điều top-k nhiều khối của PyTorch, <code>mbtopk</code>, làm. Phép chọn là chủ đề của
          module X1, ở <a href={href('x1', 'X1.4')}>X1.4, "Top-k là một phép chọn, không phải một phép sắp xếp"</a>.
          </p>
        }
      >
        <p>
          A sampler's top-k needs the k largest probabilities, not the order of the other hundred thousand. MSD thinking
          applied to selection does exactly that: count a histogram of the top digit, walk the bins from the top until k keys
          are covered, keep every bucket above that bin, and refine only the bin where the k-th largest key lies. That is what
          llama.cpp's <code>top_k_radix_cuda</code> does (8-bit digits, per-block histograms in shared memory, a select that
          walks bins from the top), and what PyTorch's multi-block top-k, <code>mbtopk</code>, does. Selection is the
          subject of module X1's <a href={href('x1', 'X1.4')}>X1.4, "Top-k is a selection, not a sort"</a>.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
            Không có trong sách; sách không mô tả thuật toán chọn nào. Nguồn: notes/today/ch14-facts.md D2, D5.
            </p>
          }
        >
          <p>
            Not in the book, which describes no selection algorithm. Sources: notes/today/ch14-facts.md D2, D5.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
          Top-p thì khác: cách phát biểu phổ biến sắp các xác suất, lấy tổng tích lũy (scan của chương 11) và cắt ở chỗ tổng đạt p.
          Hugging Face transformers, mlx-lm và bộ lấy mẫu backend của llama.cpp đều làm theo cách này. Đó không phải cách duy nhất:
          đường Triton mặc định của vLLM tìm điểm cắt bằng một phép tìm pivot mà không sắp xếp gì cả.
          </p>
        }
      >
        <p>
          Top-p is different: the common formulation sorts the probabilities, takes a running sum (chapter 11's scan) and cuts
          where the sum reaches p. Hugging Face transformers, mlx-lm and llama.cpp's backend sampler all do it this way. It is
          not the only way: vLLM's default Triton path finds the cut with a pivot search and no sort at all.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch14-facts.md, A7, B4, C1, D4, D7)', '2026-11-20 (notes/today/ch14-facts.md, các mục A7, B4, C1, D4, D7)')}>
        <Translated
          vi={
            <p>
            Thư viện: <code>DeviceTopK</code> của CUB hiện chỉ nhận cấu hình không có thứ tự, không tất định, thứ tự trùng không xác
            định; <code>DeviceBatchedTopK</code> cần CUDA 13.5 và giới hạn một đoạn ở 2^21 phần tử. rocPRIM có <code>topk</code> và{' '}
            <code>segmented_topk</code>, không có wrapper hipCUB. Apple ghi trong tài liệu MPSGraph{' '}
            <code>topK</code> (macOS 14) và <code>MPSMatrixFindTopK</code> cho k tối đa 16. <code>topk</code> của MLX chạy một phép
            sắp xếp đầy đủ trên Metal.
            </p>
          }
        >
          <p>
            Libraries: CUB's <code>DeviceTopK</code> currently accepts only the unordered, non-deterministic,
            unspecified-ties configuration; <code>DeviceBatchedTopK</code> needs CUDA 13.5 and caps a segment at 2^21 items.
            rocPRIM has <code>topk</code> and <code>segmented_topk</code>, with no hipCUB wrapper. Apple documents MPSGraph{' '}
            <code>topK</code> (macOS 14) and <code>MPSMatrixFindTopK</code> for k up to 16. MLX's <code>topk</code> runs a
            full sort on Metal.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Gọi gì thay vì tự viết">What to call instead of writing it</Translated></h2>
      <Translated
        vi={
          <p>
          Lời khuyên cuối của sách là dùng một thư viện như Thrust hay CUB thay vì tự viết kernel sắp xếp từ đầu, và coi sắp xếp
          như một nghiên cứu điển hình. Với các việc một bộ lấy mẫu cần làm, đây là các lời gọi hiện nay. Các tên đến từ tài liệu và
          header của từng nhà cung cấp; không gì trong khóa học này biên dịch CUB hay hipCUB, nên ở đây tất cả đều là unverified.
          </p>
        }
      >
        <p>
          The book's closing advice is to use a library such as Thrust or CUB rather than write a sorting kernel from scratch,
          and to treat sorting as a case study. For the jobs a sampler has, these are the calls today. The names come from
          each vendor's documentation and headers; nothing in this course compiles CUB or hipCUB, so all of it is unverified
          here.
        </p>
      </Translated>
      <Source section="14.10" pages="347" />
      <Today checked={tr('2026-11-20 (notes/today/ch14-facts.md, A2 to A6, B1 to B5, C1)', '2026-11-20 (notes/today/ch14-facts.md, các mục A2 đến A6, B1 đến B5, C1)')}>
        <Translated
          vi={
            <p>
            Trên Metal không có gì để gọi cho một phép sắp xếp hàng ổn định trên buffer, đó là một lý do các lab của chương tự viết
            radix sort cho mọi backend. <a href={href('ch14', 'labs')}>Trang lab</a> liệt kê chúng.
            </p>
          }
        >
          <p>
            On Metal there is nothing to call for a stable row sort over buffers, which is one reason the chapter's labs write
            their own radix sort for every backend. The <a href={href('ch14', 'labs')}>labs page</a> lists them.
          </p>
        </Translated>
      </Today>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="libraries">
        <thead>
          <tr><th><Translated vi="Việc">Job</Translated></th><th>CUDA</th><th>HIP (ROCm){/* i18n-ignore: backend name */}</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="Sắp các cặp (key, token), lớn nhất trước, ổn định">Sort (key, token) pairs, largest first, stable</Translated></td>
            <td>
              <Translated vi={<><code>cub::DeviceRadixSort::SortPairsDescending</code>: tài liệu ghi là ổn định, kèm phép biến đổi khóa float được ghi trong tài liệu</>}>
                <code>cub::DeviceRadixSort::SortPairsDescending</code>: documented stable, with a documented float-key transform
              </Translated>
            </td>
            <td>
              <Translated vi={<><code>hipcub::DeviceRadixSort::SortPairsDescending</code>, dựa trên <code>rocprim::radix_sort_pairs_desc</code>: tài liệu ghi là ổn định</>}>
                <code>hipcub::DeviceRadixSort::SortPairsDescending</code>, over <code>rocprim::radix_sort_pairs_desc</code>: documented stable
              </Translated>
            </td>
            <td>
              <Translated vi={<>Không có phép sắp xếp thư viện ở mức kernel. MPSGraph <code>argSort(_:axis:descending:name:)</code> (macOS 13); không nêu tính ổn định</>}>
                No kernel-level library sort. MPSGraph <code>argSort(_:axis:descending:name:)</code> (macOS 13); stability not stated
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="Mọi hàng của một batch, lớn nhất trước, ổn định">Every row of a batch, largest first, stable</Translated></td>
            <td>
              <Translated vi={<><code>cub::DeviceSegmentedRadixSort::SortPairsDescending</code> hoặc <code>cub::DeviceSegmentedSort::StableSortPairsDescending</code>; không bao giờ dùng <code>DeviceSegmentedSort::SortPairsDescending</code>, được tài liệu ghi là không ổn định</>}>
                <code>cub::DeviceSegmentedRadixSort::SortPairsDescending</code> or <code>cub::DeviceSegmentedSort::StableSortPairsDescending</code>; never <code>DeviceSegmentedSort::SortPairsDescending</code>, documented as not stable
              </Translated>
            </td>
            <td><Translated vi="Cùng các tên đó trong hipCUB">The same names in hipCUB</Translated></td>
            <td>
              <Translated vi={<>MPSGraph <code>sort</code> hoặc <code>argSort</code> dọc theo một trục; không nêu tính ổn định, thứ tự trùng và thứ tự NaN</>}>
                MPSGraph <code>sort</code> or <code>argSort</code> along an axis; stability, tie order and NaN order not stated
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="Sắp xếp mức cao">High-level sort</Translated></td>
            <td>
              <Translated vi={<><code>thrust::sort</code>, <code>thrust::stable_sort_by_key</code>: khóa số học với less hoặc greater đi tới radix sort của CUB; một lambda so sánh tùy biến đi tới sắp xếp trộn</>}>
                <code>thrust::sort</code>, <code>thrust::stable_sort_by_key</code>: arithmetic keys with less or greater go to CUB's radix sort; a custom comparator lambda goes to merge sort
              </Translated>
            </td>
            <td><Translated vi="rocThrust: cùng cách chia đó, trên rocPRIM">rocThrust: the same split, onto rocPRIM</Translated></td>
            <td><Translated vi="Không có">None</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Top-k không cần sắp xếp đầy đủ">Top-k without a full sort</Translated></td>
            <td>
              <Translated vi={<><code>cub::DeviceTopK</code> (đến nay chỉ có loại không thứ tự, không tất định); <code>cub::DeviceBatchedTopK</code> từ CUDA 13.5</>}>
                <code>cub::DeviceTopK</code> (unordered, non-deterministic only, so far); <code>cub::DeviceBatchedTopK</code> from CUDA 13.5
              </Translated>
            </td>
            <td>
              <Translated vi={<><code>rocprim::topk</code>, <code>rocprim::segmented_topk</code>; không có wrapper hipCUB</>}>
                <code>rocprim::topk</code>, <code>rocprim::segmented_topk</code>; no hipCUB wrapper
              </Translated>
            </td>
            <td>
              <Translated vi={<>MPSGraph <code>topK</code> (macOS 14); <code>MPSMatrixFindTopK</code>, k tối đa 16</>}>
                MPSGraph <code>topK</code> (macOS 14); <code>MPSMatrixFindTopK</code>, k at most 16
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="Sắp xếp ngay trong ngôn ngữ kernel">Sort in the kernel language itself</Translated></td>
            <td>
              <Translated vi={<>Không có; <code>cub::BlockRadixSort</code> bên trong một kernel</>}>
                None; <code>cub::BlockRadixSort</code> inside a kernel
              </Translated>
            </td>
            <td>
              <Translated vi={<><code>hipcub::BlockRadixSort</code> (tham số độ rộng chữ số của nó bị bỏ qua)</>}>
                <code>hipcub::BlockRadixSort</code> (its digit-width parameter is ignored)
              </Translated>
            </td>
            <td>
              <Translated vi="Không có: MSL không có hàm sắp xếp, và mã mẫu của Apple không có mẫu sắp xếp nào">
                None: MSL has no sort, and Apple's sample code has no sorting sample
              </Translated>
            </td>
          </tr>
        </tbody>
      </table>
    </>
  );
}
