import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import {
  coRankTrace,
  knownRange,
  mergeByThreads,
  mergeStable,
  partition,
  type CoRankStep,
  type Slice,
  type Tagged,
} from '../../sim/corank';

// "Searching for the split": the book's running example (Fig. 13.1), thread 1's
// co-rank search for k = 3 (Figs. 13.6 to 13.8), every thread's partition, and the
// swapped tie rule. Every number drawn comes from src/sim/corank.ts. Six beats, one
// narration clip each (content/ch13/narration/corank-search-N.txt).
const A = [1, 7, 8, 9, 10];
const B = [7, 10, 10, 12];
const K = 3; // thread 1's first output when three threads take three outputs each
const TRACE = coRankTrace(K, A, B);
const S = TRACE.steps; // three iterations: i too high, j too high, done
const BOOK = partition(A, B, 3, 3);
const SWAP = partition(A, B, 3, 3, 'swapped');
const C_BOOK = mergeStable(A, B);
const C_SWAP = mergeByThreads(A, B, SWAP) as Tagged[];
const MOVED = C_SWAP.flatMap((e, p) => (e.src !== C_BOOK[p].src || e.idx !== C_BOOK[p].idx ? [p] : []));

const W = 720;
const CW = 46; // cell size
const AX = 70;
const BX = AX + A.length * CW + 96;
const RY = 64; // top of the A and B rows
const CX = (W - C_BOOK.length * CW) / 2;
const CY = 226; // top of the C row

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const ease = (t: number) => t * t * (3 - 2 * t);
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const ax = (b: number) => AX + b * CW;
const bx = (b: number) => BX + b * CW;
const cx = (b: number) => CX + b * CW;
const tag = (e: Tagged) => `${e.src}${e.idx}`;

type Mark = 'fail' | 'ok';
interface Pos { i: number; j: number; iLow: number; jLow: number }

interface Frame {
  beat: number;
  appear?: number; // opacity of the three arrays
  cuts?: number; // opacity of C's three slices
  unknown?: number; // opacity of the question marks over A and B
  pos?: Pos; // bar positions (fractional while moving)
  labels?: Pos; // the integers printed beside the bars
  badge?: string;
  tests?: { text: string; mark: Mark }[];
  markA?: Map<number, Mark>;
  markB?: Map<number, Mark>;
  window?: [number, number];
  outline?: boolean; // thread 1's outputs C[3..6) outlined
  slices?: Slice[]; // every thread's bands
  sliceText?: string;
  tagged?: Tagged[]; // C drawn with source tags
  badC?: Set<number>;
  headline: string;
  data?: Record<string, string | number>;
}

function Row({ x, y, name, values, mark, shade, show = 1 }: {
  x: number; y: number; name: string; values: number[]; mark?: Map<number, Mark>; shade?: (p: number) => boolean; show?: number;
}) {
  return (
    <g opacity={show}>
      <text x={x - 12} y={y + CW / 2 + 5} fontSize="15" fontWeight="650" textAnchor="end" style={{ fill: 'var(--global)' }}>{name}</text>
      {values.map((v, p) => {
        const m = mark?.get(p);
        return (
          <g key={p} data-cell={`${name}${p}`} data-mark={m ?? ''}>
            <rect x={x + p * CW} y={y} width={CW - 3} height={CW - 3} rx="3"
              fill={m ? `var(--${m}-wash)` : shade?.(p) ? 'var(--global-wash)' : 'var(--cell)'}
              stroke={m ? `var(--${m})` : shade?.(p) ? 'var(--global)' : 'var(--rule)'} strokeWidth={m ? 2 : 1} />
            <text x={x + p * CW + (CW - 3) / 2} y={y + CW / 2 + 3} fontSize="15" textAnchor="middle"
              style={{ fill: m ? `var(--${m})` : 'var(--ink)', fontWeight: m ? 700 : 450 }}>{v}</text>
            <text x={x + p * CW + (CW - 3) / 2} y={y - 4} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{p}</text>
          </g>
        );
      })}
    </g>
  );
}

/** A marker bar on a boundary between cells, as in Figs. 13.6 to 13.8. */
function Bar({ x, y, label, low }: { x: number; y: number; label: string; low?: boolean }) {
  const X = x - 1.5;
  return (
    <g>
      <line x1={X} y1={low ? y + 6 : y - 14} x2={X} y2={low ? y + CW + 8 : y + CW - 8} stroke="var(--ink)" strokeWidth={low ? 1.6 : 3}
        strokeDasharray={low ? '4 3' : undefined} />
      <text x={X} y={low ? y + CW + 20 : y - 18} fontSize="12" fontWeight={low ? 450 : 650} textAnchor="middle">{label}</text>
    </g>
  );
}

function Stage(f: Frame) {
  const pos = f.pos;
  const lab = f.labels ?? pos;
  const cRow = f.tagged ?? C_BOOK;
  return (
    <svg viewBox={`0 0 ${W} 340`} role="img" aria-label={tr('Co-rank search on the running example', 'Tìm co-rank trên ví dụ xuyên suốt')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <g data-beat={f.beat} {...Object.fromEntries(Object.entries(f.data ?? {}).map(([k, v]) => [`data-${k}` /* i18n-ignore: data attribute name */, String(v)]))}>
        {f.slices ? (
          <g data-slices={f.slices.map((s) => `${s.iCurr},${s.jCurr}`).join(' ') + ` | ${f.slices[f.slices.length - 1].iNext},${f.slices[f.slices.length - 1].jNext}`}>
            {f.slices.map((s) => (
              <g key={s.t}>
                {s.iNext > s.iCurr ? (
                  <text x={(ax(s.iCurr) + ax(s.iNext)) / 2} y={RY - 22} fontSize="11.5" textAnchor="middle" fontWeight="600">t{s.t}</text>
                ) : null}
                {s.jNext > s.jCurr ? (
                  <text x={(bx(s.jCurr) + bx(s.jNext)) / 2} y={RY - 22} fontSize="11.5" textAnchor="middle" fontWeight="600">t{s.t}</text>
                ) : null}
              </g>
            ))}
          </g>
        ) : null}
        <Row x={AX} y={RY} name="A" values={A} mark={f.markA} show={f.appear ?? 1}
          shade={f.slices ? (p) => f.slices!.some((s) => s.t % 2 === 0 && p >= s.iCurr && p < s.iNext) : undefined} />
        <Row x={BX} y={RY} name="B" values={B} mark={f.markB} show={f.appear ?? 1}
          shade={f.slices ? (p) => f.slices!.some((s) => s.t % 2 === 0 && p >= s.jCurr && p < s.jNext) : undefined} />
        {f.unknown ? (
          <g opacity={f.unknown}>
            <text x={(ax(0) + ax(A.length)) / 2} y={RY - 16} fontSize="13" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}><Translated vi="luồng 1 bắt đầu ở đâu trong A?">where does thread 1 start in A?</Translated></text>
            <text x={(bx(0) + bx(B.length)) / 2} y={RY - 16} fontSize="13" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}><Translated vi="còn trong B?">and in B?</Translated></text>
          </g>
        ) : null}
        {pos && lab ? (
          <g>
            <Bar x={ax(pos.i)} y={RY} label={`i = ${lab.i}`} />
            <Bar x={ax(pos.iLow) + (Math.abs(pos.iLow - pos.i) < 0.05 ? 4 : 0)} y={RY} label={`i_low = ${lab.iLow}`} low />
            <Bar x={bx(pos.j)} y={RY} label={`j = ${lab.j}`} />
            <Bar x={bx(pos.jLow) + (Math.abs(pos.jLow - pos.j) < 0.05 ? 4 : 0)} y={RY} label={`j_low = ${lab.jLow}`} low />
          </g>
        ) : null}
        {f.slices ? (
          <g>
            {(['A', 'B'] as const).map((arr) => {
              // Where each thread starts in this array, and where the last one ends; threads that start at the same place share a label.
              const at = new Map<number, string[]>();
              const last = f.slices![f.slices!.length - 1];
              for (const s of f.slices!) {
                const b = arr === 'A' ? s.iCurr : s.jCurr;
                at.set(b, [...(at.get(b) ?? []), `t${s.t}`]);
              }
              const end = arr === 'A' ? last.iNext : last.jNext;
              at.set(end, [...(at.get(end) ?? []), tr('end', 'cuối')]);
              const x = arr === 'A' ? ax : bx;
              return [...at].map(([b, names]) => (
                <g key={arr + b}>
                  <line x1={x(b) - 1.5} y1={RY - 10} x2={x(b) - 1.5} y2={RY + CW} stroke="var(--ink)" strokeWidth="2.4" />
                  <text x={x(b) - 1.5} y={RY + CW + 16} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{names.join(', ')}: {b}</text>
                </g>
              ));
            })}
          </g>
        ) : null}
        {f.window ? (
          <g data-window={`${f.window[0]}-${f.window[1]}`}>
            <line x1={ax(f.window[0]) - 1.5} y1={RY + CW + 32} x2={ax(f.window[1]) - 1.5 + (f.window[0] === f.window[1] ? 0.01 : 0)} y2={RY + CW + 32} stroke="var(--ink-2)" strokeWidth="2" />
            {Array.from({ length: f.window[1] - f.window[0] + 1 }, (_, q) => (
              <circle key={q} cx={ax(f.window![0] + q) - 1.5} cy={RY + CW + 32} r="4" fill="var(--ink)" />
            ))}
            <text x={AX - 30} y={RY + CW + 50} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
              <Translated vi={<>{f.window[1] - f.window[0] + 1} ứng viên cho i</>}>
                {f.window[1] - f.window[0] + 1} candidate{f.window[0] === f.window[1] ? '' : 's'} for i
              </Translated>
            </text>
          </g>
        ) : null}
        {f.badge ? (
          <g>
            <rect x={W - 150} y={RY + CW + 48} width={130} height={24} rx="3" fill="var(--ok-wash)" stroke="var(--ok)" />
            <text x={W - 85} y={RY + CW + 64} fontSize="12.5" textAnchor="middle" fontWeight="650" style={{ fill: 'var(--ok)' }} data-badge>{f.badge}</text>
          </g>
        ) : null}
        {f.tests ? (
          <g>
            {f.tests.map((t, q) => (
              <text key={q} x={AX - 30} y={RY + CW + 70 + q * 18} fontSize="12.5" fontWeight="600" style={{ fill: `var(--${t.mark})` }} data-test={t.mark}>{t.text}</text>
            ))}
          </g>
        ) : null}
        {f.sliceText ? (
          <text x={W / 2} y={RY + CW + 70} fontSize="12" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{f.sliceText}</text>
        ) : null}

        {/* C */}
        <g opacity={f.appear ?? 1}>
          <Row x={CX} y={CY} name="C" values={cRow.map((e) => e.key)}
            mark={f.badC ? new Map([...f.badC].map((p) => [p, 'fail' as Mark])) : undefined}
            shade={f.outline ? (p) => p >= K && p < K + 3 : undefined} />
          {f.tagged || f.slices ? (
            <g data-order={cRow.map(tag).join(' ')}>
              {cRow.map((e, p) => (
                <text key={p} x={cx(p) + (CW - 3) / 2} y={CY + CW + 12} fontSize="10.5" textAnchor="middle"
                  style={{ fill: f.badC?.has(p) ? 'var(--fail)' : 'var(--ink-2)', fontWeight: f.badC?.has(p) ? 700 : 450 }}>{tag(e)}</text>
              ))}
            </g>
          ) : null}
          {f.outline ? (
            <g>
              <rect x={cx(K) - 4} y={CY - 4} width={3 * CW + 5} height={CW + 5} rx="4" fill="none" stroke="var(--ink)" strokeWidth="2" />
              <text x={cx(K) + 1.5 * CW} y={CY + CW + 18} fontSize="11.5" textAnchor="middle"><Translated vi="luồng 1: k = 3, 4, 5">thread 1: k = 3, 4, 5</Translated></text>
            </g>
          ) : null}
        </g>
        {f.cuts ? (
          <g opacity={f.cuts}>
            {[0, 3, 6, 9].map((k) => (
              <line key={k} x1={cx(k) - 1.5} y1={CY - 12} x2={cx(k) - 1.5} y2={CY + CW + 2} stroke="var(--ink)" strokeWidth="2.6" />
            ))}
            {[0, 1, 2].map((t) => (
              <text key={t} x={cx(3 * t) + 1.5 * CW} y={CY + CW + (f.tagged || f.slices ? 28 : 16)} fontSize="11.5" textAnchor="middle">
                <Translated vi={<>luồng {t}: k = {3 * t}</>}>
                  thread {t}: k = {3 * t}
                </Translated>
              </text>
            ))}
          </g>
        ) : null}
        <text x={W / 2} y={330} fontSize="13.5" fontWeight="600" textAnchor="middle" data-headline>{f.headline}</text>
      </g>
    </svg>
  );
}

/** The two tests of one iteration, as the scene prints them. */
function testLines(st: CoRankStep): { text: string; mark: Mark }[] {
  const out: { text: string; mark: Mark }[] = [];
  if (st.t1.inBounds)
    out.push({
      text: st.t1.holds
        ? tr(`A[${st.i - 1}] = ${st.t1.left} > B[${st.j}] = ${st.t1.right}: i is too high`, `A[${st.i - 1}] = ${st.t1.left} > B[${st.j}] = ${st.t1.right}: i quá cao`)
        : tr(`A[${st.i - 1}] = ${st.t1.left} ≤ B[${st.j}] = ${st.t1.right}: first test passes`, `A[${st.i - 1}] = ${st.t1.left} ≤ B[${st.j}] = ${st.t1.right}: phép thử thứ nhất qua`),
      mark: st.t1.holds ? 'fail' : 'ok',
    });
  if (st.t2?.inBounds)
    out.push({
      text: st.t2.holds
        ? tr(`B[${st.j - 1}] = ${st.t2.left} ≥ A[${st.i}] = ${st.t2.right}: j is too high`, `B[${st.j - 1}] = ${st.t2.left} ≥ A[${st.i}] = ${st.t2.right}: j quá cao`)
        : tr(`B[${st.j - 1}] = ${st.t2.left} < A[${st.i}] = ${st.t2.right}: second test passes`, `B[${st.j - 1}] = ${st.t2.left} < A[${st.i}] = ${st.t2.right}: phép thử thứ hai qua`),
      mark: st.t2.holds ? 'fail' : 'ok',
    });
  return out;
}

function marksOf(st: CoRankStep, upto: number): { a: Map<number, Mark>; b: Map<number, Mark> } {
  const a = new Map<number, Mark>();
  const b = new Map<number, Mark>();
  const ts = [st.t1, st.t2].slice(0, upto);
  ts.forEach((t, q) => {
    if (!t?.inBounds) return;
    const m: Mark = t.holds ? 'fail' : 'ok';
    if (q === 0) {
      a.set(st.i - 1, m);
      b.set(st.j, m);
    } else {
      b.set(st.j - 1, m);
      a.set(st.i, m);
    }
  });
  return { a, b };
}

const stateOf = (s: CoRankStep): Pos => ({ i: s.i, j: s.j, iLow: s.iLow, jLow: s.jLow });
const between = (x: Pos, y: Pos, t: number): Pos => ({ i: lerp(x.i, y.i, t), j: lerp(x.j, y.j, t), iLow: lerp(x.iLow, y.iLow, t), jLow: lerp(x.jLow, y.jLow, t) });
const sum = (s: Pos) => `i + j = ${s.i} + ${s.j} = ${s.i + s.j}`;
const data = (s: CoRankStep, shown: boolean) => ({ i: s.i, j: s.j, ilow: s.iLow, jlow: s.jLow, verdict: shown ? s.test : '' });

/** One iteration: the bars move in from the previous state, then the tests light up. */
function iteration(n: number, p: number, headline: (shown: boolean) => string): Frame {
  const st = S[n];
  const from = n > 0 ? stateOf(S[n - 1]) : stateOf(st);
  const moving = n > 0 ? ease(ramp(p, 0.05, 0.38)) : 1;
  const shown = p >= (n > 0 ? 0.48 : 0.42);
  const firstOnly = shown && p < (n > 0 ? 0.68 : 1) && st.t1.holds === false && st.t2 != null;
  const lines = testLines(st);
  const upto = firstOnly ? 1 : 2;
  const marks = shown ? marksOf(st, upto) : undefined;
  return {
    beat: n + 2,
    pos: between(from, stateOf(st), moving),
    labels: moving < 1 ? from : stateOf(st),
    badge: sum(moving < 1 ? from : stateOf(st)),
    tests: shown ? lines.slice(0, upto) : undefined,
    markA: marks?.a,
    markB: marks?.b,
    window: knownRange(TRACE, shown ? n : n - 1),
    outline: true,
    headline: headline(shown),
    data: data(st, shown && !firstOnly),
  };
}

const say = (s: CoRankStep) => (s.delta ? `delta = ${s.delta}` : '');

const beats: Beat[] = [
  {
    clip: 'corank-search-1',
    get label() {
      return tr('Three threads', 'Ba luồng');
    },
    draw: (p) => (
      <Stage beat={1} cuts={ramp(p, 0.3, 0.45)} unknown={ramp(p, 0.62, 0.75)}
        headline={p < 0.55 ? tr('nine outputs, three threads, three outputs each', 'chín đầu ra, ba luồng, mỗi luồng ba đầu ra') : tr('each thread knows where its outputs start, not where its inputs start', 'mỗi luồng biết đầu ra của nó bắt đầu ở đâu, nhưng không biết đầu vào của nó bắt đầu ở đâu')} />
    ),
  },
  {
    clip: 'corank-search-2',
    get label() {
      return tr('A first guess', 'Lần đoán đầu tiên');
    },
    draw: (p) => <Stage {...iteration(0, p, (shown) => (shown ? tr(`${say(S[0])}: move i down and j up`, `${say(S[0])}: dời i xuống và j lên`) : tr('start with as much of A as possible: i = min(k, m)', 'bắt đầu với nhiều phần của A nhất có thể: i = min(k, m)')))} />,
  },
  {
    clip: 'corank-search-3',
    get label() {
      return tr('Halve the gap', 'Chia đôi khoảng cách');
    },
    draw: (p) => <Stage {...iteration(1, p, (shown) => (shown ? tr(`${say(S[1])}: now move j down and i up`, `${say(S[1])}: giờ dời j xuống và i lên`) : tr(`i and j move by ${S[0].delta} together; their sum stays 3`, `i và j cùng dời ${S[0].delta}; tổng của chúng vẫn là 3`)))} />,
  },
  {
    clip: 'corank-search-4',
    get label() {
      return tr('The answer', 'Đáp án');
    },
    draw: (p) => (
      <Stage {...iteration(2, p, (shown) =>
        shown
          ? tr(`co-rank of 3: i = ${TRACE.result}, j = ${K - TRACE.result}; ${TRACE.candidates} candidates to 1 in ${S.length} iterations`, `co-rank của 3: i = ${TRACE.result}, j = ${K - TRACE.result}; từ ${TRACE.candidates} ứng viên còn 1 sau ${S.length} lần lặp`)
          : tr(`i and j move by ${S[1].delta}`, `i và j dời ${S[1].delta}`))} />
    ),
  },
  {
    clip: 'corank-search-5',
    get label() {
      return tr('Every thread at once', 'Mọi luồng cùng lúc');
    },
    draw: (p) => {
      const t1 = BOOK[1];
      const t2 = BOOK[2];
      return (
        <Stage beat={5} slices={BOOK} cuts={1}
          sliceText={p > 0.45 ? tr(`thread 1: A[${t1.iCurr}..${t1.iNext}) and ${t1.jNext - t1.jCurr} of B;  thread 2: ${t2.iNext - t2.iCurr} of A and B[${t2.jCurr}..${t2.jNext})`, `luồng 1: A[${t1.iCurr}..${t1.iNext}) và ${t1.jNext - t1.jCurr} phần tử của B;  luồng 2: ${t2.iNext - t2.iCurr} phần tử của A và B[${t2.jCurr}..${t2.jNext})`) : undefined}
          headline={p < 0.7 ? tr('every thread searches for its own start, at the same time', 'mọi luồng cùng lúc tìm điểm bắt đầu của riêng nó') : tr('then each merges its own slices: no thread waits for another', 'rồi mỗi luồng trộn các lát của riêng nó: không luồng nào phải chờ luồng khác')} />
      );
    },
  },
  {
    clip: 'corank-search-6',
    get label() {
      return tr('Why the tests are lopsided', 'Vì sao hai phép thử lệch nhau');
    },
    draw: (p) => {
      const swapped = p >= 0.42;
      return (
        <Stage beat={6} slices={swapped ? SWAP : BOOK} cuts={1} tagged={swapped ? C_SWAP : C_BOOK}
          badC={swapped && p >= 0.6 ? new Set(MOVED) : undefined}
          data={{ rule: swapped ? 'swapped' : 'book' }}
          headline={
            !swapped
              ? tr('the book\'s tests: equal keys keep their order, A first', 'các phép thử của sách: khóa bằng nhau giữ nguyên thứ tự, A trước')
              : p < 0.6
                ? tr('swap > and >= in the two tests: the slices move', 'đổi > và >= trong hai phép thử: các lát dịch chuyển')
                : tr(`keys still sorted, but ${tag(C_SWAP[MOVED[0]])} now comes before ${tag(C_SWAP[MOVED[1]])}: not stable`, `khóa vẫn đã sắp xếp, nhưng ${tag(C_SWAP[MOVED[0]])} giờ đứng trước ${tag(C_SWAP[MOVED[1]])}: không ổn định`)
          } />
      );
    },
  },
];

export function CoRankScene() {
  return <Scene chapter="ch13" title={tr('Searching for the split', 'Tìm điểm chia')} beats={beats} />;
}
