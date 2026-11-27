import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { defineLabs } from '../../components/LabKit';
import { cuStatus } from './labStatus';
import { Beyond, Bench, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { excerptFunction } from '../../lib/excerpt';
import { useReports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  BOOK_SHAPE,
  fmas,
  readsPerFma,
  registersFor,
  reuse,
  run,
  sharedReadsV1,
  sharedReadsV2,
  smallInputs,
  stateAt,
  type Shape,
  type State,
} from '../../sim/gemmreads';
import { OuterProductScene } from './OuterProductScene';

// ---------------------------------------------------------------- the two mm functions, from the labs
export const LABS154 = defineLabs('ch15', [
  { num: '15.1', dir: 'lab1_coarse', name: 'coarsened', vi: { name: 'gộp việc cho luồng' } },
  { num: '15.2', dir: 'lab2_regtile', name: 'register tiling', vi: { name: 'phân ô thanh ghi' } },
]);
const [LAB1, LAB2] = LABS154;
export const MM_FILES = {
  v1: `${LAB1.id}/solution/kernel.cu`,
  v2: `${LAB2.id}/solution/kernel.cu`,
};

const TIERS: Record<string, Tier> = {
  a: 'shared', b: 'shared', a_r: 'reg', b_r: 'reg', c: 'reg', row: 'reg', col: 'reg', i: 'reg',
};

/**
 * One function of a lab file, located by name (lib/excerpt.ts), from the line that declares
 * it to its closing brace, without the comment above it, with the file's own line numbers.
 */
export function functionOf(text: string, name: string): { code: string; firstLine: number } | null {
  const e = excerptFunction(text, name, { comment: false });
  return e ? { code: e.text.replace(/\n$/, ''), firstLine: e.firstLine } : null;
}

function TwoMm() {
  const { reports } = useReports();
  const f1 = useLabFile(MM_FILES.v1);
  const f2 = useLabFile(MM_FILES.v2);
  if (!f1 || !f2) return null;
  const m1 = functionOf(f1, 'mm');
  const m2 = functionOf(f2, 'mm');
  return (
    <div className="wide" data-two-mm>
      <Code
        tabs={[{ label: tr('mm, lab 15.1 (Fig. 15.6)', 'mm, lab 15.1 (Hình 15.6)'), code: m1 ? m1.code : f1, tiers: TIERS, status: cuStatus(reports, LAB1) }]}
        firstLine={m1?.firstLine ?? 1}
        caption={
          <Translated vi={<>Từ <code>labs/{MM_FILES.v1}</code>: <code>i</code> trong cùng, hai lần đọc bộ nhớ chia sẻ mỗi phép nhân-cộng.</>}>
            <>From <code>labs/{MM_FILES.v1}</code>: <code>i</code> innermost, two shared reads per multiply-add.</>
          </Translated>
        }
      />
      <Code
        tabs={[{ label: tr('mm, lab 15.2 (Fig. 15.9)', 'mm, lab 15.2 (Hình 15.9)'), code: m2 ? m2.code : f2, tiers: TIERS, status: cuStatus(reports, LAB2) }]}
        firstLine={m2?.firstLine ?? 1}
        caption={
          <Translated vi={<>Từ <code>labs/{MM_FILES.v2}</code>: <code>i</code> ngoài cùng, các dải nằm trong thanh ghi.</>}>
            <>From <code>labs/{MM_FILES.v2}</code>: <code>i</code> outermost, strips in registers.</>
          </Translated>
        }
      />
    </div>
  );
}

// ---------------------------------------------------------------- explorer and stepper: two orders of the same loops
const PRESETS: { label: string; s: Shape }[] = [
  // Getters: the labels are read in the selected language when the buttons render.
  { get label() { return tr('the book: 8 × 8, bK 8', 'của sách: 8 × 8, bK 8'); }, s: { tM: 8, tN: 8, bK: 8 } },
  { label: '16 × 4', s: { tM: 16, tN: 4, bK: 8 } },
  { label: '32 × 2', s: { tM: 32, tN: 2, bK: 8 } },
  { get label() { return tr('small: 3 × 3, bK 4', 'nhỏ: 3 × 3, bK 4'); }, s: { tM: 3, tN: 3, bK: 4 } },
];

/** One order's picture: the thread's A rows and B columns, the strips (Fig. 15.9 only), the patch. */
function OrderPanel({ s, st, order, inputs }: { s: Shape; st: State; order: 'v1' | 'v2'; inputs: { a: number[]; b: number[] } }) {
  const cs = Math.max(7, Math.min(20, Math.floor(190 / Math.max(s.tM, s.tN, s.bK))));
  const big = cs >= 15;
  const gap = 6;
  const ax = 4, bx = ax + s.bK * cs + gap + cs + gap, by = 16, ay = by + s.bK * cs + gap + cs + gap;
  const arx = ax + s.bK * cs + gap, bry = by + s.bK * cs + gap;
  const last = st.step[st.step.length - 1];
  const loads = st.step.filter((o) => o.kind !== 'fma');
  const hotA = new Set<number>(), hotB = new Set<number>(), hotAR = new Set<number>(), hotBR = new Set<number>();
  let hotC = -1;
  if (last?.kind === 'fma') {
    hotC = last.row * s.tN + last.col;
    if (order === 'v1') {
      hotA.add(last.row * s.bK + last.i);
      hotB.add(last.i * s.tN + last.col);
    } else {
      hotAR.add(last.row);
      hotBR.add(last.col);
    }
  }
  for (const o of loads) {
    if (o.kind === 'loadA') { hotA.add(o.row * s.bK + o.i); hotAR.add(o.row); }
    if (o.kind === 'loadB') { hotB.add(o.i * s.tN + o.col); hotBR.add(o.col); }
  }
  const cell = (key: string, x: number, y: number, v: ReactNode, fill: string, stroke: string, ink: string, sub?: string, attrs?: Record<string, string>) => (
    <g key={key} {...attrs}>
      <rect x={x + 0.5} y={y + 0.5} width={cs - 1.5} height={cs - 1.5} rx="1.5" fill={fill} stroke={stroke} strokeWidth="0.7" />
      {big ? <text x={x + cs / 2} y={y + cs / 2 + 3.5} fontSize="9.5" textAnchor="middle" style={{ fill: ink }}>{v}</text> : null}
      {big && sub ? <text x={x + cs - 2} y={y + cs - 2} fontSize="6.5" textAnchor="end" style={{ fill: ink === 'var(--page)' ? 'var(--page)' : 'var(--ink-3)' }}>{sub}</text> : null}
    </g>
  );
  const els: ReactNode[] = [];
  for (let r = 0; r < s.tM; r++)
    for (let k = 0; k < s.bK; k++) {
      const i = r * s.bK + k, n = st.aReads[i], h = hotA.has(i);
      els.push(cell('a' + i, ax + k * cs, ay + r * cs, inputs.a[i], h ? 'var(--shared)' : n ? 'var(--shared-wash)' : 'var(--cell)', 'var(--shared)', h ? 'var(--page)' : 'var(--ink)', n > 1 ? `${n}` : undefined, { 'data-a': String(i), 'data-reads': String(n) }));
    }
  for (let k = 0; k < s.bK; k++)
    for (let c = 0; c < s.tN; c++) {
      const i = k * s.tN + c, n = st.bReads[i], h = hotB.has(i);
      els.push(cell('b' + i, bx + c * cs, by + k * cs, inputs.b[i], h ? 'var(--shared)' : n ? 'var(--shared-wash)' : 'var(--cell)', 'var(--shared)', h ? 'var(--page)' : 'var(--ink)', n > 1 ? `${n}` : undefined, { 'data-b': String(i), 'data-reads': String(n) }));
    }
  if (order === 'v2') {
    for (let r = 0; r < s.tM; r++) {
      const v = st.aR[r], h = hotAR.has(r);
      els.push(cell('ar' + r, arx, ay + r * cs, v ?? '', h ? 'var(--reg)' : v != null ? 'var(--reg-wash)' : 'var(--cell)', 'var(--reg)', h ? 'var(--page)' : 'var(--ink)'));
    }
    for (let c = 0; c < s.tN; c++) {
      const v = st.bR[c], h = hotBR.has(c);
      els.push(cell('br' + c, bx + c * cs, bry, v ?? '', h ? 'var(--reg)' : v != null ? 'var(--reg-wash)' : 'var(--cell)', 'var(--reg)', h ? 'var(--page)' : 'var(--ink)'));
    }
  }
  for (let r = 0; r < s.tM; r++)
    for (let c = 0; c < s.tN; c++) {
      const i = r * s.tN + c, v = st.C[i], u = st.updates[i] > 0, h = i === hotC;
      els.push(cell('c' + i, bx + c * cs, ay + r * cs, u ? v : '', h ? 'var(--reg)' : u ? 'var(--reg-wash)' : 'var(--cell)', 'var(--reg)', h ? 'var(--page)' : 'var(--ink)', undefined, { 'data-c': String(i), 'data-v': String(v) }));
    }
  const W = bx + s.tN * cs + 22, H = ay + s.tM * cs + 6;
  return (
    <div className="panel" data-order={order}>
      <div className="mx-label">{order === 'v1' ? tr('Fig. 15.6: i innermost', 'Hình 15.6: i trong cùng') : tr('Fig. 15.9: i outermost, strips in registers', 'Hình 15.9: i ngoài cùng, các dải nằm trong thanh ghi')}</div>
      <div className="ui" style={{ marginBottom: '0.4rem' }}>
        <Translated vi={<>số lần đọc bộ nhớ chia sẻ đến giờ: <b data-reads-total>{st.sharedReads}</b></>}>
          shared reads so far: <b data-reads-total>{st.sharedReads}</b>
        </Translated>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, height: 'auto', display: 'block' }} role="img" aria-label={order === 'v1' ? tr('First loop order', 'Thứ tự vòng lặp đầu tiên') : tr('Interchanged loop order', 'Thứ tự vòng lặp đã hoán đổi')}>
          <text x={bx} y={10} fontSize="9" style={{ fill: 'var(--shared)' }}>b (B_s)</text>
          <text x={ax} y={ay - 4} fontSize="9" style={{ fill: 'var(--shared)' }}>a (A_s)</text>
          {order === 'v2' ? <text x={arx} y={ay - 4} fontSize="9" style={{ fill: 'var(--reg)' }}>a_r</text> : null}
          {order === 'v2' ? <text x={bx + s.tN * cs + 3} y={bry + cs - 5} fontSize="9" style={{ fill: 'var(--reg)' }}>b_r</text> : null}
          <text x={bx + s.tN * cs + 3} y={ay + 9} fontSize="9" style={{ fill: 'var(--reg)' }}>c</text>
          {els}
        </svg>
      </div>
    </div>
  );
}

function TwoOrders() {
  const [s, setS] = useState<Shape>(BOOK_SHAPE);
  const [k, setK] = useState(0);
  const [playing, setPlaying] = useState(false);
  const inputs = useMemo(() => smallInputs(s), [s]);
  const t1 = useMemo(() => run(s, inputs, 'v1'), [s, inputs]);
  const t2 = useMemo(() => run(s, inputs, 'v2'), [s, inputs]);
  const total = fmas(s);
  const at = Math.min(k, total);
  const st1 = useMemo(() => stateAt(t1, at), [t1, at]);
  const st2 = useMemo(() => stateAt(t2, at), [t2, at]);
  const strip = s.tM * s.tN;
  useEffect(() => setK(0), [s]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setK((x) => x + 1), 180);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);
  const set = (part: Partial<Shape>) => setS({ ...s, ...part });
  const r1 = sharedReadsV1(s), r2 = sharedReadsV2(s);
  const same = at >= total && st1.C.every((v, j) => v === st2.C[j]);
  return (
    <Bench
      title={tr('Two orders of the same loops', 'Hai thứ tự của cùng các vòng lặp')}
      task={tr(
        "Change the thread's tile shape and read the counts; then step one thread through both orders side by side.",
        'Thay đổi hình dạng ô của luồng và đọc các con số; rồi cho một luồng chạy từng bước qua cả hai thứ tự cạnh nhau.',
      )}
      foot={
        <Translated
          vi={
            <>
              Các con số tính cho mỗi luồng, mỗi cặp ô, đúng như mã nguồn phát lệnh; các giá trị là số nguyên nhỏ để bạn có thể tự
              kiểm tra một tổng bằng tay. <TierLegend tiers={['reg', 'shared']} />
            </>
          }
        >
          <>
            Counts are per thread per tile pair, as the source code issues them; values are small integers so you can check a
            sum by hand. <TierLegend tiers={['reg', 'shared']} />
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        {(['tM', 'tN', 'bK'] as const).map((key) => (
          <label className="field" key={key}>
            {key} <input type="range" min={1} max={key === 'bK' ? 16 : 32} value={s[key]} aria-label={key} onChange={(e) => set({ [key]: +e.target.value })} />
            <b className="mono" style={{ minWidth: '1.6rem' }} data-val={key}>{s[key]}</b>
          </label>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span className="seg" role="group" aria-label={tr('Presets', 'Thiết lập sẵn')}>
          {PRESETS.map((p) => (
            <button key={p.label} aria-pressed={p.s.tM === s.tM && p.s.tN === s.tN && p.s.bK === s.bK} onClick={() => setS(p.s)}>{p.label}</button>
          ))}
        </span>
      </div>
      <table className="data" style={{ background: 'var(--page)', margin: '0 0 1rem' }} data-counts>
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}>{tr('per thread, per tile pair', 'mỗi luồng, mỗi cặp ô')}</th>
            <th className="num">{tr('Fig. 15.6 order', 'thứ tự Hình 15.6')}</th>
            <th className="num">{tr('Fig. 15.9 order', 'thứ tự Hình 15.9')}</th>
          </tr>
        </thead>
        <tbody>
          <tr><td style={{ paddingLeft: '0.6rem' }}>{tr('shared-memory reads', 'số lần đọc bộ nhớ chia sẻ')}</td><td className="num" data-v="r1">{r1}</td><td className="num" data-v="r2">{r2}</td></tr>
          <tr><td style={{ paddingLeft: '0.6rem' }}>{tr('multiply-adds', 'số phép nhân-cộng')}</td><td className="num">{total}</td><td className="num" data-v="fmas">{total}</td></tr>
          <tr><td style={{ paddingLeft: '0.6rem' }}>{tr('shared reads per multiply-add', 'số lần đọc chia sẻ mỗi phép nhân-cộng')}</td><td className="num">{readsPerFma(s, 'v1')}</td><td className="num" data-v="per">{+readsPerFma(s, 'v2').toFixed(5)}</td></tr>
          <tr><td style={{ paddingLeft: '0.6rem' }}>{tr('registers: accumulators (+ strips)', 'thanh ghi: biến tích lũy (+ các dải)')}</td><td className="num">{registersFor(s, 'v1')}</td><td className="num" data-v="regs">{registersFor(s, 'v2')}</td></tr>
        </tbody>
      </table>
      <p style={{ margin: '0 0 1rem' }} data-ratio>
        <Translated
          vi={
            <>
              2·{s.tM}·{s.tN}·{s.bK} = {r1} so với {s.bK}·({s.tM} + {s.tN}) = {r2}: ít hơn <b>{+(r1 / r2).toFixed(2)}×</b>. Mỗi
              giá trị <code>a_r</code> phục vụ {reuse(s).a} phép nhân-cộng và mỗi giá trị <code>b_r</code> phục vụ {reuse(s).b}.
            </>
          }
        >
          2·{s.tM}·{s.tN}·{s.bK} = {r1} against {s.bK}·({s.tM} + {s.tN}) = {r2}: <b>{+(r1 / r2).toFixed(2)}×</b> fewer. Each{' '}
          <code>a_r</code> value serves {reuse(s).a} multiply-adds and each <code>b_r</code> value {reuse(s).b}.
        </Translated>
      </p>

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setK(0); }} disabled={at === 0}>
          <Translated vi="Bắt đầu lại">Restart</Translated>
        </button>
        <button onClick={() => { setPlaying(false); setK(at - 1); }} disabled={at === 0}>
          <Translated vi="Lùi lại">Back</Translated>
        </button>
        <button className="primary" onClick={() => { setPlaying(false); setK(at + 1); }} disabled={at >= total}>
          <Translated vi="Bước">Step</Translated>
        </button>
        <button onClick={() => { setPlaying(false); setK(Math.min(total, (Math.floor(at / strip) + 1) * strip)); }} disabled={at >= total}>
          <Translated vi="Cập nhật hạng 1 kế tiếp">Next rank-1 update</Translated>
        </button>
        <button onClick={() => { setPlaying(false); setK(total); }} disabled={at >= total}>
          <Translated vi="Chạy đến cuối">Run to end</Translated>
        </button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Chạy')}</button>
        <span className="muted" aria-live="polite" data-step>
          <Translated vi={<>phép nhân-cộng {at} trên {total}</>}>multiply-add {at} of {total}</Translated>
        </span>
      </div>
      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }} data-narrate>
        {at === 0 ? (
          <Translated vi={<>Chưa có gì chạy. Cả hai thứ tự sẽ làm cùng {total} phép nhân-cộng; bấm Bước để làm một phép ở mỗi bên.</>}>
            <>Nothing has run. Both orders will do the same {total} multiply-adds; press Step to do one in each.</>
          </Translated>
        ) : (
          <>
            {(() => {
              const f1 = st1.step[st1.step.length - 1];
              const f2 = st2.step[st2.step.length - 1];
              const loads = st2.step.length - 1;
              return (
                <>
                  {f1?.kind === 'fma' ? (
                    <Translated vi={<>Hình 15.6: c[{f1.row}][{f1.col}] += a[{f1.row}][{f1.i}] · b[{f1.i}][{f1.col}], hai lần đọc từ bộ nhớ chia sẻ. </>}>
                      <>Fig. 15.6: c[{f1.row}][{f1.col}] += a[{f1.row}][{f1.i}] · b[{f1.i}][{f1.col}], two reads from shared memory. </>
                    </Translated>
                  ) : null}
                  {f2?.kind === 'fma' ? (
                    <Translated
                      vi={
                        <>
                          Hình 15.9{loads > 0 ? <>, bước i = {f2.i}: trước tiên {loads} giá trị dải được chép vào thanh ghi, rồi</> : ':'} c[{f2.row}][{f2.col}] += a_r[{f2.row}] · b_r[{f2.col}], không đọc bộ nhớ chia sẻ.
                        </>
                      }
                    >
                      <>
                        Fig. 15.9{loads > 0 ? <>, step i = {f2.i}: first {loads} strip values copied to registers, then</> : ':'} c[{f2.row}][{f2.col}] += a_r[{f2.row}] · b_r[{f2.col}], no shared read.
                      </>
                    </Translated>
                  ) : null}
                </>
              );
            })()}
          </>
        )}
        {same ? <><br /><span className="chip ok" data-same>
          <Translated vi={<>Cả hai thứ tự giữ cùng {s.tM * s.tN} giá trị.</>}>Both orders hold the same {s.tM * s.tN} values.</Translated>
        </span></> : null}
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(16rem, 1fr))' }}>
        <OrderPanel s={s} st={st1} order="v1" inputs={inputs} />
        <OrderPanel s={s} st={st2} order="v2" inputs={inputs} />
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const per = (tM: number, tN: number) => readsPerFma({ tM, tN, bK: 8 }, 'v2');

export const questionsS154: Record<string, Question> = {
  'ch15/15.4/q-shape': {
    kind: 'choice',
    id: 'ch15/15.4/q-shape',
    prompt: (
      <Translated
        vi={
          <>
            Một luồng giữ 64 biến tích lũy. Bạn có thể tạo hình ô của nó là 8 × 8, 16 × 4 hoặc 32 × 2 (tM × tN). Với các vòng lặp đã
            hoán đổi, hình dạng nào cần ít lần đọc bộ nhớ chia sẻ nhất cho mỗi phép nhân-cộng?
          </>
        }
      >
        <>
          A thread keeps 64 accumulators. You can shape its tile as 8 × 8, 16 × 4 or 32 × 2 (tM × tN). With the
          interchanged loops, which shape needs the fewest shared-memory reads per multiply-add?
        </>
      </Translated>
    ),
    options: [
      { label: '8 × 8', correct: true },
      {
        label: '16 × 4',
        why: <Translated vi={`(16 + 4) / 64 = ${per(16, 4)} lần đọc mỗi phép nhân-cộng, so với ${per(8, 8)}.`}>{`(16 + 4) / 64 = ${per(16, 4)} reads per multiply-add, against ${per(8, 8)}.`}</Translated>,
      },
      {
        label: '32 × 2',
        why: <Translated vi={`(32 + 2) / 64 = ${per(32, 2)}: gấp đôi so với hình vuông.`}>{`(32 + 2) / 64 = ${per(32, 2)}: twice as many as the square.`}</Translated>,
      },
      {
        label: <Translated vi={"Đều như nhau: mỗi cái 64 biến tích lũy."}>{'All the same: 64 accumulators each.'}</Translated>,
        why: <Translated vi={"Số phép nhân-cộng như nhau; các dải thì không. Độ dài của chúng là tM + tN."}>{'The multiply-adds are the same; the strips are not. Their length is tM + tN.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi bước một luồng đọc tM + tN giá trị dải và làm tM·tN phép nhân-cộng, nên nó đọc (tM + tN) / (tM·tN) lần cho mỗi phép
            nhân-cộng: {per(8, 8)}, {per(16, 4)} và {per(32, 2)}. Với số biến tích lũy cố định, tổng nhỏ nhất khi ô là hình vuông:
            mỗi giá trị dải được dùng lại số lần bằng độ dài của dải kia.
          </p>
        }
      >
        <p>
          Per step a thread reads tM + tN strip values and does tM·tN multiply-adds, so it reads (tM + tN) / (tM·tN) per
          multiply-add: {per(8, 8)}, {per(16, 4)} and {per(32, 2)}. For a fixed number of accumulators the sum is smallest
          when the tile is square: each strip value is reused as often as the other strip is long.
        </p>
      </Translated>
    ),
    source: 'our count from Fig. 15.9 (p. 359); notes/ch15.md §15.4 [inferred]',
    vi: { source: "phép đếm của khóa học này từ Hình 15.9 (p. 359); notes/ch15.md §15.4 [suy ra]" },
  },
  'ch15/15.4/q-reuse': {
    kind: 'numeric',
    id: 'ch15/15.4/q-reuse',
    prompt: (
      <Translated
        vi={
          <>
            tM = tN = bK = 8. Khi <code>a_r[row]</code> đã giữ giá trị của nó cho bước i, bao nhiêu phép nhân-cộng của bước đó dùng
            nó?
          </>
        }
      >
        <>
          tM = tN = bK = 8. Once <code>a_r[row]</code> holds its value for step i, how many multiply-adds of that step use
          it?
        </>
      </Translated>
    ),
    answer: reuse(BOOK_SHAPE).a,
    explain: (
      <Translated
        vi={
          <p>
            Mọi cột của mảng: <code>c[row][col] += a_r[row]*b_r[col]</code> với col = 0 đến 7, nên tN ={' '}
            {reuse(BOOK_SHAPE).a}. Theo thứ tự của Hình 15.6, mỗi lần dùng trong tám lần đó là một lần đọc bộ nhớ chia sẻ riêng.
          </p>
        }
      >
        <p>
          Every column of the patch: <code>c[row][col] += a_r[row]*b_r[col]</code> for col = 0 to 7, so tN ={' '}
          {reuse(BOOK_SHAPE).a}. In Fig. 15.6's order each of those eight uses was a separate read of shared memory.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.4, Fig. 15.9, p. 359',
  },
  'ch15/15.4/q-what-rises': {
    kind: 'choice',
    id: 'ch15/15.4/q-what-rises',
    prompt: (
      <Translated
        vi={
          <>
            Sách ghi công cho phân ô thanh ghi việc tăng cường độ tính toán (p. 364). Phát biểu nào nói chính xác mỗi bước đã mua
            được gì?
          </>
        }
      >
        <>
          The book credits register tiling with increasing arithmetic intensity (p. 364). Which statement says precisely
          what each step bought?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"FLOP trên mỗi byte toàn cục đến từ ô đầu ra 128 × 128 (mục 15.2); phân ô thanh ghi cắt số lần đọc chia sẻ mỗi phép nhân-cộng từ 2 xuống 0.25."}>{'Global FLOP per byte came from the 128 × 128 output tile (section 15.2); register tiling cut shared reads per multiply-add from 2 to 0.25.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Phân ô thanh ghi nâng FLOP trên mỗi byte toàn cục lên gấp tám, từ 32 lên 256, vì mỗi giá trị nạp từ bộ nhớ toàn cục giờ được dùng lại từ thanh ghi."}>{'Register tiling raised the FLOP per global byte eightfold, from 32 to 256, because each value loaded from global memory is now reused from registers.'}</Translated>,
        why: <Translated vi={"Lưu lượng toàn cục mỗi cặp ô không đổi: cùng các ô A và B được nạp vào bộ nhớ chia sẻ theo cả hai cách."}>{'Global traffic per tile pair is unchanged: the same A and B tiles are loaded into shared memory either way.'}</Translated>,
      },
      {
        label: <Translated vi={"Phân ô thanh ghi nâng cường độ tính toán vì 64 biến tích lũy không còn phải đi ra bộ nhớ toàn cục ở mỗi bước."}>{'Register tiling raised intensity because the 64 accumulators no longer travel to global memory at every step.'}</Translated>,
        why: <Translated vi={"Chúng chưa bao giờ phải đi: C_r đã nằm trong thanh ghi từ mục 15.3."}>{'They never did: C_r has been in registers since section 15.3.'}</Translated>,
      },
      {
        label: <Translated vi={"Cả hai là một hiệu ứng nhìn ở hai nơi, nên cách diễn đạt của sách là chính xác."}>{'Both are one effect seen in two places, so the book\'s wording is exact.'}</Translated>,
        why: <Translated vi={"Chúng tác động lên các bộ nhớ khác nhau: một lên lưu lượng DRAM mỗi FLOP, một lên lưu lượng bộ nhớ chia sẻ mỗi FLOP."}>{'They act on different memories: one on DRAM traffic per FLOP, the other on shared-memory traffic per FLOP.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Cường độ tính toán của mục 15.2, 0.5·m·n / (m + n), chỉ phụ thuộc vào ô đầu ra. Hoán đổi các vòng lặp không thay đổi gì
            đi qua ranh giới bộ nhớ toàn cục; nó thay đổi tần suất một luồng tới bộ nhớ chia sẻ: từ{' '}
            {readsPerFma(BOOK_SHAPE, 'v1')} lần đọc mỗi phép nhân-cộng xuống {readsPerFma(BOOK_SHAPE, 'v2')}. Văn bản của sách không
            tách bạch hai hiệu ứng này.
          </p>
        }
      >
        <p>
          Section 15.2's intensity, 0.5·m·n / (m + n), depends only on the output tile. Swapping the loops changes nothing
          that crosses the global-memory boundary; it changes how often a thread goes to shared memory: from{' '}
          {readsPerFma(BOOK_SHAPE, 'v1')} reads per multiply-add to {readsPerFma(BOOK_SHAPE, 'v2')}. The book's text does
          not separate the two effects.
        </p>
      </Translated>
    ),
    source: 'ch. 15 pp. 351, 359, 364; notes/ch15.md "Unclear in the PDF" item 12',
    vi: { source: "ch. 15 pp. 351, 359, 364; notes/ch15.md \"Unclear in the PDF\" mục 12" },
  },
  'ch15/15.4/q-cutlass-strips': {
    kind: 'choice',
    id: 'ch15/15.4/q-cutlass-strips',
    prompt: (
      <Translated
        vi={
          <>
            CUTLASS giữ hai bộ dải thanh ghi cho mỗi warp ("two fragments are allocated within registers", hai fragment được cấp
            phát trong thanh ghi) và luân phiên giữa chúng. Bộ thứ hai giúp làm được điều gì?
          </>
        }
      >
        <>
          CUTLASS keeps two sets of register strips per warp ("two fragments are allocated within registers") and
          alternates between them. What does the second set make possible?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Đọc các dải của bước i + 1 từ bộ nhớ chia sẻ trong khi các phép nhân-cộng của bước i vẫn đang dùng bộ hiện tại."}>{'Reading step i + 1\'s strips from shared memory while step i\'s multiply-adds still use the current set.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Gấp đôi số biến tích lũy mỗi luồng, để mỗi warp có thể sở hữu một ô đầu ra lớn hơn."}>{'Twice as many accumulators per thread, so each warp can own a larger tile of the output.'}</Translated>,
        why: <Translated vi={"Các fragment chứa dải đầu vào; các biến tích lũy là riêng."}>{'The fragments hold input strips; the accumulators are separate.'}</Translated>,
      },
      {
        label: <Translated vi={"Ít lần đọc bộ nhớ chia sẻ hơn mỗi phép nhân-cộng: mỗi dải đọc từ bộ nhớ chia sẻ giờ phục vụ hai bước i thay vì một."}>{'Fewer shared-memory reads per multiply-add: each strip read from shared memory now serves two steps of i instead of one.'}</Translated>,
        why: <Translated vi={"Mỗi giá trị dải vẫn được đọc một lần mỗi bước; chỉ thời điểm đọc là thay đổi."}>{'Each strip value is still read once per step; only when it is read changes.'}</Translated>,
      },
      {
        label: <Translated vi={"Các lần đọc dải không còn xung đột bank, vì hai bộ thanh ghi luân phiên nhau."}>{'Strip reads free of bank conflicts, since the two sets of registers alternate.'}</Translated>,
        why: <Translated vi={"Một lần đọc chạm vào bank nào phụ thuộc vào địa chỉ của nó, không phụ thuộc thanh ghi nào nhận nó."}>{'Which banks a read hits depends on its addresses, not on which registers receive it.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Với một bộ, các lần đọc của bước kế tiếp sẽ ghi đè lên thanh ghi mà bước hiện tại vẫn đang đọc: một xung đột ghi-sau-đọc
            (write-after-read), chính là phụ thuộc giả mà mục 15.8 loại bỏ cho bộ nhớ chia sẻ bằng hai buffer. CUTLASS mô tả đệm kép
            ở cả hai phạm vi, ô bộ nhớ chia sẻ và fragment thanh ghi, như cách của nó để chồng lấp truy cập bộ nhớ với tính toán bên
            trong một luồng.
          </p>
        }
      >
        <p>
          With one set, the next step's reads would overwrite registers the current step is still reading: a
          write-after-read conflict, the false dependence that section 15.8 removes for shared memory with two buffers.
          CUTLASS describes double buffering at both scopes, shared-memory tiles and register fragments, as its way to
          overlap memory accesses with computation inside a thread.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch15-facts.md A12 (CUTLASS, "Pipelining"); ch. 15 §15.8 for the dependence (p. 366)',
    vi: { source: "notes/today/ch15-facts.md A12 (CUTLASS, \"Pipelining\"); ch. 15 §15.8 cho phụ thuộc này (p. 366)" },
  },
};

// ---------------------------------------------------------------- the section
export function S154() {
  const r1 = sharedReadsV1(BOOK_SHAPE);
  const r2 = sharedReadsV2(BOOK_SHAPE);
  return (
    <>
      <Translated
        vi={
          <p>
            Phân ô bộ nhớ chia sẻ (shared memory) đã gỡ nút thắt bộ nhớ toàn cục (global memory) bằng cách nạp mỗi phần tử đầu vào
            vào bộ nhớ chia sẻ một lần và đọc nó ở đó nhiều lần: một luồng nạp, nhiều luồng dùng. Bộ nhớ chia sẻ nhanh hơn bộ nhớ
            toàn cục nhiều, nhưng vẫn chậm hơn một thanh ghi (register). Khi băng thông toàn cục đã thôi là giới hạn, thời gian đọc
            bộ nhớ chia sẻ là giới hạn kế tiếp.
          </p>
        }
      >
        <p>
          Shared-memory tiling fixed the global-memory bottleneck by loading each input element into shared memory once and
          reading it there many times: one thread loads it, many threads use it. Shared memory is much faster than global
          memory, but it is still slower than a register. Once global bandwidth has stopped being the limit, the time spent
          reading shared memory is the next one.
        </p>
      </Translated>
      <Source section="15.4" pages="357–358" />
      <Translated
        vi={
          <p>
            Và <code>mm</code> của mục 15.3 đọc nó rất nhiều. Một luồng dùng mỗi giá trị trong 8 hàng của nó ở{' '}
            <code>A_s</code> cho 8 đầu ra khác nhau, mỗi cột của mảng một đầu ra, và lần nào cũng nạp nó từ bộ nhớ chia sẻ. "Ô đầu
            vào mức luồng" chỉ là một cách mô tả những giá trị một luồng cần; chưa có gì từng đưa các giá trị đó tới nơi nào gần hơn
            các mảng của khối.
          </p>
        }
      >
        <p>
          And the <code>mm</code> of section 15.3 reads it a lot. A thread uses each value of its 8 rows of{' '}
          <code>A_s</code> for 8 different outputs, one per column of its patch, and fetches it from shared memory every
          time. The "thread-level input tile" was a way of describing which values a thread needs; nothing ever put those
          values anywhere nearer than the block's arrays.
        </p>
      </Translated>
      <Source pages="358" />
      <Predict
        id="ch15/15.4/p-ratio"
        prompt={
          <Translated
            vi={
              <>
                Giả sử mỗi luồng chép mỗi giá trị chia sẻ nó cần vào một thanh ghi một lần và dùng lại từ đó. Với tM = tN = bK = 8, một
                luồng sẽ phát ra ít hơn bao nhiêu lần số lần đọc bộ nhớ chia sẻ mỗi cặp ô?
              </>
            }
          >
            <>
              Suppose each thread copied every shared value it needs into a register once and reused it from there. With tM
              = tN = bK = 8, how many times fewer shared-memory reads would one thread issue per tile pair?
            </>
          </Translated>
        }
        unit={tr('× fewer', '× ít hơn')}
      >
        {(v) => (
          <p>
            <Translated
              vi={
                <>
                  <b>{r1 / r2} lần.</b> Ở thứ tự đầu tiên, mỗi phép nhân-cộng đọc hai giá trị: 2 · 8 · 8 · 8 = {r1} lần đọc cho{' '}
                  {fmas(BOOK_SHAPE)} phép nhân-cộng. Nếu mỗi giá trị chỉ đọc một lần, các lát của luồng chứa 8 · (8 + 8) = {r2} giá trị.
                </>
              }
            >
              <b>{r1 / r2} times.</b> In the first order every multiply-add reads two values: 2 · 8 · 8 · 8 = {r1} reads for{' '}
              {fmas(BOOK_SHAPE)} multiply-adds. Read once each, the thread's slices hold 8 · (8 + 8) = {r2} values.
            </Translated>{' '}
            {v === r1 / r2
              ? tr('You had it.', 'Bạn đã đúng.')
              : v === 64
                ? tr('64 would mean each value fed every output of the patch; a value of A feeds only the 8 outputs in its row.', '64 nghĩa là mỗi giá trị nuôi mọi đầu ra của mảng; một giá trị của A chỉ nuôi 8 đầu ra trong hàng của nó.')
                : tr('The explorer below shows the counts for any shape.', 'Trình khám phá bên dưới cho thấy các con số với mọi hình dạng.')}{' '}
            <Translated vi="Sách nêu ý tưởng, không nêu các con số này; chúng là của khóa học này.">
              The book states the idea, not these counts; they are ours.
            </Translated>
          </p>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Trình khám phá đếm cả hai thứ tự cho mọi ô của luồng, và trình chạy từng bước của nó cho một luồng chạy qua cả hai cùng
            lúc, mỗi bước một phép nhân-cộng, trên các số nguyên nhỏ.
          </p>
        }
      >
        <p>
          The explorer counts both orders for any thread tile, and its stepper runs one thread through both at once, one
          multiply-add per step, on small integers.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một luồng, một cặp ô, số lần đọc tính đúng như mã nguồn phát lệnh. Không đo thời gian, không độ trễ, không xung đột bank
              (mục 15.6), và không nói gì về việc trình biên dịch làm gì với các lần nạp.
            </p>
          }
        >
          <p>
            One thread, one tile pair, reads counted as the source issues them. No timing, no latency, no bank conflicts
            (section 15.6), and nothing about what the compiler does with the loads.
          </p>
        </Translated>
      </Simplification>
      <TwoOrders />

      <h2>
        <Translated vi="Đổi chỗ các vòng lặp">Swap the loops</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Cách sửa là lấy mỗi giá trị cần thiết từ bộ nhớ chia sẻ một lần, giữ nó trong một thanh ghi, và dùng nó cho mọi đầu ra
            mà nó nuôi. Trong mã, đó là một phép <b>hoán đổi vòng lặp</b> (loop interchange). Trong Hình 15.6, vòng lặp theo chiều
            trong i là trong cùng; hãy chuyển nó ra ngoài cùng. Với một giá trị của i, luồng khi đó cần một cột trong 8 hàng của nó ở
            A, được chép vào <code>a_r</code>, và một hàng trong 8 cột của nó ở B, được chép vào <code>b_r</code>. Rồi mỗi đầu ra
            của mảng nhận một tích: <code>c[row][col] += a_r[row]*b_r[col]</code>. Đó là tích ngoài (outer product) của hai vector 8
            phần tử, một cập nhật hạng 1 của ô 8 × 8, và 8 lần như vậy liên tiếp tạo thành phần đóng góp của cặp ô.
          </p>
        }
      >
        <p>
          The fix is to take each needed value from shared memory once, hold it in a register, and use it for every output
          it feeds. In code that is a <b>loop interchange</b>. In Fig. 15.6 the loop over the inner dimension i is the
          innermost; move it outermost. For one value of i the thread then needs one column of its 8 rows of A, which it
          copies into <code>a_r</code>, and one row of its 8 columns of B, copied into <code>b_r</code>. Then every output
          of the patch takes one product: <code>c[row][col] += a_r[row]*b_r[col]</code>. That is an outer product of two
          8-element vectors, a rank-1 update of the 8 × 8 tile, and 8 of them in a row make the tile pair's contribution.
        </p>
      </Translated>
      <Source section="15.4" pages="358–359" figs={tr('Figs. 15.8, 15.9', 'Hình 15.8, 15.9')} />
      <Translated
        vi={
          <p>
            Giờ sách gọi chính các dải là ô đầu vào mức luồng. Giữ chúng mỏng như vậy là điều cho phép chúng vừa trong thanh ghi: 8 +
            8 giá trị bên cạnh 64 biến tích lũy. Và vì cùng lý do như ở mục 15.3, cả năm vòng lặp của <code>mm</code> mới đều được
            trải hết, để <code>a_r</code>, <code>b_r</code> và mảng chỉ được đánh chỉ số bằng hằng số và trình biên dịch có thể giữ
            cả ba trong thanh ghi.
          </p>
        }
      >
        <p>
          The book now calls the strips themselves the thread-level input tiles. Keeping them that thin is what lets them
          fit in registers: 8 + 8 values beside the 64 accumulators. And for the same reason as in section 15.3, all five
          loops of the new <code>mm</code> are fully unrolled, so that <code>a_r</code>, <code>b_r</code> and the patch
          are indexed only by constants and the compiler can keep all three in registers.
        </p>
      </Translated>
      <Source pages="359" />
      <Translated
        vi={
          <p>
            Hoạt hình theo một luồng qua một cặp ô: các lần đọc của thứ tự đầu tiên, việc đổi chỗ các vòng lặp, các tích ngoài lấp
            đầy mảng của nó, và cách cùng ý tưởng đó lặp lại ở một cấp cao hơn.
          </p>
        }
      >
        <p>
          The animation follows one thread through one tile pair: the reads of the first order, the swap of the loops,
          the outer products that fill its patch, and how the same idea repeats one level up.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Hoạt hình được vẽ từ cùng mô hình với trình khám phá ở trên: một luồng, một cặp ô, số nguyên nhỏ, số lần đọc tính đúng
              như mã nguồn phát lệnh. Không đo thời gian và không xung đột bank.
            </p>
          }
        >
          <p>
            The animation is drawn from the same model as the explorer above: one thread, one tile pair, small integers,
            reads counted as the source issues them. No timing and no bank conflicts.
          </p>
        </Translated>
      </Simplification>
      <OuterProductScene />
      <TwoMm />
      <Translated
        vi={
          <p>
            Hai thứ tự cộng cùng các tích vào mỗi <code>c[row][col]</code>, và theo cùng thứ tự của i, nên kết quả khớp chính xác.
            Điều thay đổi là các toán hạng đến từ đâu.
          </p>
        }
      >
        <p>
          The two orders add the same products into each <code>c[row][col]</code>, and in the same order of i, so the
          results agree exactly. What changes is where the operands come from.
        </p>
      </Translated>
      <Ask q={questionsS154['ch15/15.4/q-reuse']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS154['ch15/15.4/q-shape']} />

      <h2>
        <Translated vi="Nó nâng cái gì, và không nâng cái gì">What it raises, and what it does not</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách ghi công cho phân ô thanh ghi việc nâng cường độ tính toán và giải tỏa một nút thắt thực thi lệnh. Hãy chính xác
            về cường độ nào. Tỉ số FLOP trên byte nạp từ bộ nhớ toàn cục đã được ấn định ở mục 15.2 bởi kích thước của ô đầu ra, và
            phép hoán đổi vòng lặp để nó nguyên chỗ cũ: cùng các ô A và B đi vào bộ nhớ chia sẻ. Điều giảm xuống là lưu lượng ở một
            cấp thấp hơn, từ {r1} lần đọc chia sẻ mỗi cặp ô xuống {r2}, từ {readsPerFma(BOOK_SHAPE, 'v1')} mỗi phép nhân-cộng xuống{' '}
            {readsPerFma(BOOK_SHAPE, 'v2')}.
          </p>
        }
      >
        <p>
          The book credits register tiling with a higher arithmetic intensity and with relieving an instruction-execution
          bottleneck. Be precise about which intensity. The ratio of FLOP to bytes loaded from global memory was set in
          section 15.2 by the size of the output tile, and the loop interchange leaves it exactly where it was: the same A
          and B tiles cross into shared memory. What falls is the traffic one level down, from {r1} shared reads per tile
          pair to {r2}, from {readsPerFma(BOOK_SHAPE, 'v1')} per multiply-add to {readsPerFma(BOOK_SHAPE, 'v2')}.
        </p>
      </Translated>
      <Source pages="359, 364" />
      <Ask q={questionsS154['ch15/15.4/q-what-rises']} />

      <h2>
        <Translated vi="Phân ô ở mọi cấp">Tiling at every level</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Phân ô bộ nhớ chia sẻ và phân ô thanh ghi là một ý tưởng áp dụng hai lần. Ở bất cứ đâu hai bộ nhớ khác nhau về độ trễ và
            thông lượng, hãy đưa phần dữ liệu được dùng lại vào bộ nhớ nhanh hơn và dùng lại nó ở đó. Sách gọi sự kết hợp ở đây là
            kỹ thuật <b>phân cấp</b> (hierarchical): ô mức khối trong bộ nhớ chia sẻ, ô mức luồng trong thanh ghi. Với ma trận quá
            lớn cho một GPU, cùng bước đi đó tiếp tục lên trên, tới ô cho mỗi GPU và mỗi nút của một cụm máy.
          </p>
        }
      >
        <p>
          Shared-memory tiling and register tiling are one idea applied twice. Wherever two memories differ in latency and
          throughput, bring the reused part of the data into the faster one and reuse it there. The book calls the
          combination here the <b>hierarchical</b> technique: block-level tiles in shared memory, thread-level tiles in
          registers. For matrices too big for one GPU the same move continues upwards, to tiles per GPU and per node of a
          cluster.
        </p>
      </Translated>
      <Source section="15.4" pages="359–360" />
      <Translated
        vi={
          <p>
            Đó cũng là cách thư viện template của chính NVIDIA mô tả GEMM mức luồng của nó: mỗi luồng phát một chuỗi lệnh toán độc
            lập và "computes an accumulated outer product" (tính một tích ngoài được cộng dồn). Các lệnh tensor core của mục 15.9 đi
            thêm một bước và làm một phép nhân ma trận nhỏ trên cả ô trong một lệnh duy nhất.
          </p>
        }
      >
        <p>
          That is also how NVIDIA's own template library describes its thread-level GEMM: each thread issues a sequence of
          independent math instructions and "computes an accumulated outer product". The tensor-core instructions of section
          15.9 go one step further and do a small matrix multiplication of whole tiles in a single instruction.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch15-facts.md A12)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              CUTLASS, "Efficient GEMM in CUDA": các luồng "cannot access each other's registers, so we choose an organization that
              enables reuse of values held in registers" (không thể truy cập thanh ghi của nhau, nên chúng tôi chọn một cách tổ chức
              cho phép dùng lại các giá trị giữ trong thanh ghi); "a 2D tiled structure within a thread" (một cấu trúc phân ô 2D bên
              trong một luồng).
            </p>
          }
        >
          <p>
            CUTLASS, "Efficient GEMM in CUDA": threads "cannot access each other's registers, so we choose an organization
            that enables reuse of values held in registers"; "a 2D tiled structure within a thread".
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            CUTLASS đi thêm một bước so với Hình 15.9: nó giữ hai bộ dải thanh ghi và luân phiên giữa chúng, giống như mục 15.8 luân
            phiên giữa hai bộ ô bộ nhớ chia sẻ.
          </p>
        }
      >
        <p>
          CUTLASS goes one step past Fig. 15.9: it keeps two sets of register strips and alternates between them, the way
          section 15.8 alternates between two sets of shared-memory tiles.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              CUTLASS dùng đệm kép ở hai phạm vi: ô bộ nhớ chia sẻ, và "two fragments are allocated within registers" (hai fragment
              được cấp phát trong thanh ghi). Hình 15.9 của sách có một bộ dải.
            </p>
          }
        >
          <p>
            CUTLASS double-buffers at two scopes: shared-memory tiles, and "two fragments are allocated within registers".
            The book's Fig. 15.9 has one set of strips.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS154['ch15/15.4/q-cutlass-strips']} />

      <h2>
        <Translated vi="Đến lượt bạn">Your turn</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lab 15.2 yêu cầu bạn thực hiện phép hoán đổi trong kernel lab 15.1 của mình. Trước khi đo hiệu năng, hãy để ý một điều
            mà phép đếm ở trên bỏ qua. Trong cách bố trí luồng này, các lần đọc dải của một warp vẫn va nhau trong bộ nhớ chia sẻ:
            trên 32 bank, dải A là xung đột bank (bank conflict) 2 chiều và dải B là 4 chiều. Ít hơn tám lần số lần đọc không nhất
            thiết nghĩa là ít hơn tám lần thời gian đọc. Hãy chốt dự đoán của bạn trên <a href={href('ch15', 'labs')}>trang lab</a>;
            mục 15.5 và 15.6 xử lý các xung đột.
          </p>
        }
      >
        <p>
          Lab 15.2 has you make the interchange in your lab 15.1 kernel. Before you benchmark it, note one thing the count
          above ignores. In this thread layout a warp's strip reads still collide in shared memory: on 32 banks, the A strip
          is a 2-way bank conflict and the B strip a 4-way one. Eight times fewer reads need not mean eight times less time
          spent reading. Commit your prediction on the <a href={href('ch15', 'labs')}>labs page</a>; sections 15.5 and 15.6
          deal with the conflicts.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Mức độ xung đột là phép tính của khóa học này với quy tắc bank được NVIDIA công bố (notes/today/ch15-facts.md E1, A2).
              Sách chỉ phân tích bank cho cách bố trí của mục 15.5.
            </p>
          }
        >
          <p>
            The conflict degrees are our arithmetic with NVIDIA's documented bank rule (notes/today/ch15-facts.md E1, A2).
            The book analyses banks only for the layout of section 15.5.
          </p>
        </Translated>
      </Beyond>
    </>
  );
}
