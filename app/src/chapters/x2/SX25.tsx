import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import {
  ACT_DIST,
  OUTLIER_J,
  PATH_LABEL,
  ROW_LENGTHS,
  bitwiseAgreement,
  exactDequantDot,
  exactFloatDot,
  hex,
  histogram,
  makeCase,
  rowErrors,
  sci,
  show,
  tracePath,
  type ActDist,
  type Counts,
  type Path,
  type PathRun,
  type QStep,
} from '../../sim/qdot';
import { IntegerPathScene } from './IntegerPathScene';

// Section X2.5: one block's dot product along the float path the Metal kernels use and the
// integer path llama.cpp's CUDA and HIP decode uses. Everything shown is computed by
// src/sim/qdot.ts and pinned in qdot.test.ts. Module X2 is ours: no book pages.
const FACTS = 'notes/today/x2-facts.md';
const CHECKED = `2026-11-20 (${FACTS}`;
const loaded = (t: string | null) => (t && !t.startsWith('// could not load') ? t : null);
/** The dot4_i8 helper, cut from lab X2.3's tested solution: from its comment to its closing brace. */
function helperOf(src: string | null): string | null {
  if (!src) return null;
  const i = src.indexOf('// Four signed byte products');
  const end = i < 0 ? -1 : src.indexOf('\n}\n', i);
  return end < 0 ? null : src.slice(i, end + 2);
}

// ---------------------------------------------------------------- the stepper
const COUNT_ROWS: { k: keyof Counts; label: string }[] = [
  { k: 'mul', get label() { return tr('float multiplies', 'phép nhân số thực'); } },
  { k: 'add', get label() { return tr('float adds', 'phép cộng số thực'); } },
  { k: 'cmp', get label() { return tr('float compares', 'phép so sánh số thực'); } },
  { k: 'int', get label() { return tr('integer operations', 'phép toán số nguyên'); } },
  { k: 'conv', get label() { return tr('conversions', 'phép chuyển đổi'); } },
];
const list = (v: number[], d = 4) => v.map((x) => show(x, d)).join(', ');

function describe(s: QStep, run: PathRun, d: number): string {
  const lanes = s.lanes ?? [];
  const js = lanes.map((l) => l.j).join(', ');
  switch (s.kind) {
    case 'group':
      return tr(
        `Weights ${js}: codes ${lanes.map((l) => l.code).join(', ')}, so (q − 8)·d = ${list(lanes.map((l) => l.wv))} (exact in float32). Four multiply-adds with x = ${list(lanes.map((l) => l.xv))}: acc = ${show(s.acc, 7)}.`,
        `Trọng số ${js}: mã ${lanes.map((l) => l.code).join(', ')}, nên (q − 8)·d = ${list(lanes.map((l) => l.wv))} (chính xác ở float32). Bốn phép nhân-cộng với x = ${list(lanes.map((l) => l.xv))}: acc = ${show(s.acc, 7)}.`,
      );
    case 'prepare':
      return tr(
        `Once per token for this block, shared by every row: Σx = ${show(s.vals!.sumx, 7)}, and each activation divided by 1, 256, 16 or 4096 to match the mask that will meet it (exact: powers of two).`,
        `Một lần cho mỗi token với khối này, dùng chung cho mọi hàng: Σx = ${show(s.vals!.sumx, 7)}, và mỗi activation được chia cho 1, 256, 16 hoặc 4096 để khớp với mask sẽ gặp nó (chính xác: lũy thừa của hai).`,
      );
    case 'word':
      return tr(
        `16-bit word ${hex(s.word!, 4)} masked four ways (0x000F, 0x0F00, 0x00F0, 0xF000) gives weights ${js} as ${lanes.map((l) => l.wv).join(', ')}: codes ${lanes.map((l) => l.code).join(', ')} times 1, 256, 16, 4096, never shifted. Times the pre-divided activations, each product is exactly code × x: ${list(lanes.map((l) => l.prod))}. Four accumulators: ${list(s.accs!, 5)}.`,
        `Word 16 bit ${hex(s.word!, 4)} được mask theo bốn cách (0x000F, 0x0F00, 0x00F0, 0xF000) cho ra trọng số ${js} là ${lanes.map((l) => l.wv).join(', ')}: mã ${lanes.map((l) => l.code).join(', ')} nhân 1, 256, 16, 4096, không hề dịch bit. Nhân với các activation đã chia trước, mỗi tích đúng bằng mã × x: ${list(lanes.map((l) => l.prod))}. Bốn bộ tích lũy: ${list(s.accs!, 5)}.`,
      );
    case 'fold':
      return tr(
        `The offset, once per block: −8 · Σx = ${show(s.vals!.fold, 7)}, plus the four accumulators = ${show(s.acc, 7)}. No weight ever had 8 subtracted from it.`,
        `Độ dời, một lần cho mỗi khối: −8 · Σx = ${show(s.vals!.fold, 7)}, cộng bốn bộ tích lũy = ${show(s.acc, 7)}. Không trọng số nào từng bị trừ đi 8.`,
      );
    case 'rescale':
      return tr(
        `Times the block's scale d = ${show(d, 6)}: the block's result is ${show(s.acc, 8)}.`,
        `Nhân với hệ số tỉ lệ của khối d = ${show(d, 6)}: kết quả của khối là ${show(s.acc, 8)}.`,
      );
    case 'quantise-max':
      return tr(
        `Once per token for this block: a 32-wide max finds amax = ${show(s.vals!.amax, 6)}; d8 = amax / 127 = ${show(s.vals!.dFloat, 6)}, stored as FP16 ${show(s.vals!.d, 6)}.`,
        `Một lần cho mỗi token với khối này: một phép max rộng 32 tìm ra amax = ${show(s.vals!.amax, 6)}; d8 = amax / 127 = ${show(s.vals!.dFloat, 6)}, lưu dưới dạng FP16 là ${show(s.vals!.d, 6)}.`,
      );
    case 'quantise-round': {
      const q = run.q8!;
      return tr(
        `Each x / d8 rounded to a signed byte: ${q.qs.slice(0, 8).join(', ')}, … Their sum is ${s.vals!.sum}; s = d8 · ${s.vals!.sum} = ${show(s.vals!.s, 6)} in FP16.`,
        `Mỗi x / d8 được làm tròn thành một byte có dấu: ${q.qs.slice(0, 8).join(', ')}, … Tổng của chúng là ${s.vals!.sum}; s = d8 · ${s.vals!.sum} = ${show(s.vals!.s, 6)} ở FP16.`,
      );
    }
    case 'dp4a':
      return tr(
        `Word ${hex(s.word!)}${s.shift ? ' shifted right by 4' : ''}, masked with 0x0F0F0F0F: ${hex(s.masked!)}, the unsigned codes ${lanes.map((l) => l.wv).join(', ')} of weights ${js}. dp4a with the bytes ${lanes.map((l) => l.xv).join(', ')} adds ${lanes.reduce((t, l) => t + l.prod, 0)}: sumi = ${s.acc}.`,
        `Word ${hex(s.word!)}${s.shift ? ' dịch phải 4 bit' : ''}, mask với 0x0F0F0F0F: ${hex(s.masked!)}, các mã không dấu ${lanes.map((l) => l.wv).join(', ')} của trọng số ${js}. dp4a với các byte ${lanes.map((l) => l.xv).join(', ')} cộng thêm ${lanes.reduce((t, l) => t + l.prod, 0)}: sumi = ${s.acc}.`,
      );
    case 'correct':
      return tr(
        `The correction, once per block: sumi · d8 − 8 · s = ${s.vals!.sumi} · ${show(s.vals!.d8, 6)} − 8 · ${show(s.vals!.s, 6)} = ${show(s.vals!.scaled, 7)} − ${show(s.vals!.off, 7)} = ${show(s.acc, 7)}.`,
        `Bước hiệu chỉnh, một lần cho mỗi khối: sumi · d8 − 8 · s = ${s.vals!.sumi} · ${show(s.vals!.d8, 6)} − 8 · ${show(s.vals!.s, 6)} = ${show(s.vals!.scaled, 7)} − ${show(s.vals!.off, 7)} = ${show(s.acc, 7)}.`,
      );
  }
}

function ThreePaths() {
  const [path, setPath] = useState<Path>('A');
  const [outlier, setOutlier] = useState(false);
  const [n, setN] = useState(0);
  const c = useMemo(() => makeCase(outlier), [outlier]);
  const runs = useMemo(() => ({ A: tracePath('A', c), B: tracePath('B', c), C: tracePath('C', c) }), [c]);
  const exact = useMemo(() => exactDequantDot(c.block, c.x), [c]);
  const exactW = useMemo(() => exactFloatDot(c.w, c.x), [c]);
  const run = runs[path];
  const total = run.steps.length;
  const at = Math.min(n, total);
  const cur = at > 0 ? run.steps[at - 1] : undefined;
  const sofar = run.steps.slice(0, at);
  const sum = (perRow: boolean, k: keyof Counts) => sofar.filter((s) => s.perRow === perRow).reduce((t, s) => t + (s.counts[k] ?? 0), 0);
  const choose = (p: Path) => {
    setPath(p);
    setN(0);
  };
  const done = at >= total;
  return (
    <Bench
      title={tr('One block, three paths', 'Một khối, ba đường')}
      task={tr(
        'Step through path A, then B, then C. Then tick the large activation and run C again.',
        'Chạy từng bước đường A, rồi B, rồi C. Sau đó đánh dấu activation lớn và chạy lại C.',
      )}
      foot={
        <Translated
          vi={
            <>
              Một khối Q4_0 gồm 32 trọng số (lượng tử hóa theo quy tắc tham chiếu của llama.cpp) và 32 activation số thực,
              sinh từ seed cố định. Số học số thực là float32, làm tròn ở mọi bước mà mã nguồn C làm tròn. Số đếm là số phép
              toán ở mức mã nguồn cho khối này: theo mỗi hàng, và theo mỗi token (dùng chung cho mọi hàng đầu ra đọc cùng các
              activation).
            </>
          }
        >
        <>
          One Q4_0 block of 32 weights (quantised by llama.cpp's reference rule) and 32 float activations, seeded.
          Float arithmetic is float32, rounded at every step the C source rounds. Counts are source-level operations for
          this block: per row, and per token (shared by every output row that reads the same activations).
        </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Path', 'Đường')}>
          {(['A', 'B', 'C'] as Path[]).map((p) => (
            <button key={p} aria-pressed={path === p} onClick={() => choose(p)}>{tr(PATH_LABEL[p], ({ A: 'A. giải lượng tử hóa, rồi nhân', B: 'B. số thực, độ dời được gộp ra ngoài', C: 'C. số nguyên, activation về int8' } as Record<Path, string>)[p])}</button>
          ))}
        </span>
        <label className="check">
          <input type="checkbox" checked={outlier} onChange={(e) => { setOutlier(e.target.checked); setN(0); }} /> <Translated vi={<>activation lớn ở kênh {OUTLIER_J}</>}>large activation in channel {OUTLIER_J}</Translated>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setN(0)} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => setN(at - 1)} disabled={at === 0}><Translated vi="Lùi lại">Back</Translated></button>
        <button className="primary" onClick={() => setN(at + 1)} disabled={done}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => setN(total)} disabled={done}><Translated vi="Chạy đến cuối">Run to the end</Translated></button>
        <span className="muted" aria-live="polite" data-at={at}><Translated vi={<>bước {at} trên {total}</>}>step {at} of {total}</Translated></span>
      </div>
      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }} data-narrate="">
        {cur ? describe(cur, run, c.block.d) : (
          <Translated
            vi={
              <>
                Chưa có gì chạy. Hệ số tỉ lệ của khối là d = {show(c.block.d, 6)} (FP16) và các mã của nó chạy từ 0 đến 15.
                {path === 'A' ? ' Đường A biến mọi mã thành một trọng số số thực trước.' : path === 'B' ? ' Đường B không bao giờ trừ 8 khỏi một mã.' : ' Đường C trước hết biến các activation thành byte.'} Hãy nhấn Bước.
              </>
            }
          >
          <>
            Nothing has run yet. The block's scale is d = {show(c.block.d, 6)} (FP16) and its codes run from 0 to 15.
            {path === 'A' ? ' Path A turns every code into a float weight first.' : path === 'B' ? ' Path B never subtracts 8 from a code.' : ' Path C first turns the activations into bytes.'} Press Step.
          </>
          </Translated>
        )}
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)' }}>
        <div>
          {cur?.lanes ? (
            <div style={{ overflowX: 'auto' }}>
            <table className="data" style={{ margin: 0 }} data-lanes="">
              <thead>
                <tr>
                  <th><Translated vi="trọng số">weight</Translated></th>
                  <th className="num"><Translated vi="mã">code</Translated></th>
                  <th className="num">{path === 'A' ? '(q − 8)·d' : path === 'B' ? tr('masked', 'đã mask') : tr('nibble byte', 'byte nibble')}</th>
                  <th className="num">{path === 'A' ? 'x' : path === 'B' ? tr('x pre-divided', 'x đã chia trước') : tr('x as int8', 'x dạng int8')}</th>
                  <th className="num"><Translated vi="tích">product</Translated></th>
                </tr>
              </thead>
              <tbody>
                {cur.lanes.map((l) => (
                  <tr key={l.j}>
                    <td>{l.j}{outlier && l.j === OUTLIER_J ? tr(' (large)', ' (lớn)') : ''}</td>
                    <td className="num t-global">{l.code}</td>
                    <td className="num">{show(l.wv, 6)}</td>
                    <td className="num">{show(l.xv, 6)}</td>
                    <td className="num">{show(l.prod, 6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          ) : (
            <p className="muted" style={{ margin: 0 }}>{cur ? tr('This step works on the whole block, not on four weights.', 'Bước này làm việc trên cả khối, không phải trên bốn trọng số.') : ''}</p>
          )}
          <p style={{ margin: '0.8rem 0 0' }}>
            <Translated vi="giá trị đang chạy:">running value:</Translated> <b className="t-reg" data-acc="">{cur ? show(cur.acc, 8) : '0'}</b>{' '}
            <span className="muted">{path === 'C' && cur?.kind === 'dp4a' ? tr('(int32, exact)', '(int32, chính xác)') : '(float32)'}</span>
          </p>
        </div>
        <table className="data" style={{ margin: 0 }} data-counts="">
          <thead>
            <tr><th><Translated vi="đến lúc này">so far</Translated></th><th className="num"><Translated vi="mỗi hàng">per row</Translated></th><th className="num"><Translated vi="mỗi token">per token</Translated></th></tr>
          </thead>
          <tbody>
            {COUNT_ROWS.map((r) => (
              <tr key={r.k}>
                <td>{r.label}</td>
                <td className="num" data-count={r.k}>{sum(true, r.k)}</td>
                <td className="num">{sum(false, r.k)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {done ? (
        <table className="data" style={{ margin: '1rem 0 0' }} data-results="">
          <thead>
            <tr><th><Translated vi="đường">path</Translated></th><th className="num"><Translated vi="kết quả khối">block result</Translated></th><th className="num"><Translated vi="trừ giá trị chính xác">minus exact</Translated></th><th className="num"><Translated vi="phép nhân số thực mỗi hàng">float multiplies per row</Translated></th></tr>
          </thead>
          <tbody>
            {(['A', 'B', 'C'] as Path[]).map((p) => (
              <tr key={p} style={p === path ? { fontWeight: 620 } : undefined} data-path={p} data-result={String(runs[p].result)}>
                <td>{tr(PATH_LABEL[p], ({ A: 'A. giải lượng tử hóa, rồi nhân', B: 'B. số thực, độ dời được gộp ra ngoài', C: 'C. số nguyên, activation về int8' } as Record<Path, string>)[p])}</td>
                <td className="num">{show(runs[p].result, 9)}</td>
                <td className="num">{runs[p].result === exact ? '0' : sci(runs[p].result - exact)}</td>
                <td className="num">{runs[p].perRow.mul}</td>
              </tr>
            ))}
            <tr>
              <td><Translated vi="chính xác, trọng số đã giải lượng tử hóa">exact, dequantised weights</Translated></td>
              <td className="num" data-exact={String(exact)}>{show(exact, 9)}</td>
              <td className="num">0</td>
              <td />
            </tr>
            <tr>
              <td className="muted"><Translated vi="chính xác, trọng số float32 trước khi lượng tử hóa">exact, the float32 weights before quantising</Translated></td>
              <td className="num muted">{show(exactW, 9)}</td>
              <td className="num muted">{sci(exactW - exact)}</td>
              <td />
            </tr>
          </tbody>
        </table>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the explorer
function ActivationCost() {
  const [n, setN] = useState(4096);
  const [dist, setDist] = useState<ActDist>('normal');
  const sel = useMemo(() => rowErrors(n, dist), [n, dist]);
  const other = useMemo(() => rowErrors(n, dist === 'normal' ? 'outliers' : 'normal'), [n, dist]);
  const LO = -7;
  const HI = -1;
  const bins = histogram(sel.actErr, LO, HI, 4);
  const obins = histogram(other.actErr, LO, HI, 4);
  const max = Math.max(...bins.map((b) => b.count), ...obins.map((b) => b.count), 1);
  const gw = 620;
  const gh = 150;
  const bw = gw / bins.length;
  const xl = (e: number) => 40 + ((Math.log10(e) - LO) / (HI - LO)) * gw;
  const T = 34; // top of the bars
  return (
    <Bench
      title={tr('What activation quantisation costs', 'Lượng tử hóa activation tốn gì')}
      task={tr(
        'Compare the two distributions at each row length. Where does the error of a large channel go?',
        'So sánh hai phân phối ở mỗi độ dài hàng. Sai số của một kênh lớn đi về đâu?',
      )}
      foot={
        <Translated
          vi={
            <>
              200 hàng sinh từ seed: trọng số phân phối chuẩn tắc lượng tử hóa về Q4_0, activation theo lựa chọn. Sai số tương
              đối là |C − B| chia cho tổng |w·x| trên hàng (với trọng số đã giải lượng tử hóa). Các kênh lớn là cấu trúc của
              khóa học này: một kênh trong mỗi 128, độ lớn khoảng 20 so với một phân phối chuẩn tắc. Sai số dưới 1e−7 được
              đếm vào cột đầu tiên.
            </>
          }
        >
        <>
          200 seeded rows: standard-normal weights quantised to Q4_0, activations as chosen. Relative error is |C − B|
          divided by the sum of |w·x| over the row (the dequantised weights). The large channels are ours: one in every 128,
          of magnitude about 20 against a standard normal. Errors below 1e−7 are counted in the first bar.
        </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Độ dài hàng">Row length</Translated>
          <select aria-label={tr('Row length', 'Độ dài hàng')} value={n} onChange={(e) => setN(+e.target.value)}>
            {ROW_LENGTHS.map((v) => <option key={v} value={v}>{v.toLocaleString('en-US')}</option>)}
          </select>
        </label>
        <span className="seg" role="group" aria-label={tr('Activations', 'Activation')}>
          {ACT_DIST.map((d) => (
            <button key={d.id} aria-pressed={dist === d.id} onClick={() => setDist(d.id)}>{tr(d.label, d.id === 'normal' ? 'chuẩn tắc' : 'chuẩn tắc, với vài kênh lớn')}</button>
          ))}
        </span>
      </div>
      <svg viewBox="0 0 700 226" role="img" aria-label={tr('Histogram of the relative error of path C against path B', 'Histogram sai số tương đối của đường C so với đường B')} style={{ width: '100%', maxWidth: 700, display: 'block', minWidth: 540 }}>
        {obins.map((b, i) => (
          <rect key={'o' + i} x={40 + i * bw + 1} y={T + gh - (b.count / max) * gh} width={bw - 2} height={(b.count / max) * gh}
            fill="none" stroke="var(--ink-3)" strokeDasharray="3 2" />
        ))}
        {bins.map((b, i) => (
          <rect key={i} x={40 + i * bw + 3} y={T + gh - (b.count / max) * gh} width={bw - 6} height={(b.count / max) * gh} fill="var(--ink-2)" data-bin={i} data-count={b.count} />
        ))}
        <line x1={40} x2={40 + gw} y1={T + gh} y2={T + gh} stroke="var(--rule)" />
        {Array.from({ length: HI - LO + 1 }, (_, k) => LO + k).map((e) => (
          <text key={e} x={xl(10 ** e)} y={T + gh + 16} fontSize="10.5" textAnchor="middle">1e{e}</text>
        ))}
        <line x1={xl(sel.median)} x2={xl(sel.median)} y1={T - 4} y2={T + gh} stroke="var(--ink)" strokeWidth="2" />
        <text x={xl(sel.median)} y={12} fontSize="10.5" textAnchor="middle"><Translated vi="trung vị">median</Translated></text>
        <line x1={xl(sel.weightMedian)} x2={xl(sel.weightMedian)} y1={T - 4} y2={T + gh} stroke="var(--global)" strokeWidth="2" strokeDasharray="5 3" />
        <text x={xl(sel.weightMedian)} y={26} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--global)' }}><Translated vi="trọng số về 4 bit (trung vị)">weights to 4 bits (median)</Translated></text>
        <text x={40} y={222} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>đặc: {dist === 'normal' ? 'chuẩn tắc' : 'chuẩn tắc, với vài kênh lớn'}; viền nét đứt: {dist === 'normal' ? 'chuẩn tắc, với vài kênh lớn' : 'chuẩn tắc'}</>}>solid: {ACT_DIST.find((d) => d.id === dist)!.label}; dashed outline: {ACT_DIST.find((d) => d.id !== dist)!.label}</Translated></text>
      </svg>
      <p style={{ margin: '0.6rem 0 0' }} data-summary="" data-median={String(sel.median)} data-p95={String(sel.p95)}>
        <Translated
          vi={
            <>
              Đường C so với đường B: trung vị <b>{sci(sel.median)}</b>, phân vị thứ 95 <b>{sci(sel.p95)}</b>{' '}
              (phân phối còn lại: trung vị {sci(other.median)}). Cái giá mà việc lượng tử hóa trọng số về Q4_0 đã gây ra, so
              với trọng số float32: trung vị <b className="t-global">{sci(sel.weightMedian)}</b>.
            </>
          }
        >
        Path C against path B: median <b>{sci(sel.median)}</b>, 95th percentile <b>{sci(sel.p95)}</b>{' '}
        (the other distribution: median {sci(other.median)}). What quantising the weights to Q4_0 already cost, against
        the float32 weights: median <b className="t-global">{sci(sel.weightMedian)}</b>.
        </Translated>
      </p>
    </Bench>
  );
}

function BitwiseReveal({ predicted }: { predicted: number }) {
  const r = bitwiseAgreement(40);
  return (
    <>
      <Translated
        vi={
          <p data-bitwise={r.same}>
            Trong mô hình, <b>{r.same} trên 40</b> khối sinh từ seed cho cùng các bit float32 trên cả hai đường
            {predicted === r.same ? ', đúng như bạn nói' : ''}. Khoảng chênh lớn nhất, tương đối so với tổng |w·x| của khối,
            là {sci(r.worstGap)}: làm tròn số thực, không hơn.
          </p>
        }
      >
      <p data-bitwise={r.same}>
        In the model, <b>{r.same} of 40</b> seeded blocks give the same float32 bits on both paths
        {predicted === r.same ? ', as you said' : ''}. The largest gap, relative to the block's sum of |w·x|, is{' '}
        {sci(r.worstGap)}: float rounding, nothing more.
      </p>
      </Translated>
      <Translated
        vi={
          <p>
            Trong số học chính xác, hai đường cho cùng một số với mọi khối (unit test kiểm tra 80 khối). Ở float32, chúng
            làm tròn các giá trị trung gian khác nhau: đường A làm tròn tích của từng trọng số đã nhân hệ số vào một tổng,
            đường B làm tròn mã nhân activation đã chia trước vào bốn tổng và gộp độ dời ở cuối. Vì vậy một kernel chuyển
            từ cách này sang cách kia sẽ đổi các bit thấp của đầu ra, và các bài kiểm tra của nó cần một dung sai, không
            phải phép so sánh bằng.
          </p>
        }
      >
      <p>
        In exact arithmetic the two are the same number for every block (the unit test checks 80 of them). In float32
        they round different intermediate values: path A rounds each scaled weight's product into one sum, path B
        rounds code times pre-divided activation into four sums and folds the offset at the end. So a kernel that
        switches from one to the other changes the low bits of its output, and its tests need a tolerance, not equality.
      </p>
      </Translated>
    </>
  );
}

function ActErrorReveal({ predicted }: { predicted: number }) {
  const r = rowErrors(4096, 'normal');
  const pct = r.median * 100;
  return (
    <Translated
      vi={
        <p data-act-reveal="">
          Trung vị của mô hình trên 200 hàng dài 4,096 là <b>{pct.toPrecision(2)}%</b> của tổng |w·x|
          {predicted > pct * 10 ? ', thấp hơn nhiều so với dự đoán của bạn' : predicted < pct / 10 ? ', cao hơn nhiều so với dự đoán của bạn' : ', gần với dự đoán của bạn'}. Hãy
          dùng trình khám phá bên dưới để xem nó thay đổi thế nào theo độ dài hàng và khi có các kênh lớn.
        </p>
      }
    >
    <p data-act-reveal="">
      The model's median over 200 rows of 4,096 is <b>{pct.toPrecision(2)}%</b> of the sum of |w·x|
      {predicted > pct * 10 ? ', well under your guess' : predicted < pct / 10 ? ', well above your guess' : ', near your guess'}. Use
      the explorer below to see how it moves with the row length and with large channels.
    </p>
    </Translated>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSX25: Record<string, Question> = {
  'x2/X2.5/q-fold': {
    kind: 'choice',
    id: 'x2/X2.5/q-fold',
    prompt: (
      <Translated
        vi={
          <>
            Một khối Q4_0 lưu các mã q trong 0..15 và mang nghĩa (q − 8)·d. Cách viết lại nào đưa −8 ra khỏi vòng lặp qua 32
            trọng số, và nó tốn gì cho mỗi khối?
          </>
        }
      >
      <>
        A Q4_0 block stores codes q in 0..15 and means (q − 8)·d. Which rewriting takes the −8 out of the loop over
        the 32 weights, and what does it cost per block?
      </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="d·Σ q·x − 8·d·Σ x: một phép nhân-cộng mỗi khối, với Σ x dùng chung cho mọi hàng.">{'d·Σ q·x − 8·d·Σ x: one multiply-add per block, with Σ x shared by every row.'}</Translated>,
        correct: true,
      },
      { label: <Translated vi="d·Σ q·x − 8·Σ x: miễn phí, vì d chỉ nhân với số hạng đầu tiên trong hai số hạng.">{'d·Σ q·x − 8·Σ x: free, because d scales only the first of the two terms.'}</Translated>, why: <Translated vi="Số 8 cũng tính theo đơn vị d: phần hiệu chỉnh là −8·d·Σx, và đó là lý do các kernel nhân với d sau khi gộp.">{'The 8 is in units of d as well: the correction is −8·d·Σx, which is why the kernels multiply by d after folding.'}</Translated> },
      { label: <Translated vi="Không cách nào: phép trừ là theo từng trọng số, nên nó nằm lại trong vòng lặp 32 trọng số.">{'None: the subtraction is per weight, so it stays inside the 32-weight loop.'}</Translated>, why: <Translated vi="Tổng tuyến tính theo các mã, nên phần hằng của mọi số hạng có thể được cộng riêng.">{'The sum is linear in the codes, so the constant part of every term can be summed separately.'}</Translated> },
      { label: <Translated vi="Chỉ với activation int8: phép gộp cần tổng số nguyên chính xác mới hợp lệ.">{'Only with int8 activations: the fold needs exact integer sums to be valid.'}</Translated>, why: <Translated vi="Đó là đại số thuần túy trên mọi số: các kernel Metal gộp nó với activation số thực.">{'It is plain algebra on any numbers: the Metal kernels fold it with float activations.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Σ (d·q + b)·x = d·Σ q·x + b·Σ x với b = −8·d. Khi đó vòng lặp trong chỉ chạm vào mã và activation. Σ x chỉ phụ
            thuộc vào các activation, nên một tổng cho mỗi khối 32 phục vụ cho mọi hàng trong hàng nghìn hàng đọc cùng các
            activation. Kernel Metal của llama.cpp giữ nó là <code>sumy</code>, kernel CUDA của nó giữ bên trong{' '}
            <code>s</code> của Q8_1, MLX giữ là <code>sum</code> (với bias của nó thay cho −8·d).
          </p>
        }
      >
      <p>
        Σ (d·q + b)·x = d·Σ q·x + b·Σ x with b = −8·d. The inner loop then touches only codes and activations. Σ x
        depends on the activations alone, so one sum per block of 32 serves every one of the thousands of rows that
        read the same activations. llama.cpp's Metal kernel keeps it as <code>sumy</code>, its CUDA kernel inside
        Q8_1's <code>s</code>, MLX as <code>sum</code> (with its bias in place of −8·d).
      </p>
      </Translated>
    ),
    source: `ours (algebra); the kernels that do it: ${FACTS} D3 to D6, E4`,
    vi: { source: `của khóa học này (đại số); các kernel làm điều đó: ${FACTS} D3 đến D6, E4` },
  },
  'x2/X2.5/q-why-not-int-row': {
    kind: 'choice',
    id: 'x2/X2.5/q-why-not-int-row',
    prompt: <Translated vi={<>Đường C giữ một tổng số nguyên chính xác bên trong mỗi khối rồi nhân lại hệ số. Sao không giữ một tổng số nguyên trên cả hàng và nhân hệ số một lần ở cuối?</>}><>Path C keeps an exact integer sum inside each block and rescales it. Why not keep one integer sum over the whole row and rescale once at the end?</></Translated>,
    options: [
      { label: <Translated vi="Tổng số nguyên của mỗi khối có đơn vị riêng của nó, d_w × d_x.">{'Each block\'s integer sum is in its own units, d_w × d_x.'}</Translated>, correct: true },
      { label: <Translated vi="Một int32 sẽ tràn ở đâu đó dọc một hàng rộng 4,096.">{'An int32 would overflow somewhere along a 4,096-wide row.'}</Translated>, why: <Translated vi="Ngay cả ở trường hợp xấu nhất 16,129 cho mỗi tích, 133,144 tích vẫn vừa; tích thứ 133,145 mới tràn, nhiều hơn hẳn một hàng rộng 4,096. Tràn số không phải lý do.">{'Even at the worst case of 16,129 per product, 133,144 products still fit; the 133,145th overflows, far more than a 4,096-wide row. Overflow is not the reason.'}</Translated> },
      { label: <Translated vi="dp4a không thể cộng vào một bộ tích lũy quá tám lần.">{'dp4a cannot add into one accumulator more than eight times.'}</Translated>, why: <Translated vi="Nó có thể được gọi bao nhiêu lần cũng được; nó chỉ là một phép cộng vào một số nguyên.">{'It can be called any number of times; it is just an add into an integer.'}</Translated> },
      { label: <Translated vi="Chuyển một int32 lớn sang số thực ở cuối làm mất các bit thấp của nó.">{'Converting one large int32 to float at the end loses its low bits.'}</Translated>, why: <Translated vi="Một lần chuyển đổi là một lần làm tròn, ít hơn 128 lần mà việc nhân hệ số theo từng khối gây ra. Vấn đề thật là tổng đó chẳng mang nghĩa gì.">{'One conversion is one rounding, smaller than the 128 a per-block rescale makes. The real problem is that the sum means nothing.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Tổng số nguyên của một khối mang nghĩa d<sub>w</sub>·d<sub>x</sub> nhân với số nguyên đó. Cộng số nguyên của hai
            khối trước khi nhân hệ số là giả định rằng các hệ số của chúng bằng nhau, mà chúng không bằng: mỗi bộ 32 trọng số
            và mỗi bộ 32 activation có hệ số riêng. Hàng là Σ<sub>blocks</sub> d<sub>w</sub>·d<sub>x</sub>·(số nguyên), nên
            nó là số nguyên bên trong một khối và số thực giữa các khối. Mục X2.6 cho thấy vì sao đó cũng là độ rộng đúng cho
            mỗi phần.
          </p>
        }
      >
      <p>
        A block's integer sum means d<sub>w</sub>·d<sub>x</sub> times that integer. Adding two blocks' integers before
        scaling assumes their scales are equal, which they are not: each set of 32 weights and each set of 32
        activations got its own. The row is Σ<sub>blocks</sub> d<sub>w</sub>·d<sub>x</sub>·(integer), so it is
        integer inside a block and float across blocks. Section X2.6 shows why that is also the right width for each.
      </p>
      </Translated>
    ),
    source: `ours; the kernel's structure: ${FACTS} D3, E3`,
    vi: { source: `của khóa học này; cấu trúc của kernel: ${FACTS} D3, E3` },
  },
  'x2/X2.5/q-unsigned': {
    kind: 'choice',
    id: 'x2/X2.5/q-unsigned',
    prompt: <Translated vi={<>Kernel CUDA của llama.cpp đưa các mã không dấu 0..15, không phải −8..7, vào một phép tích vô hướng byte có dấu. Vì sao điều đó được phép, và số 8 đi đâu?</>}><>llama.cpp's CUDA kernel feeds the unsigned codes 0..15, not −8..7, into a signed byte dot product. Why is that allowed, and where does the 8 go?</></Translated>,
    options: [
      { label: <Translated vi="0..15 vừa một byte có dấu, nên dp4a có dấu đọc được; 8·s được trừ đi một lần mỗi khối.">{'0..15 fits a signed byte, so signed dp4a reads it; 8·s comes off once per block.'}</Translated>, correct: true },
      { label: <Translated vi="dp4a mở rộng dấu cho mỗi giá trị 4 bit, nên −8..7 được khôi phục miễn phí.">{'dp4a sign-extends each 4-bit value, so −8..7 is recovered for free.'}</Translated>, why: <Translated vi="dp4a làm việc trên byte. Sau mask, mỗi byte giữ một giá trị từ 0 đến 15, mà ở dạng byte có dấu thì chính là số dương đó.">{'dp4a works on bytes. After the mask each byte holds a value from 0 to 15, which as a signed byte is just that positive number.'}</Translated> },
      { label: <Translated vi="Intrinsic CUDA có một overload hỗn hợp không dấu nhân có dấu cho việc này.">{'The CUDA intrinsic has a mixed unsigned-by-signed overload for this.'}</Translated>, why: <Translated vi="Nó không có overload hỗn hợp (PTX thì có). Cũng không cần: 0..15 không âm trong một byte có dấu.">{'It has no mixed overload (PTX has one). None is needed: 0..15 is non-negative in a signed byte.'}</Translated> },
      { label: <Translated vi="Số 8 được gộp vào hệ số tỉ lệ FP16 d một lần, khi nạp mô hình.">{'The 8 is folded into the FP16 scale d once, when the model is loaded.'}</Translated>, why: <Translated vi="Một độ dời hằng không thể gộp vào một hệ số tỉ lệ; nó phải bị trừ ở đâu đó, và đó là số hạng s.">{'A constant offset cannot be folded into a scale; it has to be subtracted somewhere, and that is the s term.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Với mã không dấu, tổng số nguyên là Σ q·x<sub>q</sub>; giá trị thật của khối cần Σ (q − 8)·x<sub>q</sub>, tức là
            tổng đó trừ 8·Σ x<sub>q</sub>. Bộ lượng tử hóa activation đã tính sẵn Σ x<sub>q</sub> và lưu nó, nhân với
            d<sub>x</sub>, thành s (s = d<sub>x</sub>·Σ q<sub>x</sub>). Vì vậy kernel trả về d<sub>w</sub>·(sumi·d<sub>x</sub> − 8·s): phép trừ
            diễn ra một lần mỗi khối, ở số thực.
          </p>
        }
      >
      <p>
        With unsigned codes the integer sum is Σ q·x<sub>q</sub>; the true block value needs Σ (q − 8)·x<sub>q</sub>,
        which is that minus 8·Σ x<sub>q</sub>. The activation quantiser already computed Σ x<sub>q</sub> and stored
        it, times d<sub>x</sub>, as s (s = d<sub>x</sub>·Σ q<sub>x</sub>). So the kernel returns d<sub>w</sub>·(sumi·d<sub>x</sub> − 8·s): the subtraction
        happens once per block, in float.
      </p>
      </Translated>
    ),
    source: `${FACTS} A2 (dp4a's overloads), D3 (vec_dot_q4_0_q8_1, quantize_q8_1)`,
    vi: { source: `${FACTS} A2 (các overload của dp4a), D3 (vec_dot_q4_0_q8_1, quantize_q8_1)` },
  },
  'x2/X2.5/q-prescale-exact': {
    kind: 'choice',
    id: 'x2/X2.5/q-prescale-exact',
    prompt: <Translated vi={<>Đường B chia activation cho 16, 256 và 4096 để các nibble đã mask, không dịch bit, cho ra đúng tích. Vì sao phép chia đó không làm mất gì ở float32?</>}><>Path B divides activations by 16, 256 and 4096 so that masked, unshifted nibbles give the right products. Why does that division lose nothing in float32?</></Translated>,
    options: [
      { label: <Translated vi="Chia cho một lũy thừa của hai chỉ đổi số mũ, trừ khi bị underflow.">{'Dividing by a power of two changes only the exponent, barring underflow.'}</Translated>, correct: true },
      { label: <Translated vi="Mask nhân với cùng hệ số đó, nên triệt tiêu mọi sai số làm tròn.">{'The mask multiplies by the same factor, which cancels any rounding.'}</Translated>, why: <Translated vi="Mask đúng là cung cấp hệ số đó, nhưng như vậy không hoàn tác được một lần làm tròn xảy ra ở phép chia. Mà cũng chẳng có lần nào để hoàn tác.">{'The mask does supply the factor, but that would not undo a rounding made at the division. There is none to undo.'}</Translated> },
      { label: <Translated vi="Activation dùng ít hơn 24 bit phần định trị, nên còn dư chỗ.">{'Activations use fewer than 24 significand bits, leaving room to spare.'}</Translated>, why: <Translated vi="Không phải vậy; một activation float32 có thể dùng cả 24 bit.">{'They do not; a float32 activation can use all 24.'}</Translated> },
      { label: <Translated vi="Nó mất một chút, nhưng luôn ít hơn mức dung sai của các bài kiểm tra cho phép.">{'It loses a little, but always less than the tests’ tolerance allows.'}</Translated>, why: <Translated vi="Nó không mất gì với mọi activation có độ lớn thông thường: unit test kiểm tra mọi activation nó sinh ra.">{'It loses nothing for any activation of ordinary size: the unit test checks every one it generates.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            x / 4096 có cùng phần định trị 24 bit với x và số mũ nhỏ hơn 12 đơn vị. Chỉ một giá trị vốn đã gần số chuẩn nhỏ nhất của
            float (khoảng 1.2e−38) mới hết số mũ. Vì vậy mã × (x / 16<sup>k</sup>) × 16<sup>k</sup>, chính là thứ một nibble đã
            mask nhân với một activation đã chia trước tính ra, bằng mã × x đến từng bit, trước khi bộ tích lũy làm tròn nó. Ở
            độ chính xác half, cùng mẹo đó không an toàn: mục X2.6 cho thấy vì sao.
          </p>
        }
      >
      <p>
        x / 4096 has the same 24-bit significand as x and an exponent 12 smaller. Only a value already near
        float's smallest normal (about 1.2e−38) would run out of exponent. So code × (x / 16<sup>k</sup>) × 16<sup>k</sup>,
        which is what a masked nibble times a pre-divided activation computes, equals code × x to the bit, before the
        accumulator rounds it. In half precision the same trick is not safe: section X2.6 shows why.
      </p>
      </Translated>
    ),
    source: `${FACTS} E4, D4`,
  },
  'x2/X2.5/q-hip-sdot4': {
    kind: 'choice',
    id: 'x2/X2.5/q-hip-sdot4',
    prompt: <Translated vi={<>Bạn build một kernel HIP cho GPU RDNA3 với một hàm trợ giúp gọi <code>__builtin_amdgcn_sdot4</code>. Điều gì xảy ra?</>}><>You build a HIP kernel for an RDNA3 GPU with a helper that calls <code>__builtin_amdgcn_sdot4</code>. What happens?</></Translated>,
    options: [
      { label: <Translated vi="Một lỗi biên dịch: RDNA3 và RDNA4 thiếu tính năng của sdot4; chúng có sudot4.">{'A compile error: RDNA3 and RDNA4 lack sdot4\'s feature; they have sudot4.'}</Translated>, correct: true },
      { label: <Translated vi="Nó biên dịch thành bốn phép nhân vô hướng: kết quả đúng, chỉ chậm hơn.">{'It compiles to four scalar multiplies: correct results, just slower.'}</Translated>, why: <Translated vi="Gọi một builtin trên target không có tính năng của nó là lỗi biên dịch, không phải một đường chậm; chính bài kiểm tra của clang ghi “needs target feature dot1-insts”.">{'Calling a builtin on a target without its feature is a compile error, not a slow path; clang’s own test says "needs target feature dot1-insts".'}</Translated> },
      { label: <Translated vi="Nó biên dịch và chạy, vì HIP ánh xạ builtin này sang __dp4a.">{'It compiles and runs, because HIP maps the builtin onto __dp4a.'}</Translated>, why: <Translated vi="HIP hoàn toàn không có __dp4a.">{'HIP has no __dp4a at all.'}</Translated> },
      { label: <Translated vi="Nó biên dịch được, nhưng cho kết quả sai với các mã nibble không dấu.">{'It compiles, but gives wrong answers for the unsigned nibble codes.'}</Translated>, why: <Translated vi="Vấn đề đến sớm hơn: nó không biên dịch được cho target đó.">{'The problem comes earlier: it does not compile for that target.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mô tả target của trình biên dịch giao phép tích vô hướng bốn chiều 8 bit có dấu (dot1) cho gfx906, CDNA và RDNA2,
            và dạng dấu hỗn hợp (dot8) cho RDNA3 và RDNA4, mà llama.cpp gọi với cả hai toán hạng được đánh dấu là có dấu.{' '}
            <code>ggml_cuda_dp4a</code> của nó chọn bằng một <code>#if</code> theo kiến trúc; gfx900 và RDNA1 dùng inline
            assembly.
          </p>
        }
      >
      <p>
        The compiler's target description gives the signed 8-bit four-way dot (dot1) to gfx906, CDNA and RDNA2, and the
        mixed-sign form (dot8) to RDNA3 and RDNA4, which llama.cpp calls with both operands marked signed. Its <code>ggml_cuda_dp4a</code> chooses with an{' '}
        <code>#if</code> on the architecture; gfx900 and RDNA1 get inline assembly.
      </p>
      </Translated>
    ),
    source: `${FACTS} B1, B2`,
  },
  'x2/X2.5/q-metal': {
    kind: 'choice',
    id: 'x2/X2.5/q-metal',
    prompt: <Translated vi={<>Metal không có phép tích vô hướng số nguyên theo từng luồng. llama.cpp, MLX và PyTorch làm gì trong các kernel ma trận-vector 4 bit trên Metal của họ?</>}><>Metal has no per-thread integer dot product. What do llama.cpp, MLX and PyTorch do in their Metal 4-bit matrix-vector kernels?</></Translated>,
    options: [
      { label: <Translated vi="Ở lại với số thực: nibble đã mask nhân activation đã chia trước.">{'Stay in float: masked nibbles times pre-divided activations.'}</Translated>, correct: true },
      { label: <Translated vi="Lượng tử hóa activation về int8 theo từng khối và nhân các byte trong một vòng lặp.">{'Quantise the activations to int8 per block and multiply bytes in a loop.'}</Translated>, why: <Translated vi="Không ai trong số họ lượng tử hóa activation trên Metal; đó là đường CUDA và HIP của llama.cpp.">{'None of them quantises the activations on Metal; that is llama.cpp’s CUDA and HIP path.'}</Translated> },
      { label: <Translated vi="Gọi tensor API của Apple với toán hạng int4b_format cho trọng số.">{'Call Apple\'s tensor API with int4b_format operands for the weights.'}</Translated>, why: <Translated vi="Apple đã thêm toán hạng số nguyên 4 bit vào tensor API, nhưng một lượt tìm kiếm mã không thấy llama.cpp hay MLX dùng int4b_format.">{'Apple added 4-bit integer operands to its tensor API, but a code search found neither llama.cpp nor MLX using int4b_format.'}</Translated> },
      { label: <Translated vi="Giải lượng tử hóa mỗi trọng số về half và tích lũy bằng dot trên vector half4.">{'Dequantise each weight to half and accumulate with dot on half4 vectors.'}</Translated>, why: <Translated vi="Như vậy sẽ tích lũy ở half (dot của half4 trả về half). llama.cpp chuyển sang float4 trước khi dot.">{'That would accumulate in half (dot of half4 returns half). llama.cpp converts to float4 before dot.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Cả ba đều đọc các word 16 bit chứa mã, mask chúng mà không dịch bit, và chia trước các activation tương ứng cho cùng
            các lũy thừa của hai, nên mọi tích đều đúng bằng mã × activation, rồi gộp độ dời một lần mỗi khối. Đó là đường B
            trong trình chạy từng bước, và là cách làm nên chép theo trên Metal.
          </p>
        }
      >
      <p>
        All three read 16-bit words of codes, mask them without shifting, and divide the matching activations by the
        same powers of two first, so every product is code × activation exactly, and fold the offset once per block.
        It is path B in the stepper, and the idiom to copy on Metal.
      </p>
      </Translated>
    ),
    source: `${FACTS} D4, D5, D6, C1, C2`,
  },
};

// ---------------------------------------------------------------- the page
export function SX25() {
  const dot4 = helperOf(loaded(useLabFile('x2/lab3_qhunt/solution/kernel.cu')));
  return (
    <>
      <Translated
        vi={
        <p>
          Một GEMV lượng tử hóa (mục X2.4) dành vòng lặp trong của nó cho phép tích vô hướng của một khối: 32 mã bốn bit, một
          hệ số tỉ lệ, 32 activation. Có hai họ cách tính nó. Giải lượng tử hóa các mã thành số thực rồi nhân ở số thực, hoặc
          biến cả các activation thành số nguyên nhỏ rồi nhân ở số nguyên. Kernel dùng cách nào tùy vào backend, và cả hai đều
          có một mẹo khiến chúng rẻ hơn vẻ ngoài.
        </p>
        }
      >
      <p>
        A quantised GEMV (section X2.4) spends its inner loop on one block's dot product: 32 four-bit codes, one
        scale, 32 activations. There are two families of ways to compute it. Dequantise the codes to floats and
        multiply in float, or turn the activations into small integers too and multiply in integers. Which one a
        kernel uses depends on the backend, and both have a trick that makes them cheaper than they look.
      </p>
      </Translated>
      <Beyond title={tr("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
        <Translated
          vi={
          <p>
            Module X2 là của khóa học này: sách không có định dạng số nguyên. Mọi điều nói ở đây về llama.cpp, MLX, PyTorch và
            các nhà sản xuất đã được kiểm tra trong lượt kiểm tra dữ kiện ngày 2026-11-20 ({FACTS}).
          </p>
          }
        >
        <p>
          Module X2 is ours: the book has no integer formats. Everything said here about llama.cpp, MLX, PyTorch and the
          vendors was checked in the facts pass on 2026-11-20 ({FACTS}).
        </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
        <p>
          Một điều cần biết trước khi vào chi tiết. Khi llama.cpp decode một mô hình Q4_0 trên GPU CUDA hoặc ROCm, chính vector
          activation của bạn bị lượng tử hóa về int8, mỗi lần 32 giá trị, cho mọi token, để các trọng số có thể được nhân dưới
          dạng byte. Trên GPU của Apple, cùng mô hình đó vẫn ở số thực.
        </p>
        }
      >
      <p>
        One thing to know before the details. When llama.cpp decodes a Q4_0 model on a CUDA or ROCm GPU, your
        activation vector is itself quantised to int8, 32 values at a time, for every token, so that the weights can
        be multiplied as bytes. On Apple GPUs the same model stays in float.
      </p>
      </Translated>
      <Today checked={`${CHECKED}, D3, D4)`}>
        <Translated
          vi={
          <p>
            Đường ma trận-vector CUDA của llama.cpp gọi <code>quantize_row_q8_1_cuda</code> trên các activation trước kernel
            của nó; bản build HIP biên dịch cùng mã nguồn đó. Các kernel decode Metal của nó không bao giờ lượng tử hóa
            activation.
          </p>
          }
        >
        <p>
          llama.cpp's CUDA matrix-vector path calls <code>quantize_row_q8_1_cuda</code> on the activations before its
          kernel; the HIP build compiles the same source. Its Metal decode kernels never quantise the activations.
        </p>
        </Translated>
      </Today>

      <h2><Translated vi="Ba cách đến một con số">Three ways to one number</Translated></h2>
      <Translated
        vi={
        <p>
          <b>Đường A</b> là định nghĩa: biến mọi mã thành trọng số của nó, (q − 8)·d, rồi nhân-cộng với activation. Nó đúng và
          nó lãng phí: một phép trừ, một phép chuyển đổi và một phép nhân với d cho mỗi trọng số, trước cả phép nhân làm việc
          thật.
        </p>
        }
      >
      <p>
        <b>Path A</b> is the definition: turn every code into its weight, (q − 8)·d, then multiply-add with the
        activation. It is correct and it is wasteful: a subtraction, a conversion and a multiplication by d for every
        weight, before the multiply that does the work.
      </p>
      </Translated>
      <Translated
        vi={
        <p>
          <b>Đường B</b> đưa mọi thứ là hằng ra khỏi vòng lặp. Vì Σ d·(q − 8)·x = d·(Σ q·x − 8·Σ x), vòng lặp chỉ cần mã nhân
          activation; số 8 được trừ một lần, thông qua Σ x, và d được áp dụng một lần. Σ x chỉ phụ thuộc vào các activation, nên
          nó được tính một lần cho mỗi khối mỗi token và dùng chung cho mọi hàng của ma trận trọng số. Mẹo thứ hai loại bỏ các
          phép dịch bit: đọc hai byte thành một word 16 bit rồi mask nó theo bốn cách sẽ để lại mỗi nibble nhân với 1, 16, 256
          hoặc 4096. Chia trước activation tương ứng cho cùng lũy thừa của hai đó và tích sẽ là mã × activation, chính xác.
        </p>
        }
      >
      <p>
        <b>Path B</b> moves everything constant out of the loop. Because Σ d·(q − 8)·x = d·(Σ q·x − 8·Σ x), the loop
        only needs codes times activations; the 8 is subtracted once, through Σ x, and d is applied once. Σ x depends
        only on the activations, so it is computed once per block per token and shared by every row of the weight
        matrix. A second trick removes the shifts: reading two bytes as one 16-bit word and masking it four ways leaves
        each nibble multiplied by 1, 16, 256 or 4096. Divide the matching activation by the same power of two in
        advance and the product is code × activation, exactly.
      </p>
      </Translated>
      <Translated
        vi={
        <p>
          <b>Đường C</b> là decode CUDA và HIP của llama.cpp. Trước hết 32 activation trở thành một khối Q8_1: một phép max rộng 32,
          d = max / 127, mọi activation làm tròn thành một byte có dấu, và s = d × (tổng các byte). Sau đó các mã trọng số, vẫn
          không dấu, gặp các byte activation trong một phép tích vô hướng byte bốn chiều (dp4a), tám lần mỗi khối, vào một số
          nguyên chính xác. Một bước hiệu chỉnh và một lần nhân hệ số mỗi khối biến nó trở lại thành số thực. Phép max và phép
          tổng là phép rút gọn (reduction) trên warp của chương 10, áp dụng cho các nhóm 32 bất kể độ rộng warp của phần cứng.
        </p>
        }
      >
      <p>
        <b>Path C</b> is llama.cpp's CUDA and HIP decode. First the 32 activations become a Q8_1 block: a 32-wide max,
        d = max / 127, every activation rounded to a signed byte, and s = d × (sum of the bytes). Then the weight codes,
        still unsigned, meet the activation bytes in a four-way byte dot product (dp4a), eight times per block, into an
        exact integer. One correction and one rescale per block turn it back into a float. The max and the sum are
        chapter 10's warp reduction, applied to groups of 32 whatever the hardware's warp width.
      </p>
      </Translated>
      <Source section="10.7" pages="237–241" />
      <Today checked={tr(`${CHECKED}, D3 to D6)`, `${CHECKED}, D3 đến D6)`)}>
        <Translated
          vi={
          <p>
            Đường B là cách các kernel <code>mul_mv</code> của llama.cpp, <code>qmv</code> của MLX và{' '}
            <code>int4pack_mm</code> trên MPS của PyTorch làm: activation chia trước cho 16, 256 và 4096, độ dời gộp qua tổng các
            activation. Đường C là <code>vec_dot_q4_0_q8_1</code> của llama.cpp.
          </p>
          }
        >
        <p>
          Path B is what llama.cpp's <code>mul_mv</code> kernels, MLX's <code>qmv</code> and PyTorch's MPS{' '}
          <code>int4pack_mm</code> do: activations pre-divided by 16, 256 and 4096, offset folded through the sum of the
          activations. Path C is llama.cpp's <code>vec_dot_q4_0_q8_1</code>.
        </p>
        </Translated>
      </Today>
      <Translated
        vi={
        <p>
          Trình chạy từng bước chạy cả ba trên cùng một khối. Hãy theo dõi hai thứ trong lúc đi: giá trị đang chạy, và số đếm
          công việc mỗi hàng so với công việc mỗi token.
        </p>
        }
      >
      <p>
        The stepper runs all three on the same block. Watch two things as you go: the running value, and the counts
        of work per row against work per token.
      </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
          <p>
            Số phép toán tính ở mức mã nguồn, không phải lệnh máy: một FMA tính là một phép nhân và một phép cộng, một dp4a là một
            phép toán số nguyên. Không có đo thời gian. Lab X2.3 đo thứ thật trên thiết bị của bạn.
          </p>
          }
        >
        <p>
          Operation counts are source-level, not instructions: an FMA counts as one multiply and one add, a dp4a as one
          integer operation. There is no timing. Lab X2.3 measures the real thing on your device.
        </p>
        </Translated>
      </Simplification>
      <ThreePaths />
      <Predict
        id="x2/X2.5/p-bitwise"
        prompt={
          <Translated
            vi={
              <>
                Đường A và B bằng nhau trong số học chính xác. Trên 40 khối sinh từ seed tính ở float32, có bao nhiêu khối
                chúng cho ra cùng các bit?
              </>
            }
          >
          <>
            Paths A and B are equal in exact arithmetic. Over 40 seeded blocks computed in float32, in how many do they
            give the same bits?
          </>
          </Translated>
        }
        unit={tr('of 40', 'trên 40')}
      >
        {(v) => <BitwiseReveal predicted={v} />}
      </Predict>
      <Ask q={questionsSX25['x2/X2.5/q-fold']} />
      <Ask q={questionsSX25['x2/X2.5/q-prescale-exact']} />

      <h2><Translated vi="Đường số nguyên tốn gì">What the integer path costs</Translated></h2>
      <Translated
        vi={
        <p>
          Đường C mua số học số nguyên chính xác cho 32 tích với cái giá là ba thứ, và bạn nên kể ra được chúng trước khi đọc
          tiếp. Một lượt đi thêm qua các activation cho mọi token. Việc làm tròn mọi activation về một trong 255 mức, do activation
          lớn nhất trong khối của nó quyết định. Và việc làm tròn FP16 của s, tổng được lưu mà bước hiệu chỉnh dùng đến.
        </p>
        }
      >
      <p>
        Path C buys exact integer arithmetic for 32 products at the price of three things, and you should be able to
        name them before reading on. An extra pass over the activations for every token. The rounding of every
        activation to one of 255 levels, set by the largest one in its block. And the FP16 rounding of s, the stored
        sum that the correction uses.
      </p>
      </Translated>
      <Predict
        id="x2/X2.5/p-act-error"
        prompt={
          <Translated
            vi={
              <>
                Với một hàng rộng 4,096 có trọng số và activation phân phối chuẩn tắc, bạn dự đoán sai số của đường C so với
                đường B lớn đến đâu, tính theo phần trăm của tổng |w·x|? (Đường B chỉ làm tròn ở float32.)
              </>
            }
          >
          <>
            For a 4,096-wide row with standard-normal weights and activations, how large do you expect path C's error
            against path B to be, as a percentage of the sum of |w·x|? (Path B rounds only in float32.)
          </>
          </Translated>
        }
        unit="%"
      >
        {(v) => <ActErrorReveal predicted={v} />}
      </Predict>
      <ActivationCost />
      <Translated
        vi={
        <p>
          Một activation lớn quyết định hệ số tỉ lệ của khối nó. Mọi activation khác trong khối khi đó bị làm tròn về một lưới
          thô hơn, nên toàn bộ phần đóng góp của khối trở nên tệ hơn, không chỉ số hạng lớn. Phân phối thứ hai của trình khám phá
          đặt một kênh lớn trong mỗi 128, và sai số tăng tương ứng. Bên cạnh là sai số mà việc lượng tử hóa trọng số về 4 bit đã
          gây ra: ở đây, việc làm tròn activation là cái nhỏ hơn trong hai.
        </p>
        }
      >
      <p>
        A large activation sets its block's scale. Every other activation in the block is then rounded to a coarser
        grid, so the block's whole contribution gets worse, not just the large term. The explorer's second
        distribution puts one large channel in every 128, and the error rises accordingly. Next to it is the error that
        quantising the weights to 4 bits already introduced: the activation rounding is the smaller of the two here.
      </p>
      </Translated>
      <Ask q={questionsSX25['x2/X2.5/q-why-not-int-row']} />
      <Ask q={questionsSX25['x2/X2.5/q-unsigned']} />
      <Translated
        vi={
        <p>
          Cảnh bên dưới theo một khối đi dọc đường C, từ phép max trên warp quyết định hệ số tỉ lệ của các activation đến kết quả
          của hàng trên cả hai đường, và cho thấy mỗi cái giá trong ba cái giá đó nảy sinh ở đâu.
        </p>
        }
      >
      <p>
        The scene below follows one block down path C, from the warp max that sets the activations' scale to the row's
        result on both paths, and shows where each of those three costs arises.
      </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
          <p>
            Một khối sinh từ seed, rồi một hàng 8 khối trên 4 lane, không phải lượng của cả một warp. Mọi giá trị đến từ{' '}
            <code>qdot.ts</code>, với cách làm tròn Q8_1 của <code>quantize_q8_1</code> trong llama.cpp. Không có gì được đo thời gian.
          </p>
          }
        >
        <p>
          One seeded block, then a row of 8 blocks over 4 lanes, not a warp's worth. Every value comes from{' '}
          <code>qdot.ts</code>, with the Q8_1 rounding of llama.cpp's <code>quantize_q8_1</code>. Nothing is timed.
        </p>
        </Translated>
      </Simplification>
      <IntegerPathScene />

      <h2><Translated vi="Prefill giải lượng tử hóa từng ô">Prefill dequantises tiles</Translated></h2>
      <Translated
        vi={
        <p>
          Mọi thứ ở trên là decode: một token, một phép nhân ma trận-vector, bị giới hạn bởi băng thông. Với prefill, nhiều token
          cùng lúc, <code>mul_mm</code> trên Metal của llama.cpp giải lượng tử hóa từng ô (tile) trọng số vào bộ nhớ threadgroup rồi
          nhân các ô ở số thực bằng các ma trận SIMD group. Đó lại là đường A, nhưng ở mức ô: mỗi trọng số đã giải lượng tử hóa được
          mọi token trong ô dùng lại, nên chi phí của nó được dàn mỏng. Đó là hình dạng mà phép GEMM phân ô của lab 15.6 sẽ có nếu
          các ô trọng số của nó được nạp từ các khối Q4_0.
        </p>
        }
      >
      <p>
        Everything above is decode: one token, a matrix-vector product, bandwidth-bound. For prefill, many tokens at
        once, llama.cpp's Metal <code>mul_mm</code> dequantises each tile of weights into threadgroup memory and then
        multiplies tiles in float with SIMD-group matrices. That is path A again, but at tile granularity: each
        dequantised weight is reused by every token in the tile, so its cost is spread thin. It is the shape lab
        15.6's tiled GEMM would take if its weight tiles were loaded from Q4_0 blocks.
      </p>
      </Translated>
      <Aside>
        <Translated
          vi={
          <p>
            GEMM phân ô là chương 15 (sách pp. 349–370) và <a href={href('ch15', 'labs')}>lab 15.6</a>. Đường ô của
            llama.cpp: {FACTS} D4.
          </p>
          }
        >
        <p>
          The tiled GEMM is chapter 15 (book pp. 349–370) and <a href={href('ch15', 'labs')}>lab 15.6</a>. llama.cpp's
          tile path: {FACTS} D4.
        </p>
        </Translated>
      </Aside>
      <Beyond title={tr('Not used here: integer matrix instructions', 'Không dùng ở đây: lệnh ma trận số nguyên')}>
        <Translated
          vi={
          <p>
            <code>mma</code> của CUDA có dạng s8 và s4 từ sm_75 và sm_80, <code>wgmma</code> của Hopper chỉ s8, và{' '}
            <code>tcgen05</code> <code>kind::i8</code> của Blackwell chỉ trên một số target cụ thể. AMD có MFMA i8 trên CDNA và WMMA
            iu8/iu4 trên RDNA3 và RDNA4. Tensor API của Metal nhân các toán hạng char và int4b_format nhưng chỉ chấp nhận hệ số tỉ lệ
            khối là lũy thừa của hai (UE8M0). Không cái nào trong số đó có trong các lab. ({FACTS} A4, B3, C3.)
          </p>
          }
        >
        <p>
          CUDA's <code>mma</code> has s8 and s4 forms from sm_75 and sm_80, Hopper's <code>wgmma</code> s8 only, and
          Blackwell's <code>tcgen05</code> <code>kind::i8</code> only on specific targets. AMD has i8 MFMA on CDNA and
          iu8/iu4 WMMA on RDNA3 and RDNA4. Metal's tensor API multiplies char and int4b_format operands but accepts only
          power-of-two (UE8M0) block scales. None of them is in the labs. ({FACTS} A4, B3, C3.)
        </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Mỗi backend cung cấp gì cho phép tích vô hướng byte">What each backend offers for a byte dot product</Translated></h2>
      <Translated
        vi={
        <p>
          Ở đây câu chuyện về tính di động của khóa học có thêm một mục mới. Một tệp <code>.cu</code> build cho HIP không cần sửa
          thông qua <code>pmpp/vendors/hip.h</code> với luồng, khối, bộ nhớ chia sẻ, rào chắn, atomic kiểu cũ và các lời gọi
          runtime. Danh sách những thứ không như vậy đã có các intrinsic warp <code>_sync</code> (HIP muốn một lane mask 64 bit),
          cooperative groups, header half-precision, functor CUB và intrinsic ma trận. Thêm vào đó phép tích vô hướng số nguyên
          theo từng luồng: <code>__dp4a</code> của CUDA không tồn tại dưới HIP, và các builtin AMD thay thế nó phụ thuộc vào thế
          hệ GPU.
        </p>
        }
      >
      <p>
        Here the portability story of the course gets a new entry. One <code>.cu</code> file builds for HIP unchanged
        through <code>pmpp/vendors/hip.h</code> for threads, blocks, shared memory, barriers, legacy atomics and
        runtime calls. The list of what does not already had warp <code>_sync</code> intrinsics (HIP wants a 64-bit
        lane mask), cooperative groups, half-precision headers, CUB functors and matrix intrinsics. Add the per-thread
        integer dot product: CUDA's <code>__dp4a</code> does not exist under HIP, and the AMD builtins that replace it
        depend on the GPU generation.
      </p>
      </Translated>
      <Today checked={`${CHECKED}, A2, A3, B1, B2, B4, C1)`}>
        <Translated
          vi={
          <p>
            Các builtin của AMD chỉ được ghi trong mô tả target của trình biên dịch; không tìm thấy trang văn bản nào cho chúng hay
            cho <code>amd_mixed_dot</code>. Không gì trong bảng đã được chạy trên phần cứng CUDA hay AMD.
          </p>
          }
        >
        <p>
          AMD's builtins are documented only in the compiler's target description; no prose page was found for them or
          for <code>amd_mixed_dot</code>. Nothing in the table was run on CUDA or AMD hardware.
        </p>
        </Translated>
      </Today>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-backends="">
        <thead>
          <tr><th><Translated vi="Theo từng luồng">Per thread</Translated></th><th>CUDA</th><th>HIP (ROCm){/* i18n-ignore: backend name */}</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="Tích vô hướng int8 bốn chiều, tích lũy 32 bit">Four-way int8 dot, 32-bit accumulate</Translated></td>
            <td><Translated vi={<><code>__dp4a</code>, từ sm_61; chỉ có overload cùng dấu (PTX có cả dấu hỗn hợp)</>}><code>__dp4a</code>, from sm_61; same-sign overloads only (PTX also has mixed sign)</Translated></td>
            <td><Translated vi={<>Không có <code>__dp4a</code>. <code>__builtin_amdgcn_sdot4</code> trên gfx906, CDNA, RDNA2; <code>__builtin_amdgcn_sudot4</code> trên RDNA3 và RDNA4; dùng nhầm cái thì không biên dịch được. <code>amd_mixed_dot</code> trong header không có tài liệu</>}>No <code>__dp4a</code>. <code>__builtin_amdgcn_sdot4</code> on gfx906, CDNA, RDNA2; <code>__builtin_amdgcn_sudot4</code> on RDNA3 and RDNA4; the wrong one does not compile. The header's <code>amd_mixed_dot</code> is undocumented</Translated></td>
            <td><Translated vi={<>Không có: <code>dot</code> của MSL dành cho vector float và half</>}>None: MSL's <code>dot</code> is for float and half vectors</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Tích vô hướng half với tích lũy float">Half dot with float accumulate</Translated></td>
            <td><Translated vi={<>Không có (<code>__hfma2</code> tích lũy ở half)</>}>None (<code>__hfma2</code> accumulates in half)</Translated></td>
            <td><code>amd_mixed_dot(__half2, __half2, float, bool)</code></td>
            <td><Translated vi={<>Không có: <code>dot(half4, half4)</code> trả về half</>}>None: <code>dot(half4, half4)</code> returns half</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Tích vô hướng FP8">FP8 dot</Translated></td>
            <td><Translated vi="Không có">None</Translated></td>
            <td><Translated vi={<><code>__builtin_amdgcn_dot4_f32_fp8_fp8</code> và các biến thể, gfx11.7 và gfx12</>}><code>__builtin_amdgcn_dot4_f32_fp8_fp8</code> and variants, gfx11.7 and gfx12</Translated></td>
            <td><Translated vi="Không có">None</Translated></td>
          </tr>
        </tbody>
      </table>
      <Ask q={questionsSX25['x2/X2.5/q-hip-sdot4']} />
      <Ask q={questionsSX25['x2/X2.5/q-metal']} />
      <Translated
        vi={
        <p>
          Vậy một lab muốn có phép tích vô hướng byte trong một tệp mã nguồn có hai lựa chọn. Một hàm trợ giúp đơn giản với bốn
          tích byte, biên dịch được ở mọi nơi và là thứ các trình giả lập chạy. Hoặc một thang <code>#if</code> như{' '}
          <code>ggml_cuda_dp4a</code> của llama.cpp: dưới HIP, <code>__builtin_amdgcn_sdot4(a, b, c, false)</code> trên CDNA, RDNA2
          và gfx906, <code>__builtin_amdgcn_sudot4(true, a, true, b, c, false)</code> trên RDNA3 và RDNA4, inline assembly trên
          RDNA1 và gfx900, bốn tích byte cho các trường hợp còn lại; trên CUDA, <code>__dp4a</code> từ sm_61 và bốn tích byte dưới
          mức đó. Hàm trợ giúp của lab X2.3 là nửa CUDA của thang đó.
        </p>
        }
      >
      <p>
        So a lab that wants a byte dot product in one source file has two choices. A plain helper with four byte
        products, which compiles everywhere and is what the emulators run. Or an <code>#if</code> ladder like
        llama.cpp's <code>ggml_cuda_dp4a</code>: under HIP, <code>__builtin_amdgcn_sdot4(a, b, c, false)</code> on
        CDNA, RDNA2 and gfx906, <code>__builtin_amdgcn_sudot4(true, a, true, b, c, false)</code> on RDNA3 and RDNA4,
        inline assembly on RDNA1 and gfx900, four byte products otherwise; on CUDA, <code>__dp4a</code> from sm_61 and
        four byte products below it. Lab X2.3's helper is the CUDA half of that ladder.
      </p>
      </Translated>
      <Today checked={`${CHECKED}, B1)`}>
        <Translated
          vi={
          <p>
            Thang này được đọc từ <code>ggml-cuda/common.cuh</code> của llama.cpp. Các nhánh AMD của nó là builtin của trình biên
            dịch và inline assembly; không nhánh nào trong số đó có trong các lab của khóa học.
          </p>
          }
        >
        <p>
          The ladder is read from llama.cpp's <code>ggml-cuda/common.cuh</code>. Its AMD branches are compiler builtins
          and inline assembly; none of them is in the course's labs.
        </p>
        </Translated>
      </Today>
      {dot4 ? (
        <Code
          tabs={[{ label: 'dot4_i8', code: dot4, // i18n-ignore: function name
            status: tr('four-product branch: emulator-checked; __dp4a branch: unverified', 'nhánh bốn tích: emulator-checked; nhánh __dp4a: unverified') }]}
          caption={
            <Translated
              vi={
                <>
                  Từ lời giải tham chiếu (solution) của lab X2.3, <code>labs/x2/lab3_qhunt/solution/kernel.cu</code>. Bản build
                  HIP và các trình giả lập đi theo nhánh bốn tích. Nhánh <code>__dp4a</code> là unverified: bước kiểm tra
                  front-end CUDA của khóa học chỉ biên dịch phía host (<code>--cuda-host-only</code>), nên mã nằm dưới{' '}
                  <code>__CUDA_ARCH__</code> không bao giờ được nó phân tích, và chưa có thiết bị CUDA nào chạy nó.
                </>
              }
            >
            <>
              From the reference solution of lab X2.3, <code>labs/x2/lab3_qhunt/solution/kernel.cu</code>. The HIP build
              and the emulators take the four-product branch. The <code>__dp4a</code> branch is unverified: the
              course's CUDA front-end check compiles the host side only (<code>--cuda-host-only</code>), so code under{' '}
              <code>__CUDA_ARCH__</code> is never parsed by it, and no CUDA device has run it yet.
            </>
            </Translated>
          }
        />
      ) : (
        <Translated
          vi={
          <p className="muted" data-dot4-missing="">
            Hàm trợ giúp nằm trong lời giải tham chiếu của lab X2.3, <code>labs/x2/lab3_qhunt/solution/kernel.cu</code>, tệp
            không đọc được từ bản checkout này.
          </p>
          }
        >
        <p className="muted" data-dot4-missing="">
          The helper is in lab X2.3's reference solution, <code>labs/x2/lab3_qhunt/solution/kernel.cu</code>, which
          could not be read from this checkout.
        </p>
        </Translated>
      )}
      <Translated
        vi={
        <p>
          Trên GPU của chính bạn, lab X2.3 (<a href={href('x2', 'labs')}>trang lab</a>) đưa đường C vào cả một GEMV và giấu
          các lỗi trong đó để bạn tìm.
        </p>
        }
      >
      <p>
        On your own GPU, lab X2.3 (<a href={href('x2', 'labs')}>labs page</a>) puts path C into a whole GEMV and hides
        defects in it for you to find.
      </p>
      </Translated>
    </>
  );
}
