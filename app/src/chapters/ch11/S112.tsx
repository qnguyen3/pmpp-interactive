import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, NoteGroup, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  EXERCISE_1,
  FIG_11_3,
  KIND_LABEL,
  SCHEDULES,
  SCHEDULE_LABEL,
  additionsAt,
  divergence,
  finalCount,
  iterations,
  kernelSource,
  levels,
  naiveAdditions,
  run,
  strides,
  totalAdditions,
  treeOf,
  warpStates,
  type Config,
  type Schedule,
} from '../../sim/ksblock';
import { functionSource } from '../../sim/toolchain';
import { KsDoublingScene } from './KsDoublingScene';
import { KsGrid, TraceView, fromLevels } from './KsGrid';

const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const WRAP: CSSProperties = { flexWrap: 'wrap' };

// ---------------------------------------------------------------- stride by stride
const SIXTEEN = [3, 1, 7, 0, 4, 1, 6, 3, 5, 2, 8, 1, 0, 6, 2, 4];
const ROWS = [
  {
    id: 'ex1',
    get label() {
      return tr('exercise 1', 'bài tập 1');
    },
    values: EXERCISE_1,
  },
  {
    id: 'sixteen',
    get label() {
      return tr('sixteen values', 'mười sáu giá trị');
    },
    values: SIXTEEN,
  },
];
function parseRow(s: string): number[] | null {
  const parts = s.split(/[\s,]+/).filter(Boolean);
  if (parts.length < 2 || parts.length > 16) return null;
  const v = parts.map((p) => Number(p.replace('−', '-')));
  return v.every((x) => Number.isInteger(x) && Math.abs(x) <= 999) ? v : null;
}

function StrideStepper() {
  const [values, setValues] = useState<number[]>(EXERCISE_1);
  const [text, setText] = useState(EXERCISE_1.join(' '));
  const lv = useMemo(() => levels(values), [values]);
  const B = values.length;
  const L = lv.length - 1;
  const [k, setK] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [tree, setTree] = useState<number>(B - 1);
  useEffect(() => {
    setK(0);
    setTree(values.length - 1);
    setPlaying(false);
  }, [values]);
  useEffect(() => {
    if (!playing) return;
    if (k >= L) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setK((x) => x + 1), 1100);
    return () => window.clearTimeout(id);
  }, [playing, k, L]);
  const t = useMemo(() => (tree >= 0 ? treeOf(tree, B) : undefined), [tree, B]);
  const g = fromLevels(lv, k, { tree: t, hot: k, carries: true });
  const done = lv.slice(1, k + 1).reduce((s, l) => s + l.added.filter(Boolean).length, 0);
  const bad = parseRow(text) === null;
  const pow2 = Number.isInteger(Math.log2(B));

  return (
    <Bench
      title={tr('Kogge-Stone, stride by stride', 'Kogge-Stone, từng stride một')}
      task={tr('Step through the strides. Before each step, say which positions will be final after it.', 'Đi qua từng stride. Trước mỗi bước, hãy nói những vị trí nào sẽ xong sau bước đó.')}
      foot={
        <Translated
          vi={
            <>
              Mọi vị trí của một bước cộng cùng lúc, đọc hàng phía trên (bức tranh lockstep của Fig. 11.2). Ô tô màu giữ giá trị
              cuối cùng. Dưới mỗi giá trị: các đầu vào mà nó tính.
            </>
          }
        >
          <>
            All positions of a step add at the same time, reading the row above (the lockstep picture of Fig. 11.2). Shaded
            cells hold their final value. Under each value: the inputs it counts.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ flex: '1 1 16rem' }}>
          <Translated vi="Đầu vào (2 đến 16 số nguyên)">Input (2 to 16 whole numbers)</Translated>
          <input type="text" value={text} aria-label={tr('Input', 'Đầu vào')} aria-invalid={bad}
            style={{ width: '100%', fontVariantNumeric: 'tabular-nums', borderColor: bad ? 'var(--fail)' : undefined }}
            onChange={(e) => {
              setText(e.target.value);
              const v = parseRow(e.target.value);
              if (v) setValues(v);
            }} />
        </label>
        <span className="seg" role="group" aria-label={tr('Rows', 'Các hàng')} style={WRAP}>
          {ROWS.map((r) => (
            <button key={r.id} onClick={() => { setValues(r.values); setText(r.values.join(' ')); }}>{r.label}</button>
          ))}
        </span>
        <label className="field">
          <Translated vi="Hiện cây phía sau">Show the tree behind</Translated>
          <select value={tree} onChange={(e) => setTree(Number(e.target.value))} aria-label={tr('Tree behind output', 'Cây phía sau đầu ra')}>
            <option value={-1}>{tr('no output', 'không đầu ra nào')}</option>
            {values.map((_, i) => <option key={i} value={i}>y{i}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setK(0); }} disabled={k === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => { setPlaying(false); setK(k - 1); }} disabled={k === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => { setPlaying(false); setK(k + 1); }} disabled={k >= L}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => setPlaying(!playing)} disabled={k >= L}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Phát')}</button>
        <span className="muted" data-readout="level">{k === 0 ? tr('the input', 'đầu vào') : tr(`after stride ${lv[k].stride} (step ${k - 1})`, `sau stride ${lv[k].stride} (bước ${k - 1})`)}</span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <KsGrid rows={g.rows} labels={g.labels} arrows={g.arrows} marks={g.marks} cellW={B > 8 ? 50 : 66} label={tr('Kogge-Stone, level by level', 'Kogge-Stone, từng tầng một')} />
      </div>
      <p style={{ marginBottom: 0 }} data-readout="stride" data-final={finalCount(B, k)} data-additions={done}>
        {k === 0 ? (
          <Translated vi={<>Vị trí 0 đã có sẵn đáp án, y0 = x0. Mọi vị trí khác chỉ chứa đầu vào của chính nó.</>}>Position 0 already holds its answer, y0 = x0. Every other position holds only its own input.</Translated>
        ) : (
          <Translated
            vi={
              <>
                Stride {lv[k].stride}: {additionsAt(B, lv[k].stride!)} phép cộng, tổng cộng đến giờ là {done}.{' '}
                <b>{finalCount(B, k)}</b> vị trí đầu tiên đã xong; mỗi vị trí còn lại chứa {2 ** k} đầu vào kết thúc tại nó.
              </>
            }
          >
            <>
              Stride {lv[k].stride}: {additionsAt(B, lv[k].stride!)} additions, {done} so far. The first{' '}
              <b>{finalCount(B, k)}</b> positions are final; every other one holds the {2 ** k} inputs ending at it.
            </>
          </Translated>
        )}{' '}
        {k === L ? (
          <Translated
            vi={
              <>
                Xong sau {L} bước với <b>{totalAdditions(B)}</b> phép cộng
                {pow2 ? ` = ${B}·${L} − ${B - 1}` : ''}, so với {B - 1} của vòng lặp.
              </>
            }
          >
            <>
              Done in {L} steps with <b>{totalAdditions(B)}</b> additions
              {pow2 ? ` = ${B}·${L} − ${B - 1}` : ''}, against {B - 1} for the loop.
            </>
          </Translated>
        ) : null}
        {t && tree > 0 && k === L ? (
          <Translated vi={<> Cây phía sau y{tree} dùng {t.nodes.filter((nd) => nd.level > 0 && nd.pos >= 2 ** (nd.level - 1)).length} phép cộng trong số đó.</>}>
            <> The tree behind y{tree} uses {t.nodes.filter((nd) => nd.level > 0 && nd.pos >= 2 ** (nd.level - 1)).length} of them.</>
          </Translated>
        ) : null}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- which warps diverge
const SIZES = [64, 128, 256, 512, 1024];
function WarpExplorer() {
  const [B, setB] = useState(256);
  const [width, setWidth] = useState(32);
  const ss = strides(B);
  const [si, setSi] = useState(2);
  const s = ss[Math.min(si, ss.length - 1)];
  const ws = warpStates(B, width, s);
  const d = divergence(B, width);
  const W = 640;
  const bw = Math.min(80, (W - 8) / ws.length);
  return (
    <Bench
      title={tr('Which warps diverge?', 'Warp nào phân kỳ?')}
      task={tr('Find every stride at which some warp runs both paths of the if. Then change the warp width.', 'Tìm mọi stride mà tại đó có warp chạy cả hai nhánh của if. Rồi đổi độ rộng warp.')}
      foot={
        <Translated vi={<>Một warp phân kỳ khi một số lane của nó thỏa <code>threadIdx.x &gt;= stride</code> còn một số thì không. Các lane được đánh số theo thứ tự luồng.</>}>
          <>A warp diverges when some of its lanes pass <code>threadIdx.x &gt;= stride</code> and some do not. Lanes are numbered in thread order.</>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi="Số luồng mỗi khối">Threads per block</Translated>
          <select value={B} onChange={(e) => setB(Number(e.target.value))} aria-label={tr('Threads per block', 'Số luồng mỗi khối')}>
            {SIZES.map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>
        <span style={FIELD}>
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <span className="seg" role="group" aria-label={tr('Warp width', 'Độ rộng warp')}>
            <button aria-pressed={width === 32} onClick={() => setWidth(32)}><Translated vi="32: sách, NVIDIA">32: the book, NVIDIA</Translated></button>
            <button aria-pressed={width === 64} onClick={() => setWidth(64)}><Translated vi="64: AMD gfx9">64: AMD gfx9</Translated></button>
          </span>
        </span>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          stride = {s}
          <input type="range" min={0} max={ss.length - 1} value={Math.min(si, ss.length - 1)} aria-label={tr('Stride', 'Stride')}
            onChange={(e) => setSi(Number(e.target.value))} style={{ width: '100%' }} />
        </label>
      </div>
      <svg viewBox={`0 0 ${W} 64`} role="img" aria-label={tr('Warps of the block', 'Các warp của khối')} data-part="warps" style={{ width: '100%', maxWidth: W, display: 'block' }}>
        {ws.map((w) => (
          <g key={w.warp} data-part="warp" data-warp={w.warp} data-state={w.state} data-active={w.activeLanes}>
            <rect x={4 + w.warp * bw} y={8} width={bw - 3} height={30} rx={2} fill="var(--cell)"
              stroke={w.state === 'divergent' ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={w.state === 'divergent' ? 2.2 : 1}
              strokeDasharray={w.state === 'idle' ? '3 2' : undefined} />
            <rect x={4 + w.warp * bw + ((bw - 3) * (w.width - w.activeLanes)) / w.width} y={8} width={((bw - 3) * w.activeLanes) / w.width} height={30} rx={2} fill="var(--ink-3)" opacity={0.55} />
            {ws.length <= 16 || w.warp % 4 === 0 ? (
              <text x={4 + w.warp * bw + (bw - 3) / 2} y={54} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{w.warp}</text>
            ) : null}
          </g>
        ))}
      </svg>
      <Translated
        vi={
          <p className="muted" style={{ margin: '0.2rem 0 0.6rem' }}>
            Tô xám: các lane của warp thỏa phép kiểm tra (các lane thấp bị loại trước). Viền đậm: phân kỳ. Nét đứt: cả warp bỏ
            qua phép cộng.
          </p>
        }
      >
        <p className="muted" style={{ margin: '0.2rem 0 0.6rem' }}>
          Grey fill: the lanes of a warp that pass the test (the low lanes drop out first). Heavy outline: divergent.
          Dashed: the whole warp skips the addition.
        </p>
      </Translated>
      <p style={{ margin: 0 }} data-readout="warps" data-divergent={ws.filter((w) => w.state === 'divergent').length}>
        <Translated vi={<>Tại stride {s}:</>}>At stride {s}:</Translated>{' '}
        {ws.some((w) => w.state === 'divergent')
          ? tr(`warp 0 has ${ws[0].activeLanes} of ${ws[0].width} lanes active, so it runs both paths; every other warp is full.`, `warp 0 có ${ws[0].activeLanes} trên ${ws[0].width} lane hoạt động, nên nó chạy cả hai nhánh; mọi warp khác đều đầy.`)
          : tr(`no warp is divergent: ${ws.filter((w) => w.state === 'idle').length} whole warps skip the addition and the rest are full.`, `không warp nào phân kỳ: ${ws.filter((w) => w.state === 'idle').length} warp nguyên vẹn bỏ qua phép cộng và số còn lại đều đầy.`)}
      </p>
      <p style={{ marginBottom: 0 }} data-readout="loop" data-iterations={d.iterations} data-pairs={d.pairs}>
        <Translated
          vi={
            <>
              Trên toàn bộ vòng lặp ({ss.length} lần lặp), có warp phân kỳ trong <b>{d.iterations}</b> lần lặp: các stride{' '}
              {d.strides.join(', ')}, luôn là warp 0 (tổng cộng {d.pairs} cặp warp-lần lặp phân kỳ).
            </>
          }
        >
          Over the whole loop ({ss.length} iterations), some warp diverges in <b>{d.iterations}</b> iterations: strides{' '}
          {d.strides.join(', ')}, always warp 0 ({d.pairs} divergent warp-iterations in all).
        </Translated>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the race behind barrier 2
function RaceHunt() {
  const [b1, setB1] = useState(true);
  const [b2, setB2] = useState(false);
  const [direct, setDirect] = useState(false);
  const [schedule, setSchedule] = useState<Schedule>('book-race');
  const cfg: Config = useMemo(() => ({ ...FIG_11_3, threads: 8, barrier1: b1, barrier2: b2, temp: !direct }), [b1, b2, direct]);
  const locale = useLocale();
  const trace = useMemo(() => run(cfg, EXERCISE_1, schedule), [cfg, schedule, locale]);
  return (
    <Bench
      title={tr('The race behind barrier 2', 'Race condition đằng sau rào chắn 2')}
      task={tr('Barrier 2 starts removed. Press Go to the first failure, then try the other thread orders.', 'Rào chắn 2 bị bỏ từ đầu. Bấm Tới lỗi đầu tiên, rồi thử các thứ tự luồng khác.')}
      foot={
        <Translated
          vi={
            <>
              Fig. 11.3 trên một khối 8 luồng, với các giá trị của bài tập 1. Kernel hiển thị là mã của biến thể bạn chọn; bốn
              biến thể có tên nằm trong labs/ch11/snippets và chạy trên trình giả lập (emulator) của khóa học với cùng kết quả.
            </>
          }
        >
          <>
            Fig. 11.3 on one block of 8 threads, exercise 1's values. The kernel shown is the text of the variant you
            picked; the four named variants are in labs/ch11/snippets and run on the course's emulator with the same
            results.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="check"><input type="checkbox" checked={b1} onChange={(e) => setB1(e.target.checked)} /> <Translated vi="rào chắn 1 (dòng 10)">barrier 1 (line 10)</Translated></label>
        <label className="check"><input type="checkbox" checked={b2} onChange={(e) => setB2(e.target.checked)} /> <Translated vi="rào chắn 2">barrier 2</Translated> {direct ? tr('(after the store)', '(sau lần ghi)') : tr('(line 14)', '(dòng 14)')}</label>
        <label className="check"><input type="checkbox" checked={direct} onChange={(e) => setDirect(e.target.checked)} /> <Translated vi="cộng thẳng vào buffer_s, không dùng temp">add straight into buffer_s, no temp</Translated></label>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Thứ tự luồng">Thread order</Translated>
          <select value={schedule} onChange={(e) => setSchedule(e.target.value as Schedule)} aria-label={tr('Thread order', 'Thứ tự luồng')}>
            {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
          </select>
        </label>
      </div>
      <TraceView trace={trace} />
    </Bench>
  );
}

const VARIANTS: { label: string; over: Partial<Config> }[] = [
  {
    get label() {
      return tr('Fig. 11.3 as printed', 'Fig. 11.3 như bản in');
    },
    over: {},
  },
  {
    get label() {
      return tr('barrier 2 removed', 'bỏ rào chắn 2');
    },
    over: { barrier2: false },
  },
  {
    get label() {
      return tr('barrier 1 removed', 'bỏ rào chắn 1');
    },
    over: { barrier1: false },
  },
  {
    get label() {
      return tr('straight into buffer_s, both barriers', 'thẳng vào buffer_s, đủ cả hai rào chắn');
    },
    over: { temp: false },
  },
];
function Outcomes() {
  return (
    <Bench
      title={tr('Every thread order at once', 'Mọi thứ tự luồng cùng lúc')}
      task={tr('Before reading the table: which order will hide every bug?', 'Trước khi đọc bảng: thứ tự nào sẽ che giấu mọi lỗi?')}
      foot={tr('Each cell runs the whole block in the model above and compares the output with the scan of exercise 1.', 'Mỗi ô chạy cả khối trong mô hình ở trên và so đầu ra với scan của bài tập 1.')}
    >
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-part="outcomes">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}><Translated vi="Kernel">Kernel</Translated></th>
              {SCHEDULES.map((s) => <th key={s}>{SCHEDULE_LABEL[s]}</th>)}
            </tr>
          </thead>
          <tbody>
            {VARIANTS.map((v) => (
              <tr key={v.label}>
                <td style={{ paddingLeft: '0.6rem' }}>{v.label}</td>
                {SCHEDULES.map((s) => {
                  const r = run({ ...FIG_11_3, threads: 8, ...v.over }, EXERCISE_1, s);
                  const kinds = [...new Set(r.violations.map((x) => x.kind))];
                  return (
                    <td key={s} data-outcome={r.correct ? 'correct' : 'wrong'} data-wrong={r.wrong.length}>
                      {r.correct ? <span className="chip ok"><Translated vi="đúng">correct</Translated></span> : <span className="chip fail"><Translated vi={<>{r.wrong.length} trên 8 sai</>}>{r.wrong.length} of 8 wrong</Translated></span>}
                      {kinds.map((k) => <div key={k} className="muted">{KIND_LABEL[k]}</div>)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS112: Record<string, Question> = {
  'ch11/11.2/q-naive-work': {
    kind: 'numeric',
    id: 'ch11/11.2/q-naive-work',
    prompt: (
      <Translated
        vi={
          <>
            Scan song song ngây thơ giao cho mỗi luồng trong 1024 luồng một đầu ra riêng, và mỗi luồng cộng dồn tiền tố của mình
            bằng một vòng lặp tuần tự. Tổng cộng các luồng làm bao nhiêu phép cộng?
          </>
        }
      >
        <>
          The naive parallel scan gives each of 1024 threads its own output, and each thread adds up its prefix with a
          sequential loop. How many additions do the threads do in total?
        </>
      </Translated>
    ),
    answer: naiveAdditions(1024),
    unit: 'additions',
    vi: { unit: 'phép cộng' },
    explain: (
      <Translated
        vi={
          <p>
            Luồng <i>i</i> làm <i>i</i> phép cộng, nên tổng là 0 + 1 + … + 1023 = 1024·1023/2 = 523,776, so với 1023 của vòng
            lặp. Và nó thậm chí không nhanh hơn: riêng luồng cuối đã làm 1023 phép cộng liên tiếp, nên độ dài đường găng (span)
            bằng của vòng lặp.
          </p>
        }
      >
        <p>
          Thread <i>i</i> does <i>i</i> additions, so the total is 0 + 1 + … + 1023 = 1024·1023/2 = 523,776, against 1023
          for the loop. And it is not even faster: the last thread alone does 1023 additions in a row, so the span is the
          loop's.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.2, p. 254',
  },
  'ch11/11.2/q-ex1': {
    kind: 'numeric',
    id: 'ch11/11.2/q-ex1',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 1: Kogge-Stone trên 4 6 7 1 2 8 5 2. Sau bước stride 2, giá trị ở vị trí 5 là bao nhiêu? (Vị trí đếm từ
            0.)
          </>
        }
      >
        <>
          Exercise 1: Kogge-Stone on 4 6 7 1 2 8 5 2. Which value is at position 5 after the stride-2 step? (Positions count
          from 0.)
        </>
      </Translated>
    ),
    answer: levels(EXERCISE_1)[2].values[5],
    explain: (
      <Translated
        vi={
          <p>
            Sau stride 1: 4 10 13 8 3 10 13 7. Sau stride 2, mỗi vị trí cộng thêm giá trị cách nó hai ô về bên trái: vị trí 5 =
            10 + 8 = 18, tính x2 đến x5 (7 + 1 + 2 + 8). Sau stride 4, mảng là 4 10 17 18 20 28 33 35.
          </p>
        }
      >
        <p>
          After stride 1: 4 10 13 8 3 10 13 7. After stride 2 each position adds the value two to its left: position 5 =
          10 + 8 = 18, which counts x2 to x5 (7 + 1 + 2 + 8). After stride 4 the array is 4 10 17 18 20 28 33 35.
        </p>
      </Translated>
    ),
    source: 'ch. 11 exercise 1, p. 286',
    vi: { source: 'ch. 11 bài tập 1, p. 286' },
  },
  'ch11/11.2/q-final': {
    kind: 'numeric',
    id: 'ch11/11.2/q-final',
    prompt: (
      <Translated vi={<>Một scan Kogge-Stone 16 phần tử đã chạy các stride 1, 2 và 4. Bao nhiêu vị trí giữ giá trị cuối cùng?</>}>
        <>A 16-element Kogge-Stone scan has run the strides 1, 2 and 4. How many positions hold their final value?</>
      </Translated>
    ),
    answer: finalCount(16, 3),
    unit: 'positions',
    vi: { unit: 'vị trí' },
    explain: (
      <Translated
        vi={
          <p>
            Sau <i>k</i> bước, vị trí <i>i</i> chứa tối đa 2<sup>k</sup> đầu vào kết thúc tại <i>i</i>. Các vị trí 0 đến 7 có
            tối đa 8 đầu vào cần tính, nên sau 3 bước chúng đã đầy đủ; các vị trí 8 đến 15 còn chờ bước stride 8.
          </p>
        }
      >
        <p>
          After <i>k</i> steps position <i>i</i> holds up to 2<sup>k</sup> inputs ending at <i>i</i>. Positions 0 to 7 have
          at most 8 inputs to count, so after 3 steps they are complete; positions 8 to 15 still wait for the stride-8 step.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.2, pp. 255–256',
  },
  'ch11/11.2/q-race': {
    kind: 'choice',
    id: 'ch11/11.2/q-race',
    prompt: (
      <Translated
        vi={
          <>
            Rào chắn 2 bị bỏ và, trong lần lặp stride 2, luồng 4 ghi giá trị mới trước khi luồng 6 đọc{' '}
            <code>buffer_s[4]</code>. Cuối cùng vị trí 6 tính hai lần những đầu vào nào?
          </>
        }
      >
        <>
          Barrier 2 is removed and, in the stride-2 iteration, thread 4 stores its new value before thread 6 reads{' '}
          <code>buffer_s[4]</code>. Which inputs does position 6 count twice at the end?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="x1 và x2">x1 and x2</Translated>, correct: true },
      {
        label: <Translated vi="x3 và x4">x3 and x4</Translated>,
        why: <Translated vi="Đó là những gì luồng 6 lẽ ra phải đọc. Nó đã đọc nhiều hơn thế.">Those are what thread 6 should have read. It read more than that.</Translated>,
      },
      {
        label: <Translated vi="x0, x1 và x2">x0, x1 and x2</Translated>,
        why: <Translated vi="x0 chỉ đến một lần, ở bước stride 4.">x0 arrives once, in the stride-4 step.</Translated>,
      },
      {
        label: <Translated vi="không cái nào: luồng 6 chỉ mất x3 và x4">none: thread 6 just loses x3 and x4</Translated>,
        why: <Translated vi="Nó vẫn nhận x3 và x4, nằm bên trong giá trị lớn hơn mà luồng 4 đã ghi.">It still gets x3 and x4, inside the larger value thread 4 stored.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Luồng 6 lẽ ra phải cộng x3..x4 vào x5..x6 của nó. Thay vào đó nó gặp x1..x4 mới của luồng 4 và ghi x1..x6. Bước
            stride 4 sau đó cộng vị trí 2, đang chứa x0..x2, nên x1 và x2 đến lần thứ hai: x0 + 2x1 + 2x2 + x3 + x4 + x5 + x6.
            Sách in kết quả thiếu x0, có vẻ là một sơ suất; còn ý chính, x1 và x2 bị tính hai lần, vẫn đúng.
          </p>
        }
      >
        <p>
          Thread 6 should add x3..x4 to its x5..x6. It finds thread 4's new x1..x4 instead and stores x1..x6. The stride-4
          step then adds position 2, which holds x0..x2, so x1 and x2 arrive a second time: x0 + 2x1 + 2x2 + x3 + x4 + x5 +
          x6. The book prints the result without x0, which looks like a slip; the point, x1 and x2 counted twice, stands.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.2, pp. 257–258; the missing x0: notes/ch11.md, "Unclear in the PDF"',
    vi: { source: 'ch. 11 §11.2, pp. 257–258; x0 bị thiếu: notes/ch11.md, "Unclear in the PDF"' },
  },
  'ch11/11.2/q-ex2-64': {
    kind: 'numeric',
    id: 'ch11/11.2/q-ex2-64',
    prompt: (
      <Translated
        vi={
          <>
            Một khối 256 luồng chạy Fig. 11.3 trên một thiết bị AMD với wavefront rộng 64. Trong bao nhiêu lần lặp của vòng lặp
            có wavefront bị phân kỳ?
          </>
        }
      >
        <>
          A 256-thread block runs Fig. 11.3 on an AMD device with 64-wide wavefronts. In how many iterations of the loop is
          some wavefront divergent?
        </>
      </Translated>
    ),
    answer: divergence(256, 64).iterations,
    unit: 'iterations',
    vi: { unit: 'lần lặp', source: 'ch. 11 bài tập 2, p. 286; độ rộng wavefront: notes/today/ch05-facts.md B5, ch10-facts B2' },
    explain: (
      <Translated
        vi={
          <p>
            Wavefront 0 bị chia bởi <code>threadIdx.x &gt;= stride</code> khi stride còn dưới 64: các stride 1, 2, 4, 8, 16 và
            32, sáu lần lặp. Từ stride 64 trở đi, cả wavefront bị loại cùng lúc. Với warp rộng 32 của sách thì là năm lần lặp,
            các stride đến 16 mà bài tập 2 nêu.
          </p>
        }
      >
        <p>
          Wavefront 0 is split by <code>threadIdx.x &gt;= stride</code> while the stride is below 64: strides 1, 2, 4, 8,
          16 and 32, six iterations. From stride 64 on, whole wavefronts drop out together. With the book's 32-wide warps it
          is five iterations, the strides up to 16 that exercise 2 names.
        </p>
      </Translated>
    ),
    source: 'ch. 11 exercise 2, p. 286; wavefront width: notes/today/ch05-facts.md B5, ch10-facts B2',
  },
  'ch11/11.2/q-ex3': {
    kind: 'numeric',
    id: 'ch11/11.2/q-ex3',
    prompt: (
      <Translated vi={<>Bài tập 3: kernel của Fig. 11.3 thực hiện bao nhiêu phép cộng trên 2048 phần tử?</>}>
        <>Exercise 3: how many additions does the kernel of Fig. 11.3 perform on 2048 elements?</>
      </Translated>
    ),
    answer: totalAdditions(2048),
    unit: 'additions',
    vi: { unit: 'phép cộng', source: 'ch. 11 bài tập 3, p. 286; p. 266; hai cách hiểu: notes/ch11.md, exercises' },
    explain: (
      <Translated
        vi={
          <p>
            Coi như một đoạn 2048 phần tử, công thức ở p. 266 cho 2048·11 − 2047 = 20,481. Một lần khởi chạy thật không làm thế
            được: một khối chứa tối đa 1024 luồng, nên đầu vào thành hai khối 1024, làm 2·(1024·10 − 1023) = 18,434 phép cộng
            (và để lại hai scan riêng cho mục 11.9 ghép lại). Bài tập không nói nó muốn cách nào.
          </p>
        }
      >
        <p>
          As one 2048-element segment, p. 266's formula gives 2048·11 − 2047 = 20,481. A real launch cannot do that: one
          block holds at most 1024 threads, so the input becomes two blocks of 1024, which do 2·(1024·10 − 1023) = 18,434
          additions (and leave two separate scans for section 11.9 to stitch). The exercise does not say which it means.
        </p>
      </Translated>
    ),
    source: 'ch. 11 exercise 3, p. 286; p. 266; the two readings: notes/ch11.md, exercises',
  },
  'ch11/11.2/q-why-not-reduction': {
    kind: 'choice',
    id: 'ch11/11.2/q-why-not-reduction',
    prompt: (
      <Translated vi={<>Vì sao phép rút gọn tại chỗ của chương 10 chỉ cần một rào chắn mỗi lần lặp, còn scan này cần hai?</>}>
        <>Why did chapter 10's in-place reduction need one barrier per iteration, while this scan needs two?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Khác với phép rút gọn, trong Kogge-Stone một luồng đang hoạt động đọc một ô mà một luồng hoạt động khác ghi đè trong cùng lần lặp.">
            Unlike the reduction, in Kogge-Stone an active thread reads a slot another active thread overwrites in the same iteration.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Phép rút gọn giữ các tổng riêng phần trong thanh ghi giữa các lần lặp; scan giữ chúng trong bộ nhớ chia sẻ.">
            The reduction kept its partial sums in registers between iterations; the scan keeps them in shared memory.
          </Translated>
        ),
        why: (
          <Translated vi="Không bên nào giữ tổng riêng phần trong thanh ghi: cả hai đều làm tại chỗ trong một mảng mà mọi luồng đọc (ở đây là bộ nhớ chia sẻ, còn ở Figs. 10.6 và 10.7 là bộ nhớ toàn cục).">
            Neither keeps its partial sums in registers: both work in place in an array every thread reads (shared memory here, global memory in Figs. 10.6 and 10.7).
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Scan giữ mọi luồng hoạt động ở mọi lần lặp, còn phép rút gọn giảm một nửa mỗi lần.">
            The scan keeps all of its threads active in every iteration, where the reduction halves them each time.
          </Translated>
        ),
        why: (
          <Translated vi="Kogge-Stone cũng tắt các luồng dưới stride. Điều quan trọng là các luồng hoạt động dùng chung những ô nào.">
            Kogge-Stone switches off the threads below the stride too. What matters is which slots the active threads share.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Phép rút gọn cũng cần hai; chương 10 bỏ một cái vì race condition đó hiếm khi lộ ra.">
            The reduction needed two as well; chapter 10 left one out because that race rarely shows.
          </Translated>
        ),
        why: (
          <Translated vi="Chương 10 đúng: hãy xem các luồng hoạt động của nó đọc và ghi những ô nào.">
            It was right: check which slots its active threads read and write.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong phép rút gọn hội tụ, mỗi luồng hoạt động đọc ô của chính nó và một ô cách <i>stride</i> về bên phải, và không
            luồng hoạt động nào ghi các ô bên phải đó trong lần lặp ấy. Trong Kogge-Stone, luồng <i>i</i> đọc ô{' '}
            <i>i</i> − stride, mà luồng <i>i</i> − stride sắp ghi đè: nhiều cây rút gọn chạy cùng lúc và dùng tổng riêng phần
            của nhau. Sự tái sử dụng đó làm nó nhanh, và cũng làm rào chắn thứ hai trở nên cần thiết.
          </p>
        }
      >
        <p>
          In the convergent reduction each active thread reads its own slot and one <i>stride</i> to the right, and no
          active thread writes those right-hand slots in that iteration. In Kogge-Stone thread <i>i</i> reads slot{' '}
          <i>i</i> − stride, which thread <i>i</i> − stride is about to overwrite: many reduction trees run at once and use
          each other's partial sums. That reuse is what makes it fast, and what makes the second barrier necessary.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.2, pp. 257–258; ch. 10 Figs. 10.6 and 10.7',
    vi: { source: 'ch. 11 §11.2, pp. 257–258; ch. 10 Figs. 10.6 và 10.7' },
  },
};

// ---------------------------------------------------------------- the section
export function S112() {
  const file = useLabFile('ch11/snippets/s112_kogge_stone.cu');
  const fn = file ? functionSource(file, 'kogge_stone_scan_kernel') : null;
  const lines = kernelSource(FIG_11_3).line;
  const q = questionsS112;
  return (
    <>
      <Translated
        vi={
          <p>
            Cách hiển nhiên để scan song song là mỗi đầu ra một luồng, mỗi luồng cộng dồn tiền tố của mình bằng một vòng lặp.
            Mọi đầu ra được tính cùng lúc, nhưng luồng cuối vẫn phải cộng N giá trị liên tiếp, nên scan mất thời gian ngang vòng
            lặp tuần tự, và các luồng cộng lại làm N(N − 1)/2 phép cộng thay vì N − 1. Thêm phần cứng mà không nhanh hơn.
          </p>
        }
      >
        <p>
          The obvious way to scan in parallel is one thread per output, each adding up its own prefix with a loop. Every
          output is computed at once, but the last thread still adds N values in a row, so the scan takes as long as the
          sequential loop, and the threads together do N(N − 1)/2 additions instead of N − 1. More hardware, no speedup.
        </p>
      </Translated>
      <Source section="11.2" pages="254" />
      <Ask q={q['ch11/11.2/q-naive-work']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Một khởi đầu tốt hơn là ý tưởng của chương 10: tính mỗi đầu ra bằng một cây rút gọn, trong log₂ bước. Điều đó cần một
            toán tử có tính kết hợp (<a href={href('ch10', '10.2')}>mục 10.2</a>), và cả giao hoán nếu cây sắp xếp lại các
            toán hạng. Nhưng một cây riêng cho đầu ra <i>i</i> vẫn làm <i>i</i> phép cộng, nên tổng vẫn là bậc hai, trừ khi các
            cây dùng chung tổng riêng phần.
          </p>
        }
      >
        <p>
          A better start is chapter 10's idea: compute each output with a reduction tree, in log₂ steps. That needs an
          associative operator (<a href={href('ch10', '10.2')}>section 10.2</a>), and commutative too if the tree reorders
          operands. But a separate tree for output <i>i</i> still does <i>i</i> additions, so the total stays quadratic, unless
          the trees share their partial sums.
        </p>
      </Translated>
      <Source section="11.2" pages="254–255" />

      <h2><Translated vi="Dùng chung tổng riêng phần: Kogge-Stone">Sharing partial sums: Kogge-Stone</Translated></h2>
      <Translated
        vi={
          <p>
            <b>Kogge-Stone</b> là một sơ đồ dùng chung như vậy, được phát minh vào thập niên 1970 cho các mạch cộng nhanh. Nó làm
            tại chỗ: mảng bắt đầu là các đầu vào và biến thành các đầu ra. Ở bước <i>k</i> (đếm từ 0), mọi vị trí <i>i</i>{' '}
            cách đầu mảng ít nhất 2<sup>k</sup> cộng thêm giá trị cách nó 2<sup>k</sup> về bên trái, tất cả cùng lúc. Sau{' '}
            <i>k</i> bước, vị trí <i>i</i> chứa tổng của tối đa 2<sup>k</sup> đầu vào kết thúc tại <i>i</i>, và các vị trí 0
            đến 2<sup>k</sup> − 1 đã xong.
          </p>
        }
      >
        <p>
          <b>Kogge-Stone</b> is such a sharing scheme, invented in the 1970s for fast adder circuits. It works in place: the
          array starts as the inputs and turns into the outputs. In step <i>k</i> (counting from 0), every position <i>i</i>{' '}
          at least 2<sup>k</sup> from the start adds the value 2<sup>k</sup> to its left, all of them at once. After{' '}
          <i>k</i> steps position <i>i</i> holds the sum of up to 2<sup>k</sup> inputs ending at <i>i</i>, and positions 0
          to 2<sup>k</sup> − 1 are final.
        </p>
      </Translated>
      <Source section="11.2" pages="255–256" figs="Fig. 11.2" />
      <StrideStepper />
      <Ask q={q['ch11/11.2/q-ex1']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={q['ch11/11.2/q-final']} />
      <Translated
        vi={
          <p>
            Chọn y15 trong bộ chọn cây với mười sáu giá trị: 15 phép cộng của nó đúng là một cây rút gọn trên x0 đến x15, các nét
            đậm của Fig. 11.2. Cây của các đầu ra khác chạy cùng lúc và dùng chung những ô đó. Sự dùng chung ấy đưa tổng xuống
            N·log₂N − (N − 1) phép cộng, 49 với 16 phần tử. Cảnh bên dưới cho thấy sự tăng trưởng đó, và điều gì hỏng khi thời
            điểm bị lệch.
          </p>
        }
      >
        <p>
          Pick y15 in the tree selector on the sixteen values: its 15 additions are exactly a reduction tree over x0 to
          x15, the heavy lines of Fig. 11.2. The other outputs' trees run at the same time and share those boxes. That
          sharing brings the total down to N·log₂N − (N − 1) additions, 49 for 16 elements. The scene below shows the
          growth, and what goes wrong when the timing slips.
        </p>
      </Translated>
      <NoteGroup>
        <Aside>
          <Translated
            vi={
              <p>
                Phần văn của sách đếm lần lặp từ 1 ("iteration 2" có stride 2), còn các hình đếm từ 0 ("Step 1 (stride = 2)"). Các
                trang này đếm từ 0, như mã nguồn, và gọi tên một bước theo stride của nó khi cần.
              </p>
            }
          >
            <p>
              The book's prose counts iterations from 1 ("iteration 2" has stride 2), its figures from 0 ("Step 1 (stride =
              2)"). These pages count from 0, as the code does, and name a step by its stride when it matters.
            </p>
          </Translated>
        </Aside>
        <Simplification>
          <Translated
            vi={
              <p>
                Cảnh này vẽ theo mô hình của trình chạy từng bước: mọi vị trí của một bước đọc các giá trị từ trước bước đó, theo
                lockstep. Race condition của nó là một lịch chạy duy nhất, của sách, với rào chắn 2 bị bỏ; không có warp, không có
                thời gian.
              </p>
            }
          >
            <p>
              The scene draws the stepper's model: every position of a step reads the values from before it, in lockstep.
              Its race is one schedule, the book's, with barrier 2 removed; no warps, no timing.
            </p>
          </Translated>
        </Simplification>
      </NoteGroup>
      <KsDoublingScene />

      <h2><Translated vi="Một khối, một đoạn">One block, one segment</Translated></h2>
      <Translated
        vi={
          <p>
            Fig. 11.3 là kernel đầu tiên. Mỗi khối scan đoạn riêng gồm <code>SEG_SIZE</code> phần tử với đúng{' '}
            <code>SEG_SIZE</code> luồng, mỗi phần tử một luồng, trong một buffer chia sẻ; việc ghép các đoạn thành một scan phải
            chờ đến mục 11.9. Các luồng ngoài phạm vi nạp 0, phần tử đơn vị, nên chúng không thay đổi gì. Mỗi luồng giữ phần tử
            của mình biến đổi qua vòng lặp, rồi ghi nó ra.
          </p>
        }
      >
        <p>
          Fig. 11.3 is the first kernel. Each block scans its own segment of <code>SEG_SIZE</code> elements with exactly{' '}
          <code>SEG_SIZE</code> threads, one per element, in a shared buffer; stitching the segments into one scan waits
          until section 11.9. Out-of-range threads load 0, the identity, so they change nothing. Each thread keeps its
          element evolving through the loop, then writes it out.
        </p>
      </Translated>
      <Source section="11.2" pages="256–257" figs="Fig. 11.3" />
      {fn ? (
        <Code
          tabs={[{ label: 'CUDA', code: fn.text, status: tr('emulator-checked and front-end-checked; unverified on CUDA and HIP hardware', 'emulator-checked và front-end-checked; unverified trên phần cứng CUDA và HIP') }]}
          highlight={[...(lines.barrier1 ?? []), ...(lines.barrier2 ?? [])]}
          caption={
            <Translated
              vi={
                <>
                  Fig. 11.3 từ labs/ch11/snippets/s112_kogge_stone.cu, đánh số dòng như trên trang sách. Dòng 14 được in là{' '}
                  <code>__ syncthreads();</code> trong sách, với một dấu cách khiến nó không biên dịch được; ở đây đã sửa. Sách gọi
                  kernel là <code>scan_kernel</code>.
                </>
              }
            >
              <>
                Fig. 11.3 from labs/ch11/snippets/s112_kogge_stone.cu, numbered as on the page. Line 14 is printed{' '}
                <code>__ syncthreads();</code> in the book, with a space that would not compile; it is fixed here. The book
                calls the kernel <code>scan_kernel</code>.
              </>
            </Translated>
          }
        />
      ) : null}
      <Translated
        vi={
          <p>
            Vòng lặp có hai rào chắn, và chúng làm hai việc khác nhau. Rào chắn thứ nhất, dòng 10, làm mọi lần ghi của lần lặp
            trước hiển thị trước khi ai đó đọc: cùng loại rào chắn như trong các kernel rút gọn của chương 10. Rào chắn thứ hai,
            dòng 14, buộc mọi lần đọc của lần lặp này xong trước khi ai đó ghi đè: đó là lý do mỗi luồng cộng vào một biến{' '}
            <code>temp</code> riêng trước và chỉ ghi nó sau rào chắn.
          </p>
        }
      >
        <p>
          The loop has two barriers, and they do different jobs. The first, line 10, makes every write of the previous
          iteration visible before anyone reads: the same barrier as in the reduction kernels of chapter 10. The second,
          line 14, makes every read of this iteration finish before anyone overwrites: that is why each thread adds into a
          private <code>temp</code> first and stores it only after the barrier.
        </p>
      </Translated>
      <Source section="11.2" pages="257" />
      <Translated
        vi={
          <p>
            HIP chấp nhận <code>__shared__</code> và <code>__syncthreads()</code> nguyên văn, và header của khóa học ánh xạ các
            lời gọi runtime, nên cùng kernel đó lẽ ra build được cho HIP (unverified: nó chưa được build hay chạy trên phần cứng
            AMD). Trên Metal, buffer là một mảng <code>threadgroup</code> khai báo trong kernel và rào chắn là{' '}
            <code>threadgroup_barrier(mem_flags::mem_threadgroup)</code>; lab 11.1 có cả hai phiên bản.
          </p>
        }
      >
        <p>
          HIP accepts <code>__shared__</code> and <code>__syncthreads()</code> as written, and the course's header maps the
          runtime calls, so the same kernel should build for HIP (unverified: it has not been built or run on AMD hardware).
          On Metal the buffer is a <code>threadgroup</code> array declared in the kernel and
          the barrier is <code>threadgroup_barrier(mem_flags::mem_threadgroup)</code>; lab 11.1 has both versions.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch05-facts C1, C2; ch11-facts C2)', '2026-11-20 (ch05-facts C1 và C2; ch11-facts C2)')}>
        <Translated
          vi={
            <p>
              MSL chỉ cho phép biến <code>threadgroup</code> trong các hàm kernel; một hàm phụ trợ thì nhận một con trỏ{' '}
              <code>threadgroup</code> thay vào đó.
            </p>
          }
        >
          <p>
            MSL allows <code>threadgroup</code> variables only in kernel functions; a helper takes a{' '}
            <code>threadgroup</code> pointer instead.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Phân kỳ chỉ ở mức vừa phải">Divergence is modest</Translated></h2>
      <Translated
        vi={
          <p>
            Phép kiểm tra <code>threadIdx.x &gt;= stride</code> tắt các luồng từ đầu khối. Khi stride còn nhỏ hơn một warp, các
            luồng bị tắt là một số lane của warp 0, nên chỉ warp 0 chạy cả hai nhánh. Khi stride đạt độ rộng warp, cả warp bị
            loại cùng lúc và không có gì phân kỳ. Trong một khối nhiều warp, điều đó hầu như không đáng kể.
          </p>
        }
      >
        <p>
          The test <code>threadIdx.x &gt;= stride</code> switches threads off from the front of the block. While the stride
          is smaller than a warp, the threads switched off are some of warp 0's lanes, so only warp 0 runs both paths. Once
          the stride reaches the warp width, whole warps drop out together and nothing diverges. In a block of many warps
          that is hardly felt.
        </p>
      </Translated>
      <Source section="11.2" pages="257" />
      <Translated
        vi={
          <p>
            Bài tập 2 yêu cầu bạn chứng minh điều đó với warp 32. Kiểm tra trong trình khám phá, rồi đặt độ rộng thành 64,
            wavefront của các thiết bị gfx9 của AMD, và đếm lại.
          </p>
        }
      >
        <p>
          Exercise 2 asks you to show it for warps of 32. Check it in the explorer, then set the width to 64, the
          wavefront of AMD's gfx9 devices, and count again.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Trình khám phá cho biết lane nào hoạt động, không cho biết một warp phân kỳ tốn bao nhiêu (ch04-facts A9, B7, C4). Độ
              rộng SIMD group của Metal được đọc từ pipeline, không có trong tài liệu (ch04-facts C2).
            </p>
          }
        >
          <p>
            The explorer says which lanes are active, not what a divergent warp costs (ch04-facts A9, B7, C4). Metal's
            SIMD-group width is read from the pipeline, not documented (ch04-facts C2).
          </p>
        </Translated>
      </Simplification>
      <WarpExplorer />
      <Ask q={q['ch11/11.2/q-ex2-64']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Vì sao scan cần rào chắn 2">Why the scan needs barrier 2</Translated></h2>
      <Translated
        vi={
          <p>
            Phép rút gọn của chương 10 chỉ cần một rào chắn mỗi lần lặp (<a href={href('ch10', '10.3')}>mục 10.3</a>). Ở đó,
            những ô mà một luồng hoạt động đọc ở khoảng cách <i>stride</i> không bao giờ bị một luồng hoạt động khác ghi trong
            cùng lần lặp. Trong Kogge-Stone thì có. Luồng <i>i</i> đọc ô <i>i</i> − stride trong khi luồng <i>i</i> − stride
            đang tính một giá trị mới cho ô đó, và nếu luồng kia ghi trước, luồng <i>i</i> cộng nhầm.
          </p>
        }
      >
        <p>
          Chapter 10's reduction got by with one barrier per iteration (<a href={href('ch10', '10.3')}>section 10.3</a>).
          There, the slots an active thread reads at distance <i>stride</i> are never written by another active thread in
          the same iteration. In Kogge-Stone they are. Thread <i>i</i> reads slot <i>i</i> − stride while thread <i>i</i> −
          stride is computing a new value for it, and if that thread stores first, thread <i>i</i> adds the wrong thing.
        </p>
      </Translated>
      <Source section="11.2" pages="257–258" />
      <Translated
        vi={
          <p>
            Ví dụ của sách: trong lần lặp stride 2, luồng 6 phải cộng <code>buffer_s[4]</code> cũ, x3 + x4, vào x5 + x6 của nó.
            Nếu luồng 4 ghi x1 + … + x4 trước, luồng 6 ghi x1 + … + x6, và bước stride 4 cộng thêm x0 + x1 + x2, nên x1 và x2
            bị tính hai lần. Nếu luồng 6 tình cờ đọc trước, kết quả đúng. Kernel đúng hay sai phụ thuộc vào thời điểm, và có thể
            đổi giữa các lần chạy; đó là điều khiến những lỗi như vậy khó tìm.
          </p>
        }
      >
        <p>
          The book's example: in the stride-2 iteration, thread 6 must add the old <code>buffer_s[4]</code>, x3 + x4, to its
          own x5 + x6. If thread 4 stores x1 + … + x4 first, thread 6 stores x1 + … + x6, and the stride-4 step adds x0 + x1
          + x2 on top, so x1 and x2 are counted twice. If thread 6 happens to read first, the answer is right. Whether the
          kernel is correct depends on timing, and can change from run to run; that is what makes such bugs hard to find.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Sách in giá trị bị hỏng là 2x1 + 2x2 + x3 + x4 + x5 + x6. Tính kỹ thì bước stride 4 còn cộng x0, nên đúng ra phải là
              x0 + 2x1 + 2x2 + x3 + x4 + x5 + x6 (notes, "Unclear in the PDF"). Mô hình bên dưới hiển thị biểu thức đầy đủ.
            </p>
          }
        >
          <p>
            The book prints the corrupted value as 2x1 + 2x2 + x3 + x4 + x5 + x6. Working it through, the stride-4 step also
            adds x0, so it should read x0 + 2x1 + 2x2 + x3 + x4 + x5 + x6 (notes, "Unclear in the PDF"). The model below
            shows the full expression.
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Cuộc truy lùng bên dưới chạy Fig. 11.3 từng luồng một trên một khối 8 luồng. Rào chắn 2 bị bỏ từ đầu và thứ tự là
            của sách: luồng 4 chạy trước ở bước stride 2. Mỗi giá trị mang theo các đầu vào mà nó tính, nên bạn có thể thấy x1 và
            x2 đến hai lần.
          </p>
        }
      >
        <p>
          The hunt below runs Fig. 11.3 thread by thread on one block of 8 threads. Barrier 2 starts removed and the order
          is the book's: thread 4 runs ahead in the stride-2 step. Every value carries the inputs it counts, so you can see
          x1 and x2 arrive twice.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi luồng tiến một thao tác ở mức mã nguồn mỗi lần, không có warp và không có thời gian; phép cộng thẳng vào{' '}
              <code>buffer_s</code> được tách thành các lần nạp và lần ghi. Bộ nhớ chia sẻ chưa khởi tạo là NaN. "Lockstep" là
              một warp đơn lý tưởng hóa.
            </p>
          }
        >
          <p>
            A thread advances one source-level operation at a time, with no warps and no timing; the add straight into{' '}
            <code>buffer_s</code> is split into its loads and its store. Uninitialised shared memory is NaN. "Lockstep" is
            an idealised single warp.
          </p>
        </Translated>
      </Simplification>
      <RaceHunt />
      <Translated
        vi={
          <p>
            Ba điều rút ra từ cuộc truy lùng. Khi bỏ rào chắn 2, thứ tự của sách hỏng đúng như mô tả: chỉ vị trí 6 sai. "Luồng
            cuối chạy trước" tình cờ đúng, vì mọi lần đọc đều đến trước lần ghi có thể làm hỏng nó. Và lockstep che giấu mọi lỗi
            loại này, đó là lý do một bài kiểm thử qua được trên GPU chứng minh rất ít. Rào chắn 1 bảo vệ một phụ thuộc khác: bỏ
            nó thay vào đó, và thứ tự đảo ngược đọc những ô chưa ai ghi.
          </p>
        }
      >
        <p>
          Three things to take from the hunt. With barrier 2 removed, the book's order fails exactly as described: only
          position 6 is wrong. "Last thread first" happens to be right, because every read comes before the store that
          would spoil it. And lockstep hides every one of these bugs, which is why a test on a GPU that passes proves little.
          Barrier 1 guards a different dependence: remove it instead, and the reversed order reads slots nobody has written
          yet.
        </p>
      </Translated>
      <Outcomes />
      <Ask q={q['ch11/11.2/q-race']} />
      <Ask q={q['ch11/11.2/q-why-not-reduction']} />
      <Translated
        vi={
          <p>
            Nguyên nhân gốc rễ chính là sự dùng chung. Kogge-Stone chạy nhiều cây rút gọn cùng lúc và cho chúng tái sử dụng tổng
            riêng phần của nhau, nên giá trị mà một luồng sắp thay thế lại là đầu vào cho cây của luồng khác. Mục 11.5 đếm xem
            sự dùng chung đó tiết kiệm được bao nhiêu, và mục 11.3 bỏ rào chắn 2 bằng một con đường khác.
          </p>
        }
      >
        <p>
          The root cause is the sharing itself. Kogge-Stone runs many reduction trees at once and lets them reuse each
          other's partial sums, so the value one thread is about to replace is an input to another thread's tree. Section
          11.5 counts what that sharing saves, and section 11.3 removes barrier 2 by a different route.
        </p>
      </Translated>
      <Source section="11.2" pages="258" />
      <Ask q={q['ch11/11.2/q-ex3']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Cùng race condition đó trong mã production">The same race in production code</Translated></h2>
      <Translated
        vi={
          <p>
            Kernel scan của PyTorch cho GPU Apple mang một tổng chạy từ khúc này sang khúc kế tiếp của một hàng dài qua một mảng
            chia sẻ nhỏ. Mã nguồn của nó có một rào chắn ngay sau khi mọi luồng đọc giá trị mang sang đó, kèm một chú thích gọi
            đúng tên lỗi này: không có nó, các lần ghi của khúc kế tiếp có thể ghi đè giá trị mang sang trước khi mọi luồng đọc
            xong. (Ví dụ này là của chúng tôi; sách không xem xét các backend thật.) Lab 11.3 truy lùng cùng loại race condition
            trong các kernel scan trên trình giả lập.
          </p>
        }
      >
        <p>
          PyTorch's scan kernel for Apple GPUs carries a running total from one chunk of a long row to the next through a
          small shared array. Its source has a barrier right after every thread reads that carry, with a comment that names
          this exact bug: without it, the next chunk's writes can overwrite the carry before everyone has read it. (The
          example is ours; the book does not look at real backends.) Lab 11.3 hunts the same kind of race in scan kernels
          on the emulator.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch11-facts D4)', '2026-11-20 (ch11-facts D4)')}>
        <Translated
          vi={
            <p>
              <code>ScanKernel.metal</code>: "Ensure every thread has read the carry before the next iteration overwrites
              simdgroup_sums; otherwise long (multi-iteration) scans race." (Bảo đảm mọi luồng đã đọc giá trị mang sang trước khi
              lần lặp kế tiếp ghi đè simdgroup_sums; nếu không, các scan dài nhiều lần lặp sẽ bị race.)
            </p>
          }
        >
          <p>
            <code>ScanKernel.metal</code>: "Ensure every thread has read the carry before the next iteration overwrites
            simdgroup_sums; otherwise long (multi-iteration) scans race."
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Fig. 11.3 trả {iterations(1024) * 2} rào chắn cho một khối 1024 phần tử, hai rào chắn trong mỗi lần lặp trong số{' '}
            {iterations(1024)} lần lặp. Mục <a href={href('ch11', '11.3')}>11.3</a> giảm con số đó đi một nửa.
          </p>
        }
      >
        <p>
          Fig. 11.3 pays {iterations(1024) * 2} barriers for one block of 1024 elements, two in each of its{' '}
          {iterations(1024)} iterations. Section <a href={href('ch11', '11.3')}>11.3</a> halves that.
        </p>
      </Translated>
    </>
  );
}
