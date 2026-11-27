import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, MATMUL_TIERS, useLabFile } from '../../components/Code';
import { defineLabs } from '../../components/LabKit';
import { cuStatus, metalStatus } from './labStatus';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { BOOK_SHAPE, sharedBytes } from '../../sim/gemmbudget';
import {
  HAZARD_LABEL,
  SCHEDULES,
  SCHEDULE_LABEL,
  SCHEMES,
  SCHEME_LABEL,
  VARIANTS,
  VARIANT_LABEL,
  baselineStream,
  firstFailure,
  hazardsOf,
  loopSource,
  nextRelease,
  roles,
  run,
  speedupLimit,
  stateAt,
  timeline,
  where,
  type DepKind,
  type Op,
  type Schedule,
  type Step,
  type Variant,
} from '../../sim/gemmpipe';
import { LANES_H, Lanes, PipelineScene } from './PipelineScene';

const P = 4; // threads in the stepper's block
const NT = 4; // tiles
const PIPE_SMEM = sharedBytes({ ...BOOK_SHAPE, pad: 1, buffers: 2 });
const [LAB155] = defineLabs('ch15', [{ num: '15.5', dir: 'lab5_pipelined', name: 'pipelined', vi: { name: 'pipeline hóa' } }]);
const LAB_DIR = LAB155.id;

// ---------------------------------------------------------------- true or false?
function BarrierQuiz() {
  const locale = useLocale();
  const stream = useMemo(() => baselineStream(4), [locale]);
  const [pick, setPick] = useState<Record<number, DepKind>>({});
  const bars = stream.filter((x) => x.kind === 'sync');
  const answered = bars.filter((b) => pick[b.barrier!] != null).length;
  const right = bars.filter((b) => pick[b.barrier!] === b.dep).length;
  return (
    <Bench
      title={tr('True or false dependence?', 'Phụ thuộc thật hay giả?')}
      task={
        <Translated vi="Gắn nhãn cho mọi rào chắn của Fig. 15.12(a): thứ theo sau nó cần dữ liệu, hay chỉ cần vùng nhớ?">
          Label every barrier of Fig. 15.12(a): does what follows it need data, or only the memory?
        </Translated>
      }
      foot={
        <Translated vi="Bốn cặp ô, như trong hình của sách. Một rào chắn bảo vệ một phụ thuộc thật khi các lệnh sau nó đọc những gì các lệnh trước nó đã ghi, và một phụ thuộc giả khi chúng chỉ ghi đè lên những gì vẫn đang được đọc.">
          {"Four tile pairs, as in the book's figure. A barrier guards a true dependence when the instructions after it read what those before it wrote, and a false one when they would only overwrite what is still being read."}
        </Translated>
      }
    >
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0, background: 'var(--page)' }}>
          <tbody>
            {stream.map((x, k) =>
              x.kind === 'sync' ? (
                <tr key={k} data-barrier={x.barrier} data-dep={x.dep} data-pick={pick[x.barrier!] ?? ''}>
                  <td style={{ paddingLeft: '0.6rem', whiteSpace: 'nowrap', width: '1%' }}>
                    <code style={{ whiteSpace: 'nowrap', wordBreak: 'normal', overflowWrap: 'normal' }}>__syncthreads()</code>
                  </td>
                  <td style={{ whiteSpace: 'nowrap', width: '1%' }}>
                    <span className="seg" role="group" aria-label={tr(`barrier ${x.barrier! + 1}`, `rào chắn ${x.barrier! + 1}`)}>
                      {(['true', 'false'] as DepKind[]).map((d) => (
                        <button key={d} aria-pressed={pick[x.barrier!] === d} onClick={() => setPick({ ...pick, [x.barrier!]: d })}>
                          <Translated vi={d === 'true' ? 'thật' : 'giả'}>{d}</Translated>
                        </button>
                      ))}
                    </span>
                  </td>
                  <td>
                    {pick[x.barrier!] == null ? (
                      <span className="muted">
                        <Translated vi={<>rào chắn {x.barrier! + 1} trên {bars.length}</>}>
                          barrier {x.barrier! + 1} of {bars.length}
                        </Translated>
                      </span>
                    ) : (
                      <>
                        <span className={'chip ' + (pick[x.barrier!] === x.dep ? 'ok' : 'fail')}>
                          {pick[x.barrier!] === x.dep ? tr('right', 'đúng') : tr(`no: ${x.dep}`, `không: ${x.dep === 'true' ? 'thật' : 'giả'}`)}
                        </span>{' '}
                        <span className="muted">
                          <Translated
                            vi={x.dep === 'true'
                              ? `Phép tính C${x.tile} đọc các phần tử của A${x.tile} và B${x.tile} mà các luồng khác đã nạp: dữ liệu phải đến trước khi bất kỳ ai dùng nó.`
                              : `Việc nạp A${x.tile + 1} và B${x.tile + 1} không cần gì từ phép tính C${x.tile}. Nó chờ chỉ vì nó sẽ ghi đè lên A${x.tile} và B${x.tile}, trong cùng các mảng chia sẻ, trong khi các luồng khác vẫn đang đọc chúng.`}
                          >
                            {x.why}
                          </Translated>
                        </span>
                      </>
                    )}
                  </td>
                </tr>
              ) : (
                <tr key={k}>
                  <td style={{ paddingLeft: '0.6rem' }} colSpan={3}>
                    <span className={'chip ' + (x.kind === 'load' ? 'global' : 'reg')}>
                      {tr(x.label, x.kind === 'load' ? `Nạp A${x.tile} và B${x.tile}` : `Tính C${x.tile}`)}
                    </span>
                    <span className="muted">
                      {' '}
                      {x.kind === 'load'
                        ? tr('from global memory into shared memory', 'từ bộ nhớ toàn cục (global memory) vào bộ nhớ chia sẻ (shared memory)')
                        : tr('reads the shared tiles, accumulates in registers', 'đọc các ô chia sẻ, tích lũy trong thanh ghi (register)')}
                    </span>
                  </td>
                </tr>
              ),
            )}
          </tbody>
        </table>
      </div>
      <p style={{ margin: '0.8rem 0 0' }} data-out="score" data-right={right} data-answered={answered}>
        {answered < bars.length ? (
          <span className="muted">
            <Translated vi={<>Đã gắn nhãn {answered} trên {bars.length}.</>}>
              {answered} of {bars.length} labelled.
            </Translated>
          </span>
        ) : (
          <>
            <span className={'chip ' + (right === bars.length ? 'ok' : 'fail')}>
              <Translated vi={<>đúng {right} trên {bars.length}</>}>
                {right} of {bars.length} right
              </Translated>
            </span>{' '}
            <Translated vi="Bốn rào chắn sau một lần nạp là phụ thuộc thật; ba rào chắn sau một lần tính là phụ thuộc giả. Chỉ những phụ thuộc giả có thể được loại bỏ bằng cách cho cặp tiếp theo vùng nhớ khác.">
              The four after a load are true dependences; the three after a compute are false ones. Only the false ones can
              be removed by giving the next pair other memory.
            </Translated>
          </>
        )}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the Gantt explorer
const CHECKPOINTS: { key: string; label: string; n: number; L: number; C: number; s: number }[] = [
  {
    key: 'book',
    get label() {
      return tr("The book's figure: n = 4, L = C = 1", 'Hình của sách: n = 4, L = C = 1');
    },
    n: 4, L: 1, C: 1, s: 0,
  },
  {
    key: 'slow-load',
    get label() {
      return tr('Loads 3 × compute: n = 8, L = 3, C = 1', 'Nạp gấp 3 × tính: n = 8, L = 3, C = 1');
    },
    n: 8, L: 3, C: 1, s: 0,
  },
  {
    key: 'slow-compute',
    get label() {
      return tr('Compute 4 × loads: n = 8, L = 1, C = 4', 'Tính gấp 4 × nạp: n = 8, L = 1, C = 4');
    },
    n: 8, L: 1, C: 4, s: 0,
  },
];
const pct = (v: number) => `${Math.round(v * 100)}%`;
const num2 = (v: number) => String(Number(v.toFixed(2)));

function GanttExplorer() {
  const [L, setL] = useState(1);
  const [C, setC] = useState(1);
  const [n, setN] = useState(4);
  const [s, setS] = useState(0);
  const tls = SCHEMES.map((sc) => timeline(sc, n, L, C, s));
  const maxT = Math.max(...tls.map((t) => t.total));
  const scale = Math.min(80, 560 / maxT);
  const gap = 26;
  const H = tls.length * (LANES_H + gap) + 6;
  const slider = (label: string, value: number, set: (v: number) => void, min: number, max: number, step: number, unit = '') => (
    <label className="field" style={{ minWidth: '10rem' }}>
      <span>
        {label} <output data-k={label}>{num2(value)}</output>
        {unit}
      </span>
      <input type="range" min={min} max={max} step={step} value={value} aria-label={label} onChange={(e) => set(Number(e.target.value))} />
    </label>
  );
  const same = (c: (typeof CHECKPOINTS)[number]) => c.n === n && c.L === L && c.C === C && c.s === s;
  const base = tls[0].total;
  return (
    <Bench
      title={tr('Pipelining Gantt', 'Biểu đồ Gantt của pipeline')}
      task={
        <Translated vi="Tái tạo con số 8 và 5 của sách, rồi tìm điểm mà pipeline thôi còn có lợi.">
          {"Reproduce the book's 8 and 5, then find where pipelining stops paying off."}
        </Translated>
      }
      foot={
        <Translated vi="Đơn vị thời gian tùy bạn chọn; chỉ các tỉ số mới có ý nghĩa. Mỗi làn là một loại phần cứng cho cả khối. Các lần nạp được vẽ bằng màu bộ nhớ toàn cục, các lần tính bằng màu thanh ghi.">
          Time units are whatever you like; only ratios mean anything. Each lane is one kind of hardware for the whole block. Loads are drawn in the global-memory colour, computes in the register colour.
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-start' }}>
        {slider(tr('load time L', 'thời gian nạp L'), L, setL, 0.25, 4, 0.25)}
        {slider(tr('compute time C', 'thời gian tính C'), C, setC, 0.25, 4, 0.25)}
        {slider(tr('tile pairs n', 'số cặp ô n'), n, setN, 1, 12, 1)}
        {slider(tr('barrier cost s', 'chi phí rào chắn s'), s, setS, 0, 1, 0.05)}
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }} role="group" aria-label={tr('Checkpoints', 'Điểm mốc')}>
        <span className="muted">
          <Translated vi="Điểm mốc">Checkpoints</Translated>
        </span>
        {CHECKPOINTS.map((c) => (
          <button key={c.key} className="quiet" data-checkpoint={c.key} aria-pressed={same(c)}
            onClick={() => { setN(c.n); setL(c.L); setC(c.C); setS(c.s); }}>
            {c.label}
          </button>
        ))}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 720 ${H}`} role="img" aria-label={tr('Three schedules of the tile loop', 'Ba lịch chạy của vòng lặp ô')} style={{ width: '100%', minWidth: 560, maxWidth: 720, display: 'block' }}>
          {tls.map((tl, i) => (
            <Lanes
              key={tl.scheme} tl={tl} x={92} y={18 + i * (LANES_H + gap)} scale={scale} deps="all" /* i18n-ignore: prop value */ bufs={tl.scheme !== 'two-barriers' && scale * Math.min(L, C) >= 26}
              title={tr(SCHEME_LABEL[tl.scheme], { 'two-barriers': 'Một buffer, hai rào chắn (Fig. 15.12a)', 'double-buffered': 'Hai buffer, không xoay vòng (Fig. 15.12b)', pipelined: 'Hai buffer, pipeline (Fig. 15.12c)' }[tl.scheme])}
            />
          ))}
        </svg>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: '0.6rem 0 0', background: 'var(--page)' }}>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>
                <Translated vi="Vòng lặp">Loop</Translated>
              </th>
              <th className="num">
                <Translated vi="Tổng">Total</Translated>
              </th>
              <th className="num">
                <Translated vi="Tăng tốc">Speed-up</Translated>
              </th>
              <th className="num">
                <Translated vi="Bộ nhớ bận">Memory busy</Translated>
              </th>
              <th className="num">
                <Translated vi="Tính toán bận">Compute busy</Translated>
              </th>
              <th className="num">
                <Translated vi="Rào chắn">Barriers</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            {tls.map((tl) => (
              <tr key={tl.scheme} data-scheme={tl.scheme} data-total={tl.total} data-speedup={num2(base / tl.total)}>
                <td style={{ paddingLeft: '0.6rem' }}>
                  {tr(SCHEME_LABEL[tl.scheme], { 'two-barriers': 'Một buffer, hai rào chắn (Fig. 15.12a)', 'double-buffered': 'Hai buffer, không xoay vòng (Fig. 15.12b)', pipelined: 'Hai buffer, pipeline (Fig. 15.12c)' }[tl.scheme])}
                </td>
                <td className="num">{num2(tl.total)}</td>
                <td className="num">{num2(base / tl.total)}×</td>
                <td className="num">{pct(tl.memUtil)}</td>
                <td className="num">{pct(tl.compUtil)}</td>
                <td className="num">{tl.barriers.length}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ margin: '0.8rem 0 0' }} data-out="limit" data-limit={num2(speedupLimit(L, C))}>
        <Translated
          vi={
            <>
              Dù có bao nhiêu cặp, sự chồng lấp chỉ có thể che tối đa pha ngắn hơn trong hai pha, nên với rào chắn miễn phí,
              vòng lặp pipeline tiến tới (L + C) / max(L, C) = <b>{num2(speedupLimit(L, C))}×</b> ở đây, và chỉ đạt 2× khi
              hai pha bằng nhau.
            </>
          }
        >
          However many pairs there are, overlap can hide at most the shorter of the two phases, so with free barriers the
          pipelined loop approaches (L + C) / max(L, C) = <b>{num2(speedupLimit(L, C))}×</b> here, and 2× only when the
          two are equal.
        </Translated>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the stepper
const STEP_TIERS = { buf: 'shared', tiles: 'global', sum: 'reg', tile: 'reg', numTiles: 'reg' } as const;
const digits = (v: number) => String(v).padStart(NT, '0');
const OP_NAME: Record<Op['kind'], string> = {
  load: 'loadTile',
  mm: 'mm',
  get barrier() {
    return tr('the barrier', 'rào chắn');
  },
};

function describe(s: Step, v: Variant): string {
  if (s.release) {
    if (s.op.phase === 'prologue')
      return tr(
        'The barrier after the prologue opens: every thread has loaded its slot of tile 0.',
        'Rào chắn sau phần mở đầu (prologue) mở ra: mọi luồng đã nạp ô nhớ của mình trong tile 0.',
      );
    const b = roles(s.op.iter);
    if (v === 'barrier-before-load')
      return tr(
        `The barrier in iteration ${s.op.iter} opens: every thread has added up buffer ${b.curr}. The loads come after it, unguarded.`,
        `Rào chắn trong vòng lặp ${s.op.iter} mở ra: mọi luồng đã cộng xong buffer ${b.curr}. Các lần nạp đến sau nó, không được bảo vệ.`,
      );
    const curr = v === 'single-buffer' ? 0 : b.curr;
    return tr(
      `The barrier at the end of iteration ${s.op.iter} opens: every thread has added up buffer ${curr} and loaded its slot of the next tile.`,
      `Rào chắn ở cuối vòng lặp ${s.op.iter} mở ra: mọi luồng đã cộng xong buffer ${curr} và nạp ô nhớ của mình trong tile tiếp theo.`,
    );
  }
  const who = tr(
    `Thread ${s.thread}, ${where(s.op)}`,
    `Luồng ${s.thread}, ${s.op.phase === 'prologue' ? 'phần mở đầu' : s.op.phase === 'epilogue' ? 'sau vòng lặp' : `vòng lặp ${s.op.iter}`}`,
  );
  if (s.op.kind === 'barrier') return tr(`${who}: reaches the barrier and waits for the others.`, `${who}: đến rào chắn và chờ các luồng khác.`);
  if (s.write)
    return tr(
      `${who}: loads slot ${s.write.slot} of tile ${s.write.tile} into buffer ${s.write.buf}.`,
      `${who}: nạp ô nhớ ${s.write.slot} của tile ${s.write.tile} vào buffer ${s.write.buf}.`,
    );
  const found = s.reads!.map((r) => (r.tile < 0 ? tr('empty', 'trống') : r.tile)).join(', ');
  return tr(
    `${who}: adds up the ${s.reads!.length} slots of buffer ${s.op.buf} and finds tiles ${found}. Its sum is now ${digits(s.sum!)}.`,
    `${who}: cộng ${s.reads!.length} ô nhớ của buffer ${s.op.buf} và thấy các tile ${found}. Tổng của nó giờ là ${digits(s.sum!)}.`,
  );
}

function Stepper() {
  const [variant, setVariant] = useState<Variant>('correct');
  const [schedule, setSchedule] = useState<Schedule>('in-order');
  const [n, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const locale = useLocale();
  const trace = useMemo(() => run({ threads: P, tiles: NT, variant }, schedule), [variant, schedule, locale]);
  const total = trace.steps.length;
  const at = Math.min(n, total);
  const st = useMemo(() => stateAt(trace, at), [trace, at]);
  const cur = at > 0 ? trace.steps[at - 1] : undefined;
  const bad = firstFailure(trace);
  const src = loopSource(variant);
  const single = variant === 'single-buffer';

  useEffect(() => setStep(0), [variant, schedule]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setStep((x) => x + 1), 380);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);

  const slotCls = (b: number, j: number) => {
    if (cur?.write && cur.write.buf === b && cur.write.slot === j) return 'hot';
    const r = cur?.reads?.find((x) => x.buf === b && x.slot === j);
    if (r) return r.ok ? 'hot' : 'bad';
    return '';
  };
  const doing = (t: number) => {
    const th = st.threads[t];
    if (th.done) return tr('finished', 'đã xong');
    if (th.waiting)
      return tr(
        `waits at the barrier (${where(th.last!)})`,
        `chờ ở rào chắn (${th.last!.phase === 'prologue' ? 'phần mở đầu' : th.last!.phase === 'epilogue' ? 'sau vòng lặp' : `vòng lặp ${th.last!.iter}`})`,
      );
    const next = trace.ops[th.pc];
    return tr(
      `${th.last ? '' : 'not started; '}${where(next)}, ${OP_NAME[next.kind]} next`,
      `${th.last ? '' : 'chưa bắt đầu; '}${next.phase === 'prologue' ? 'phần mở đầu' : next.phase === 'epilogue' ? 'sau vòng lặp' : `vòng lặp ${next.iter}`}, tiếp theo là ${OP_NAME[next.kind]}`,
    );
  };
  const buffersOf = (t: number) => {
    const th = st.threads[t];
    const op = th.done ? trace.ops[trace.ops.length - 1] : trace.ops[th.pc];
    if (!op) return '';
    if (single) return tr('buffer 0', 'buffer 0');
    if (op.phase === 'prologue') return tr('fills 0', 'nạp 0');
    if (op.phase === 'epilogue') return `curr ${op.buf}`; // i18n-ignore: code variable name
    const r = roles(op.iter);
    return `curr ${r.curr}, next ${r.next}`; // i18n-ignore: code variable names
  };

  return (
    <Bench
      title={tr('Step through the pipelined loop', 'Chạy từng bước vòng lặp pipeline')}
      task={
        <Translated vi="Giữ thứ tự mà mỗi luồng chạy xa hết mức có thể, thử từng biến thể hỏng, và đi tới lỗi đầu tiên. Sau đó chuyển sang chạy đồng bộ từng bước (lockstep).">
          Keep the order where each thread runs as far as it can, try each broken variant, and go to the first failure. Then switch to lockstep.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Bốn luồng, bốn tile. Ô nhớ j của mọi tile chứa 10 mũ số thứ tự của tile, nên một tổng đọc được theo từng chữ số:
              chữ số cuối đếm số ô nhớ của tile 0 mà một luồng đã cộng, chữ số đầu đếm của tile 3. Mọi luồng phải kết thúc ở{' '}
              {digits(trace.reference)}. Một ô nhớ trống được tính là 0 ở đây; bộ nhớ chia sẻ thật sẽ chứa bất cứ thứ gì còn
              sót lại trong đó.
            </>
          }
        >
          <>
            Four threads, four tiles. Slot j of every tile holds 10 to the power of the tile's number, so a sum reads digit by
            digit: the last digit counts the slots of tile 0 a thread has added, the first those of tile 3. Every thread
            should end at {digits(trace.reference)}. An empty slot counts as 0 here; real shared memory would hold whatever
            was left in it.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          {tr('Kernel', 'Kernel')}
          <select value={variant} onChange={(e) => setVariant(e.target.value as Variant)} aria-label={tr('kernel', 'kernel')} style={{ maxWidth: '100%' }}>
            {VARIANTS.map((v) => <option key={v} value={v}>{tr(VARIANT_LABEL[v], {
                      correct: 'như trong Fig. 15.14',
                      'no-loop-barrier': 'bỏ rào chắn trong vòng lặp',
                      'barrier-before-load': 'chuyển rào chắn lên trước loadTile',
                      'no-epilogue': 'thiếu mm sau vòng lặp',
                      'wrong-tile': 'nạp tile thay vì tile + 1',
                      'single-buffer': 'một buffer, một rào chắn',
                      'load-first': 'nạp trước, rồi mới mm',
                    }[v])}</option>)}
          </select>
        </label>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Thứ tự luồng">Thread order</Translated>
          <select value={schedule} onChange={(e) => setSchedule(e.target.value as Schedule)} aria-label={tr('thread order', 'thứ tự luồng')} style={{ maxWidth: '100%' }}>
            {SCHEDULES.map((s) => (
              <option key={s} value={s}>
                {tr(
                  SCHEDULE_LABEL[s],
                  {
                    lockstep: 'mọi luồng chạy đồng bộ từng bước (lockstep)',
                    'in-order': 'mỗi luồng chạy xa hết mức có thể, luồng 0 trước',
                    reversed: 'mỗi luồng chạy xa hết mức có thể, luồng cuối trước',
                    shuffled: 'xen kẽ ngẫu nhiên',
                  }[s],
                )}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setStep(0); }} disabled={at === 0}>
          <Translated vi="Bắt đầu lại">Restart</Translated>
        </button>
        <button onClick={() => { setPlaying(false); setStep(at - 1); }} disabled={at === 0}>
          <Translated vi="Lùi">Back</Translated>
        </button>
        <button className="primary" onClick={() => { setPlaying(false); setStep(at + 1); }} disabled={at >= total}>
          <Translated vi="Bước">Step</Translated>
        </button>
        <button onClick={() => { setPlaying(false); setStep(nextRelease(trace, at)); }} disabled={at >= total}>
          <Translated vi="Chạy tới rào chắn kế tiếp">Run to next barrier</Translated>
        </button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Chạy')}</button>
        {bad ? (
          <button onClick={() => { setPlaying(false); setStep(bad); }}>
            <Translated vi="Đi tới lỗi đầu tiên">Go to the first failure</Translated>
          </button>
        ) : null}
        <span className="muted" aria-live="polite">
          <Translated vi={<>bước {at} trên {total}</>}>
            step {at} of {total}
          </Translated>
        </span>
      </div>
      <div className={'narrate' + (cur?.violations ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {cur ? (
          <>
            {describe(cur, variant)}
            {cur.violations ? (
              <>
                <br />
                <Translated
                  vi={
                    <b className="t-fail">
                      Lỗi ({cur.violations[0].kind === 'read-before-write' ? 'đọc trước khi ghi' : 'bị ghi đè trước khi đọc'}):{' '}
                      {(() => {
                        const r = cur.reads!.find((x) => !x.ok)!;
                        const t = cur.thread;
                        const w = cur.op.phase === 'prologue' ? 'phần mở đầu' : cur.op.phase === 'epilogue' ? 'sau vòng lặp' : `vòng lặp ${cur.op.iter}`;
                        return r.hazard === 'read-before-write'
                          ? `luồng ${t} cộng buffer ${cur.op.buf} trong ${w} trước khi luồng ${r.slot} nạp tile ${r.meant} vào ô nhớ ${r.slot}: ${r.tile < 0 ? 'nó vẫn còn trống' : `nó vẫn chứa tile ${r.tile}`}`
                          : `luồng ${t} cần tile ${r.meant} ở ô nhớ ${r.slot} của buffer ${cur.op.buf} trong ${w}, nhưng luồng ${r.slot} đã nạp tile ${r.tile} đè lên nó`;
                      })()}
                      .
                    </b>
                  }
                >
                  <b className="t-fail">Failure ({HAZARD_LABEL[cur.violations[0].kind]}): {cur.violations[0].message}.</b>
                </Translated>
                {cur.violations.length > 1 ? (
                  <span className="t-fail">
                    {' '}
                    <Translated
                      vi={
                        <>
                          Tương tự với ô nhớ{' '}
                          {cur.reads!.filter((r) => !r.ok).slice(1).map((r) => r.slot).join(', ').replace(/, (\d+)$/, ' và $1')}.
                        </>
                      }
                    >
                      The same for slot{cur.violations.length > 2 ? 's' : ''}{' '}
                      {cur.reads!.filter((r) => !r.ok).slice(1).map((r) => r.slot).join(', ').replace(/, (\d+)$/, ' and $1')}.
                    </Translated>
                  </span>
                ) : null}
              </>
            ) : null}
          </>
        ) : (
          <Translated vi={<>Chưa có gì chạy. Mỗi luồng sẽ nạp ô nhớ của riêng nó trong một tile và cộng cả bốn ô nhớ của buffer mà nó dùng để tính. Bấm Bước.</>}>
            <>Nothing has run yet. Each thread will load its own slot of a tile and add up all four slots of the buffer it computes with. Press Step.</>
          </Translated>
        )}
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 5fr)' }}>
        <div>
          <Code
            tabs={[{ label: tr('the loop', 'vòng lặp'), code: src.code, tiers: STEP_TIERS }]}
            highlight={cur ? [cur.op.line] : []}
          />
        </div>
        <div>
          {(single ? [0] : [0, 1]).map((b) => (
            <div key={b} style={{ marginBottom: '0.8rem' }}>
              <div className="mx-label t-shared">
                {single ? 'buf[0]' : `buf[${b}]`} <Translated vi="(chia sẻ)">(shared)</Translated>
              </div>
              <div className="mx shared" data-buffer={b} style={{ gridTemplateColumns: `repeat(${P}, auto)` }}>
                {st.buffers[b].map((sl, j) => (
                  <span key={j} className={slotCls(b, j)} data-tile={sl.tile}>
                    {sl.tile < 0 ? '·' : sl.tile}
                  </span>
                ))}
              </div>
            </div>
          ))}
          <div style={{ overflowX: 'auto' }}>
          <table className="threads" aria-label={tr('threads', 'các luồng')}>
            <thead>
              <tr>
                <th><Translated vi="luồng">thread</Translated></th>
                <th><Translated vi="buffer">buffers</Translated></th>
                <th><Translated vi="đang làm">doing</Translated></th>
                <th><Translated vi="tổng">sum</Translated></th>
              </tr>
            </thead>
            <tbody>
              {st.threads.map((th, t) => (
                <tr key={t} className={cur?.thread === t ? 'cur' : ''} data-thread={t} data-sum={th.sum}>
                  <td>{t}</td>
                  <td>{buffersOf(t)}</td>
                  <td>{doing(t)}</td>
                  <td className="t-reg">{digits(th.sum)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
          <Translated
            vi={
              <p className="muted" style={{ margin: '0.5rem 0 0', fontSize: '0.85rem' }}>
                Các số trong ô nhớ là tile mà chúng đang chứa; một dấu chấm là ô nhớ chưa được nạp gì.
              </p>
            }
          >
            <p className="muted" style={{ margin: '0.5rem 0 0', fontSize: '0.85rem' }}>
              The numbers in the slots are the tiles they hold; a dot is a slot nothing has been loaded into yet.
            </p>
          </Translated>
        </div>
      </div>
      {at >= total ? (
        <p style={{ margin: '0.9rem 0 0' }} data-k="result">
          {trace.correct ? (
            <span className="chip ok">
              <Translated vi={<>Mọi luồng kết thúc với {digits(trace.reference)}.</>}>
                Every thread ends with {digits(trace.reference)}.
              </Translated>
            </span>
          ) : (
            <span className="chip fail">
              <Translated vi={<>{trace.wrong.length} trên {P} luồng kết thúc với tổng sai.</>}>
                {trace.wrong.length} of {P} threads end with a wrong sum.
              </Translated>
            </span>
          )}{' '}
          <span className="muted">
            <Translated
              vi={
                <>
                  Các tổng: {trace.sums.map(digits).join(', ')}.
                  {trace.violations.length ? ` ${trace.violations.length} lần đọc sai.` : ''}
                </>
              }
            >
              Sums: {trace.sums.map(digits).join(', ')}.
              {trace.violations.length ? ` ${trace.violations.length} bad read${trace.violations.length === 1 ? '' : 's'}.` : ''}
            </Translated>
          </span>
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the outcome table
function Outcomes() {
  const [guess, setGuess] = useState<Set<Variant>>(new Set());
  const [shown, setShown] = useState(false);
  const results = useMemo(() => VARIANTS.map((v) => ({ v, cells: SCHEDULES.map((s) => run({ threads: P, tiles: NT, variant: v }, s)) })), []);
  const everywhere = results.filter((r) => r.cells.every((c) => c.correct)).map((r) => r.v);
  const toggle = (v: Variant) => {
    const g = new Set(guess);
    if (g.has(v)) g.delete(v);
    else g.add(v);
    setGuess(g);
  };
  const same = shown && everywhere.length === guess.size && everywhere.every((v) => guess.has(v));
  return (
    <Bench
      title={tr('Every thread order at once', 'Mọi thứ tự luồng cùng lúc')}
      task={
        <Translated vi="Đánh dấu các vòng lặp bạn cho là đúng dưới mọi thứ tự, rồi mở bảng.">
          Tick the loops you expect to be correct under every order, then reveal the table.
        </Translated>
      }
      foot={
        <Translated vi="Mỗi ô chạy vòng lặp bốn luồng, bốn tile của trình chạy từng bước dưới một thứ tự và so tổng của mọi luồng với tổng mong đợi.">
          {"Each cell runs the four-thread, four-tile loop of the stepper under one order and compares every thread's sum with the expected one."}
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }} role="group" aria-label={tr('your prediction', 'dự đoán của bạn')}>
        {VARIANTS.map((v) => (
          <label key={v} className="check">
            <input type="checkbox" checked={guess.has(v)} disabled={shown} onChange={() => toggle(v)} /> {tr(VARIANT_LABEL[v], {
                      correct: 'như trong Fig. 15.14',
                      'no-loop-barrier': 'bỏ rào chắn trong vòng lặp',
                      'barrier-before-load': 'chuyển rào chắn lên trước loadTile',
                      'no-epilogue': 'thiếu mm sau vòng lặp',
                      'wrong-tile': 'nạp tile thay vì tile + 1',
                      'single-buffer': 'một buffer, một rào chắn',
                      'load-first': 'nạp trước, rồi mới mm',
                    }[v])}
          </label>
        ))}
      </div>
      {shown ? (
        <>
          <p style={{ margin: '0 0 0.8rem' }} data-k="prediction">
            <span className={'chip ' + (same ? 'ok' : 'fail')}>{same ? tr('as you predicted', 'đúng như bạn dự đoán') : tr('not quite', 'chưa đúng hẳn')}</span>{' '}
            <Translated vi={<>Đúng dưới mọi thứ tự: {everywhere.map((v) => ({
                      correct: 'như trong Fig. 15.14',
                      'no-loop-barrier': 'bỏ rào chắn trong vòng lặp',
                      'barrier-before-load': 'chuyển rào chắn lên trước loadTile',
                      'no-epilogue': 'thiếu mm sau vòng lặp',
                      'wrong-tile': 'nạp tile thay vì tile + 1',
                      'single-buffer': 'một buffer, một rào chắn',
                      'load-first': 'nạp trước, rồi mới mm',
                    }[v])).join('; ')}.</>}>
              Correct under every order: {everywhere.map((v) => VARIANT_LABEL[v]).join('; ')}.
            </Translated>
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table className="data" style={{ background: 'var(--page)', margin: 0 }}>
              <thead>
                <tr>
                  <th style={{ paddingLeft: '0.6rem' }}>
                    <Translated vi="Vòng lặp">Loop</Translated>
                  </th>
                  {SCHEDULES.map((s) => (
                    <th key={s}>
                      {tr(
                        SCHEDULE_LABEL[s],
                        {
                          lockstep: 'mọi luồng chạy đồng bộ từng bước (lockstep)',
                          'in-order': 'mỗi luồng chạy xa hết mức có thể, luồng 0 trước',
                          reversed: 'mỗi luồng chạy xa hết mức có thể, luồng cuối trước',
                          shuffled: 'xen kẽ ngẫu nhiên',
                        }[s],
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.map(({ v, cells }) => (
                  <tr key={v} data-variant={v}>
                    <td style={{ paddingLeft: '0.6rem' }}>{tr(VARIANT_LABEL[v], {
                      correct: 'như trong Fig. 15.14',
                      'no-loop-barrier': 'bỏ rào chắn trong vòng lặp',
                      'barrier-before-load': 'chuyển rào chắn lên trước loadTile',
                      'no-epilogue': 'thiếu mm sau vòng lặp',
                      'wrong-tile': 'nạp tile thay vì tile + 1',
                      'single-buffer': 'một buffer, một rào chắn',
                      'load-first': 'nạp trước, rồi mới mm',
                    }[v])}</td>
                    {cells.map((r, k) => {
                      const kinds = hazardsOf(r);
                      return (
                        <td key={k} data-outcome={r.correct ? 'correct' : 'wrong'} data-kinds={kinds.join(' ')}>
                          {r.correct ? (
                            <span className="chip ok"><Translated vi="đúng">correct</Translated></span>
                          ) : (
                            <span className="chip fail">
                              <Translated vi={<>sai {r.wrong.length} trên {P}</>}>
                                {r.wrong.length} of {P} wrong
                              </Translated>
                            </span>
                          )}
                          {kinds.map((x) => (
                            <div key={x} className="muted">
                              {tr(HAZARD_LABEL[x], x === 'read-before-write' ? 'đọc trước khi ghi' : 'bị ghi đè trước khi đọc')}
                            </div>
                          ))}
                          {!r.correct && !kinds.length ? (
                            <div className="muted">
                              <Translated vi="không có race: lần nào cũng sai">no race: wrong every time</Translated>
                            </div>
                          ) : null}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <button className="primary" onClick={() => setShown(true)}>
          <Translated vi="Mở bảng">Reveal the table</Translated>
        </button>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- the lab's code
const GEMM_TIERS = { ...MATMUL_TIERS, A_s: 'shared', B_s: 'shared', C_r: 'reg', a_r: 'reg', b_r: 'reg', tile: 'reg' } as const;

function PipelineListing() {
  const { reports } = useReports();
  const cu = useLabFile(`${LAB_DIR}/solution/kernel.cu`);
  const metal = useLabFile(`${LAB_DIR}/solution/kernel.metal`);
  if (!cu || !metal) return null;
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: cu, tiers: GEMM_TIERS, status: cuStatus(reports, LAB155) },
        { label: 'Metal', code: metal, tiers: GEMM_TIERS, status: metalStatus(reports, LAB155) },
      ]}
      caption={
        <Translated
          vi={
            <>
              Lời giải tham chiếu (solution) của lab 15.5, nạp từ <code>labs/{LAB_DIR}/solution/</code>. Đó là Fig. 15.14
              biến thành mã, và khác trang sách ở những chỗ trang sách chỉ là sơ đồ (notes, mục 8): sách lược bỏ các đối số,
              theo thứ tự khác với Figs. 15.5 và 15.6, dùng kiểu con trỏ 2D cho các buffer mà Fig. 15.3 khai báo 1D, và
              không khai báo cả buffer lẫn <code>numTiles</code>. Lab khai báo hai bản của mỗi ô,{' '}
              <code>A_s[2][bM * LDA_S]</code> và <code>B_s[2][bK * bN]</code>, chọn cặp theo tính chẵn lẻ của chỉ số tile
              như lab 6.5 đã làm thay vì hoán đổi con trỏ, và đếm tile bằng một <code>int</code> có dấu, bỏ qua vòng lặp
              khi K = 0. Hãy thử <a href={href('ch15', 'labs')}>lab</a> trước khi đọc nó.
            </>
          }
        >
          <>
            The reference solution of lab 15.5, loaded from <code>labs/{LAB_DIR}/solution/</code>. It is Fig. 15.14 made
            into code, and differs from the page where the page is only a schematic (notes, item 8): the book elides the
            arguments, in another order than Figs. 15.5 and 15.6, uses 2D pointer types for buffers that Fig. 15.3 declares
            1D, and declares neither the buffers nor <code>numTiles</code>. The lab declares two of each tile,{' '}
            <code>A_s[2][bM * LDA_S]</code> and <code>B_s[2][bK * bN]</code>, picks the pair by the parity of the tile
            index as lab 6.5 did instead of swapping pointers, and counts tiles with a signed <code>int</code>, skipping
            the loop when K = 0. Try the <a href={href('ch15', 'labs')}>lab</a> before you read it.
          </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- the reader's shared-memory limit
function SharedLimits({ need }: { need: number }) {
  const { reports, loading } = useReports();
  if (loading) return null;
  const rows = Object.entries(reports)
    .filter(([b]) => b !== 'emu' && b !== 'race')
    .flatMap(([b, r]) => {
      const key = r.device.maxThreadgroupMemoryLength != null ? 'maxThreadgroupMemoryLength' : 'sharedMemPerBlock';
      const limit = Number(r.device[key]);
      return isFinite(limit) && limit > 0 ? [{ b, key, limit, device: String(r.device.name ?? b) }] : [];
    });
  if (!rows.length)
    return (
      <Translated
        vi={
          <p data-smem-limit="none">
            Chưa thiết bị nào của bạn báo cáo giới hạn của nó. Hãy chạy <code>python3 labs/pmpp.py doctor</code> trên một
            máy có GPU, rồi tải lại.
          </p>
        }
      >
        <p data-smem-limit="none">
          None of your devices has reported its limit yet. Run <code>python3 labs/pmpp.py doctor</code> on a machine with a
          GPU, then reload.
        </p>
      </Translated>
    );
  return (
    <>
      {rows.map((r) => (
        <p key={r.b} data-smem-limit={r.limit} data-backend={r.b}>
          <Translated
            vi={
              <>
                {r.device} ({r.b}) báo cáo <code>{r.key}</code> = {r.limit.toLocaleString('en-US')} byte:{' '}
                {need <= r.limit
                  ? `${need.toLocaleString('en-US')} byte vừa, còn dư ${(r.limit - need).toLocaleString('en-US')}.`
                  : `${need.toLocaleString('en-US')} byte không vừa.`}
              </>
            }
          >
            {r.device} ({r.b}) reports <code>{r.key}</code> = {r.limit.toLocaleString('en-US')} bytes:{' '}
            {need <= r.limit
              ? `the ${need.toLocaleString('en-US')} bytes fit, with ${(r.limit - need).toLocaleString('en-US')} to spare.`
              : `the ${need.toLocaleString('en-US')} bytes do not fit.`}
          </Translated>
        </p>
      ))}
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS158: Record<string, Question> = {
  'ch15/15.8/q-false': {
    kind: 'choice',
    id: 'ch15/15.8/q-false',
    prompt: (
      <Translated vi={<>Trong Fig. 15.12(a), những rào chắn nào bảo vệ một phụ thuộc giả?</>}>
        <>In Fig. 15.12(a), which barriers guard a false dependence?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Những rào chắn sau mỗi lần tính, trước lần nạp tiếp theo.">
            {'The ones after each compute, before the next load.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Những rào chắn sau mỗi lần nạp: phép tính chờ ở đó chỉ vì ô tình cờ nằm trong bộ nhớ chia sẻ.">
            {'The ones after each load: the compute waits there only because the tile happens to sit in shared memory.'}
          </Translated>
        ),
        why: (
          <Translated vi="Phép tính đọc các phần tử mà luồng khác đã nạp. Đó là dữ liệu chảy từ bên ghi sang bên đọc, một phụ thuộc thật, và không lượng bộ nhớ nào loại bỏ được nó.">
            {'The compute reads elements other threads loaded. That is data flowing from writer to reader, a true dependence, and no amount of memory removes it.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Tất cả: mỗi rào chắn tồn tại chỉ vì các ô được chia sẻ.">
            {'All of them: each barrier exists only because the tiles are shared.'}
          </Translated>
        ),
        why: (
          <Translated vi="Chia sẻ là mục đích của các ô. Bên đọc cần những gì bên ghi tạo ra; phần đó là thật.">
            {'Sharing is the point of the tiles. Readers need what writers produce; that part is real.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không cái nào: mọi rào chắn trong một vòng lặp phân ô đều bảo vệ dữ liệu mà luồng nào đó cần.">
            {'None: every barrier in a tiled loop protects data that some thread needs.'}
          </Translated>
        ),
        why: (
          <Translated vi="Lần nạp cặp i + 1 không dùng gì mà phép tính i tạo ra. Nó chờ chỉ để tránh ghi đè.">
            {'The load of pair i + 1 uses nothing compute i produced. It waits only to avoid overwriting.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Cùng mẫu với rào chắn 1 và 2 của kernel phân ô chương 5 và vòng lặp trao đổi của mục 6.7: một rào chắn trước
            lần đọc những gì luồng khác đã ghi là phụ thuộc đọc-sau-ghi (thật); một rào chắn trước lần ghi đè lên dữ liệu
            luồng khác vẫn đang đọc là phụ thuộc ghi-sau-đọc (giả), và nó biến mất nếu lần ghi đi tới chỗ khác.
          </p>
        }
      >
        <p>
          The same pattern as barriers 1 and 2 of chapter 5's tiled kernel and the exchange loop of section 6.7: a
          barrier before a read of other threads' writes is a read-after-write (true) dependence; a barrier before a write
          over data others still read is a write-after-read (false) dependence, and it disappears if the write goes
          somewhere else.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.8, p. 366, Fig. 15.12(a)',
    vi: { source: 'ch. 15 §15.8, p. 366, Hình 15.12(a)' },
  },
  'ch15/15.8/q-between': {
    kind: 'choice',
    id: 'ch15/15.8/q-between',
    prompt: (
      <Translated
        vi={
          <>
            Fig. 15.14 không có rào chắn giữa <code>mm(curr)</code> và <code>loadTile(next)</code>. Vì sao điều đó an toàn?
          </>
        }
      >
        <>
          Fig. 15.14 has no barrier between <code>mm(curr)</code> and <code>loadTile(next)</code>. Why is that safe?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Chúng dùng các buffer khác nhau, và rào chắn kết thúc vòng lặp i ngăn mọi luồng nạp lại một buffer cho đến khi tất cả đã đọc xong nó.">
            {'They use different buffers, and the barrier ending iteration i keeps every thread from refilling a buffer until all have finished reading it.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Vì mm chỉ đọc bộ nhớ chia sẻ và loadTile chỉ ghi vào nó, mà đọc và ghi không bao giờ xung đột.">
            {'Because mm only reads shared memory and loadTile only writes it, and reads and writes never conflict.'}
          </Translated>
        ),
        why: (
          <Translated vi="Một lần ghi đè lên ô nhớ mà luồng khác chưa đọc chính là xung đột mà rào chắn phụ thuộc giả sinh ra để ngăn.">
            {'A write over a slot that another thread has yet to read is exactly the conflict the false-dependence barrier was for.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Vì mọi luồng của một khối thực thi mm và loadTile đồng bộ từng bước, nên không luồng nào chạy trước được.">
            {'Because all threads of a block execute mm and loadTile in lockstep, so none can run ahead.'}
          </Translated>
        ),
        why: (
          <Translated vi="Chỉ một warp mới có thể như vậy, và khối có tám warp. Thứ tự lockstep của trình chạy từng bước cho thấy giả định đó sẽ che giấu điều gì.">
            {'Only a warp might, and the block has eight. The stepper\'s lockstep order shows what that assumption would hide.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Tự nó thì không an toàn; trình biên dịch chèn một rào chắn ở nơi các buffer được chia sẻ.">
            {'It is not safe on its own; the compiler inserts a barrier where the buffers are shared.'}
          </Translated>
        ),
        why: (
          <Translated vi="Trình biên dịch không chèn rào chắn. Bảng kết quả cho thấy vòng lặp đúng dưới mọi thứ tự luồng.">
            {'Compilers do not insert barriers. The outcome table shows the loop is correct under every thread order.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong một vòng lặp, hai lời gọi dùng các buffer khác nhau, nên chúng không thể va chạm. Giữa các vòng lặp, lần
            ghi tiếp theo vào buffer <i>curr</i> xảy ra ở vòng lặp i + 1, bởi một luồng đã qua rào chắn ở cuối vòng lặp i;
            không luồng nào qua được nó cho đến khi tất cả đã xong <code>mm</code> của vòng lặp i. Một rào chắn bảo vệ cả
            việc tile tiếp theo đến nơi lẫn việc tái sử dụng tile cũ. Sách nêu nhiệm vụ thứ nhất; nhiệm vụ thứ hai là cách
            đọc của chúng tôi, và bảng kết quả kiểm chứng nó.
          </p>
        }
      >
        <p>
          Within one iteration the two calls use different buffers, so they cannot collide. Across iterations, the next
          write into buffer <i>curr</i> happens in iteration i + 1, by a thread that has passed the barrier at the end of
          iteration i; no thread passes it until all have finished iteration i's <code>mm</code>. The one barrier guards
          both the arrival of the next tile and the reuse of the old one. The book states the first job; the second is our
          reading, and the outcome table tests it.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.8, p. 367, Fig. 15.14; notes/ch15.md (the second job is marked [inferred])',
    vi: { source: 'ch. 15 §15.8, p. 367, Fig. 15.14; notes/ch15.md (nhiệm vụ thứ hai được đánh dấu [inferred])' },
  },
  'ch15/15.8/q-fig1513': {
    kind: 'numeric',
    id: 'ch15/15.8/q-fig1513',
    prompt: (
      <Translated
        vi={
          <>
            Bốn cặp ô; nạp một cặp mất một đơn vị và tính với nó mất một đơn vị; rào chắn không tốn gì. Mức tăng tốc của
            lịch pipeline so với lịch hai rào chắn là bao nhiêu?
          </>
        }
      >
        <>
          Four tile pairs; loading a pair takes one unit and computing with it one unit; barriers cost nothing. What is the
          speed-up of the pipelined schedule over the two-barrier one?
        </>
      </Translated>
    ),
    answer: 1.6,
    tol: 0.01,
    unit: '×',
    explain: (
      <Translated
        vi={
          <p>
            Hai rào chắn: 4 × (1 + 1) = 8 đơn vị, các làn thay phiên nhau. Pipeline: lần nạp ở phần mở đầu (1), ba vòng lặp
            mỗi vòng làm một lần tính và lần nạp tiếp theo song song (3 × 1), lần tính cuối (1): 5 đơn vị. 8 / 5 = 1.6.
            Fig. 15.13 của sách được vẽ gần đúng với các độ rộng này; nó là sơ đồ, không phải phép đo.
          </p>
        }
      >
        <p>
          Two barriers: 4 × (1 + 1) = 8 units, the lanes taking turns. Pipelined: the prologue load (1), three iterations
          each doing a compute and the next load side by side (3 × 1), the last compute (1): 5 units. 8 / 5 = 1.6. The
          book's Fig. 15.13 is drawn to about these widths; it is a schematic, not a measurement.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.8, pp. 365–366, Fig. 15.13; the arithmetic is the model above',
    vi: { source: 'ch. 15 §15.8, pp. 365–366, Fig. 15.13; phép tính là mô hình ở trên' },
  },
  'ch15/15.8/q-unbalanced': {
    kind: 'numeric',
    id: 'ch15/15.8/q-unbalanced',
    prompt: (
      <Translated
        vi={
          <>
            Tám cặp ô; một lần nạp mất 3 đơn vị và một lần tính mất 1; rào chắn không tốn gì. Mức tăng tốc của lịch pipeline
            là bao nhiêu?
          </>
        }
      >
        <>
          Eight tile pairs; a load takes 3 units and a compute 1; barriers cost nothing. What is the speed-up of the
          pipelined schedule?
        </>
      </Translated>
    ),
    answer: 1.28,
    tol: 0.01,
    unit: '×',
    explain: (
      <Translated
        vi={
          <p>
            Hai rào chắn: 8 × (3 + 1) = 32. Pipeline: 3 cho phần mở đầu, 7 vòng lặp mỗi vòng max(3, 1) = 3, và 1 cho lần
            tính cuối: 3 + 21 + 1 = 25. 32 / 25 = 1.28. Sự chồng lấp chỉ che được pha ngắn hơn; làn bộ nhớ bận 24 trên 25
            đơn vị và quyết định nhịp độ.
          </p>
        }
      >
        <p>
          Two barriers: 8 × (3 + 1) = 32. Pipelined: 3 for the prologue, 7 iterations of max(3, 1) = 3, and 1 for the last
          compute: 3 + 21 + 1 = 25. 32 / 25 = 1.28. Overlap can hide only the shorter phase; the memory lane is busy 24 of
          the 25 units and sets the pace.
        </p>
      </Translated>
    ),
    source: 'our arithmetic with the model of Fig. 15.13',
    vi: { source: 'phép tính của chúng tôi với mô hình của Fig. 15.13' },
  },
  'ch15/15.8/q-k0': {
    kind: 'choice',
    id: 'ch15/15.8/q-k0',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel tính <code>unsigned int numTiles = (K + bK - 1) / bK;</code> và chạy vòng lặp của Fig. 15.14,{' '}
            <code>for (unsigned int tile = 0; tile &lt; numTiles - 1; ++tile)</code>. Điều gì xảy ra với K = 0?
          </>
        }
      >
        <>
          A kernel computes <code>unsigned int numTiles = (K + bK - 1) / bK;</code> and runs the loop of Fig. 15.14,{' '}
          <code>for (unsigned int tile = 0; tile &lt; numTiles - 1; ++tile)</code>. What happens for K = 0?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="numTiles - 1 quay vòng thành 4,294,967,295: vòng lặp chạy, và các lần nạp với biên không dấu đọc xa ra ngoài biên.">
            {'numTiles - 1 wraps to 4,294,967,295: the loop runs, and loads with unsigned bounds read far out of bounds.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="numTiles là 0, nên numTiles - 1 là -1 và vòng lặp không bao giờ chạy; chỉ mm của phần kết (epilogue) thực thi, trên một ô toàn số 0, và C toàn số 0.">
            {'numTiles is 0, so numTiles - 1 is -1 and the loop never runs; only the epilogue\'s mm executes, on a zero-filled tile, and C is all zeros.'}
          </Translated>
        ),
        why: (
          <Translated vi="Đó là điều vòng lặp của Fig. 15.3, tile < numTiles, sẽ làm. Lấy 0 không dấu trừ 1 không cho ra -1.">
            {'That is what Fig. 15.3\'s loop, tile < numTiles, would do. Subtracting 1 from an unsigned 0 does not give -1.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một phép chia cho 0 trong số tile, gây trap trước khi vòng lặp bắt đầu.">
            {'A division by zero in the tile count, which traps before the loop starts.'}
          </Translated>
        ),
        why: (
          <Translated vi="Số chia là bK, bằng 8. Số bị chia mới là 0.">
            {'The divisor is bK, which is 8. The dividend is what is 0.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Phép trừ không dấu quay vòng: 0 − 1 là 2³² − 1. Hãy chặn trường hợp này trước phần mở đầu, hoặc đếm tile bằng
            một <code>int</code> có dấu. Đây không phải errata: sách không bao giờ khai báo <code>numTiles</code>, nên cái
            bẫy thuộc về người điền vào sơ đồ. Lab 15.5 làm cả hai; kernel đồ chơi của trình chạy từng bước bên dưới làm
            cách thứ nhất. Lab 15.5 và 15.6 kiểm tra K = 0 với giới hạn 30 giây: khi biên của các lần nạp có dấu, như trong
            các lab, vòng lặp bị quay vòng không đọc gì sai chỗ nhưng chạy bốn tỉ vòng, và giới hạn thời gian dừng nó lại.
          </p>
        }
      >
        <p>
          Unsigned subtraction wraps: 0 − 1 is 2³² − 1. Guard the case before the prologue, or count tiles with a signed{' '}
          <code>int</code>. This is not an erratum: the book never declares <code>numTiles</code>, so the trap belongs to
          whoever fills in the schematic. Lab 15.5 does both; the toy kernel of the stepper below does the first. Labs
          15.5 and 15.6 test K = 0 under a 30-second limit: where the loads' bounds are signed, as in the labs, the
          wrapped loop reads nothing it should not but runs four billion iterations, and the limit stops it.
        </p>
      </Translated>
    ),
    source: 'ours; the loop condition is the book\'s (Fig. 15.14, p. 367), numTiles is not declared there (notes, item 8)',
    vi: { source: 'của khóa học này; điều kiện vòng lặp là của sách (Fig. 15.14, p. 367), numTiles không được khai báo ở đó (notes, mục 8)' },
  },
  'ch15/15.8/q-smem': {
    kind: 'numeric',
    id: 'ch15/15.8/q-smem',
    prompt: (
      <Translated
        vi={
          <>
            Kernel giữ một ô A có đệm (128 hàng, mỗi hàng 8 + 1 float) và một ô B 8 × 128, và vòng lặp pipeline cần hai bản
            của mỗi ô. Mỗi khối cần bao nhiêu byte bộ nhớ chia sẻ?
          </>
        }
      >
        <>
          The kernel keeps a padded A tile (128 rows of 8 + 1 floats) and an 8 × 128 B tile, and the pipelined loop needs
          two of each. How many bytes of shared memory per block?
        </>
      </Translated>
    ),
    answer: PIPE_SMEM,
    unit: 'bytes',
    explain: (
      <>
        <Translated
          vi={
            <p>
              A_s: 128 × 9 × 4 = 4,608 byte. B_s: 8 × 128 × 4 = 4,096. Một cặp: 8,704. Hai cặp: 17,408. Nó phải vừa giới
              hạn mỗi khối của thiết bị bạn, được đọc từ báo cáo của bạn bên dưới. Kể cả khi nhân đôi, bộ nhớ chia sẻ không
              phải thứ giới hạn occupancy của kernel này: thanh ghi mới là thứ đó (mục 15.7).
            </p>
          }
        >
          <p>
            A_s: 128 × 9 × 4 = 4,608 bytes. B_s: 8 × 128 × 4 = 4,096. One pair: 8,704. Two pairs: 17,408. It has to fit
            your device's per-block limit, which is read from your reports below. Even doubled, shared memory is not what
            limits this kernel's occupancy: registers are (section 15.7).
          </p>
        </Translated>
        <SharedLimits need={PIPE_SMEM} />
      </>
    ),
    source: 'our arithmetic (notes/today/ch15-facts.md E1); limits from your reports',
    vi: { unit: 'byte', source: 'phép tính của chúng tôi (notes/today/ch15-facts.md E1); giới hạn lấy từ báo cáo của bạn' },
  },
  'ch15/15.8/q-no-async': {
    kind: 'choice',
    id: 'ch15/15.8/q-no-async',
    prompt: (
      <Translated vi={<>Trên HIP và trên Metal, cái gì nạp buffer kế tiếp trong khi buffer hiện tại đang được dùng?</>}>
        <>On HIP and on Metal, what fills the next buffer while the current one is being used?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Các lệnh nạp thông thường và lệnh ghi vào bộ nhớ chia sẻ do chính các luồng đang tính thực hiện: pipeline phần mềm của mục này.">
            {'Ordinary loads and shared-memory stores by the computing threads themselves: this section\'s software pipeline.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated
            vi={
              <>
                Một copy engine bất đồng bộ, thông qua{' '}
                <code style={{ overflowWrap: 'anywhere' }}>__builtin_amdgcn_async_work_group_copy</code> trên HIP.
              </>
            }
          >
            <>
              An asynchronous copy engine, through{' '}
              <code style={{ overflowWrap: 'anywhere' }}>__builtin_amdgcn_async_work_group_copy</code> on HIP.
            </>
          </Translated>
        ),
        why: (
          <Translated vi="Tài liệu của AMD có nêu tên intrinsic đó, nhưng nó không tồn tại trong trình biên dịch ROCm 10.2.">
            {'AMD\'s documentation names that intrinsic, but it does not exist in the ROCm 10.2 compiler.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Tensor memory accelerator, mà GPU AMD và Apple truy cập qua các giao diện sao chép riêng của chúng.">
            {'The tensor memory accelerator, which AMD and Apple GPUs reach through their own copy interfaces.'}
          </Translated>
        ),
        why: (
          <Translated vi="TMA là phần cứng NVIDIA, compute capability 9.0 trở về sau.">
            {'TMA is NVIDIA hardware, compute capability 9.0 and later.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một lần prefetch vào bộ nhớ threadgroup mà Metal tự lên lịch khi một kernel đọc cùng một buffer device trong vòng lặp.">
            {'A prefetch into threadgroup memory that Metal schedules by itself when a kernel reads the same device buffer in a loop.'}
          </Translated>
        ),
        why: (
          <Translated vi="Apple không công bố điều đó: đặc tả Metal Shading Language hoàn toàn không có sao chép bất đồng bộ.">
            {'Apple documents none: the Metal Shading Language specification has no asynchronous copy at all.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            AMD mô tả các engine di chuyển dữ liệu trên CDNA 3 và 4 và nêu tên một intrinsic cho chúng mà trình biên dịch
            không có; các lệnh nạp từ bộ nhớ toàn cục vào LDS mà trình biên dịch có lại không nằm trong tài liệu HIP. Đặc
            tả của Metal không có sao chép bất đồng bộ. Trên cả hai, vòng lặp của Fig. 15.14 là thứ bạn viết, và sự chồng lấp
            đến từ lịch lệnh của trình biên dịch, đó chính là ý của sách.
          </p>
        }
      >
        <p>
          AMD describes data movement engines on CDNA 3 and 4 and names an intrinsic for them that the compiler does not
          have; the global-to-LDS loads the compiler does have are not in the HIP documentation. Metal's specification has
          no asynchronous copy. On both, Fig. 15.14's loop is what you write, and the overlap comes from the compiler's
          instruction schedule, which is the book's point.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch15-facts.md B5; ch06-facts.md C7 (not in the book)',
    vi: { source: 'notes/today/ch15-facts.md B5; ch06-facts.md C7 (không có trong sách)' },
  },
  'ch15/15.8/q-warpspec': {
    kind: 'choice',
    id: 'ch15/15.8/q-warpspec',
    prompt: (
      <Translated vi={<>Vì sao chuyên biệt hóa warp (warp specialisation) không phụ thuộc vào việc trình biên dịch sắp xếp lệnh tốt?</>}>
        <>Why does warp specialisation not depend on the compiler ordering instructions well?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Các warp khác nhau phát lệnh nạp và lệnh FMA, và phần cứng xen kẽ các warp.">
            {'Different warps issue the loads and the FMAs, and the hardware interleaves warps.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Các warp chuyên biệt không cần rào chắn, nên không gì ngăn bộ lập lịch chạy các lần nạp sớm.">
            {'Specialised warps need no barriers, so nothing stops the scheduler from running the loads early.'}
          </Translated>
        ),
        why: (
          <Translated vi="Các warp tính toán vẫn phải chờ một ô hoàn tất trước khi đọc nó.">
            {'The compute warps still have to wait for a tile to be complete before reading it.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Trình biên dịch không tham gia vào mã của các warp chuyên biệt: phần cứng sinh ra thứ tự lệnh của chúng.">
            {'The compiler is not involved in code for specialised warps: the hardware generates their instruction order.'}
          </Translated>
        ),
        why: (
          <Translated vi="Nó biên dịch mã của chúng như mọi mã khác. Điều thay đổi là không thứ tự lệnh của một warp đơn lẻ nào phải chứa cả hai loại việc.">
            {'It compiles their code like any other. What changes is that no single warp\'s instruction order has to contain both kinds of work.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong pipeline phần mềm, luồng lệnh của một warp chứa cả lệnh nạp lẫn lệnh FMA, nên chúng chồng lấp tốt đến đâu
            phụ thuộc vào chỗ trình biên dịch đặt chúng, mà trình biên dịch biết rất ít về điều kiện lúc chạy. Với chuyên
            biệt hóa warp, một số warp chỉ nạp và số khác chỉ tính, và phần cứng chuyển giữa chúng khi chúng sẵn sàng.
          </p>
        }
      >
        <p>
          In the software pipeline one warp's instruction stream holds both the loads and the FMAs, so how well they
          overlap depends on where the compiler put them, and the compiler knows little about run-time conditions. With
          warp specialisation, some warps only load and others only compute, and the hardware switches between them as
          they become ready.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.8, pp. 367–368',
  },
};

// ---------------------------------------------------------------- the section
export function S158() {
  return (
    <>
      <Translated
        vi={
          <p>
            Mục 15.7 để lại cho mỗi SM một khối làm việc theo lượt: mọi warp của nó nạp ô, rồi tất cả cùng tính. Trong khi
            các ô được nạp, các đơn vị số học không có gì làm; trong khi khối tính, hệ thống bộ nhớ không có gì làm. Mục này
            chồng lấp hai việc đó, để độ trễ dài của các lần nạp ẩn sau các phép nhân-cộng gộp (FMA).
          </p>
        }
      >
        <p>
          Section 15.7 left each SM with one block that takes turns: all its warps load tiles, then all of them compute.
          While the tiles load, the arithmetic units have nothing to do; while the block computes, the memory system has
          nothing to do. This section overlaps the two, so that the long latency of the loads hides behind fused
          multiply-adds.
        </p>
      </Translated>
      <Source section="15.8" pages="364" />

      <h2>
        <Translated vi="Thời gian đi đâu">Where the time goes</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Viết vòng lặp ô của mục 15.3 ra thành một chuỗi lệnh. Nạp các ô A0 và B0 từ bộ nhớ toàn cục (global memory) vào
            bộ nhớ chia sẻ (shared memory). Một rào chắn (barrier), để không luồng nào tính trước khi mọi luồng đã nạp xong.
            Tính phần đóng góp của cặp này, C0. Một rào chắn thứ hai, để không luồng nào nạp cặp tiếp theo đè lên A0 và B0
            trong khi các luồng khác vẫn đang đọc chúng. Rồi lặp lại như vậy cho A1 và B1. Vẽ thành biểu đồ thời gian, phần
            cứng bộ nhớ và các đơn vị dấu phẩy động không bao giờ bận cùng lúc.
          </p>
        }
      >
        <p>
          Write the tile loop of section 15.3 out as a stream of instructions. Load the tiles A0 and B0 from global memory
          into shared memory. A barrier, so that no thread computes before every thread has loaded. Compute this pair's
          contribution, C0. A second barrier, so that no thread loads the next pair over A0 and B0 while others still
          read them. Then the same again for A1 and B1. Drawn as a timing diagram, the memory hardware and the
          floating-point units are never busy at the same time.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Trang 365 nói A0 và B0 được nạp "from shared memory". Chúng đến từ bộ nhớ toàn cục vào bộ nhớ chia sẻ, như
              trang 366 nói (notes, mục 3).
            </p>
          }
        >
          <p>
            Page 365 says A0 and B0 are loaded "from shared memory". They come from global memory into shared memory, as
            page 366 says (notes, item 3).
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Lý tưởng là điều ngược lại: các lần nạp toàn cục xen kẽ với lệnh dấu phẩy động, với các FMA độc lập đặt sau mỗi
            lần nạp có độ trễ dài, để cả hai loại phần cứng đều bận. Trình biên dịch có thể lên lịch lệnh như vậy, nhưng
            không thể vượt qua một rào chắn ngăn cách các lần nạp với phép tính.
          </p>
        }
      >
        <p>
          The ideal is the opposite: global loads interleaved with floating-point instructions, with independent FMAs
          placed after each long-latency load, so that both kinds of hardware stay busy. A compiler can schedule
          instructions that way, but not across a barrier that separates the loads from the compute.
        </p>
      </Translated>
      <Source section="15.8" pages="365–366" figs="Figs. 15.12(a), 15.13(a)" />

      <h2>
        <Translated vi="Những rào chắn nào phải ở lại?">Which barriers have to stay?</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn đã gặp câu hỏi này hai lần: rào chắn 1 và 2 của <a href={href('ch05', '5.4')}>kernel phân ô chương 5</a>, và
            vòng lặp trao đổi của <a href={href('ch06', '6.7')}>mục 6.7</a>. Hãy gắn nhãn cho từng rào chắn của chuỗi lệnh
            trước khi đọc tiếp.
          </p>
        }
      >
        <p>
          You have met this question twice: barriers 1 and 2 of <a href={href('ch05', '5.4')}>chapter 5's tiled kernel</a>,
          and the exchange loop of <a href={href('ch06', '6.7')}>section 6.7</a>. Label each barrier of the stream before
          reading on.
        </p>
      </Translated>
      <BarrierQuiz />
      <Ask q={questionsS158['ch15/15.8/q-false']} />

      <h2>
        <Translated vi="Hai buffer">Two buffers</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một rào chắn bảo vệ phụ thuộc giả sẽ biến mất nếu cặp tiếp theo được ghi vào chỗ khác. Đó là đệm kép, từ mục
            6.7: nạp cặp tiếp theo vào một bộ ô thứ hai. Hai bộ là đủ, một cho các vòng lặp chẵn và một cho các vòng lẻ, và
            rào chắn sau mỗi lần tính biến mất.
          </p>
        }
      >
        <p>
          A barrier that guards a false dependence goes away if the next pair is written somewhere else. That is double
          buffering, from section 6.7: load the next pair into a second set of tiles. Two sets are enough, one for even
          iterations and one for odd, and the barrier after each compute is gone.
        </p>
      </Translated>
      <Source section="15.8" pages="366" figs="Fig. 15.12(b)" />
      <Translated
        vi={
          <p>
            Tuy vậy, riêng đệm kép không chồng lấp được gì. Lần nạp cặp i + 1 vẫn nằm trong vòng lặp i + 1, sau phép tính
            của cặp i, và trình biên dịch không xen kẽ các lệnh nằm ở những vòng lặp khác nhau.
          </p>
        }
      >
        <p>
          Double buffering alone overlaps nothing, though. The load of pair i + 1 still sits in iteration i + 1, after
          the compute of pair i, and the compiler does not interleave instructions that sit in different iterations of
          the loop.
        </p>
      </Translated>

      <h2>
        <Translated vi="Xoay vòng lặp">Rotate the loop</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Pipeline phần mềm sắp xếp lại vòng lặp để một vòng chứa cả hai: vòng lặp i tính với cặp i và nạp cặp i + 1.
            Trước vòng lặp, một phần mở đầu (prologue) nạp cặp 0; sau nó, thêm một lần tính xử lý cặp cuối. Trong một vòng,
            các lần nạp và các FMA độc lập với nhau, nên trình biên dịch được tự do xen kẽ chúng, làm cho các chuỗi lệnh độc
            lập dài hơn và các lần dừng hiếm hơn.
          </p>
        }
      >
        <p>
          Software pipelining rearranges the loop so that one iteration holds both: iteration i computes with pair i and
          loads pair i + 1. Before the loop, a prologue loads pair 0; after it, one more compute handles the last pair.
          Inside an iteration the loads and the FMAs are independent, so the compiler is free to interleave them, which
          makes the runs of independent instructions longer and stalls rarer.
        </p>
      </Translated>
      <Source section="15.8" pages="366" figs="Figs. 15.12(c), 15.13(b)" />
      <PipelineScene />
      <Translated
        vi={
          <p>
            Trình khám phá gắn con số vào bức tranh. Mỗi cặp ô mất L đơn vị để nạp và C đơn vị để tính, và mỗi rào chắn s
            đơn vị. Bắt đầu từ hình của sách, rồi làm cho các lần nạp chậm hơn nhiều so với phép tính, hoặc nhanh hơn nhiều,
            và tìm điểm mà pipeline thôi còn có lợi.
          </p>
        }
      >
        <p>
          The explorer puts numbers on the picture. Each tile pair takes L units to load and C units to compute, and each
          barrier s units. Start at the book's figure, then make the loads much slower than the compute, or much faster,
          and find where pipelining stops paying off.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Thời gian cố định cho mỗi ô, hai làn chồng lấp hoàn hảo, không có bộ nhớ đệm (cache), một khối mỗi SM. Sự xen
              kẽ thật là lịch lệnh của trình biên dịch (p. 367), và các lần nạp thật cũng chồng lấp lẫn nhau.
            </p>
          }
        >
          <p>
            Fixed times per tile, perfect overlap of the two lanes, no caches, one block per SM. The real interleaving is
            the compiler's instruction schedule (p. 367), and real loads overlap with each other too.
          </p>
        </Translated>
      </Simplification>
      <GanttExplorer />
      <Ask q={questionsS158['ch15/15.8/q-fig1513']} label={tr('Work it out', 'Tính ra')} />
      <Ask q={questionsS158['ch15/15.8/q-unbalanced']} label={tr('Work it out', 'Tính ra')} />

      <h2>
        <Translated vi="Mã nguồn">The code</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Fig. 15.14 là kết quả, thay cho vòng lặp ô của Fig. 15.3. Nạp cặp 0 vào các buffer hiện tại và đồng bộ hóa. Sau
            đó, với mọi tile trừ tile cuối: <code>mm</code> trên các buffer hiện tại, <code>loadTile</code> của cặp tiếp theo
            vào các buffer kia, một rào chắn, và hoán đổi buffer nào là hiện tại. Sau vòng lặp, một lần <code>mm</code>{' '}
            cuối cùng. Cố ý không có rào chắn giữa <code>mm</code> và <code>loadTile</code>: chúng dùng các buffer khác
            nhau, và việc bỏ nó đi là điều cho phép trình biên dịch xen kẽ hai hàm đã được inline.
          </p>
        }
      >
        <p>
          Fig. 15.14 is the result, in place of the tile loop of Fig. 15.3. Load pair 0 into the current buffers and
          synchronise. Then, for every tile but the last: <code>mm</code> on the current buffers, <code>loadTile</code> of
          the next pair into the other buffers, one barrier, and swap which buffers are current. After the loop, one last{' '}
          <code>mm</code>. There is deliberately no barrier between <code>mm</code> and <code>loadTile</code>: they use
          different buffers, and leaving it out is what lets the compiler interleave the two inlined functions.
        </p>
      </Translated>
      <Source section="15.8" pages="367" figs="Fig. 15.14" />
      <PipelineListing />
      <Translated
        vi={
          <p>
            Một cái bẫy khi điền vào hình là của chúng tôi chứ không phải của sách, vì sách không bao giờ khai báo{' '}
            <code>numTiles</code>. Vòng lặp chạy khi <code>tile &lt; numTiles - 1</code>.
          </p>
        }
      >
        <p>
          One trap in filling in the figure is ours rather than the book's, because the book never declares{' '}
          <code>numTiles</code>. The loop runs while <code>tile &lt; numTiles - 1</code>.
        </p>
      </Translated>
      <Ask q={questionsS158['ch15/15.8/q-k0']} />

      <h2>
        <Translated vi="Một rào chắn, hai nhiệm vụ">One barrier, two jobs</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Vòng lặp có một rào chắn duy nhất, ở cuối mỗi vòng. Nó bảo vệ phụ thuộc thật: không luồng nào tính với cặp i + 1
            cho đến khi mọi luồng đã nạp phần của mình. Nó cũng bảo vệ việc tái sử dụng vùng nhớ. Buffer được đọc ở vòng
            lặp i được nạp lại ở vòng lặp i + 1, và không luồng nào bắt đầu vòng lặp i + 1 trước khi mọi luồng đã qua rào
            chắn, vốn nằm sau <code>mm</code> của chính nó ở vòng lặp i.
          </p>
        }
      >
        <p>
          The loop has a single barrier, at the end of each iteration. It guards the true dependence: no thread computes
          with pair i + 1 until every thread has loaded its part of it. It also guards the reuse of memory. The buffer
          read in iteration i is refilled in iteration i + 1, and no thread starts iteration i + 1 before every thread
          has passed the barrier, which is after its own <code>mm</code> of iteration i.
        </p>
      </Translated>
      <Beyond title={tr('Our reading', 'Cách đọc của chúng tôi')}>
        <Translated
          vi={
            <p>
              Sách nêu nhiệm vụ thứ nhất. Nhiệm vụ thứ hai là suy luận của chúng tôi (notes, [inferred]); trình chạy từng
              bước và bảng bên dưới kiểm chứng nó.
            </p>
          }
        >
          <p>
            The book states the first job. The second is our inference (notes, [inferred]); the stepper and the table
            below test it.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy một phiên bản đồ chơi của vòng lặp với bốn luồng. Mỗi luồng nạp một ô nhớ của một tile
            mỗi vòng lặp, và <code>mm</code> của nó cộng cả bốn ô nhớ của buffer hiện tại, ba trong số đó do luồng khác ghi.
            Với vòng lặp như trong Fig. 15.14, mã trong trình chạy từng bước, từng dòng một, chính là vòng lặp của{' '}
            <code>labs/ch15/snippets/s158_pipeline_toy.cu</code>, và mã của mỗi biến thể hỏng là vòng lặp trong tệp riêng
            của nó bên cạnh (<code>s158_no_loop_barrier.cu</code> và các tệp khác). Bỏ rào chắn, di chuyển nó, hoặc bỏ một
            buffer, và xem thứ tự luồng nào bắt được lỗi.
          </p>
        }
      >
        <p>
          The stepper runs a toy version of the loop with four threads. Each thread loads one slot of a tile per iteration,
          and its <code>mm</code> adds up all four slots of the current buffer, three of which other threads wrote. For
          the loop as in Fig. 15.14, the listing in the stepper is, line for line, the loop of{' '}
          <code>labs/ch15/snippets/s158_pipeline_toy.cu</code>, and each broken variant&rsquo;s listing is the loop of
          its own file beside it (<code>s158_no_loop_barrier.cu</code> and so on). Remove the barrier, move it, or drop a
          buffer, and see which thread orders catch the mistake.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một luồng tiến từng thao tác ở mức mã nguồn một, và cả một <code>mm</code> là một bước. "Lockstep" là một warp
              đơn được lý tưởng hóa. Không có thời gian: mô hình cho biết một luồng thấy những tile nào.
            </p>
          }
        >
          <p>
            A thread advances one source-level operation at a time, and a whole <code>mm</code> is one step. "Lockstep" is
            an idealised single warp. No timing: the model says which tiles a thread sees.
          </p>
        </Translated>
      </Simplification>
      <Stepper />
      <Outcomes />
      <Ask q={questionsS158['ch15/15.8/q-between']} />

      <h2>
        <Translated vi="Nơi nó chưa đủ">Where it falls short</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Pipeline phần mềm đặt hai loại lệnh cạnh nhau; chúng có thật sự chồng lấp hay không phụ thuộc vào ước tính của
            trình biên dịch về thời gian mỗi lệnh mất khi nó lên lịch, mà trình biên dịch biết rất ít về điều kiện lúc chạy.
            Hai lựa chọn khác giao việc xen kẽ cho phần cứng. Trong chuyên biệt hóa warp (warp specialisation), một số warp
            của khối chỉ làm truy cập bộ nhớ và số khác chỉ làm tính toán, và bộ lập lịch warp xen kẽ chúng như với mọi warp.
            Hoặc phần cứng di chuyển dữ liệu giữa bộ nhớ toàn cục và bộ nhớ chia sẻ ở chế độ nền, đó là mục 15.9.
          </p>
        }
      >
        <p>
          Software pipelining puts the two kinds of instruction side by side; whether they really overlap depends on the
          compiler's estimate of how long each instruction takes when it schedules them, and the compiler knows little
          about conditions at run time. Two alternatives hand the interleaving to hardware. In warp specialisation, some
          warps of the block do only the memory accesses and others only the computation, and the warp scheduler
          interleaves them as it would any warps. Or the hardware moves the data between global and shared memory in the
          background, which is section 15.9.
        </p>
      </Translated>
      <Source section="15.8" pages="367–368" />
      <Ask q={questionsS158['ch15/15.8/q-warpspec']} />

      <h2>
        <Translated vi="Thư viện làm gì, và HIP với Metal có gì">What libraries do, and what HIP and Metal offer</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Thư viện template của NVIDIA đưa ý tưởng này xuống sâu thêm một tầng. Nó đệm kép các ô bộ nhớ chia sẻ, như Fig.
            15.14 làm, và cả các mảnh thanh ghi mà mỗi warp đọc từ đó: trong khi một bộ dải đang cấp cho các FMA, bộ tiếp theo
            được đọc từ bộ nhớ chia sẻ. Đó là cấu trúc cần tìm khi bạn đọc CUTLASS, và từ Hopper trở đi nó còn chia khối thành
            một nhóm warp sản xuất (producer), chuyên nạp, và một nhóm warp tiêu thụ (consumer), chuyên tính: chuyên biệt hóa
            warp.
          </p>
        }
      >
        <p>
          NVIDIA's template library takes the idea one level further down. It double-buffers the shared-memory tiles, as
          Fig. 15.14 does, and also the register fragments each warp reads from them: while one set of strips feeds the
          FMAs, the next is read from shared memory. That is the structure to look for when you read CUTLASS, and from
          Hopper on it also splits the block into a producer warp group, which loads, and a consumer warp group, which
          computes: warp specialisation.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch15-facts A8, A12)', '2026-11-20 (ch15-facts mục A8, A12)')}>
        <Translated
          vi={
            <p>
              CUTLASS: "two tiles are allocated in shared memory" (hai ô được cấp phát trong bộ nhớ chia sẻ) và "two
              fragments are allocated within registers" (hai mảnh được cấp phát trong thanh ghi); từ Hopper, các warp sản
              xuất nạp bằng TMA. NVIDIA khuyên phát "several stages of TMA loads up front" (vài tầng lệnh nạp TMA ngay từ
              đầu).
            </p>
          }
        >
          <p>
            CUTLASS: "two tiles are allocated in shared memory" and "two fragments are allocated within registers"; from
            Hopper, producer warps load with TMA. NVIDIA advises issuing "several stages of TMA loads up front".
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            HIP và Metal không có bộ sao chép phần cứng nào được công bố để đặt vào vòng lặp này, nên Fig. 15.14 bằng phần
            mềm là thứ bạn viết ở đó. Thống kê pipeline của Xcode báo cáo thời gian một lần dispatch tính toán chờ ở rào
            chắn, đó là một cách để thấy pipeline đã tiết kiệm được gì.
          </p>
        }
      >
        <p>
          HIP and Metal have no documented hardware copy to put in this loop, so Fig. 15.14 in software is what you write
          there. Xcode's pipeline statistics report the time a compute dispatch spends waiting at barriers, which is one
          way to see what the pipeline saved.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch15-facts B5, C3; ch06-facts C7)', '2026-11-20 (ch15-facts mục B5, C3; ch06-facts mục C7)')}>
        <Translated
          vi={
            <p>
              Tài liệu của AMD nêu tên <code>__builtin_amdgcn_async_work_group_copy</code>; trình biên dịch ROCm 10.2 không có
              builtin đó. MSL 4.1 không có sao chép bất đồng bộ.
            </p>
          }
        >
          <p>
            AMD's docs name <code>__builtin_amdgcn_async_work_group_copy</code>; the ROCm 10.2 compiler has no such
            builtin. MSL 4.1 has no asynchronous copy.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS158['ch15/15.8/q-no-async']} />
      <Translated
        vi={
          <p>
            Cái giá của hai buffer là bộ nhớ chia sẻ. Với cột đệm của mục 15.6, hãy tính xem các ô của kernel pipeline chiếm
            bao nhiêu, và đối chiếu với thiết bị của bạn.
          </p>
        }
      >
        <p>
          The price of two buffers is shared memory. With the padding column of section 15.6, work out what the pipelined
          kernel's tiles take, and check it against your device.
        </p>
      </Translated>
      <Ask q={questionsS158['ch15/15.8/q-smem']} label={tr('Work it out', 'Tính ra')} />
    </>
  );
}
