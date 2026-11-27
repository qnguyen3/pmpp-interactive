import { useEffect, useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, MATMUL_TIERS, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Aside, Bench, Beyond, Simplification, Source, TierLegend, Today } from '../../components/notes';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { benchRows, fmt, num, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  CORRECT,
  SCHEDULE_LABEL,
  makeProblem,
  runAll,
  runBlock,
  stateAt,
  type Config,
  type Schedule,
  type Step,
} from '../../sim/tiled';

// The kernel as the stepper shows it: the body of labs/ch05/lab2_tiled/solution/kernel.cu,
// which is compiled and tested. Line numbers are what the stepper highlights.
const BODY = `__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
int tx = threadIdx.x, ty = threadIdx.y;
int Row = blockIdx.y * TILE_WIDTH + ty;
int Col = blockIdx.x * TILE_WIDTH + tx;
float Pvalue = 0;
for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {
    Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
    Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
    __syncthreads();   // barrier 1
    for (int k = 0; k < TILE_WIDTH; ++k)
        Pvalue += Mds[ty][k] * Nds[k][tx];
    __syncthreads();   // barrier 2
}
P[Row * Width + Col] = Pvalue;`;
const LINE: Record<Step['op'], number> = {
  loadM: 8, loadN: 9, barrier1: 10, release1: 10, mul: 12, barrier2: 13, release2: 13, store: 15, done: 15,
};

const SIZES = [
  {
    get label() {
      return tr('4 × 4 matrices, 2 × 2 tiles', 'ma trận 4 × 4, ô 2 × 2');
    },
    width: 4,
    tile: 2,
  },
  {
    get label() {
      return tr('6 × 6 matrices, 2 × 2 tiles', 'ma trận 6 × 6, ô 2 × 2');
    },
    width: 6,
    tile: 2,
  },
  {
    get label() {
      return tr('6 × 6 matrices, 3 × 3 tiles', 'ma trận 6 × 6, ô 3 × 3');
    },
    width: 6,
    tile: 3,
  },
];
const SCHEDULES: Schedule[] = ['lockstep', 'in-order', 'reversed', 'shuffled'];
const show = (v: number) => (Number.isNaN(v) ? '?' : String(v));

/** One sentence saying what a step did, with the index arithmetic spelled out. */
function describe(s: Step, cfg: Config, bx: number, by: number): string {
  const T = cfg.tile;
  const W = cfg.width;
  if (s.thread < 0)
    return s.op === 'release1'
      ? tr(
          `Barrier 1 opens: every thread of the block has loaded its two elements for phase ${s.ph}. Now, and only now, is it safe to read the tiles.`,
          `Rào chắn 1 mở: mọi luồng của khối đã nạp xong hai phần tử của mình cho pha ${s.ph}. Bây giờ, và chỉ bây giờ, mới an toàn để đọc các ô.`,
        )
      : tr(
          `Barrier 2 opens: every thread has finished reading the tiles for phase ${s.ph}. Now it is safe to overwrite them.`,
          `Rào chắn 2 mở: mọi luồng đã đọc xong các ô cho pha ${s.ph}. Bây giờ mới an toàn để ghi đè lên chúng.`,
        );
  const who = tr(`Thread (tx ${s.tx}, ty ${s.ty})`, `Luồng (tx ${s.tx}, ty ${s.ty})`);
  const Row = by * T + s.ty;
  const Col = bx * T + s.tx;
  switch (s.op) {
    case 'loadM':
      return tr(
        `${who}, phase ${s.ph}: loads M[${Row}][${s.ph * T + s.tx}] into Mds[${s.ty}][${s.tx}]. Index: Row·Width + ph·T + tx = ${Row}·${W} + ${s.ph}·${T} + ${s.tx} = ${s.global!.index}.`,
        `${who}, pha ${s.ph}: nạp M[${Row}][${s.ph * T + s.tx}] vào Mds[${s.ty}][${s.tx}]. Chỉ số: Row·Width + ph·T + tx = ${Row}·${W} + ${s.ph}·${T} + ${s.tx} = ${s.global!.index}.`,
      );
    case 'loadN':
      return tr(
        `${who}, phase ${s.ph}: loads N[${s.ph * T + s.ty}][${Col}] into Nds[${s.ty}][${s.tx}]. Index: (ph·T + ty)·Width + Col = (${s.ph}·${T} + ${s.ty})·${W} + ${Col} = ${s.global!.index}.`,
        `${who}, pha ${s.ph}: nạp N[${s.ph * T + s.ty}][${Col}] vào Nds[${s.ty}][${s.tx}]. Chỉ số: (ph·T + ty)·Width + Col = (${s.ph}·${T} + ${s.ty})·${W} + ${Col} = ${s.global!.index}.`,
      );
    case 'barrier1':
      return tr(
        `${who} reaches barrier 1 and waits for the rest of the block to finish loading.`,
        `${who} đến rào chắn 1 và chờ phần còn lại của khối nạp xong.`,
      );
    case 'barrier2':
      return tr(
        `${who} reaches barrier 2 and waits for the rest of the block to finish reading.`,
        `${who} đến rào chắn 2 và chờ phần còn lại của khối đọc xong.`,
      );
    case 'mul': {
      const [a, b] = s.reads!;
      return tr(
        `${who}, phase ${s.ph}, k = ${s.k}: Pvalue += Mds[${a.r}][${a.c}] × Nds[${b.r}][${b.c}] = ${show(a.value)} × ${show(b.value)}; running sum ${show(s.pvalue!)}.`,
        `${who}, pha ${s.ph}, k = ${s.k}: Pvalue += Mds[${a.r}][${a.c}] × Nds[${b.r}][${b.c}] = ${show(a.value)} × ${show(b.value)}; tổng đang cộng dồn ${show(s.pvalue!)}.`,
      );
    }
    case 'store':
      return tr(
        `${who} stores its sum ${show(s.pvalue!)} to P[${Row}][${Col}].`,
        `${who} ghi tổng ${show(s.pvalue!)} của nó vào P[${Row}][${Col}].`,
      );
    default:
      return '';
  }
}

function Stepper() {
  const [size, setSize] = useState(0);
  const [bx, setBx] = useState(0);
  const [by, setBy] = useState(0);
  const [schedule, setSchedule] = useState<Schedule>('in-order');
  const [b1, setB1] = useState(true);
  const [b2, setB2] = useState(true);
  const [n, setN] = useState(0);
  const [playing, setPlaying] = useState(false);

  const { width, tile } = SIZES[size];
  const cfg: Config = useMemo(() => ({ ...CORRECT, width, tile, barrier1: b1, barrier2: b2 }), [width, tile, b1, b2]);
  const prob = useMemo(() => makeProblem(width), [width]);
  const locale = useLocale();
  const trace = useMemo(() => runBlock(cfg, prob, bx, by, schedule, 5), [cfg, prob, bx, by, schedule, locale]);
  const total = trace.steps.length;
  const at = Math.min(n, total);
  const st = useMemo(() => stateAt(cfg, trace, at), [cfg, trace, at]);
  const cur: Step | undefined = at > 0 ? trace.steps[at - 1] : undefined;
  const firstBad = trace.violations.length ? trace.violations[0].step + 1 : null;
  const g = width / tile;

  useEffect(() => setN(0), [size, bx, by, schedule, b1, b2]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setN((x) => x + 1), 420);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);

  const nextBarrier = () => {
    for (let i = at; i < total; i++)
      if (trace.steps[i].thread < 0) return setN(i + 1);
    setN(total);
  };

  // Which cells to light up.
  const tileCol0 = cur ? cur.ph * tile : 0;
  const inMTile = (r: number, c: number) => cur && r >= by * tile && r < (by + 1) * tile && c >= tileCol0 && c < tileCol0 + tile;
  const inNTile = (r: number, c: number) => cur && c >= bx * tile && c < (bx + 1) * tile && r >= tileCol0 && r < tileCol0 + tile;
  const hotGlobal = (arr: 'M' | 'N' | 'P', i: number) => cur?.global?.array === arr && cur.global.index === i;
  const slotCls = (arr: 'Mds' | 'Nds', r: number, c: number) => {
    if (cur?.write && cur.write.array === arr && cur.write.r === r && cur.write.c === c) return 'hot';
    const rd = cur?.reads?.find((x) => x.array === arr && x.r === r && x.c === c);
    if (rd) return rd.ok ? 'hot' : 'bad';
    return '';
  };
  const stored = new Map(st.stores.map((s) => [s.index, s.value]));

  return (
    <Bench
      title={tr('Step through one block', 'Chạy từng bước một khối')}
      task={tr(
        'Follow one thread through phase 0, then watch the whole block reach each barrier.',
        'Theo một luồng qua pha 0, rồi xem cả khối đến từng rào chắn.',
      )}
      foot={
        <Translated
          vi={
            <>
              Các giá trị là số nguyên nhỏ để bạn có thể tự kiểm tra các tổng. Số nhỏ ở góc một vị trí trong ô là pha mà
              giá trị nó đang giữ thuộc về. <TierLegend />
            </>
          }
        >
          <>
            Values are small integers so you can check the sums by hand. A small number in the corner of a tile slot is
            the phase whose value it holds. <TierLegend />
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Bài toán">Problem</Translated>
          <select value={size} onChange={(e) => { setSize(+e.target.value); setBx(0); setBy(0); }}>
            {SIZES.map((s, i) => <option key={i} value={i}>{s.label}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Khối (x, y)">Block (x, y)</Translated>
          <select value={`${bx},${by}`} onChange={(e) => { const [x, y] = e.target.value.split(',').map(Number); setBx(x); setBy(y); }}>
            {Array.from({ length: g * g }, (_, i) => [i % g, Math.floor(i / g)]).map(([x, y]) => (
              <option key={`${x},${y}`} value={`${x},${y}`}>({x}, {y})</option>
            ))}
          </select>
        </label>
        <label className="field">
          <Translated vi="Thứ tự luồng">Thread order</Translated>
          <select value={schedule} onChange={(e) => setSchedule(e.target.value as Schedule)}>
            {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
          </select>
        </label>
        <label className="check"><input type="checkbox" checked={b1} onChange={(e) => setB1(e.target.checked)} /> <Translated vi="rào chắn 1">barrier 1</Translated></label>
        <label className="check"><input type="checkbox" checked={b2} onChange={(e) => setB2(e.target.checked)} /> <Translated vi="rào chắn 2">barrier 2</Translated></label>
      </div>

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setN(0); }} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => { setPlaying(false); setN(at - 1); }} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => { setPlaying(false); setN(at + 1); }} disabled={at >= total}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => { setPlaying(false); nextBarrier(); }} disabled={at >= total}><Translated vi="Chạy đến rào chắn tiếp theo">Run to next barrier</Translated></button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play', 'Chạy')}</button>
        {firstBad ? (
          <button onClick={() => { setPlaying(false); setN(firstBad); }}><Translated vi="Đến lỗi đầu tiên">Go to the first failure</Translated></button>
        ) : null}
        <span className="muted" aria-live="polite"><Translated vi={<>bước {at} trên {total}</>}>step {at} of {total}</Translated></span>
      </div>

      <div className={'narrate' + (cur?.violation ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {cur ? (
          <>
            {describe(cur, cfg, bx, by)}
            {cur.violation ? <><br /><b className="t-fail"><Translated vi={<>Lỗi: {cur.violation.message}.</>}>Failure: {cur.violation.message}.</Translated></b></> : null}
          </>
        ) : (
          <Translated
            vi={
              <>Chưa có gì chạy. Khối ({bx}, {by}) sở hữu các hàng {by * tile} đến {by * tile + tile - 1} và các cột {bx * tile} đến {bx * tile + tile - 1} của P. Bấm Bước.</>
            }
          >
            <>Nothing has run yet. Block ({bx}, {by}) owns rows {by * tile} to {by * tile + tile - 1} and columns {bx * tile} to {bx * tile + tile - 1} of P. Press Step.</>
          </Translated>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 5fr) minmax(0, 6fr)' }}>
        <div>
          <Code tabs={[{ label: tr('kernel body', 'thân kernel'), code: BODY, tiers: MATMUL_TIERS }]} highlight={cur ? [LINE[cur.op]] : []} />
          <table className="threads" aria-label={tr('Threads of the block', 'Các luồng của khối')}>
            <thead>
              <tr>
                <th><Translated vi="luồng">thread</Translated></th>
                <th><Translated vi="sở hữu">owns</Translated></th>
                <th><Translated vi="đang làm">doing</Translated></th>
                <th>Pvalue{/* i18n-ignore: variable name */}</th>
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: tile * tile }, (_, t) => {
                const tx = t % tile;
                const ty = Math.floor(t / tile);
                const last = st.last[t] >= 0 ? trace.steps[st.last[t]] : undefined;
                const doing = st.done[t]
                  ? tr('finished', 'đã xong')
                  : st.waiting[t]
                    ? tr(`waiting at barrier ${last?.op === 'barrier1' ? 1 : 2}`, `đang chờ ở rào chắn ${last?.op === 'barrier1' ? 1 : 2}`)
                    : last
                      ? tr(`phase ${last.ph}`, `pha ${last.ph}`)
                      : tr('not started', 'chưa bắt đầu');
                return (
                  <tr key={t} className={cur?.thread === t ? 'cur' : ''}>
                    <td>(tx {tx}, ty {ty}){/* i18n-ignore: thread index names */}</td>
                    <td>P[{by * tile + ty}][{bx * tile + tx}]</td>
                    <td>{doing}</td>
                    <td className="t-reg">{show(st.pvalue[t])}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div>
          <div className="row" style={{ alignItems: 'flex-start', gap: '1.2rem' }}>
            {(['M', 'N'] as const).map((name) => (
              <div key={name}>
                <div className="mx-label t-global"><Translated vi={<>{name} (bộ nhớ toàn cục)</>}>{name} (global memory)</Translated></div>
                <div className="mx global" style={{ gridTemplateColumns: `repeat(${width}, auto)` }}>
                  {(name === 'M' ? prob.M : prob.N).map((v, i) => {
                    const r = Math.floor(i / width);
                    const c = i % width;
                    const on = name === 'M' ? inMTile(r, c) : inNTile(r, c);
                    return <span key={i} className={hotGlobal(name, i) ? 'hot' : on ? 'on' : ''}>{v}</span>;
                  })}
                </div>
              </div>
            ))}
          </div>
          <div className="row" style={{ alignItems: 'flex-start', gap: '1.2rem', marginTop: '1rem' }}>
            {(['Mds', 'Nds'] as const).map((name) => (
              <div key={name}>
                <div className="mx-label t-shared"><Translated vi={<>{name} (chia sẻ, khối này)</>}>{name} (shared, this block)</Translated></div>
                <div className="mx shared" style={{ gridTemplateColumns: `repeat(${tile}, auto)` }}>
                  {(name === 'Mds' ? st.Mds : st.Nds).map((s, i) => (
                    <span key={i} className={slotCls(name, Math.floor(i / tile), i % tile)}>
                      {s.phase < 0 ? '·' : show(s.value)}
                      {s.phase >= 0 ? <sub>{s.phase}</sub> : null}
                    </span>
                  ))}
                </div>
              </div>
            ))}
            <div>
              <div className="mx-label t-global"><Translated vi="P (bộ nhớ toàn cục)">P (global memory)</Translated></div>
              <div className="mx global" style={{ gridTemplateColumns: `repeat(${width}, auto)` }}>
                {prob.ref.map((ref, i) => {
                  const r = Math.floor(i / width);
                  const c = i % width;
                  const mine = r >= by * tile && r < (by + 1) * tile && c >= bx * tile && c < (bx + 1) * tile;
                  const v = stored.get(i);
                  const cls = v === undefined ? (mine ? 'on' : 'void') : v === ref ? 'good' : 'bad';
                  return <span key={i} className={cls}>{v === undefined ? '' : show(v)}</span>;
                })}
              </div>
            </div>
          </div>
          {at >= total ? (
            <p style={{ margin: '0.9rem 0 0' }}>
              {st.stores.every((s) => s.value === prob.ref[s.index]) ? (
                <span className="chip ok"><Translated vi="Kết quả của khối này khớp với M × N.">This block's results match M × N.</Translated></span>
              ) : (
                <span className="chip fail"><Translated vi="Kết quả của khối này sai.">This block's results are wrong.</Translated></span>
              )}{' '}
              {trace.violations.length
                ? tr(`${trace.violations.length} bad reads of the tiles.`, `${trace.violations.length} lần đọc sai từ các ô.`)
                : ''}
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

function Outcomes() {
  const [size, setSize] = useState(0);
  const { width, tile } = SIZES[size];
  const prob = useMemo(() => makeProblem(width), [width]);
  const variants: { label: string; over: Partial<Config> }[] = [
    { label: tr('both barriers', 'cả hai rào chắn'), over: {} },
    { label: tr('barrier 1 removed', 'bỏ rào chắn 1'), over: { barrier1: false } },
    { label: tr('barrier 2 removed', 'bỏ rào chắn 2'), over: { barrier2: false } },
    { label: tr('both removed', 'bỏ cả hai'), over: { barrier1: false, barrier2: false } },
  ];
  return (
    <Bench
      title={tr('Every thread order at once', 'Mọi thứ tự luồng cùng lúc')}
      task={tr(
        'Before you read the table: which column do you expect to be all correct, whatever the barriers?',
        'Trước khi đọc bảng: bạn nghĩ cột nào sẽ đúng hết, bất kể các rào chắn?',
      )}
      foot={tr(
        'Each cell runs the whole multiplication in the model above and compares P with M × N.',
        'Mỗi ô chạy cả phép nhân trong mô hình ở trên và so sánh P với M × N.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Bài toán">Problem</Translated>
          <select value={size} onChange={(e) => setSize(+e.target.value)}>
            {SIZES.map((s, i) => <option key={i} value={i}>{s.label}</option>)}
          </select>
        </label>
      </div>
      <table className="data" style={{ background: 'var(--page)', margin: 0 }}>
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}><Translated vi="Kernel">Kernel</Translated></th>
            {SCHEDULES.map((s) => <th key={s}>{SCHEDULE_LABEL[s]}</th>)}
          </tr>
        </thead>
        <tbody>
          {variants.map((v) => (
            <tr key={v.label}>
              <td style={{ paddingLeft: '0.6rem' }}>{v.label}</td>
              {SCHEDULES.map((s) => {
                const r = runAll({ ...CORRECT, width, tile, ...v.over }, prob, s, 5);
                const kinds = new Set(r.violations.map((x) => x.kind));
                return (
                  <td key={s} data-outcome={r.correct ? 'correct' : 'wrong'}>
                    {r.correct ? <span className="chip ok"><Translated vi="đúng">correct</Translated></span> : <span className="chip fail"><Translated vi={<>sai {r.wrong.length} trên {width * width}</>}>{r.wrong.length} of {width * width} wrong</Translated></span>}
                    {kinds.has('read-before-write') ? <div className="muted"><Translated vi="đọc trước khi ghi">read before write</Translated></div> : null}
                    {kinds.has('overwritten-before-read') ? <div className="muted"><Translated vi="bị ghi đè trước khi đọc">overwritten before read</Translated></div> : null}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </Bench>
  );
}

function MeasuredSpeedup({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = realBackends(reports).flatMap((b) => {
    const rows = benchRows(reports[b], 'ch05/lab2_tiled');
    const naive = num(rows.find((r) => r.name === 'naive')?.gflops);
    const t16 = num(rows.find((r) => r.name === 'tiled' && r.tile === 16)?.gflops);
    return naive && t16 ? [{ b, device: String(reports[b].device.name ?? b), speedup: t16 / naive, all: rows.filter((r) => r.name === 'tiled') , naive }] : [];
  });
  if (!found.length)
    return (
      <Translated
        vi={
          <p>
            Chưa có số đo nào trên máy này. Hãy chạy <code>python3 labs/pmpp.py run 5.2 --solution --bench</code>, rồi{' '}
            <code>./verify.sh</code> để ghi lại, và quay lại đây: ô này sẽ so sánh dự đoán của bạn với kết quả.
          </p>
        }
      >
        <p>
          No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 5.2 --solution --bench</code>, then{' '}
          <code>./verify.sh</code> to record it, and come back: this box will compare your prediction with the result.
        </p>
      </Translated>
    );
  return (
    <>
      {found.map((f) => (
        <p key={f.b}>
          <Translated
            vi={
              <>
                Trên <b>{f.device}</b> ({f.b}), lời giải tham chiếu (solution) đo được <b>{fmt(f.speedup)}×</b> ở T = 16
              </>
            }
          >
            On <b>{f.device}</b> ({f.b}) the reference solution measured <b>{fmt(f.speedup)}×</b> at T = 16
          </Translated>
          {f.all.length > 1 ? (
            <Translated vi={<> (theo độ rộng ô: {f.all.map((r) => `T=${r.tile}: ${fmt((num(r.gflops) ?? 0) / f.naive)}×`).join(', ')})</>}>
              <> (by tile width: {f.all.map((r) => `T=${r.tile}: ${fmt((num(r.gflops) ?? 0) / f.naive)}×`).join(', ')})</>
            </Translated>
          ) : null}.{' '}
          {predicted > f.speedup * 2
            ? tr(
                'You expected considerably more. The factor of T is the reduction in what the kernel asks of global memory; where caches were already serving most of the naive kernel\'s repeated reads, there was less left to save.',
                'Bạn đã kỳ vọng nhiều hơn đáng kể. Hệ số T là mức giảm những gì kernel yêu cầu từ bộ nhớ toàn cục; ở nơi bộ nhớ đệm (cache) vốn đã phục vụ phần lớn các lần đọc lặp lại của kernel ngây thơ, phần còn lại để tiết kiệm ít hơn.',
              )
            : predicted < f.speedup / 2
              ? tr('You expected considerably less.', 'Bạn đã kỳ vọng ít hơn đáng kể.')
              : tr('That is close to what you predicted.', 'Kết quả gần với dự đoán của bạn.')}
        </p>
      ))}
      <Translated
        vi={
          <p className="muted">
            Đây là số đo từ máy của bạn, không phải giá trị kỳ vọng. Một thiết bị khác có thể chênh lệch rất nhiều theo cả
            hai hướng.
          </p>
        }
      >
        <p className="muted">These are measurements from your machine, not expected values. Another device can differ by a wide margin in either direction.</p>
      </Translated>
    </>
  );
}

export const questionsS54: Record<string, Question> = {
  'ch05/5.4/q-index-m': {
    kind: 'numeric',
    id: 'ch05/5.4/q-index-m',
    prompt: (
      <>
        <Translated
          vi={
            <>
              <code>Width = 6</code>, <code>TILE_WIDTH = 3</code>. Ở pha 1, luồng có <code>tx = 2, ty = 1</code> trong khối tại{' '}
              <code>blockIdx = (x 1, y 0)</code> thực hiện lần nạp M. Chỉ số phẳng vào <code>M</code> là bao nhiêu?
            </>
          }
        >
          <code>Width = 6</code>, <code>TILE_WIDTH = 3</code>. In phase 1, the thread with <code>tx = 2, ty = 1</code> in
          the block at <code>blockIdx = (x 1, y 0)</code> executes the M load. What is the flat index into{' '}
          <code>M</code>?
        </Translated>
      </>
    ),
    answer: 11,
    explain: (
      <Translated
        vi={
          <p>
            Row = by·T + ty = 0·3 + 1 = 1. Ô M của pha 1 bắt đầu ở cột ph·T = 3, và luồng này lấy phần tử xa hơn tx = 2 bước:
            cột 5. Chỉ số = Row·Width + 5 = 6 + 5 = 11, tức là M[1][5]. Tọa độ x của khối không xuất hiện: các cột của M mà
            một khối đi qua phụ thuộc vào pha, không phụ thuộc vào việc nó sở hữu những cột nào của P.
          </p>
        }
      >
        <p>
          Row = by·T + ty = 0·3 + 1 = 1. The M tile for phase 1 starts at column ph·T = 3, and this thread takes the
          element tx = 2 further along: column 5. Index = Row·Width + 5 = 6 + 5 = 11, which is M[1][5]. The block's x
          coordinate does not appear: the columns of M a block walks through depend on the phase, not on which columns of
          P it owns.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.4, pp. 109–110',
    vi: { source: 'chương 5 §5.4, tr. 109–110' },
  },
  'ch05/5.4/q-index-n': {
    kind: 'numeric',
    id: 'ch05/5.4/q-index-n',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Cùng luồng, cùng pha (<code>Width = 6</code>, <code>TILE_WIDTH = 3</code>, <code>tx = 2, ty = 1</code>,{' '}
              <code>blockIdx = (x 1, y 0)</code>, <code>ph = 1</code>). Chỉ số phẳng vào <code>N</code> là bao nhiêu?
            </>
          }
        >
          Same thread, same phase (<code>Width = 6</code>, <code>TILE_WIDTH = 3</code>, <code>tx = 2, ty = 1</code>,{' '}
          <code>blockIdx = (x 1, y 0)</code>, <code>ph = 1</code>). What is the flat index into <code>N</code>?
        </Translated>
      </>
    ),
    answer: 29,
    explain: (
      <Translated
        vi={
          <p>
            Col = bx·T + tx = 1·3 + 2 = 5. Ô N của pha 1 bắt đầu ở hàng ph·T = 3, và luồng này lấy phần tử thấp hơn ty = 1
            hàng: hàng 4. Chỉ số = 4·6 + 5 = 29, tức là N[4][5]. Lần này tọa độ y của khối mới là thứ không xuất hiện.
          </p>
        }
      >
        <p>
          Col = bx·T + tx = 1·3 + 2 = 5. The N tile for phase 1 starts at row ph·T = 3, and this thread takes the element
          ty = 1 further down: row 4. Index = 4·6 + 5 = 29, which is N[4][5]. Here it is the block's y coordinate that
          does not appear.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.4, p. 110',
    vi: { source: 'chương 5 §5.4, tr. 110' },
  },
  'ch05/5.4/q-dependence': {
    kind: 'choice',
    id: 'ch05/5.4/q-dependence',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Giả sử bộ nhớ chia sẻ (shared memory) dư dả và bạn cấp cho mỗi pha một cặp ô riêng thay vì dùng lại{' '}
              <code>Mds</code> và <code>Nds</code>. Khi đó bạn có thể xóa rào chắn (barrier) nào mà không làm thay đổi kết quả?
            </>
          }
        >
          Suppose shared memory were plentiful and you gave every phase its own pair of tiles instead of reusing{' '}
          <code>Mds</code> and <code>Nds</code>. Which barrier could you then delete without changing the result?
        </Translated>
      </>
    ),
    options: [
      {
        label: (
          <Translated vi="Rào chắn 1, sau các lần nạp vào Mds và Nds.">
            Barrier 1, after the loads into Mds and Nds.
          </Translated>
        ),
        why: (
          <Translated vi="Một luồng vẫn đọc những vị trí mà các luồng khác ghi. Nó không thể đọc chúng trước khi chúng được ghi, dù chúng nằm ở đâu.">
            A thread still reads slots that other threads write. It cannot read them before they are written, wherever they live.
          </Translated>
        ),
      },
      { label: <Translated vi="Rào chắn 2, sau tích vô hướng.">Barrier 2, after the dot product.</Translated>, correct: true },
      {
        label: (
          <Translated vi="Cả hai: với một cặp ô mới cho mỗi pha, không vị trí nào bị dùng lại.">
            Both: with a fresh pair of tiles per phase, no slot is ever reused.
          </Translated>
        ),
        why: (
          <Translated vi="Một trong hai bảo vệ một phụ thuộc không biến mất khi có thêm bộ nhớ: bên đọc vẫn cần giá trị của bên ghi trước.">
            One of them guards a dependence that does not go away with more memory: a reader still needs the writer's value first.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không cái nào: mỗi pha vẫn ghi các ô mà luồng khác đọc.">
            Neither: every phase still writes tiles that other threads read.
          </Translated>
        ),
        why: (
          <Translated vi="Điều này chỉ đúng với rào chắn 1. Rào chắn 2 tồn tại chỉ vì cùng những vị trí đó được dùng lại.">
            True of barrier 1 only. Barrier 2 exists only because the same locations are reused.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Rào chắn 1 bảo vệ một phụ thuộc <b>thật</b>: bên đọc cần dữ liệu do bên ghi tạo ra (đọc sau ghi). Thêm bao nhiêu bộ
            nhớ cũng không loại bỏ được nó. Rào chắn 2 bảo vệ một phụ thuộc <b>giả</b>: bên ghi của pha sau không cần gì từ bên
            đọc của pha này; chúng chỉ không được phá hủy thứ vẫn đang được đọc (ghi sau đọc). Ghi vào chỗ khác thì xung đột đó
            biến mất. Chương 6 biến nhận xét này thành một kỹ thuật,{' '}
            <a href={href('ch06', '6.7')}>phần 6.7, Buffer kép (double buffering)</a>.
          </p>
        }
      >
        <p>
          Barrier 1 protects a <b>true</b> dependence: readers need data that writers produce (read after write). No
          amount of memory removes it. Barrier 2 protects a <b>false</b> dependence: the next phase's writers need nothing
          from this phase's readers; they only must not destroy what is still being read (write after read). Write
          somewhere else and that conflict is gone. Chapter 6 turns this observation into a technique,{' '}
          <a href={href('ch06', '6.7')}>section 6.7, Double buffering</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.4, pp. 110–111',
    vi: { source: 'chương 5 §5.4, tr. 110–111' },
  },
  'ch05/5.4/q-lockstep': {
    kind: 'choice',
    id: 'ch05/5.4/q-lockstep',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Một đồng nghiệp bỏ cả hai rào chắn, kiểm thử với các ô 4×4 (16 luồng mỗi khối) trên một GPU, và mọi bài kiểm thử
              đều qua, lần nào cũng vậy. Kết luận vững chắc nhất là gì?
            </>
          }
        >
          A colleague removes both barriers, tests with 4×4 tiles (16 threads per block) on one GPU, and every test
          passes, every time. What is the soundest conclusion?
        </Translated>
      </>
    ),
    options: [
      {
        label: (
          <Translated vi="Các rào chắn là không cần thiết với ô nhỏ và có thể bỏ đi khi ô có tối đa 32 luồng.">
            The barriers are unnecessary for small tiles and can be left out when the tile has at most 32 threads.
          </Translated>
        ),
        why: (
          <Translated vi="Điều đó biến một sự tình cờ của một thiết bị thành quy tắc. Độ rộng warp không phải ở đâu cũng là 32, và thứ tự thực thi trong một warp không phải điều ngôn ngữ hứa hẹn.">
            That turns an accident of one device into a rule. The warp width is not 32 everywhere, and execution order within a warp is not something the language promises.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Kernel vẫn sai; bài kiểm thử qua vì một lý do mà mã không kiểm soát.">
            The kernel is still wrong; the test is passing for a reason the code does not control.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Trình biên dịch đã tự động chèn các rào chắn.">
            The compiler inserted the barriers automatically.
          </Translated>
        ),
        why: <Translated vi="Nó không làm vậy.">It does not.</Translated>,
      },
    ],
    explain: (
      <>
        <Translated
          vi={
            <>
              <p>
                Bảng ở trên cho thấy cơ chế: khi mọi luồng của khối tiến cùng nhau, mỗi lần nạp hoàn tất cho tất cả trước khi ai
                đó đọc, và các rào chắn không có việc gì để làm. Một khối vừa trong một warp rất có thể thực thi theo cách đó.
                Nhưng không có gì trong ngôn ngữ nói vậy, độ rộng của warp khác nhau giữa các thiết bị, và với ô lớn hơn, khối trải
                trên nhiều warp có thứ tự tương đối tùy ý.
              </p>
              <p>
                Một bài kiểm thử qua trên GPU là bằng chứng, không phải chứng minh. Đó là lý do các lab còn chạy kernel của bạn
                trên một trình giả lập (emulator) cố ý để một luồng chạy vượt xa các luồng khác.
              </p>
            </>
          }
        >
          <p>
            The table above shows the mechanism: when all threads of the block advance together, each load finishes for
            everyone before anyone reads, and the barriers have nothing to do. A block that fits in a single warp may
            well execute that way. But nothing in the language says so, the width of a warp differs between devices, and
            with a larger tile the block spans several warps whose relative order is arbitrary.
          </p>
          <p>
            A test that passes on a GPU is evidence, not proof. This is why the labs also run your kernel on an emulator
            that deliberately lets one thread run far ahead of the others.
          </p>
        </Translated>
      </>
    ),
    source: 'our analysis; warp widths per vendor in notes/today/ch05-facts.md (B5, C3); book ch. 4 §4.4 on warps',
    vi: {
      source:
        'phân tích của chúng tôi; độ rộng warp theo từng hãng trong notes/today/ch05-facts.md (B5, C3); sách chương 4 §4.4 về warp',
    },
  },
};

export function S54() {
  const cu = useLabFile('ch05/lab2_tiled/solution/kernel.cu');
  const metal = useLabFile('ch05/lab2_tiled/solution/kernel.metal');
  const mm = useLabFile('ch05/lab2_tiled/solution/launch.mm');
  // The labs page's words (LabKit's solutionTabStatus), read from this machine's runs.
  const { reports } = useReports();
  const st = (f: 'kernel.cu' | 'kernel.metal' | 'launch.mm') => solutionTabStatus(reports, 'ch05/lab2_tiled', f);
  return (
    <>
      <Translated
        vi={
          <p>
            Ý tưởng của phần trước gói gọn trong một câu: ở mỗi pha, mọi luồng của một khối nạp một phần tử của M và một của N
            vào bộ nhớ chia sẻ (shared memory), rồi tất cả cùng tính từ bộ nhớ chia sẻ. Đây là nó dưới dạng kernel. Hãy đọc
            theo màu trước: màu hổ phách duy nhất còn lại trong vòng lặp pha là một lần nạp <code>M</code> và một lần nạp{' '}
            <code>N</code> cho mỗi luồng mỗi pha.
          </p>
        }
      >
        <p>
          The idea from the last section fits in one sentence: in each phase, every thread of a block fetches one element
          of M and one of N into shared memory, and then all of them compute from shared memory. Here it is as a kernel.
          Read it by colour first: the only amber left inside the phase loop is one load of <code>M</code> and one of{' '}
          <code>N</code> per thread per phase.
        </p>
      </Translated>
      <Source section="5.4" pages="108" figs={tr('Fig. 5.9', 'Hình 5.9')} />
      {cu && metal && mm ? (
        <Code
          tabs={[
            { label: 'CUDA', code: cu, tiers: MATMUL_TIERS, status: st('kernel.cu') },
            { label: tr('Metal kernel', 'Kernel Metal'), code: metal, tiers: MATMUL_TIERS, status: st('kernel.metal') },
            { label: tr('Metal host', 'Host Metal'), code: mm, tiers: {}, status: st('launch.mm') },
          ]}
          caption={
            <Translated
              vi={
                <>
                  Lời giải tham chiếu (solution) của lab 5.2, nạp từ <code>labs/ch05/lab2_tiled/solution/</code>. Hãy thử
                  tự viết nó trong lab trước khi xem bản này. Bản dựng HIP được thiết kế để biên dịch tệp CUDA không cần
                  sửa; điều đó là unverified, vì chưa có máy HIP nào dựng nó.
                </>
              }
            >
              <>
                The reference solution of lab 5.2, loaded from <code>labs/ch05/lab2_tiled/solution/</code>. Try to write
                it yourself in the lab before studying this one. The HIP build is meant to compile the CUDA file
                unchanged; that is unverified, since no HIP machine has built it yet.
              </>
            </Translated>
          }
        />
      ) : null}

      <h2>
        <Translated vi="Ai nạp gì">Who loads what</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hai câu hỏi giải quyết phép tính chỉ số. Thứ nhất, một luồng sở hữu phần tử nào của P? Khối của nó phủ{' '}
            <code>TILE_WIDTH</code> hàng và cột, và <code>bx</code> khối bên trái đã phủ <code>bx·TILE_WIDTH</code> cột rồi, nên{' '}
            <code>Col = bx·TILE_WIDTH + tx</code> và tương tự <code>Row = by·TILE_WIDTH + ty</code>.
          </p>
        }
      >
        <p>
          Two questions settle the index arithmetic. First, which element of P does a thread own? Its block covers{' '}
          <code>TILE_WIDTH</code> rows and columns, and <code>bx</code> blocks to the left have covered{' '}
          <code>bx·TILE_WIDTH</code> columns already, so <code>Col = bx·TILE_WIDTH + tx</code> and likewise{' '}
          <code>Row = by·TILE_WIDTH + ty</code>.
        </p>
      </Translated>
      <Source section="5.4" pages="108–109" figs={tr('Fig. 5.10', 'Hình 5.10')} />
      <Translated
        vi={
          <p>
            Thứ hai, ở pha <code>ph</code> nó nạp những phần tử nào cho khối? Khối cần một mảnh vuông của M: các hàng của chính
            nó, và <code>TILE_WIDTH</code> cột bắt đầu từ <code>ph·TILE_WIDTH</code>. Nó cần một mảnh vuông của N: các cột của
            chính nó, và các hàng bắt đầu từ <code>ph·TILE_WIDTH</code>. Số luồng đúng bằng số phần tử trong mỗi mảnh, nên mỗi
            luồng lấy phần tử ở đúng vị trí của nó trong ô: dịch <code>tx</code> cột với M, xuống <code>ty</code> hàng với N.
          </p>
        }
      >
        <p>
          Second, which elements does it fetch for the block in phase <code>ph</code>? The block needs a square piece of
          M: its own rows, and the <code>TILE_WIDTH</code> columns starting at <code>ph·TILE_WIDTH</code>. It needs a
          square piece of N: its own columns, and the rows starting at <code>ph·TILE_WIDTH</code>. There are exactly as
          many threads as elements in each piece, so each thread takes the element at its own position within the tile:{' '}
          <code>tx</code> columns along for M, <code>ty</code> rows down for N.
        </p>
      </Translated>
      <Source section="5.4" pages="110" />
      <Ask q={questionsS54['ch05/5.4/q-index-m']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS54['ch05/5.4/q-index-n']} label={tr('Work it out', 'Tự tính')} />

      <h2>
        <Translated vi="Xem nó chạy">Watch it run</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy thân kernel cho mọi luồng của một khối. Hãy bắt đầu với thứ tự luồng đặt ở "mỗi luồng
            chạy xa nhất có thể": nó cho thấy rõ nhất vì sao các luồng phải dừng lại và chờ.
          </p>
        }
      >
        <p>
          The stepper runs the kernel body for all threads of one block. Start with the thread order set to "each thread
          runs as far as it can": it shows most clearly why threads have to stop and wait.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Ở đây một luồng tiến từng phép toán ở mức mã nguồn; luồng thật đan xen ở mức lệnh máy và chạy theo warp. "Lockstep"
              là một warp đơn lý tưởng hóa. Mô hình không có thời gian và không có bộ nhớ đệm (cache). Nó trung thực về một điều:
              luồng thấy giá trị nào khi đọc một vị trí trong ô.
            </p>
          }
        >
          <p>
            A thread advances one source-level operation at a time here; real threads interleave at the instruction level
            and run in warps. "Lockstep" is an idealised single warp. The model has no timing and no caches. It is
            faithful about one thing: which values a thread sees when it reads a tile slot.
          </p>
        </Translated>
      </Simplification>
      <Stepper />

      <h2>
        <Translated vi="Hai rào chắn bảo vệ điều gì">What the two barriers protect</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Rào chắn (barrier) 1 có mặt vì các vị trí một luồng đọc được các luồng khác ghi. Cho đến khi tất cả đã ghi xong,
            bên đọc có thể thấy giá trị của pha trước, hoặc chẳng thấy gì. Bên đọc phụ thuộc vào dữ liệu bên ghi tạo ra: một
            phụ thuộc <b>đọc-sau-ghi</b>, còn gọi là phụ thuộc thật.
          </p>
        }
      >
        <p>
          Barrier 1 is there because the slots a thread reads were written by other threads. Until all of them have
          written, a reader may find the previous phase's value, or nothing. The reader depends on data the writer
          produces: a <b>read-after-write</b> dependence, also called a true dependence.
        </p>
      </Translated>
      <Source section="5.4" pages="110–111" />
      <Translated
        vi={
          <p>
            Rào chắn 2 có mặt vì pha sau dùng lại cùng những vị trí đó. Một luồng xong tổng riêng phần sớm sẽ bắt đầu nạp ô
            tiếp theo đè lên những giá trị mà các luồng bên cạnh chưa đọc. Lần này bên ghi không cần gì từ bên đọc; xung đột
            tồn tại chỉ vì vị trí được dùng chung. Đó là một phụ thuộc <b>ghi-sau-đọc</b>, còn gọi là phụ thuộc giả.
          </p>
        }
      >
        <p>
          Barrier 2 is there because the next phase reuses the same slots. A thread that finishes its partial sum early
          would start loading the next tile over values its neighbours have not read yet. This time the writer needs
          nothing from the readers; the conflict exists only because the location is shared. That is a{' '}
          <b>write-after-read</b> dependence, also called a false dependence.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Trong trình chạy từng bước, bỏ chọn một rào chắn, giữ thứ tự "chạy xa nhất có thể", rồi bấm{' '}
            <i>Đến lỗi đầu tiên</i>. Sau đó đặt thứ tự về lockstep và chạy lại.
          </p>
        }
      >
        <p>
          In the stepper, untick one barrier, keep the "runs as far as it can" order, and press{' '}
          <i>Go to the first failure</i>. Then set the order to lockstep and run it again.
        </p>
      </Translated>
      <Outcomes />
      <Ask q={questionsS54['ch05/5.4/q-dependence']} />
      <Ask q={questionsS54['ch05/5.4/q-lockstep']} />
      <Aside>
        <Translated
          vi={
            <p>
              Cấu trúc vòng lặp này có tên. Cắt một vòng lặp dài thành một vòng ngoài qua các pha và một vòng trong ngắn là{' '}
              <b>strip-mining</b>; với các rào chắn quanh vòng trong, nó khiến mọi luồng của một khối làm việc trên cùng một lát
              đầu vào vào cùng một lúc. Trình biên dịch từ lâu đã dùng nó trên CPU để vector hóa vòng lặp. (Sách tr. 111.)
            </p>
          }
        >
          <p>
            The loop structure has a name. Cutting one long loop into an outer loop over phases and a short inner loop is{' '}
            <b>strip-mining</b>; with barriers around the inner loop it makes all threads of a block work on the same
            slice of the input at the same time. Compilers have long used it on CPUs to vectorise loops. (Book p. 111.)
          </p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Nó mang lại gì">What it buys</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trong mô hình của sách, giờ mỗi phần tử đầu vào được nạp một lần mỗi khối thay vì một lần mỗi luồng, nên tỉ số của
            kernel tăng lên <code>TILE_WIDTH</code> lần: từ 0.25 FLOP/B lên 4 với ô 16×16, 8 với ô 32×32. Trên thiết bị ví dụ
            của sách, ô 32×32 nâng mức trần từ khoảng 840 GFLOP/s lên 26,800, vẫn bị giới hạn bởi bộ nhớ và bằng 40% thông lượng
            đỉnh của thiết bị. Phần chênh lệch còn lại nằm ở{' '}
            <a href={href('ch15', 'start')}>chương 15, Các tối ưu nâng cao cho phép nhân ma trận</a>.
          </p>
        }
      >
        <p>
          In the book's model, each input element is now fetched once per block instead of once per thread, so the
          kernel's ratio rises by a factor of <code>TILE_WIDTH</code>: from 0.25 FLOP/B to 4 with 16×16 tiles, 8 with
          32×32. On the book's example device the 32×32 tiles lift the ceiling from about 840 GFLOP/s to 26,800, which
          is still memory-bound and 40% of the device's peak. The rest of the gap is{' '}
          <a href={href('ch15', 'start')}>chapter 15, Advanced optimizations for matrix multiplication</a>.
        </p>
      </Translated>
      <Source section="5.4" pages="111" />
      <Predict
        id="ch05/5.4/p-speedup"
        prompt={
          <Translated
            vi={
              <>
                Trên thiết bị <em>của bạn</em>, bạn kỳ vọng kernel phân ô với <code>TILE_WIDTH = 16</code> nhanh hơn kernel
                ngây thơ bao nhiêu lần? Hãy chốt một con số trước khi chạy lab 5.2.
              </>
            }
          >
            <>
              On <em>your</em> device, how many times faster than the naive kernel do you expect the tiled kernel to be
              with <code>TILE_WIDTH = 16</code>? Commit a number before running lab 5.2.
            </>
          </Translated>
        }
        unit={tr('× faster', '× nhanh hơn')}
      >
        {(v) => <MeasuredSpeedup predicted={v} />}
      </Predict>
      <Today checked={tr('2026-11-20 (notes/today/ch05-facts.md, A9)', '2026-11-20 (notes/today/ch05-facts.md, mục A9)')}>
        <Translated
          vi={
            <p>
              Sách chỉ đến cuBLAS và CUTLASS cho phép nhân ma trận trong sản phẩm thực tế. Cả hai đều còn hiện hành: cuBLAS đi kèm
              CUDA, còn CUTLASS là một dự án mã nguồn mở được duy trì tích cực gồm các template C++ cho các kernel kiểu GEMM, nên
              đáng đọc khi bạn đã học xong chương 15.
            </p>
          }
        >
          <p>
            The book points to cuBLAS and CUTLASS for production matrix multiplication. Both are current: cuBLAS ships
            with CUDA, and CUTLASS is an actively maintained open-source project of C++ templates for GEMM-like kernels,
            which makes it worth reading once you have finished chapter 15.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Hai nhận xét của sách đặt điều này vào bối cảnh. Phân ô (tiling) không chỉ có nghĩa là bộ nhớ chia sẻ: nó có nghĩa
            là giữ thứ bạn dùng lại trong một bộ nhớ nhanh hơn nơi nó đến từ đó. <code>Pvalue</code> là phần tử đầu ra, được giữ
            trong một thanh ghi (register) trong khi một luồng cập nhật nó nhiều lần, và đó cũng là phân ô (phân ô thanh ghi).
            CPU cũng làm điều tương tự vì cùng lý do, chỉ khác là một lõi CPU chạy một hoặc hai luồng và có thể trông cậy vào bộ
            nhớ đệm để giữ dữ liệu vừa dùng ở gần. Một SM của GPU chạy nhiều luồng cùng lúc, chúng tranh nhau bộ nhớ đệm, nên dữ
            liệu cần dùng lại được chủ động đặt vào bộ nhớ chia sẻ.
          </p>
        }
      >
        <p>
          Two remarks from the book put this in context. Tiling does not only mean shared memory: it means keeping what
          you reuse in a faster memory than where it came from. <code>Pvalue</code> is the output element, held in a
          register while one thread updates it repeatedly, and that is tiling too (register tiling). And CPUs do the same
          thing for the same reason, except that a CPU core runs one or two threads and can count on its cache to keep
          recently used data close. A GPU's SM runs many threads at once, they compete for cache, so data that must be
          reused is placed in shared memory on purpose.
        </p>
      </Translated>
      <Source section="5.4" pages="112" />

      <h2>
        <Translated vi="Cùng kernel trên các backend khác">The same kernel on other backends</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chuyển tab trong đoạn mã ở trên cùng. Kernel Metal là kernel CUDA với những từ khác cho cùng các khái niệm; phần
            khởi chạy mới là chỗ hai bên khác nhau về hình thức.
          </p>
        }
      >
        <p>
          Switch tabs in the listing at the top. The Metal kernel is the CUDA kernel with different words for the same
          things; the launch is where the two differ in shape.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ dùng CUDA. Mọi điều ở đây về HIP và Metal đều được đối chiếu với tài liệu của hãng
              (notes/today/ch05-facts.md) và, với Metal, đã chạy trên một thiết bị.
            </p>
          }
        >
          <p>
            The book uses CUDA only. Everything here about HIP and Metal is checked against vendor documentation
            (notes/today/ch05-facts.md) and, for Metal, run on a device.
          </p>
        </Translated>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr>
            <th><Translated vi="Khái niệm">Concept</Translated></th>
            <th>CUDA</th>
            <th>HIP</th>
            <th>Metal</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="Mảng chia sẻ trong khối">Block-shared array</Translated></td>
            <td><code>__shared__ float a[..]</code></td>
            <td><Translated vi="cách viết giống hệt">same spelling</Translated></td>
            <td><Translated vi={<><code>threadgroup float a[..]</code> trong thân kernel</>}><code>threadgroup float a[..]</code> in the kernel body</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Rào chắn">Barrier</Translated></td>
            <td><code>__syncthreads()</code></td>
            <td><Translated vi="cách viết giống hệt">same spelling</Translated></td>
            <td><code>threadgroup_barrier(mem_flags::mem_threadgroup)</code></td>
          </tr>
          <tr>
            <td><Translated vi="Chỉ số luồng và khối">Thread and block index</Translated></td>
            <td><code>threadIdx</code>, <code>blockIdx</code></td>
            <td><Translated vi="cách viết giống hệt">same spelling</Translated></td>
            <td><Translated vi={<>tham số kernel với <code>[[thread_position_in_threadgroup]]</code> và <code>[[threadgroup_position_in_grid]]</code></>}>kernel parameters with <code>[[thread_position_in_threadgroup]]</code> and <code>[[threadgroup_position_in_grid]]</code></Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Khởi chạy">Launch</Translated></td>
            <td><code>k&lt;&lt;&lt;grid, block&gt;&gt;&gt;(args)</code></td>
            <td><Translated vi="chấp nhận cùng cú pháp">same syntax accepted</Translated></td>
            <td><Translated vi={<>gắn buffer vào một encoder, rồi <code>dispatchThreadgroups:threadsPerThreadgroup:</code></>}>bind buffers on an encoder, then <code>dispatchThreadgroups:threadsPerThreadgroup:</code></Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Độ rộng của đơn vị thực thi cùng nhau">Width of the unit that executes together</Translated></td>
            <td><Translated vi={<><code>warpSize</code> (32 trên thiết bị NVIDIA)</>}><code>warpSize</code> (32 on NVIDIA devices)</Translated></td>
            <td><Translated vi={<><code>warpSize</code>: 64 trên một số kiến trúc AMD, 32 trên các kiến trúc khác</>}><code>warpSize</code>: 64 on some AMD architectures, 32 on others</Translated></td>
            <td><Translated vi={<>đọc <code>threadExecutionWidth</code> từ pipeline</>}>read <code>threadExecutionWidth</code> from the pipeline</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Mức chặt chẽ của dấu phẩy động mặc định">Floating-point strictness by default</Translated></td>
            <td colSpan={2}><Translated vi="không đề cập ở đây">not covered here</Translated></td>
            <td><Translated vi="mặc định không theo IEEE chặt chẽ, nên hãy so sánh với kết quả tham chiếu kèm dung sai">not strict IEEE by default, so compare against a reference with a tolerance</Translated></td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Một hệ quả thực tế cho mục tiêu thứ ba, đọc các backend thật: llama.cpp dựng backend HIP của nó từ chính những tệp{' '}
            <code>.cu</code> của backend CUDA, thông qua một header đổi tên các định danh CUDA sang HIP. Các lab trong khóa học
            này cũng làm vậy (<code>labs/common/include/pmpp/vendors/hip.h</code>), nên với những tính năng chương này dùng, mã
            CUDA bạn viết ở đây cũng chính là mã HIP bạn sẽ phát hành. Điều đó không còn đúng với các intrinsic của warp,
            cooperative groups, các header nửa độ chính xác và các intrinsic của matrix core, những thứ mà các chương sau xử lý
            lần lượt. Đường đi HIP của khóa học là unverified: chưa có máy HIP nào biên dịch nó; xem{' '}
            <a href={href('ch05', 'labs')}>trang bài thực hành</a>.
          </p>
        }
      >
        <p>
          One practical consequence for goal three, reading real backends: llama.cpp builds its HIP backend from the very
          same <code>.cu</code> files as its CUDA backend, through a header that renames CUDA identifiers to HIP ones.
          The labs in this course do the same (<code>labs/common/include/pmpp/vendors/hip.h</code>), so for the features
          this chapter uses the CUDA you write here is the HIP you would ship. That stops being true at warp
          intrinsics, cooperative groups, half-precision headers and matrix-core intrinsics, which later chapters
          treat one at a time. The course's HIP path is unverified: no HIP machine has compiled it yet; see the{' '}
          <a href={href('ch05', 'labs')}>labs page</a>.
        </p>
      </Translated>
    </>
  );
}
