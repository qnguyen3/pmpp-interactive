import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Code, useLabFile } from '../../components/Code';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { elements } from '../../sim/depend';
import {
  BACKENDS,
  BACKEND_LABEL,
  BOOK_PICTURE,
  WAIT_CALL,
  gridStateAt,
  overlap,
  program,
  readOutcomes,
  report,
  run,
  stmtText,
  waitCovers,
  type Backend,
  type Options,
  type ReadOutcome,
  type Stmt,
  type Timeline,
} from '../../sim/timeline';

const num = (v: number) => v.toLocaleString('en-US');
const ticks = (n: number) => tr(`${n} tick${n === 1 ? '' : 's'}`, `${n} nhịp`);


// ---------------------------------------------------------------- interactive: the two-lane timeline
/** One sentence, built from the model, about the statement that has just returned. */
function say(tl: Timeline, done: number, backend: Backend): { text: string; bad: boolean } {
  if (done === 0)
    return {
      text: tr(
        'Nothing has run yet. Execution starts as one host thread, at statement 1. Press Step.',
        'Chưa có gì chạy. Việc thực thi bắt đầu bằng một luồng host, ở câu lệnh 1. Hãy bấm Bước.',
      ),
      bad: false,
    };
  const h = tl.host[done - 1];
  const s = tl.program[h.stmt];
  const len = h.to - h.from;
  if (s.kind === 'host') {
    const launched = tl.grids.filter((g) => g.stmt < h.stmt);
    const during = launched
      .map((g) => ({ grid: g.grid, ran: Math.max(0, Math.min(h.to, g.end) - Math.max(h.from, g.start)) }))
      .filter((x) => x.ran > 0);
    const meanwhile = !launched.length
      ? tr('Nothing has been launched yet, so the device was idle.', 'Chưa có gì được khởi chạy, nên device đang rảnh.')
      : during.length
        ? tr(
            `In that time the device ran ${during.map((x) => `grid ${x.grid} for ${ticks(x.ran)}`).join(' and ')}. Neither side waited for the other.`,
            `Trong thời gian đó device chạy ${during.map((x) => `lưới ${x.grid} trong ${ticks(x.ran)}`).join(' và ')}. Không bên nào chờ bên nào.`,
          )
        : tr(
            'The device was idle: every grid launched so far had already finished.',
            'Device đang rảnh: mọi lưới đã khởi chạy cho đến giờ đều đã xong.',
          );
    return {
      text: tr(`The host thread did its own work for ${ticks(len)}. ${meanwhile}`, `Luồng host làm việc riêng của nó trong ${ticks(len)}. ${meanwhile}`),
      bad: false,
    };
  }
  if (s.kind === 'launch') {
    const ahead = tl.grids.find((g) => g.stmt < h.stmt && g.end > h.to);
    const what = backend === 'metal' ? tr('The commit', 'Lệnh commit') : tr('The launch statement', 'Câu lệnh khởi chạy');
    return {
      text:
        tr(
          `${what} returned at once (drawn as ${ticks(len)}). Grid ${s.grid} is queued: `,
          `${what} trả về ngay (vẽ thành ${ticks(len)}). Lưới ${s.grid} được xếp hàng: `,
        ) +
        (ahead
          ? tr(
              `it will start when grid ${ahead.grid} has finished. The host waited for neither of them.`,
              `nó sẽ bắt đầu khi lưới ${ahead.grid} đã xong. Host không chờ lưới nào trong hai lưới đó.`,
            )
          : tr(
              'none of its threads has run yet, and the host is already at its next statement.',
              'chưa luồng nào của nó chạy, và host đã ở câu lệnh tiếp theo.',
            )),
      bad: false,
    };
  }
  if (s.kind === 'wait') {
    const waited = tl.grids.filter((g) => g.stmt < h.stmt && g.end > h.from).map((g) => tr('grid ', 'lưới ') + g.grid);
    return {
      text: waited.length
        ? tr(
            `The host thread was blocked in ${WAIT_CALL[backend]} for ${ticks(len)}, until ${waited.join(' and ')} had finished. This is the only kind of statement in which the host waits.`,
            `Luồng host bị chặn trong ${WAIT_CALL[backend]} suốt ${ticks(len)}, cho đến khi ${waited.join(' và ')} xong. Đây là loại câu lệnh duy nhất mà host phải chờ.`,
          )
        : tr(
            `${WAIT_CALL[backend]} had nothing left to wait for and returned at once.`,
            `${WAIT_CALL[backend]} không còn gì để chờ và trả về ngay.`,
          ),
      bad: false,
    };
  }
  const outs = readOutcomes(tl, h.stmt);
  if (outs.every((o) => o.outcome === 'guaranteed'))
    return {
      text: tr(
        'The host read the results. A blocking call had returned after every launch, so it read finished values, and it would on every run.',
        'Host đọc kết quả. Một lời gọi chặn đã trả về sau mỗi lần khởi chạy, nên nó đọc được các giá trị đã hoàn tất, và lần chạy nào cũng vậy.',
      ),
      bad: false,
    };
  const parts = outs.map((o) =>
    o.outcome === 'not finished'
      ? tr(
          `Grid ${o.grid} was still ${gridStateAt(tl, o.grid, h.from)}: the host read whatever was in memory at that moment.`,
          `Lưới ${o.grid} vẫn chưa xong (${({ 'not launched': 'chưa khởi chạy', queued: 'đang xếp hàng', running: 'đang chạy', finished: 'đã xong', blocked: 'bị chặn' } as Record<string, string>)[gridStateAt(tl, o.grid, h.from)] ?? gridStateAt(tl, o.grid, h.from)}): host đọc bất cứ thứ gì đang nằm trong bộ nhớ lúc đó.`,
        )
      : o.outcome === 'finished by timing only'
        ? tr(
            `Grid ${o.grid} had finished, but only because the host work happened to take longer than the grid. Nothing in the program makes that so.`,
            `Lưới ${o.grid} đã xong, nhưng chỉ vì việc của host tình cờ kéo dài hơn lưới. Không gì trong chương trình bảo đảm điều đó.`,
          )
        : tr(`Grid ${o.grid} was safe to read.`, `Lưới ${o.grid} đọc an toàn.`),
  );
  return {
    text: tr('The host read the results with no blocking call before it. ', 'Host đọc kết quả mà không có lời gọi chặn nào trước đó. ') + parts.join(' '),
    bad: true,
  };
}

const OUTCOME_TEXT: Record<ReadOutcome, string> = {
  get guaranteed() {
    return tr('finished on every run', 'lần chạy nào cũng xong');
  },
  get 'finished by timing only'() {
    return tr('finished, by timing only', 'đã xong, chỉ nhờ thời điểm');
  },
  get 'not finished'() {
    return tr('not finished', 'chưa xong');
  },
};

const WRAP = { whiteSpace: 'normal' } as const;

/**
 * At phone widths the timeline is wider than its panel and scrolls inside it.
 * Keep x (in viewBox units) in view, so "host is here" never steps off the side.
 */
function useKeepInView(x: number, viewBoxWidth: number) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    const svg = el?.querySelector('svg');
    if (!el || !svg || el.scrollWidth <= el.clientWidth + 1) return;
    const px = (x / viewBoxWidth) * svg.getBoundingClientRect().width;
    el.scrollLeft = Math.max(0, px - el.clientWidth / 2);
  }, [x, viewBoxWidth]);
  return ref;
}

// Timeline geometry (SVG units).
const TW = 920;
const TL = 104; // left edge of time 0
const TR = 16;
const HOST_Y = 46;
const HOST_H = 26;
const DEV_Y = 118;
const DEV_H = 52;
const AXIS_Y = 190;

/** A block of threads, drawn the way the book draws one: a box with a few wavy lines in it. */
function ThreadBlock({ x, y, h, faint }: { x: number; y: number; h: number; faint: boolean }) {
  const waves = Math.max(1, Math.floor((h - 8) / 8));
  const wave = (cx: number) => {
    let d = `M ${cx} ${y + 4}`;
    for (let i = 0; i < waves; i++) d += ` q ${i % 2 ? -3 : 3} 4 0 8`;
    return d;
  };
  return (
    <g opacity={faint ? 0.4 : 0.9}>
      <rect x={x} y={y} width={16} height={h} rx="1.5" fill="none" stroke="var(--ink-2)" strokeWidth="1" />
      {[4, 8, 12].map((dx) => (
        <path key={dx} d={wave(x + dx)} fill="none" stroke="var(--ink-2)" strokeWidth="1" />
      ))}
    </g>
  );
}

function TimelineBench() {
  const [opts, setOpts] = useState<Options>(BOOK_PICTURE);
  const [backend, setBackend] = useState<Backend>('cuda');
  const [step, setStep] = useState(0);
  const prog = useMemo(() => program(opts), [opts]);
  const tl = useMemo(() => run(prog), [prog]);
  const n = prog.length;
  const done = Math.min(step, n);
  const rep = report(tl, done);
  const now = rep.time;
  const said = say(tl, done, backend);
  const reads = readOutcomes(tl, done);
  const finished = done === n;

  // Changing where the host waits changes the program, so start it again. The sliders keep the place.
  const setSwitch = (o: Partial<Options>) => {
    setOpts((cur) => ({ ...cur, ...o }));
    setStep(0);
  };

  const span = Math.max(tl.end, 24);
  const X = (t: number) => TL + (t * (TW - TL - TR)) / span;
  const every = span > 40 ? 10 : 5;
  const text = (s: Stmt, i: number) => stmtText(s, backend, waitCovers(prog, i));
  const scroller = useKeepInView(X(now), TW);

  const hostLane: ReactNode[] = [];
  const arrows: ReactNode[] = [];
  tl.host.slice(0, done).forEach((h) => {
    const x = X(h.from);
    const w = Math.max(2, X(h.to) - x - 1);
    hostLane.push(
      <g key={h.stmt} data-host={h.stmt} data-kind={h.kind} data-state={h.state} data-from={h.from} data-to={h.to}>
        <rect x={x} y={HOST_Y} width={w} height={HOST_H} rx="2"
          fill={h.state === 'blocked' ? 'url(#tl-hatch)' : h.kind === 'host' ? 'var(--ink-2)' : 'var(--ink)'}
          stroke={h.state === 'blocked' ? 'var(--ink-3)' : 'none'} strokeDasharray={h.state === 'blocked' ? '4 3' : undefined} />
        {h.state === 'blocked' && w > 54 ? (
          <text x={x + w / 2} y={HOST_Y + HOST_H / 2 + 4} fontSize="11" textAnchor="middle" fontWeight="600">{tr('blocked', 'bị chặn')}</text>
        ) : null}
        <text x={x + w / 2} y={HOST_Y - 6} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{h.stmt + 1}</text>
      </g>,
    );
    const g = tl.grids.find((q) => q.stmt === h.stmt);
    if (g)
      arrows.push(
        <line key={'launch-' + g.grid} x1={X(g.enqueued)} y1={HOST_Y + HOST_H + 1} x2={X(g.start)} y2={DEV_Y - 2}
          stroke="var(--ink)" strokeWidth="1.3" markerEnd="url(#tl-arrow)" />,
      );
    if (h.kind === 'wait') {
      const last = tl.grids.filter((q) => q.stmt < h.stmt).pop();
      if (last && last.end === h.to)
        arrows.push(
          <line key={'back-' + h.stmt} x1={X(h.to)} y1={DEV_Y - 2} x2={X(h.to)} y2={HOST_Y + HOST_H + 3}
            stroke="var(--ink)" strokeWidth="1.3" markerEnd="url(#tl-arrow)" />,
        );
    }
  });

  const launched = tl.grids.filter((g) => g.stmt < done);
  const deviceLane = launched.map((g) => {
    const state = gridStateAt(tl, g.grid, now);
    const x = X(g.start);
    const w = X(g.end) - x;
    const ran = Math.max(0, Math.min(now, g.end) - g.start);
    const blocks = Math.max(0, Math.min(4, Math.floor((w - 6) / 22)));
    const pad = blocks ? (w - blocks * 16) / (blocks + 1) : 0;
    return (
      <g key={g.grid} data-grid={g.grid} data-state={state} data-start={g.start} data-end={g.end}>
        {ran > 0 ? <rect x={x} y={DEV_Y} width={X(g.start + ran) - x} height={DEV_H} rx="2" fill="var(--rule)" /> : null}
        <rect x={x} y={DEV_Y} width={w} height={DEV_H} rx="2" fill="none" stroke="var(--ink)"
          strokeWidth="1.4" strokeDasharray={state === 'finished' ? undefined : '5 3'} />
        {Array.from({ length: blocks }, (_, b) => (
          <ThreadBlock key={b} x={x + pad + b * (16 + pad)} y={DEV_Y + 7} h={DEV_H - 14}
            faint={g.start + ((b + 0.5) * (g.end - g.start)) / blocks > now && state !== 'finished'} />
        ))}
        <text x={x + w / 2} y={DEV_Y + DEV_H + 13} fontSize="10.5" textAnchor="middle" fontWeight="600">
          <Translated vi={<>lưới {g.grid}</>}>grid {g.grid}</Translated>
          {w > 70 ? tr(`, ${state}`, `, ${({ 'not launched': 'chưa khởi chạy', queued: 'đang xếp hàng', running: 'đang chạy', finished: 'đã xong', blocked: 'bị chặn' } as Record<string, string>)[state] ?? state}`) : ''}
        </text>
      </g>
    );
  });

  return (
    <Bench
      title={tr('One host thread, grids of device threads', 'Một luồng host, các lưới luồng device')}
      task={tr(
        'Step through the program. Before each step, say what state each grid will be in when the statement returns.',
        'Chạy từng bước chương trình. Trước mỗi bước, hãy nói mỗi lưới sẽ ở trạng thái nào khi câu lệnh trả về.',
      )}
      foot={tr(
        'Solid bar: the host thread running. Hatched bar: the host thread blocked inside a call. Dashed box: a grid that has been launched and has not finished. The number above a bar is the statement it belongs to.',
        'Thanh đặc: luồng host đang chạy. Thanh gạch chéo: luồng host bị chặn bên trong một lời gọi. Hộp nét đứt: một lưới đã được khởi chạy và chưa xong. Số phía trên một thanh là câu lệnh mà thanh đó thuộc về.',
      )}
    >
      <div data-timeline data-done={done} data-n={n}>
        <div className="row" style={{ marginBottom: '0.7rem' }}>
          <span className="row" style={{ gap: '0.5rem' }}>
            <span className="muted">
              <Translated vi="Sau mỗi lần khởi chạy:">After each launch:</Translated>
            </span>
            <span className="seg" role="group" aria-label={tr('What follows each launch', 'Điều gì theo sau mỗi lần khởi chạy')}>
              <button data-follow="wait" aria-pressed={opts.waitAfterLaunch} onClick={() => setSwitch({ waitAfterLaunch: true })}>
                <Translated vi="một lời gọi chặn">a blocking call</Translated>
              </button>
              <button data-follow="work" aria-pressed={!opts.waitAfterLaunch} onClick={() => setSwitch({ waitAfterLaunch: false })}>
                <Translated vi="thêm việc cho host">more host work</Translated>
              </button>
            </span>
          </span>
          <label className="check">
            <input type="checkbox" data-b2b checked={opts.backToBack} onChange={(e) => setSwitch({ backToBack: e.target.checked })} />
            <Translated vi="khởi chạy B ngay sau A">launch B straight after A</Translated>
          </label>
          <span className="row" style={{ gap: '0.5rem' }}>
            <span className="muted">
              <Translated vi="Viết cho">Written for</Translated>
            </span>
            <span className="seg" role="group" aria-label={tr('Backend', 'Backend')}>
              {BACKENDS.map((b) => (
                <button key={b} data-backend={b} aria-pressed={backend === b} onClick={() => setBackend(b)}>
                  {BACKEND_LABEL[b]}
                </button>
              ))}
            </span>
          </span>
        </div>
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}>
            <Translated
              vi={
                <span>
                  Một lưới chạy trong <b data-out="grid-ticks">{opts.gridTicks}</b>
                </span>
              }
            >
              <span>
                A grid runs for <b data-out="grid-ticks">{opts.gridTicks}</b>
              </span>
            </Translated>
            <input type="range" min="1" max="12" step="1" value={opts.gridTicks} aria-label={tr('Ticks each grid runs for', 'Số nhịp mỗi lưới chạy')}
              onChange={(e) => setOpts((cur) => ({ ...cur, gridTicks: Number(e.target.value) }))} />
          </label>
          <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}>
            <Translated
              vi={
                <span>
                  Việc của host mất <b data-out="host-ticks">{opts.hostTicks}</b>
                </span>
              }
            >
              <span>
                Host work takes <b data-out="host-ticks">{opts.hostTicks}</b>
              </span>
            </Translated>
            <input type="range" min="1" max="12" step="1" value={opts.hostTicks} aria-label={tr('Ticks each piece of host work takes', 'Số nhịp mỗi phần việc của host mất')}
              onChange={(e) => setOpts((cur) => ({ ...cur, hostTicks: Number(e.target.value) }))} />
          </label>
          <span className="muted">
            <Translated vi="nhịp">ticks</Translated>
          </span>
        </div>

        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <button onClick={() => setStep(0)} disabled={done === 0}>
            <Translated vi="Bắt đầu lại">Restart</Translated>
          </button>
          <button onClick={() => setStep(done - 1)} disabled={done === 0}>
            <Translated vi="Lùi">Back</Translated>
          </button>
          <button className="primary" onClick={() => setStep(done + 1)} disabled={finished}>
            <Translated vi="Bước">Step</Translated>
          </button>
          <button onClick={() => setStep(n)} disabled={finished}>
            <Translated vi="Chạy đến cuối">Run to the end</Translated>
          </button>
          <span className="muted" aria-live="polite" data-out="step">
            <Translated vi={<>{done} trên {n} câu lệnh đã trả về</>}>
              {done} of {n} statements have returned
            </Translated>
          </span>
        </div>

        <div className="panel" style={{ overflowX: 'auto', marginBottom: '0.8rem' }} ref={scroller}>
          <svg viewBox={`0 0 ${TW} 208`} role="img"
            aria-label={tr(
              `Timeline after ${done} of ${n} statements: a host lane above a device lane`,
              `Dòng thời gian sau ${done} trên ${n} câu lệnh: làn host ở trên làn device`,
            )}
            style={{ width: '100%', minWidth: 620, maxWidth: TW, display: 'block' }}>
            <defs>
              <pattern id="tl-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <line x1="0" y1="0" x2="0" y2="6" stroke="var(--ink-3)" strokeWidth="1.6" />
              </pattern>
              <marker id="tl-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">{/* i18n-ignore: SVG attribute */}
                <path d="M 0 0 L 8 4 L 0 8 z" fill="var(--ink)" />
              </marker>
            </defs>
            <text x={TL - 12} y={HOST_Y + 12} fontSize="12" fontWeight="600" textAnchor="end">{tr('Host', 'Host')}</text>
            <text x={TL - 12} y={HOST_Y + 25} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{tr('one thread', 'một luồng')}</text>
            <text x={TL - 12} y={DEV_Y + 24} fontSize="12" fontWeight="600" textAnchor="end">{tr('Device', 'Device')}</text>
            <text x={TL - 12} y={DEV_Y + 37} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{tr('grids of threads', 'các lưới luồng')}</text>
            <line x1={TL} y1={HOST_Y + HOST_H / 2} x2={TW - TR} y2={HOST_Y + HOST_H / 2} stroke="var(--rule)" strokeDasharray="1 4" />
            <rect x={TL} y={DEV_Y} width={TW - TL - TR} height={DEV_H} fill="none" stroke="var(--rule-soft)" />
            {/* time axis */}
            <line x1={TL} y1={AXIS_Y} x2={TW - TR} y2={AXIS_Y} stroke="var(--rule)" />
            {Array.from({ length: Math.floor(span / every) + 1 }, (_, i) => i * every).map((t) => (
              <g key={t}>
                <line x1={X(t)} y1={AXIS_Y} x2={X(t)} y2={AXIS_Y + 4} stroke="var(--ink-3)" />
                <text x={X(t)} y={AXIS_Y + 15} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{t}</text>
              </g>
            ))}
            <text x={TL - 12} y={AXIS_Y + 15} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{tr('ticks', 'nhịp')}</text>
            {deviceLane}
            {hostLane}
            {arrows}
            {/* now */}
            <g data-now={now}>
              <line x1={X(now)} y1={20} x2={X(now)} y2={AXIS_Y} stroke="var(--ink)" strokeWidth="1" strokeDasharray="2 3" />
              <text x={Math.min(X(now), TW - TR - 40)} y={14} fontSize="10.5" fontWeight="600" textAnchor={X(now) > TW - 110 ? 'end' : 'start'}>
                <Translated vi={<>host đang ở đây, t = {now}</>}>host is here, t = {now}</Translated>
              </text>
            </g>
          </svg>
        </div>

        <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 5fr)' }}>
          <div className="panel">
            <table className="threads" aria-label={tr('The program', 'Chương trình')} style={{ whiteSpace: 'normal' }}>
              <thead>
                <tr>
                  <th>#</th>
                  <th>
                    <Translated vi="câu lệnh">statement</Translated>
                  </th>
                  <th>
                    <Translated vi="host">host</Translated>
                  </th>
                  <th>
                    <Translated vi="trả về lúc">returned</Translated>
                  </th>
                </tr>
              </thead>
              <tbody>
                {prog.map((s, i) => {
                  const h = tl.host[i];
                  const ran = i < done;
                  return (
                    <tr key={i} className={i === done - 1 ? 'cur' : ''} data-stmt={i} data-kind={s.kind} data-ran={ran ? 'true' : 'false'}>
                      <td>{i + 1}</td>
                      <td style={{ whiteSpace: 'normal' }}>
                        <code data-out="stmt-text">{text(s, i)}</code>
                      </td>
                      <td>
                        {ran ? (
                          tr(h.state, ({ 'not launched': 'chưa khởi chạy', queued: 'đang xếp hàng', running: 'đang chạy', finished: 'đã xong', blocked: 'bị chặn' } as Record<string, string>)[h.state] ?? h.state)
                        ) : i === done ? (
                          <span className="muted">
                            <Translated vi="tiếp theo">next</Translated>
                          </span>
                        ) : (
                          ''
                        )}
                      </td>
                      <td>{ran ? `t = ${h.to}` : ''}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div>
            <div className={'narrate' + (said.bad ? ' bad' : '')} data-out="said" role="status" style={{ marginBottom: '0.8rem' }}>
              {said.text}
            </div>
            <div className="panel">
              <table className="threads" aria-label={tr('State at this moment', 'Trạng thái tại thời điểm này')}>
                <tbody>
                  <tr>
                    <td style={WRAP}>
                      <Translated vi="Thời gian của host">Host time</Translated>
                    </td>
                    <td>
                      <output data-out="time">t = {now}</output>
                    </td>
                  </tr>
                  {rep.grids.map((g) => (
                    <tr key={g.grid}>
                      <td style={WRAP}>
                        <Translated vi={<>Lưới {g.grid}</>}>Grid {g.grid}</Translated>
                      </td>
                      <td>
                        <output data-out={'grid-' + g.grid} data-state={g.state}>
                          {tr(g.state, ({ 'not launched': 'chưa khởi chạy', queued: 'đang xếp hàng', running: 'đang chạy', finished: 'đã xong', blocked: 'bị chặn' } as Record<string, string>)[g.state] ?? g.state)}
                        </output>
                      </td>
                    </tr>
                  ))}
                  {rep.grids.map((g) => {
                    const r = reads.find((x) => x.grid === g.grid);
                    return (
                      <tr key={'read-' + g.grid}>
                        <td style={WRAP}>
                          <Translated vi={<>Nếu host đọc kết quả của {g.grid} lúc này</>}>
                            If the host read {g.grid}'s results now
                          </Translated>
                        </td>
                        <td>
                          <output data-out={'read-' + g.grid} data-outcome={r ? r.outcome : 'not launched'}>
                            {r ? (
                              <span className={'chip ' + (r.outcome === 'guaranteed' ? 'ok' : 'fail')}>{OUTCOME_TEXT[r.outcome]}</span>
                            ) : (
                              <span className="muted">
                                <Translated vi="chưa khởi chạy">not launched</Translated>
                              </span>
                            )}
                          </output>
                        </td>
                      </tr>
                    );
                  })}
                  {finished ? (
                    <>
                      <tr>
                        <td style={WRAP}>
                          <Translated vi="Device xong lúc">The device finishes at</Translated>
                        </td>
                        <td>
                          <output data-out="device-end">t = {tl.grids[tl.grids.length - 1].end}</output>
                        </td>
                      </tr>
                      <tr>
                        <td style={WRAP}>
                          <Translated vi="Việc host làm được trong khi một lưới chạy">Host work done while a grid ran</Translated>
                        </td>
                        <td>
                          <output data-out="overlap">{ticks(overlap(tl))}</output>
                        </td>
                      </tr>
                    </>
                  ) : null}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the program on three backends
const LAB = 'ch02/lab1_vecadd/solution/';
const KERNEL_TIERS: Record<string, Tier> = { A: 'global', B: 'global', C: 'global', i: 'reg' };

/**
 * The 1-based lines of a .cu file that are device code: each `__global__` or
 * `__device__` function, from the comment lines directly above it to its
 * closing brace in column 0. Everything else in the file is host code.
 */
export function deviceLines(src: string): number[] {
  const lines = src.split('\n');
  const out: number[] = [];
  for (let at = 0; at < lines.length; at++) {
    if (!/^\s*__(global|device)__\b/.test(lines[at])) continue;
    let first = at;
    while (first > 0 && /^\s*\/\//.test(lines[first - 1])) first--;
    let last = at;
    while (last < lines.length - 1 && !/^}/.test(lines[last])) last++;
    for (let i = first; i <= last; i++) out.push(i + 1);
    at = last;
  }
  return out;
}

// ---------------------------------------------------------------- questions
// Numbers the page states, from the models.
const THREADS = elements(8, 2048, 4096);
const LOOP_PROGRAM: Stmt[] = [
  { kind: 'launch', grid: 'A', ticks: 3 },
  ...Array.from({ length: 10 }, (_, i): Stmt => ({ kind: 'launch', grid: 'B' + i, ticks: 3 })),
  { kind: 'wait' },
];
const LOOP_GRIDS = run(LOOP_PROGRAM).grids.length;

export const questionsS22: Record<string, Question> = {
  'ch02/2.2/q-returned': {
    kind: 'choice',
    id: 'ch02/2.2/q-returned',
    prompt: (
      <Translated
        vi={
          <>
            <code>kernelA&lt;&lt;&lt;blocks, 256&gt;&gt;&gt;(x_d, n); auto t1 = now();</code> Điều gì chắc chắn đã xảy ra vào lúc <code>t1</code> được lấy?
          </>
        }
      >
        <code>kernelA&lt;&lt;&lt;blocks, 256&gt;&gt;&gt;(x_d, n); auto t1 = now();</code> What is guaranteed to have
        happened by the time <code>t1</code> is taken?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mọi luồng của lưới đã xong.">
            Every thread of the grid has finished.
          </Translated>
        ),
        why: (
          <Translated vi="Đó sẽ là một lần khởi chạy chờ lưới của nó. Khởi chạy không làm vậy.">
            That would be a launch that waits for its grid. It does not.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Lưới đã bắt đầu chạy, dù có thể chưa xong.">
            The grid has started running, though it may not have finished.
          </Translated>
        ),
        why: (
          <Translated vi="Thậm chí không đến mức đó. Tài liệu nói host không chờ kernel hoàn tất “hay thậm chí bắt đầu”.">
            Not even that. The documentation says the host does not wait for the kernel to complete "or even start".
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Lần khởi chạy đã được giao cho runtime. Không nhất thiết luồng nào của lưới đã chạy.">
            The launch has been handed to the runtime. No thread of the grid need have run.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không gì cả. Câu lệnh chỉ khai báo lần khởi chạy, và lưới chạy ở lời gọi chặn tiếp theo.">
            Nothing. The statement only declares the launch, and the grid runs at the next blocking call.
          </Translated>
        ),
        why: (
          <Translated vi="Lần khởi chạy là thật: kernel được thiết lập để thực thi và được xếp hàng. Một lời gọi chặn không khởi động nó. Lời gọi đó chỉ làm host chờ nó.">
            The launch is real: the kernel is set up for execution and queued. A blocking call does not start it. It only makes the host wait for it.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một lần khởi chạy là bất đồng bộ. Một lần đọc đồng hồ ở dòng tiếp theo đo thời gian câu lệnh khởi chạy mất, không phải thời gian kernel mất, đó chính là sai lầm mà <a href={href('ch02', 'labs')}>lab 2.2</a> được xây quanh. Để đo thời gian kernel, hãy đặt một lời gọi có chờ, như <code>cudaDeviceSynchronize()</code>, trước lần đọc đồng hồ thứ hai.
          </p>
        }
      >
        <p>
          A launch is asynchronous. A clock read on the next line measures how long the launch statement took, not how
          long the kernel takes, which is the mistake <a href={href('ch02', 'labs')}>lab 2.2</a> is built around. To
          time the kernel, put a call that waits, such as <code>cudaDeviceSynchronize()</code>, before the second clock
          reading.
        </p>
      </Translated>
    ),
    source: 'current CUDA documentation (the Today note; notes/today/ch02-facts.md, A5). The book does not say (§2.2, pp. 24–25)',
    vi: { source: 'tài liệu CUDA hiện hành (ghi chú Hôm nay; notes/today/ch02-facts.md, A5). Sách không nói (§2.2, pp. 24–25)' },
  },
  'ch02/2.2/q-inorder': {
    kind: 'choice',
    id: 'ch02/2.2/q-inorder',
    prompt: (
      <Translated
        vi={
          <>
            <code>kA&lt;&lt;&lt;g, b&gt;&gt;&gt;(x_d); kB&lt;&lt;&lt;g, b&gt;&gt;&gt;(x_d); cudaMemcpy(x_h, x_d, bytes,
            cudaMemcpyDeviceToHost);</code>{' '}
            Không có lệnh chờ tường minh nào. <code>kB</code> có thấy những gì <code>kA</code> đã ghi không, và phép sao chép có mang về kết quả của cả hai kernel không?
          </>
        }
      >
        <code>kA&lt;&lt;&lt;g, b&gt;&gt;&gt;(x_d); kB&lt;&lt;&lt;g, b&gt;&gt;&gt;(x_d); cudaMemcpy(x_h, x_d, bytes,
        cudaMemcpyDeviceToHost);</code>{' '}
        There is no explicit wait anywhere. Does <code>kB</code> see what <code>kA</code> wrote, and does the copy
        bring back the results of both kernels?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Có cho cả hai: một stream chạy chúng theo thứ tự phát lệnh, và phép sao chép là lời gọi chặn.">
            Yes to both: one stream runs them in issue order, and the copy blocks.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không: kB có thể chạy trước kA hoặc song song với nó, nên có thể bỏ lỡ những gì kA ghi.">
            No: kB may run before kA or alongside it, so it can miss kA’s writes.
          </Translated>
        ),
        why: (
          <Translated vi="Các lần khởi chạy không chỉ định stream sẽ vào một stream mặc định, và một stream chạy các thao tác của nó theo thứ tự chúng được phát lệnh.">
            Launches that name no stream go into one default stream, and a stream runs its operations in the order they were issued.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="kB thấy những gì kA ghi, nhưng phép sao chép có thể trả về trước khi kB xong.">
            kB sees kA’s writes, but the copy may come back before kB has finished.
          </Translated>
        ),
        why: (
          <Translated vi="Phép sao chép nằm trong cùng stream theo thứ tự, sau cả hai kernel, và cudaMemcpy không trả về cho đến khi sao chép hoàn tất.">
            The copy is in the same in-order stream, behind both kernels, and cudaMemcpy does not return until the copy is complete.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không điều nào, trừ khi có một cudaDeviceSynchronize() sau mỗi lần khởi chạy.">
            Neither, unless a cudaDeviceSynchronize() follows each launch.
          </Translated>
        ),
        why: (
          <Translated vi="Một lệnh chờ tường minh không gây hại ở đây, nhưng không cần thiết.">
            An explicit wait does no harm here, and it is not needed.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Ba sự thật có trong tài liệu kết hợp lại. Các lần khởi chạy không có stream được xếp vào một stream mặc định. Một stream thực thi các thao tác của nó theo thứ tự chúng được đưa vào hàng đợi. <code>cudaMemcpy</code> chặn cho đến khi việc truyền hoàn tất. Vì vậy phép sao chép không thể vượt lên trước các kernel và không thể trả về sớm. Bước cuối là một suy luận: lần kiểm tra sự thật không tìm thấy câu nào trong hướng dẫn hiện hành nói rằng một <code>cudaMemcpy</code> chờ các kernel trước đó. Đó cũng là lý do chương trình hoàn chỉnh của sách, vốn khởi chạy rồi sao chép về mà không chờ ở giữa, trả về kết quả đầy đủ. Trong trình chạy từng bước, đây là "khởi chạy B ngay sau A" với một lời gọi chặn.
          </p>
        }
      >
        <p>
          Three documented facts combine. Launches with no stream are queued into one default stream. A stream executes
          its operations in the order they were enqueued. <code>cudaMemcpy</code> blocks until the transfer is complete.
          So the copy cannot overtake the kernels and cannot return early. The last step is an inference: the facts
          check found no single sentence in the current guide that says a <code>cudaMemcpy</code> waits for earlier
          kernels. It is also why the book's finished program, which launches and then copies back with no wait between,
          returns a complete result. In the stepper this is "launch B straight after A" with a blocking call.
        </p>
      </Translated>
    ),
    source: 'current CUDA documentation (notes/today/ch02-facts.md, A5); the program shape is the book’s Fig. 2.13, p. 38',
    vi: { source: 'tài liệu CUDA hiện hành (notes/today/ch02-facts.md, A5); hình dạng chương trình là Fig. 2.13 của sách, p. 38' },
  },
  'ch02/2.2/q-grid': {
    kind: 'choice',
    id: 'ch02/2.2/q-grid',
    prompt: (
      <Translated
        vi={
          <>
            Một chương trình khởi chạy <code>kernelA</code> một lần rồi khởi chạy <code>kernelB</code> bên trong một vòng lặp host 10 lần lặp. Trong cả lần chạy, có bao nhiêu lưới, bao nhiêu luồng host, và hai lưới của chương trình này có thể chạy mã khác nhau không?
          </>
        }
      >
        A program launches <code>kernelA</code> once and then launches <code>kernelB</code> inside a host loop of 10
        iterations. Over the whole run, how many grids are there, how many host threads, and can two grids of this
        program run different code?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="2 lưới, 1 luồng host. Có, chúng có thể.">
            2 grids, 1 host thread. Yes, they can.
          </Translated>
        ),
        why: (
          <Translated vi="Một lưới thuộc về một lần khởi chạy, không thuộc về một kernel. Mười lần khởi chạy kernelB là mười lưới.">
            A grid belongs to a launch, not to a kernel. Ten launches of kernelB are ten grids.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi={`${LOOP_GRIDS} lưới, 1 luồng host. Có, chúng có thể.`}>
            {`${LOOP_GRIDS} grids, 1 host thread. Yes, they can.`}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi={`${LOOP_GRIDS} lưới, ${LOOP_GRIDS} luồng host. Có, chúng có thể.`}>
            {`${LOOP_GRIDS} grids, ${LOOP_GRIDS} host threads. Yes, they can.`}
          </Translated>
        ),
        why: (
          <Translated vi="Một lần khởi chạy tạo ra các luồng device. Host vẫn là một luồng duy nhất mà việc thực thi bắt đầu.">
            A launch creates device threads. The host is still the one thread that execution started with.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi={`${LOOP_GRIDS} lưới, 1 luồng host. Không: mọi lưới của một chương trình chạy cùng một kernel.`}>
            {`${LOOP_GRIDS} grids, 1 host thread. No: every grid of a program runs the same kernel.`}
          </Translated>
        ),
        why: (
          <Translated vi="Quy tắc cùng kernel đúng bên trong một lưới. Mỗi lần khởi chạy chỉ định kernel riêng của nó.">
            The same-kernel rule holds inside one grid. Each launch names its own kernel.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi lời gọi kernel khởi chạy một lưới mới: 1 + 10 = {LOOP_GRIDS}. Mọi luồng của một lưới chạy kernel đã được gọi, nên lưới đầu tiên chạy <code>kernelA</code> và mười lưới còn lại chạy <code>kernelB</code>. Bản thân vòng lặp host là mã tuần tự bình thường trong một luồng host duy nhất.
          </p>
        }
      >
        <p>
          Every kernel call launches a new grid: 1 + 10 = {LOOP_GRIDS}. All the threads of one grid run the kernel that
          was called, so the first grid runs <code>kernelA</code> and the other ten run <code>kernelB</code>. The host
          loop itself is ordinary sequential code in the single host thread.
        </p>
      </Translated>
    ),
    source: '§2.2, p. 24 (Fig. 2.3); one kernel per grid: §2.5, pp. 32–33',
    vi: { source: '§2.2, p. 24 (Fig. 2.3); một kernel cho mỗi lưới: §2.5, pp. 32–33' },
  },
};

export function S22() {
  const cu = useLabFile(LAB + 'kernel.cu');
  const metal = useLabFile(LAB + 'kernel.metal');
  const mm = useLabFile(LAB + 'launch.mm');
  return (
    <>
      <Translated
        vi={
          <p>
            CUDA C++ là C++ với một ít cú pháp mới và một thư viện hàm, dành cho những máy có cả CPU lẫn GPU. Sách viết ví dụ theo kiểu C đơn giản và dùng thêm tính năng C++ ở những chỗ có ích. Sách khẳng định rằng những gì bạn học với ngôn ngữ này dễ dàng mang sang các ngôn ngữ và nền tảng khác. Khóa học này kiểm chứng khẳng định đó trên HIP và Metal ở mọi chương.
          </p>
        }
      >
        <p>
          CUDA C++ is C++ with a small amount of new syntax and a library of functions, for machines that have both CPUs
          and GPUs. The book writes its examples in a plain C style and brings in C++ features where they help. Its
          claim for the language is that what you learn with it carries over easily to other languages and platforms.
          This course puts that claim to the test on HIP and Metal in every chapter.
        </p>
      </Translated>
      <Source section="2.2" pages="24" />
      <Translated
        vi={
          <p>
            Một chương trình CUDA được viết cho hai loại bộ xử lý. CPU là <b>host</b>. Một GPU là một <b>device</b>, và một máy có thể có nhiều hơn một. Mã host và mã device có thể nằm trong cùng một tệp nguồn. Mã device được đánh dấu bằng các từ khóa CUDA và chứa các <b>kernel</b>: những hàm được thực thi theo kiểu song song dữ liệu. Mọi thứ không được đánh dấu là mã host, nên mọi chương trình C++ bạn từng viết đều đã là một chương trình CUDA. Chỉ là một chương trình không có mã device nào.
          </p>
        }
      >
        <p>
          A CUDA program is written for two kinds of processor. The CPU is the <b>host</b>. A GPU is a <b>device</b>,
          and a machine may have more than one. Host code and device code can sit in the same source file. Device code
          is marked with CUDA keywords and contains the <b>kernels</b>: functions that are executed in a data-parallel
          way. Everything that is not marked is host code, so every C++ program you have written is already a CUDA
          program. It is one with no device code in it.
        </p>
      </Translated>
      <Source section="2.2" pages="24" />

      <h2>
        <Translated vi="Một luồng host, các lưới luồng device">One host thread, grids of device threads</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Việc thực thi bắt đầu trong mã host, như một luồng duy nhất. Khi host gọi một kernel, một số lượng lớn luồng được khởi chạy trên device. Tất cả các luồng được khởi chạy bởi một lời gọi là một <b>lưới (grid)</b>. Một lưới gồm các{' '}<b>khối (block)</b> luồng, mọi khối có cùng kích thước, và mọi luồng của lưới chạy kernel đã được gọi. Khi mọi luồng của nó xong, lưới kết thúc. Lời gọi kernel tiếp theo khởi chạy một lưới mới.
          </p>
        }
      >
        <p>
          Execution starts in host code, as a single thread. When the host calls a kernel, a large number of threads is
          launched on the device. All the threads launched by one call are a <b>grid</b>. A grid is made of{' '}
          <b>blocks</b> of threads, every block the same size, and every thread of the grid runs the kernel that was
          called. When all of its threads have finished, the grid terminates. The next kernel call launches a new grid.
        </p>
      </Translated>
      <Source
        section="2.2"
        pages="24–25"
        figs={tr('Fig. 2.3; same code and equal blocks: §2.5, pp. 32–33', 'Hình 2.3; cùng mã và các khối bằng nhau: §2.5, pp. 32–33')}
      />
      <Translated
        vi={
          <p>
            Luồng host làm gì trong khi một lưới chạy? Hình của sách vẽ luồng host, một mũi tên sang một lưới và một mũi tên quay về. Phần chữ của sách nói việc thực thi tiếp tục trên host khi lưới đã kết thúc, và thêm rằng các ứng dụng có thể cho CPU và GPU thực thi chồng lên nhau. Chương này của sách không bao giờ nói bản thân câu lệnh khởi chạy có chờ hay không, và hình có thể hiểu theo cả hai cách. Tài liệu của các nhà cung cấp nói rõ, và câu trả lời thay đổi cách bạn đọc mọi chương trình từ đây trở đi.
          </p>
        }
      >
        <p>
          What does the host thread do while a grid runs? The book's figure draws the host thread, an arrow across to a
          grid and an arrow back. Its text says execution continues on the host when the grid has terminated, and adds
          that applications can overlap CPU and GPU execution. This chapter of the book never says whether the launch
          statement itself waits, and the figure can be read either way. The vendors' documentation is explicit, and
          the answer changes how you read every program from here on.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch02-facts A5, B3, C4)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              Một lần khởi chạy là bất đồng bộ: host không chờ kernel hoàn tất, thậm chí không chờ nó bắt đầu. Một lời gọi chặn làm việc chờ: <code>cudaDeviceSynchronize()</code> giữ luồng host cho đến khi mọi công việc đã phát lệnh trước nó hoàn tất. HIP mô tả cùng mô hình, với <code>hipDeviceSynchronize()</code>. Trên Metal,{' '}<code>commit</code> trả về ngay và <code>waitUntilCompleted</code> là lời gọi chặn.
            </p>
          }
        >
          <p>
            A launch is asynchronous: the host does not wait for the kernel to complete, or even to start. A blocking
            call does the waiting: <code>cudaDeviceSynchronize()</code> holds the host thread until all work issued
            before it has completed. HIP documents the same model, with <code>hipDeviceSynchronize()</code>. On Metal,{' '}
            <code>commit</code> returns at once and <code>waitUntilCompleted</code> blocks.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy một chương trình có hình dạng như hình của sách: việc của host, một lần khởi chạy <code>KernelA</code>, việc của host, một lần khởi chạy <code>KernelB</code>, và cuối cùng host đọc kết quả. Nó mở đầu với một lời gọi chặn sau mỗi lần khởi chạy, cho ra sự luân phiên chặt chẽ mà phần chữ của sách mô tả. Hãy chạy từng bước cách đó trước. Sau đó chuyển sang "thêm việc cho host" và chạy từng bước lại, rồi khởi chạy B ngay sau A.
          </p>
        }
      >
        <p>
          The stepper runs a program with the shape of the book's figure: host work, a launch of <code>KernelA</code>,
          host work, a launch of <code>KernelB</code>, and at the end the host reads the results. It opens with a
          blocking call after each launch, which gives the strict alternation the book's text describes. Step through
          that first. Then switch to "more host work" and step through again, and then launch B straight after A.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Device chạy mỗi lần một lưới, theo thứ tự khởi chạy, như stream mặc định làm. Stream, thứ cho phép công việc chồng lên nhau trên device, bị bỏ qua. Trên Metal, việc một command buffer xong trước khi cái tiếp theo bắt đầu là giả định của mô hình: Apple chỉ mô tả thứ tự của hàng đợi. Thời lượng là số nhịp do bạn đặt và không có gì được đo. Một nhịp mà một lần khởi chạy mất và một nhịp trước khi một lưới bắt đầu là quy ước vẽ.
            </p>
          }
        >
          <p>
            The device runs one grid at a time, in launch order, as the default stream does. Streams, which let work
            overlap on the device, are left out. On Metal, one command buffer finishing before the next starts is the
            model's assumption: Apple documents only the order of the queue. Durations are ticks you set and nothing is
            measured. The one tick a launch takes and the one tick before a grid starts are drawing conventions.
          </p>
        </Translated>
      </Simplification>
      <TimelineBench />
      <Translated
        vi={
          <p>
            Bốn điều rút ra từ đó. Một câu lệnh khởi chạy trả về với lưới của nó đang xếp hàng: chưa luồng device nào chạy. Host chỉ chờ ở nơi một lời gọi chặn yêu cầu. Hai lần khởi chạy không có lệnh chờ ở giữa vẫn chạy lần lượt trên device, theo thứ tự khởi chạy, trong khi host chạy trước. Và đọc kết quả chỉ an toàn sau một lời gọi chặn theo sau lần khởi chạy. Không có nó, một kết quả đúng là sự tình cờ về thời điểm: kéo thời lượng lưới lên trên thời lượng việc của host và cùng chương trình đó sẽ đọc các giá trị chưa hoàn tất.
          </p>
        }
      >
        <p>
          Four things to take from it. A launch statement returns with its grid queued: no device thread has run yet.
          The host waits only where a blocking call says so. Two launches with no wait between them still run one after
          the other on the device, in launch order, while the host runs ahead. And reading results is safe only after a
          blocking call that follows the launch. Without one, a correct result is an accident of timing: drag the grid
          duration above the host work and the same program reads unfinished values.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Với CUDA và HIP, thứ tự trên device có trong tài liệu chứ không phải một tiện lợi của mô hình. Cả hai đưa mọi lần khởi chạy không chỉ định stream vào một stream mặc định, và một stream chạy các thao tác của nó theo thứ tự chúng được phát lệnh. Apple mô tả ít hơn. Một command queue giữ một danh sách có thứ tự các command buffer, và hình vẽ giả định cả hai được commit vào cùng một hàng đợi. Việc công việc của một command buffer xong trước khi cái tiếp theo bắt đầu là giả định của mô hình cho Metal: Apple mô tả thứ tự của danh sách, không mô tả rằng các buffer chạy lần lượt. Điều Apple có mô tả bao quát trường hợp quan trọng cho tính đúng đắn. Khi một lệnh sau dùng một buffer mà một lệnh trước ghi vào, Metal giữ lệnh sau lại cho đến khi việc ghi xong. Điều đó đúng với các buffer được tạo từ device, là cách các lab của chương này tạo chúng. Với lưu ý đó, công tắc backend trong trình chạy từng bước chỉ thay đổi cách viết các câu lệnh và không thay đổi gì trong hình.
          </p>
        }
      >
        <p>
          For CUDA and HIP the order on the device is documented and not a convenience of the model. Both put every
          launch that names no stream into one default stream, and a stream runs its operations in the order they were
          issued. Apple documents less. A command queue keeps an ordered list of command buffers, and the picture
          assumes both are committed to one queue. That one command buffer's work finishes before the next begins is
          the model's assumption for Metal: Apple documents the order of the list, not that the buffers run one after
          the other. What Apple does document covers the case that matters for correctness. When a later command uses a
          buffer that an earlier one writes, Metal holds the later one back until the write has finished. That holds
          for buffers created from the device, which is how this chapter's labs create them. With that caveat, the
          backend switch in the stepper changes the spelling of the statements and nothing in the picture.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Lấy từ tài liệu của các nhà cung cấp, không phải từ sách (ch02-facts A5, B3, C9; infra-facts C12). Đọc ngày 2026-11-20.
            </p>
          }
        >
          <p>
            From the vendors' documentation, not the book (ch02-facts A5, B3, C9; infra-facts C12). Read on
            2026-11-20.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Chương trình của chương này không thể vô tình đọc sai, vì host lấy kết quả bằng{' '}<code>cudaMemcpy</code>, bản thân nó là một lời gọi chặn. Có hai chỗ bạn có thể làm sai đáng biết ngay bây giờ. Một bộ đếm thời gian dừng ở dòng sau lần khởi chạy đo lần khởi chạy chứ không đo kernel. Và trên Apple silicon, một buffer Metal ở chế độ lưu trữ chia sẻ là bộ nhớ host có thể đọc bất cứ lúc nào, nên ở đó không gì ngoài lệnh chờ đứng giữa host và những giá trị kernel chưa ghi.
          </p>
        }
      >
        <p>
          The program of this chapter cannot get the read wrong by accident, because the host fetches its results with{' '}
          <code>cudaMemcpy</code>, which is itself a blocking call. Two places where you can get it wrong are worth
          knowing now. A timer stopped on the line after a launch measures the launch and not the kernel. And on Apple
          silicon a Metal buffer in shared storage is memory the host can read at any moment, so there nothing but the
          wait stands between the host and values the kernel has not written yet.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cũng lấy từ tài liệu hiện hành (ch02-facts A5; ch05-facts C6). Bạn sẽ gặp bộ đếm thời gian trong lab 2.2 và buffer chia sẻ trong <a href={href('ch02', '2.4')}>mục 2.4</a>.
            </p>
          }
        >
          <p>
            Also from current documentation (ch02-facts A5; ch05-facts C6). You meet the timer in lab 2.2 and the
            shared buffer in <a href={href('ch02', '2.4')}>section 2.4</a>.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS22['ch02/2.2/q-returned']} />
      <Ask q={questionsS22['ch02/2.2/q-inorder']} />

      <h2>
        <Translated vi="Luồng ở đây là gì">What a thread is here</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Từ luồng mang theo nhiều hàm ý cũ, nên sách định nghĩa nó. Một luồng là một cách nhìn đơn giản hóa về cách bộ xử lý thực thi một chương trình tuần tự: mã, vị trí trong mã đang được thực thi, và giá trị các biến của nó. Một luồng device là tuần tự đúng theo nghĩa này, nên bạn có thể đọc một kernel từ trên xuống dưới như câu chuyện của một luồng. Tính song song nằm ở số lượng của chúng. Với một luồng cho mỗi điểm ảnh đầu ra, một ảnh lớn nghĩa là hàng triệu luồng trong một lưới duy nhất. Một luồng cũng có thể tính nhiều đầu ra, điều sách sẽ bàn sau.
          </p>
        }
      >
        <p>
          The word thread comes with baggage, so the book defines it. A thread is a simplified view of how a processor
          executes a sequential program: the code, the point in the code that is being executed, and the values of its
          variables. A device thread is sequential in exactly this sense, so you can read a kernel from top to bottom
          as the story of one thread. The parallelism is in the number of them. With one thread for each output pixel,
          a large image means millions of threads in a single grid. A thread may also compute several outputs, which
          the book takes up later.
        </p>
      </Translated>
      <Source section="2.2" pages="24–25" />
      <Predict
        id="ch02/2.2/p-threads"
        prompt={
          <Translated
            vi={
              <>
                Một phép cộng residual, <code>y[i] = x[i] + r[i]</code>, trên một batch gồm 8 chuỗi 2,048 token với hidden size 4,096, viết với một luồng cho mỗi phần tử. Một lần khởi chạy đó tạo ra bao nhiêu luồng device?
              </>
            }
          >
            <>
              A residual add, <code>y[i] = x[i] + r[i]</code>, over a batch of 8 sequences of 2,048 tokens with hidden
              size 4,096, written with one thread per element. How many device threads does that one launch create?
            </>
          </Translated>
        }
        unit={tr('threads', 'luồng')}
      >
        {(v) => (
          <Translated
            vi={
              <p>
                8 × 2,048 × 4,096 = <b>{num(THREADS)}</b>, một luồng cho mỗi phần tử của tensor.{' '}
                {v === THREADS
                  ? 'Đúng như bạn đã đoán.'
                  : v > 0 && Math.abs(Math.log2(v / THREADS)) < 1
                    ? 'Bạn đã sai lệch không quá hai lần.'
                    : 'Nếu bạn sai nhiều, hãy kiểm tra cả ba chiều đã vào tích chưa: batch, số token và hidden size.'}{' '}
                Khung bên của sách đạt đến hàng triệu luồng với một luồng cho mỗi điểm ảnh của một ảnh lớn. Một tensor activation đạt đến hàng chục triệu theo cùng cách đó, trong một lần khởi chạy.
              </p>
            }
          >
            <p>
              8 × 2,048 × 4,096 = <b>{num(THREADS)}</b>, one thread for each element of the tensor.{' '}
              {v === THREADS
                ? 'That is what you had.'
                : v > 0 && Math.abs(Math.log2(v / THREADS)) < 1
                  ? 'You were within a factor of two.'
                  : 'If you were far off, check that all three dimensions went into the product: batch, tokens and hidden size.'}{' '}
              The book's sidebar reaches millions of threads with one thread per pixel of a large image. A tensor of
              activations reaches tens of millions in the same way, in one launch.
            </p>
          </Translated>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Con số đó sẽ thật vô lý với những luồng bạn biết. Với <code>std::thread</code> hay một thread pool, luồng đắt và ít, và bạn giao cho mỗi luồng một phần việc lớn. Đừng mang thói quen đó sang. Một luồng device gần với một lần lặp của vòng lặp có thanh ghi (register) riêng hơn là một worker mà bạn giao việc cho.
          </p>
        }
      >
        <p>
          That count would be absurd for the threads you know. With <code>std::thread</code> or a thread pool, threads
          are expensive and few, and you hand each one a large piece of work. Do not carry that habit over. A device
          thread is closer to one iteration of a loop that has its own registers than to a worker you hand tasks to.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Phép so sánh với <code>std::thread</code> và thread pool là của khóa học này. Sách nói "các luồng CPU truyền thống".
            </p>
          }
        >
          <p>
            The comparison with <code>std::thread</code> and thread pools is ours. The book says "traditional CPU
            threads".
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Lý do của sách là chi phí: luồng device mất rất ít chu kỳ xung nhịp để tạo và lập lịch, vì phần cứng hỗ trợ việc đó, trong khi một luồng CPU truyền thống thường mất hàng nghìn chu kỳ.
          </p>
        }
      >
        <p>
          The book's reason is cost: device threads take very few clock cycles to create and schedule, because the
          hardware supports it, where a traditional CPU thread typically takes thousands of cycles.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 against the CUDA 13.5 guide (notes/today/ch02-facts.md, A19)',
          '2026-11-20 đối chiếu với hướng dẫn CUDA 13.5 (notes/today/ch02-facts.md, A19)',
        )}
      >
        <Translated
          vi={
            <p>
              Không thể xác nhận. Các trang đã đọc không đưa ra số chu kỳ để tạo một luồng, nên hãy coi đây là khẳng định của sách.
            </p>
          }
        >
          <p>
            Could not be confirmed. The pages read give no cycle counts for creating a thread, so take this as the
            book's claim.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS22['ch02/2.2/q-grid']} />

      <h2>
        <Translated vi="Hai nửa của chương trình nằm ở đâu">Where the two halves of the program live</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Dưới đây là toàn bộ chương trình bạn sẽ viết trong <a href={href('ch02', 'labs')}>lab 2.1</a>, trên ba backend. Đừng đọc chi tiết vội. Chỉ nhìn xem mã host và mã device nằm ở đâu. Trong CUDA cả hai ở trong một tệp: các dòng được đánh dấu là mã device và phần còn lại là mã host. Bản build của lab được thiết lập để biên dịch cùng tệp đó cho HIP qua một header nhỏ đặt phía trước nó. Header đó include{' '}<code>&lt;hip/hip_runtime.h&gt;</code>, thứ mà trình biên dịch của AMD không tự đưa vào, và đổi tên các lời gọi{' '}<code>cuda</code> của mã host sang các bản <code>hip</code> tương ứng. Kernel không cần thay đổi. Trong Metal, hai nửa là hai tệp viết bằng hai ngôn ngữ: kernel viết bằng Metal Shading Language, và phía host là Objective-C++ gọi framework Metal.
          </p>
        }
      >
        <p>
          Below is the whole program you will write in <a href={href('ch02', 'labs')}>lab 2.1</a>, on three backends.
          Do not read the details yet. Look only at where the host code and the device code are. In CUDA both are in
          one file: the marked lines are device code and the rest is host code. The lab build is set up to compile
          that same file for HIP through a small header it puts in front of it. The header includes{' '}
          <code>&lt;hip/hip_runtime.h&gt;</code>, which AMD's compiler does not bring in by default, and renames the{' '}
          <code>cuda</code> calls of the host code to their <code>hip</code> counterparts. The kernel needs no change.
          In Metal the two halves are two files in two languages: the kernel is written in the Metal Shading Language,
          and the host side is Objective-C++ that calls the Metal framework.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ nói về CUDA. Các sự thật về HIP và Metal được đọc trong tài liệu của AMD và Apple ngày 2026-11-20 (ch02-facts B1, B4, C1). Bản build HIP của khóa học chưa được biên dịch trên phần cứng AMD: unverified.
            </p>
          }
        >
          <p>
            The book covers CUDA only. The HIP and Metal facts were read in AMD's and Apple's documentation on
            2026-11-20 (ch02-facts B1, B4, C1). The course's HIP build has not been compiled on AMD hardware:
            unverified.
          </p>
        </Translated>
      </Beyond>
      <Code
        tabs={[
          {
            label: tr('CUDA and HIP: one file', 'CUDA và HIP: một tệp'),
            code: cu ?? '// loading labs/' + LAB + 'kernel.cu',
            tiers: KERNEL_TIERS,
            highlight: cu ? deviceLines(cu) : [],
          },
          { label: tr('Metal: the kernel', 'Metal: kernel'), code: metal ?? '// loading labs/' + LAB + 'kernel.metal', tiers: KERNEL_TIERS },
          { label: tr('Metal: the host', 'Metal: phía host'), code: mm ?? '// loading labs/' + LAB + 'launch.mm', tiers: {} },
        ]}
        caption={
          <Translated
            vi={
              <>
                Lời giải tham chiếu (solution) của lab 2.1, từ <code>labs/{LAB}</code>. Trong thẻ đầu tiên các dòng được đánh dấu là mã device. Hai thẻ còn lại là các tệp riêng: <code>kernel.metal</code> toàn là mã device và{' '}<code>launch.mm</code> toàn là mã host. Mỗi tệp đã được chạy hay chưa, và chạy ở đâu, có trên{' '}<a href={href('ch02', 'labs')}>trang lab</a>.
              </>
            }
          >
            <>
              The reference solution of lab 2.1, from <code>labs/{LAB}</code>. In the first tab the marked lines are the
              device code. The other two tabs are separate files: <code>kernel.metal</code> is all device code and{' '}
              <code>launch.mm</code> is all host code. Whether each file has been run, and where, is on the{' '}
              <a href={href('ch02', 'labs')}>labs page</a>.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Mục tiếp theo bắt đầu với nửa host: hàm bao quanh một lời gọi kernel, và ba phần của nó dùng để làm gì (<a href={href('ch02', '2.3')}>mục 2.3</a>).
          </p>
        }
      >
        <p>
          The next section starts on the host half: the function that surrounds a kernel call, and what its three parts
          are for (<a href={href('ch02', '2.3')}>section 2.3</a>).
        </p>
      </Translated>
    </>
  );
}
