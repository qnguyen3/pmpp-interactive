import { useMemo, useState } from 'react';
import { Code } from '../../components/Code';
import { Bench } from '../../components/notes';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { BOOK, makeGraph } from '../../sim/bfs';
import {
  FORMS,
  FORM_IDS,
  OUTCOME_NAME,
  SCENARIOS,
  UNVISITED,
  canMove,
  classify,
  enumerate,
  isDone,
  replay,
  runOrder,
  type Enumeration,
  type FormId,
  type Move,
  type OutcomeKind,
} from '../../sim/bfsvisit';
import { BFS_TIERS, BfsGraph } from './BfsGraph';

// "Two threads, one neighbour" (section 18.3): step the threads of one level through a visit
// form, one statement at a time, or try every order. The model is src/sim/bfsvisit.ts; each
// form's listing is its region of labs/ch18/snippets/s183_variants.cu.

const TRAP_GRAPH = makeGraph('trap', 'the trap', 5, SCENARIOS.trap.edges, [ // i18n-ignore: graph name, never rendered by this stepper
  { x: 0.08, y: 0.5 }, { x: 0.4, y: 0.12 }, { x: 0.4, y: 0.88 }, { x: 0.82, y: 0.3 }, { x: 0.82, y: 0.88 },
]);
const BOUND = 12; // steps of one thread after which the stepper calls it a loop that does not end

const badge = (k: OutcomeKind) => <span className={'chip ' + (k === 'correct' ? 'ok' : 'fail')} data-outcome={k}>{OUTCOME_NAME[k]}</span>;

export function VisitStepper() {
  const [scId, setScId] = useState<'book' | 'trap'>('trap');
  const [formId, setFormId] = useState<FormId>('plain-append');
  const [moves, setMoves] = useState<Move[]>([]);
  const [spurious, setSpurious] = useState(1);
  const [all, setAll] = useState<Enumeration | null>(null);
  const sc = SCENARIOS[scId];
  const form = FORMS[formId];
  const g = scId === 'book' ? BOOK : TRAP_GRAPH;
  const locale = useLocale();
  const { states, events } = useMemo(() => replay(form, sc, moves), [form, sc, moves, locale]);
  const st = states[states.length - 1];
  const last = events[events.length - 1];
  const spinning = st.threads.findIndex((t) => t.pc !== 'done' && t.steps >= BOUND);
  const done = isDone(st);
  const outcome = done ? classify(sc, st) : null;
  const restart = (keepAll = false) => {
    setMoves([]);
    if (!keepAll) setAll(null);
  };
  const runIn = (order: 'forward' | 'reverse') => {
    const r = runOrder(form, sc, order, BOUND);
    const ids = sc.threads.map((_, i) => i);
    if (order === 'reverse') ids.reverse();
    // The same moves runOrder made: each thread to its end (or to the bound), in turn.
    const mv: Move[] = [];
    for (const id of ids) {
      const n = r.events.filter((e) => e.thread === id).length;
      for (let j = 0; j < n; j++) mv.push({ thread: id });
    }
    setMoves(mv);
  };
  const relevant = new Set([...sc.threads.flatMap((t) => [t.vertex, ...t.nbs])]);
  const vertices = [...relevant].sort((a, b) => a - b);
  return (
    <Bench
      title={tr('Two threads, one neighbour', 'Hai luồng, một đỉnh kề')}
      task={tr(
        'Pick a visit code, then step the threads in an order of your choosing. Can you make it go wrong?',
        'Chọn một đoạn mã thăm đỉnh, rồi cho các luồng chạy từng bước theo thứ tự bạn chọn. Bạn có làm nó sai được không?',
      )}
      foot={
        <Translated
          vi={
            <>
              Mỗi bước chạy một câu lệnh của một luồng. Mức của đỉnh kề được đọc, ghi hoặc hoán đổi trong bộ nhớ toàn cục
              (global memory); <code>e</code> là thanh ghi (register) riêng của mỗi luồng.
            </>
          }
        >
          <>
            Each step runs one statement of one thread. The neighbour's level is read, written or swapped in global
            memory; <code>e</code> is each thread's own register.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Tình huống">Scenario</Translated>
          <select value={scId} onChange={(e) => { setScId(e.target.value as 'book' | 'trap'); restart(); }} aria-label={tr('Scenario', 'Tình huống')} style={{ maxWidth: '100%' }}>
            {Object.values(SCENARIOS).map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </label>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Mã thăm đỉnh">Visit code</Translated>
          <select value={formId} onChange={(e) => { setFormId(e.target.value as FormId); restart(); }} aria-label={tr('Visit code', 'Mã thăm đỉnh')} style={{ maxWidth: '100%' }}>
            {FORM_IDS.map((id) => <option key={id} value={id}>{FORMS[id].label}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <button onClick={() => restart(true)} disabled={!moves.length}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => setMoves(moves.slice(0, -1))} disabled={!moves.length}><Translated vi="Lùi lại">Back</Translated></button>
        <button onClick={() => runIn('forward')}><Translated vi="Chạy theo thứ tự xuôi">Run forward order</Translated></button>
        <button onClick={() => runIn('reverse')}><Translated vi="Chạy theo thứ tự ngược">Run reverse order</Translated></button>
        <span className="muted"><Translated vi={<>{moves.length} bước</>}>{moves.length} step{moves.length === 1 ? '' : 's'}</Translated></span>
      </div>
      <div className={'narrate' + (outcome && outcome.kind !== 'correct' ? ' bad' : spinning >= 0 ? ' bad' : '')} role="status" style={{ marginBottom: '0.8rem' }} data-narrate>
        {last ? last.text : <Translated vi={<>Chưa có gì chạy. Mọi luồng sắp lấy đỉnh kề đầu tiên của mình. Hãy cho một luồng bất kỳ chạy một bước.</>}><>Nothing has run. Every thread is about to take its first neighbour. Step any thread.</></Translated>}
        {spinning >= 0 ? (
          <div style={{ marginTop: '0.3rem' }} data-final="never-terminates">
            {badge('never-terminates')}{' '}
            <Translated vi={<>Luồng trên đỉnh {sc.threads[spinning].vertex} đã đi {st.threads[spinning].steps} bước và quay lại đúng chỗ cũ: nó sẽ lặp mãi mãi.</>}>
              The thread on {sc.threads[spinning].vertex} has taken {st.threads[spinning].steps} steps and is back where it was: it will go round for ever.
            </Translated>
          </div>
        ) : outcome ? (
          <div style={{ marginTop: '0.3rem' }} data-final={outcome.kind}>
            {badge(outcome.kind)} {outcome.detail}.
          </div>
        ) : null}
      </div>

      {/* The listing takes the bench's full width, so its longest lines (the signature, the loop
          header) show whole and the stepped line is never the part cut off. */}
      <div data-visit-code>
        <Code tabs={[{ label: form.region, code: form.code.join('\n'), tiers: BFS_TIERS, status: tr('emulator-checked in forward and reverse order', 'emulator-checked theo thứ tự xuôi và ngược') }]} highlight={last ? [last.line + 1] : []} />
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 5fr)' }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ overflowX: 'auto' }}>
          <table className="threads" aria-label={tr('Threads', 'Các luồng')}>
            <thead>
              <tr><th><Translated vi="luồng trên đỉnh">thread on</Translated></th><th><Translated vi="các đỉnh kề">neighbours</Translated></th><th><Translated vi="đang ở">now at</Translated></th><th>e</th><th /></tr>
            </thead>
            <tbody>
              {sc.threads.map((t, i) => {
                const ts = st.threads[i];
                const at = ts.pc === 'done' ? tr('returned', 'đã trả về') : ts.pc === 'next' ? tr(`next neighbour (${ts.k + 1} of ${t.nbs.length})`, `đỉnh kề kế tiếp (${ts.k + 1} / ${t.nbs.length})`) : tr(`${ts.pc} on ${ts.nb}`, `${ts.pc} trên ${ts.nb}`);
                const canSp = canMove(form, st, { thread: i, spurious: true });
                return (
                  <tr key={i} className={last?.thread === i ? 'cur' : ''} data-thread={i}>
                    <td>{t.vertex}</td>
                    <td>{t.nbs.join(', ')}</td>
                    <td>{at}</td>
                    <td className="t-reg">{ts.e === UNVISITED ? 'UNVISITED' : ts.e}</td>
                    <td>
                      <span className="row" style={{ gap: '0.3rem', flexWrap: 'nowrap' }}>
                        <button onClick={() => setMoves([...moves, { thread: i }])} disabled={ts.pc === 'done'}><Translated vi={<>Bước {t.vertex}</>}>Step {t.vertex}</Translated></button>
                        {form.weak ? (
                          <button className="quiet" onClick={() => setMoves([...moves, { thread: i, spurious: true }])} disabled={!canSp}
                            title={tr('Make this weak compare-and-swap fail although the values are equal', 'Cho phép compare-and-swap yếu này thất bại dù các giá trị bằng nhau')}>
                            <Translated vi="thất bại giả">fail spuriously</Translated>
                          </button>
                        ) : null}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          </div>
        </div>
        <div style={{ minWidth: 0 }}>
          <BfsGraph
            g={g}
            label={tr("The scenario's graph with the current levels", 'Đồ thị của tình huống cùng các mức hiện tại')}
            height={200}
            look={(v) => (!relevant.has(v) && scId === 'book' ? 'dim' : st.level[v] === UNVISITED ? 'unvisited' : sc.level0[v] === UNVISITED ? 'new' : 'visited')}
            tag={(v) => (st.level[v] === UNVISITED ? null : String(st.level[v]))}
            ring={(v) => (st.writes[v] > 1 || (sc.level0[v] !== UNVISITED && st.level[v] !== sc.level0[v]) ? 'fail' : null)}
            edgeLook={(_, a) => (sc.threads.some((t) => t.vertex === a) ? 'plain' : 'dim')}
          />
          <div className="mx-label t-global" style={{ marginTop: '0.5rem' }}><Translated vi="level[] (bộ nhớ toàn cục)">level[] (global memory)</Translated></div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${vertices.length}, auto)` }} data-levels>
            {vertices.map((v) => {
              const changed = st.level[v] !== sc.level0[v];
              const bad = st.writes[v] > 1 || (sc.level0[v] !== UNVISITED && changed);
              return (
                <span key={v} data-v={v} className={bad ? 'bad' : changed ? 'hot' : ''} title={tr(`level[${v}], written ${st.writes[v]} times`, `level[${v}], được ghi ${st.writes[v]} lần`)}>
                  {st.level[v] === UNVISITED ? '·' : st.level[v]}
                  <sub>{v}</sub>
                </span>
              );
            })}
          </div>
          <p className="muted" style={{ margin: '0.3rem 0 0.6rem' }}><Translated vi="Số nhỏ là id của đỉnh; · là UNVISITED. Màu đỏ: bị ghi hai lần, hoặc một mức cũ hơn bị ghi đè.">Small numbers are vertex ids; · is UNVISITED. Red: written twice, or an older level overwritten.</Translated></p>
          {form.appends ? (
            <>
              <div className="mx-label t-global"><Translated vi={<>currFrontier (bộ nhớ toàn cục), bộ đếm = {st.counter}</>}>currFrontier (global memory), counter = {st.counter}</Translated></div>
              <div className="mx global" style={{ gridTemplateColumns: `repeat(${Math.max(1, st.frontier.length)}, auto)` }} data-frontier={st.frontier.join(',')}>
                {st.frontier.length ? st.frontier.map((v, i) => (
                  <span key={i} className={st.frontier.indexOf(v) !== i ? 'bad' : 'hot'}>{v}<sub>{i}</sub></span>
                )) : <span className="void"><Translated vi="trống">empty</Translated></span>}
              </div>
            </>
          ) : (
            <p className="muted" style={{ margin: 0 }}><Translated vi="Không có frontier: đoạn mã này chỉ gán nhãn.">No frontier: this code only labels.</Translated></p>
          )}
        </div>
      </div>

      <div className="panel" style={{ marginTop: '0.9rem' }}>
        <div className="row">
          <button className="primary" onClick={() => setAll(enumerate(form, sc, { spurious: form.weak ? spurious : 0 }))}><Translated vi="Thử mọi thứ tự">Try every order</Translated></button>
          {form.weak ? (
            <label className="field">
              <Translated vi="Số lần thất bại giả cho phép">Spurious failures allowed</Translated>
              <span className="seg" role="group" aria-label={tr('Spurious failures allowed', 'Số lần thất bại giả cho phép')}>
                {[0, 1, 2].map((k) => <button key={k} aria-pressed={spurious === k} onClick={() => { setSpurious(k); setAll(null); }}>{k}</button>)}
              </span>
            </label>
          ) : null}
        </div>
        {all ? (
          <div style={{ marginTop: '0.6rem' }} data-every={all.outcomes.map((o) => o.kind).join(',')}>
            <p style={{ margin: '0 0 0.4rem' }}>
              {all.outcomes.length === 1 && all.outcomes[0].kind === 'correct'
                ? <Translated vi={<>Mọi thứ tự đều đúng (đã khám phá {all.states} trạng thái phân biệt).</>}><>Every order is correct ({all.states} distinct states explored).</></Translated>
                : <Translated vi={<>Trên mọi thứ tự ({all.states} trạng thái phân biệt), các kết quả là:</>}><>Over every order ({all.states} distinct states), the outcomes are:</></Translated>}
            </p>
            {all.outcomes.length > 1 || all.outcomes[0].kind !== 'correct' ? (
              <ul style={{ margin: 0, paddingLeft: '1.2rem' }}>
                {all.outcomes.map((o) => (
                  <li key={o.kind} style={{ marginBottom: '0.3rem' }}>
                    {badge(o.kind)} {o.detail}.{' '}
                    {o.example.length ? (
                      <button className="quiet" onClick={() => setMoves(o.example)}><Translated vi={<>Nạp thứ tự này ({o.example.length} bước)</>}>Load this order ({o.example.length} steps)</Translated></button>
                    ) : null}
                  </li>
                ))}
              </ul>
            ) : null}
          </div>
        ) : null}
      </div>
    </Bench>
  );
}
