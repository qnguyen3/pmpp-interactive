import { useState, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { solutionTabStatus } from '../../components/LabKit';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  ANSWERS,
  BACKENDS,
  BACKEND_NAME,
  CALLEES,
  CALLEE_TEXT,
  CALLERS,
  CALLER_TEXT,
  DYNAMIC_PARALLELISM,
  EXECUTOR_TEXT,
  METAL_FNS,
  METAL_FN_TEXT,
  PROGRAM,
  QUALIFIER_TEXT,
  canRunOnNewArchitecture,
  cleanArchs,
  compilesOnUsersMachine,
  functionSource,
  grade,
  legalCall,
  metalCall,
  placement,
  stages,
  type Answer,
  type Backend,
  type Box,
  type Caller,
  type Chip,
  type Qualifiers,
} from '../../sim/toolchain';

const LAB = 'ch02/lab1_vecadd';
// The shared helper is shown from lab 2.3's reference solution. S27.snippets.test.ts
// checks that these functions are in the files and do what the lesson says.
const HELPER_LAB = 'ch02/lab3_elementwise';
const HELPER_CU = ['scaleShift', 'scaleShiftOnHost', 'scaleShiftKernel'] as const;
const HELPER_METAL = ['scale_shift', 'scale_shift_k'] as const;
const HELPER_TIERS: Record<string, Tier> = { X: 'global', Y: 'global', i: 'reg' };
const fnOf = (name: string) => PROGRAM.find((f) => f.name === name)!;
const facts = (ids: string[]) => ids.filter((f) => !f.startsWith('Fig.') && !f.startsWith('p.')).join(', ');

// ---------------------------------------------------------------- interactive A: follow one source file
const ON_USERS_MACHINE: Record<ReturnType<typeof compilesOnUsersMachine>, string> = {
  get always() {
    return tr(
      'always: every kernel is finished there, when its pipeline state is created',
      'luôn luôn: mọi kernel được hoàn tất ở đó, khi pipeline state của nó được tạo',
    );
  },
  get 'when-needed'() {
    return tr(
      'only when it has to: the driver compiles the PTX if no cubin in the binary suits the GPU',
      'chỉ khi buộc phải: driver biên dịch PTX nếu không cubin nào trong tệp nhị phân hợp với GPU',
    );
  },
  get never() {
    return tr(
      'never: the binary holds finished machine code for the architectures named at build time',
      'không bao giờ: tệp nhị phân chứa mã máy hoàn chỉnh cho các kiến trúc được chỉ định lúc build',
    );
  },
};

function ChipView({ chip, backend }: { chip: Chip; backend: Backend }) {
  const fn = fnOf(chip.fn);
  const both = fn.quals === 'host-device';
  const again = backend === 'metal' && chip.copy != null && placement(fn, 'metal').find((p) => p.space === chip.copy)?.writtenAgain;
  return (
    <span className="chip" data-chip={chip.fn} data-copy={chip.copy ?? 'source'}
      style={{ border: '1px solid var(--ink-3)', background: 'var(--cell)', fontWeight: both ? 700 : 500, margin: '0.15rem 0.3rem 0.15rem 0' }}>
      {chip.fn}
      {chip.copy == null ? (
        <span className="muted" style={{ fontWeight: 400 }}> {fn.quals === 'none' ? tr('no qualifier', 'không có qualifier') : QUALIFIER_TEXT[fn.quals]}</span>
      ) : both ? (
        <span className="muted" style={{ fontWeight: 400 }}>
          {' '}
          {backend === 'metal'
            ? again
              ? tr('written again here', 'được viết lại ở đây')
              : tr('written here', 'được viết ở đây')
            : tr(`${chip.copy} version`, `phiên bản ${chip.copy}`)}
        </span>
      ) : null}
    </span>
  );
}

function BoxView({ box, backend }: { box: Box; backend: Backend }) {
  const tone = box.state === 'fail' ? 'var(--fail)' : box.state === 'ok' ? 'var(--ok)' : 'var(--rule)';
  return (
    <div className="panel" data-box={box.id} data-state={box.state ?? 'plain'}
      style={{
        borderColor: tone, borderStyle: box.state === 'unused' ? 'dashed' : 'solid', minWidth: '11rem', flex: '1 1 12rem',
        background: box.state === 'fail' ? 'var(--fail-wash)' : undefined, opacity: box.state === 'unused' ? 0.7 : 1,
      }}>
      <div style={{ fontWeight: 650 }}>{box.title}</div>
      <div className="muted" style={{ marginBottom: '0.4rem' }}>{box.form}</div>
      <div>
        {box.chips.map((c) => <ChipView key={c.fn + (c.copy ?? '')} chip={c} backend={backend} />)}
        {box.chips.length === 0 ? <span className="t-fail">
            <Translated vi="không có gì để nạp">nothing to load</Translated>
          </span> : null}
      </div>
    </div>
  );
}

/** Boxes in order, with neighbours that share a group drawn inside one frame. */
function Boxes({ boxes, backend }: { boxes: Box[]; backend: Backend }) {
  const runs: { group?: string; boxes: Box[] }[] = [];
  for (const b of boxes) {
    const last = runs[runs.length - 1];
    if (last && last.group === b.group && b.group) last.boxes.push(b);
    else runs.push({ group: b.group, boxes: [b] });
  }
  return (
    <div className="row" style={{ alignItems: 'stretch', gap: '0.7rem' }}>
      {runs.map((r, k) =>
        r.group ? (
          <div key={k} data-group={r.group} style={{ border: '2px solid var(--ink-2)', borderRadius: 5, padding: '0.5rem 0.6rem 0.6rem', flex: '1 1 100%' }}>
            <div className="muted" style={{ fontWeight: 650, marginBottom: '0.4rem' }}>{r.group}</div>
            <div className="row" style={{ alignItems: 'stretch', gap: '0.7rem' }}>
              {r.boxes.map((b) => <BoxView key={b.id} box={b} backend={backend} />)}
            </div>
          </div>
        ) : (
          r.boxes.map((b) => <BoxView key={b.id} box={b} backend={backend} />)
        ),
      )}
    </div>
  );
}

function FollowFile() {
  const [backend, setBackend] = useState<Backend>('cuda');
  const [at, setAt] = useState(0);
  const [cubinFits, setCubinFits] = useState(true);
  const [archText, setArchText] = useState('');
  const [deviceArch, setDeviceArch] = useState('');
  const [metalLibrary, setMetalLibrary] = useState<'build' | 'run'>('run');

  const archs = cleanArchs(archText.split(/[\s,]+/)).slice(0, 2);
  const list = stages(backend, { cubinFits, archs, deviceArch, metalLibrary });
  const k = Math.min(at, list.length - 1);
  const stage = list[k];
  const choose = (b: Backend) => {
    setBackend(b);
    setAt(0);
  };

  return (
    <Bench
      title={tr('Follow one source file', 'Theo dấu một tệp nguồn')}
      task={tr(
        "Step through each backend's stages. At every stage, say which compiler has seen each function, and where add1 is.",
        'Đi qua từng giai đoạn của mỗi backend. Ở mỗi giai đoạn, hãy nói trình biên dịch nào đã thấy mỗi hàm, và add1 đang ở đâu.',
      )}
      foot={tr(
        "The program is this chapter's: main, the host function vecAdd, the kernel, and a one-line helper add1 that both the kernel and a CPU reference call. Each stage cites the fact it comes from (notes/today/ch02-facts.md).",
        'Chương trình là của chương này: main, hàm host vecAdd, kernel, và một hàm phụ một dòng add1 mà cả kernel lẫn bản tham chiếu CPU đều gọi. Mỗi giai đoạn dẫn ra sự thật mà nó dựa vào (notes/today/ch02-facts.md).',
      )}
    >
      <div data-follow data-backend={backend} data-stage={stage.id}>
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <span className="seg" role="group" aria-label={tr('Backend', 'Backend')}>
            {BACKENDS.map((b) => (
              <button key={b} data-backend-choice={b} aria-pressed={b === backend} onClick={() => choose(b)}>{BACKEND_NAME[b]}</button>
            ))}
          </span>
          {backend === 'cuda' ? (
            <label className="check">
              <input type="checkbox" checked={cubinFits} data-in="cubin-fits" onChange={(e) => setCubinFits(e.target.checked)} />
              <Translated vi="một cubin trong tệp nhị phân hợp với GPU mà nó chạy trên">
                a cubin in the binary suits the GPU it runs on
              </Translated>
            </label>
          ) : null}
          {backend === 'metal' ? (
            <span className="row" style={{ gap: '0.5rem' }}>
              <span className="muted">
                <Translated vi="Thư viện được tạo">The library is made</Translated>
              </span>
              <span className="seg" role="group" aria-label={tr('When the Metal library is made', 'Khi nào thư viện Metal được tạo')}>
                <button data-metal-library="build" aria-pressed={metalLibrary === 'build'} onClick={() => setMetalLibrary('build')}>
                  <Translated vi="lúc build">at build time</Translated>
                </button>
                <button data-metal-library="run" aria-pressed={metalLibrary === 'run'} onClick={() => setMetalLibrary('run')}>
                  <Translated vi="lúc chạy">at run time</Translated>
                </button>
              </span>
            </span>
          ) : null}
        </div>
        {backend === 'hip' ? (
          <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
            <label className="field">
              <Translated vi="Các kiến trúc truyền cho --offload-arch (một hoặc hai tên)">
                Architectures given to --offload-arch (one or two names)
              </Translated>
              <input type="text" value={archText} data-in="archs" placeholder={tr('type a name, or two with a comma', 'gõ một tên, hoặc hai tên cách nhau bởi dấu phẩy')} style={{ width: '17rem' }}
                onChange={(e) => setArchText(e.target.value)} />
            </label>
            <label className="field">
              <Translated vi="Kiến trúc của GPU mà chương trình chạy trên">Architecture of the GPU the program runs on</Translated>
              <input type="text" value={deviceArch} data-in="device-arch" placeholder={tr('type a name', 'gõ một tên')} style={{ width: '11rem' }}
                onChange={(e) => setDeviceArch(e.target.value)} />
            </label>
            <span className="muted" style={{ maxWidth: '24rem' }}>
              <Translated vi="AMD đặt tên kiến trúc theo dạng gfx942, ví dụ trong tài liệu của họ cho một dòng sản phẩm. Ở đây tên nào cũng được: mô hình chỉ so sánh chúng.">
                AMD names architectures in the form gfx942, its documentation's example for one product line. Any names
                will do here: the model only compares them.
              </Translated>
            </span>
          </div>
        ) : null}

        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <button onClick={() => setAt(k - 1)} disabled={k === 0}>
            <Translated vi="Lùi">Back</Translated>
          </button>
          <button className="primary" onClick={() => setAt(k + 1)} disabled={k >= list.length - 1}>
            <Translated vi="Giai đoạn tiếp">Next stage</Translated>
          </button>
          <span className="seg" role="group" aria-label={tr('Stages', 'Các giai đoạn')}>
            {list.map((s, i) => (
              <button key={s.id} data-stage-choice={s.id} aria-pressed={i === k} title={s.title} onClick={() => setAt(i)}>{i + 1}</button>
            ))}
          </span>
          <span className="muted" aria-live="polite">
            <Translated
              vi={
                <>
                  giai đoạn {k + 1} trên {list.length}, <b data-out="when">{stage.when === 'build' ? 'lúc build' : 'lúc chạy'}</b>, do{' '}
                  <span data-out="tool">{stage.tool}</span> thực hiện
                </>
              }
            >
              stage {k + 1} of {list.length}, <b data-out="when">{stage.when === 'build' ? 'build time' : 'run time'}</b>, done by{' '}
              <span data-out="tool">{stage.tool}</span>
            </Translated>
          </span>
        </div>

        <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}>
          <b data-out="title">{stage.title}{/[.?!]$/.test(stage.title) ? '' : '.'}</b> <span data-out="text">{stage.text}</span>{' '}
          <span className="muted" data-out="facts">({facts(stage.facts)})</span>
        </div>

        <Boxes boxes={stage.boxes} backend={backend} />

        <div className="panel" style={{ marginTop: '0.9rem' }}>
          <table className="threads" data-table="placement" style={{ whiteSpace: 'normal' }}>
            <thead>
              <tr>
                <th>
                  <Translated vi="Hàm">Function</Translated>
                </th>
                <th>{backend === 'metal' ? tr('Written in', 'Viết trong') : tr('Qualifier', 'Qualifier')}</th>
                <th>
                  <Translated vi="Biên dịch bởi">Compiled by</Translated>
                </th>
              </tr>
            </thead>
            <tbody>
              {PROGRAM.map((f) => {
                const where = placement(f, backend);
                return (
                  <tr key={f.name} data-fn={f.name} data-placements={where.length}>
                    <td>{f.name}</td>
                    <td>{backend === 'metal' ? where.map((p) => p.file).join(tr(' and ', ' và ')) : QUALIFIER_TEXT[f.quals]}</td>
                    <td style={{ whiteSpace: 'normal' }}>
                      {where.map((p) => p.compiler).join(tr('; and again by ', '; và lại bởi '))}
                      {where.length > 1 ? (
                        <b>
                          <Translated vi=" (hai hàm được biên dịch)"> (two compiled functions)</Translated>
                        </b>
                      ) : null}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <Translated
            vi={
              <p style={{ margin: '0.6rem 0 0' }}>
                Mã device có được biên dịch trên máy chạy chương trình không? <b data-out="users-machine">{BACKEND_NAME[backend]}</b>:{' '}
                {ON_USERS_MACHINE[compilesOnUsersMachine(backend)]}.
              </p>
            }
          >
            <p style={{ margin: '0.6rem 0 0' }}>
              Is device code compiled on the machine that runs the program? <b data-out="users-machine">{BACKEND_NAME[backend]}</b>:{' '}
              {ON_USERS_MACHINE[compilesOnUsersMachine(backend)]}.
            </p>
          </Translated>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive B: who may call it, who runs it
const cellKey = (caller: Caller, callee: Qualifiers) => `${caller}:${callee}`;
const TOTAL = CALLERS.length * CALLEES.length;
/** Short enough to read in a select one quarter of the table wide. */
const OPTION_TEXT: Record<Answer, string> = {
  get error() {
    return tr('compile error', 'lỗi biên dịch');
  },
  get 'caller-host-thread'() {
    return tr('calling host thread', 'luồng host đang gọi');
  },
  get 'caller-device-thread'() {
    return tr('calling device thread', 'luồng device đang gọi');
  },
  get 'new-grid'() {
    return tr('new grid of threads', 'lưới luồng mới');
  },
};

/** What to say about a wrong answer without giving the right one away. */
function hint(caller: Caller, callee: Qualifiers, answer: Answer): string {
  const rule = legalCall(caller, callee, 'cuda');
  if (rule.book === 'dynamic-parallelism')
    return tr('A kernel is never executed by the thread that calls it.', 'Một kernel không bao giờ được thực thi bởi chính luồng gọi nó.');
  if (rule.book === 'illegal')
    return tr(
      'This call does not compile. Ask for which side the callee was compiled, and on which side the caller runs.',
      'Lời gọi này không biên dịch được. Hãy hỏi hàm được gọi đã được biên dịch cho phía nào, và hàm gọi chạy ở phía nào.',
    );
  if (answer === 'error')
    return tr(
      'This call compiles. Ask for which side the callee was compiled, and on which side the caller runs.',
      'Lời gọi này biên dịch được. Hãy hỏi hàm được gọi đã được biên dịch cho phía nào, và hàm gọi chạy ở phía nào.',
    );
  return tr('It compiles, but that is not who executes the callee.', 'Nó biên dịch được, nhưng đó không phải là ai thực thi hàm được gọi.');
}

function WhoCalls() {
  const [answers, setAnswers] = useState<Record<string, Answer | undefined>>({});
  const [last, setLast] = useState<{ caller: Caller; callee: Qualifiers } | null>(null);
  const marks = CALLERS.flatMap((caller) => CALLEES.map((callee) => {
    const a = answers[cellKey(caller, callee)];
    return a ? grade(caller, callee, a) : null;
  }));
  const settled = marks.filter((m) => m === 'right' || m === 'depends').length;

  let feedback: ReactNode = tr('Choose an answer in any cell. Twelve cells, one rule behind them.', 'Chọn một đáp án ở ô bất kỳ. Mười hai ô, một quy tắc đằng sau chúng.');
  let bad = false;
  if (last) {
    const a = answers[cellKey(last.caller, last.callee)]!;
    const rule = legalCall(last.caller, last.callee, 'cuda');
    const mark = grade(last.caller, last.callee, a);
    const head = tr(
      `${CALLER_TEXT[last.caller]} calls a function declared ${CALLEE_TEXT[last.callee]}`,
      `${CALLER_TEXT[last.caller]} gọi một hàm được khai báo ${CALLEE_TEXT[last.callee]}`,
    );
    bad = mark === 'wrong';
    feedback =
      mark === 'wrong' ? (
        <>
          <b className="t-fail">
            <Translated vi="Không phải đáp án này.">Not this one.</Translated>
          </b>{' '}
          {head[0].toUpperCase() + head.slice(1)}: {hint(last.caller, last.callee, a)}
        </>
      ) : mark === 'depends' ? (
        <>
          <b>
            <Translated vi="Tùy vào backend.">It depends on the backend.</Translated>
          </b>{' '}
          {rule.why}{' '}
          <Translated vi={<>Bảng của sách đặt nó trong ngoặc: “{rule.row}”.</>}>
            The book's table puts it in brackets: "{rule.row}".
          </Translated>
          <span data-out="dp" style={{ display: 'block', marginTop: '0.4rem' }}>
            {BACKENDS.map((b) => `${BACKEND_NAME[b]}: ${DYNAMIC_PARALLELISM[b].text} (${DYNAMIC_PARALLELISM[b].fact})`).join('. ')}.
          </span>
        </>
      ) : (
        <>
          <b className="t-ok">
            <Translated vi="Đúng.">Right.</Translated>
          </b>{' '}
          {rule.why}{' '}
          <span className="muted">
            <Translated vi={<>Sách, Fig. 2.11: {rule.row}.</>}>Book, Fig. 2.11: {rule.row}.</Translated>
          </span>
        </>
      );
  }

  return (
    <Bench
      title={tr('Who may call it, who runs it', 'Ai được gọi nó, ai chạy nó')}
      task={tr(
        'For each caller and each kind of callee: does the call compile, and if it does, who executes the callee?',
        'Với mỗi hàm gọi và mỗi loại hàm được gọi: lời gọi có biên dịch được không, và nếu được, ai thực thi hàm được gọi?',
      )}
      foot={tr(
        "Marked against the book's Fig. 2.11 (pp. 35–36) as the model in src/sim/toolchain.ts encodes it. The two cells where a kernel is called from device code accept either answer and tell you which backend makes it true.",
        'Chấm theo Fig. 2.11 của sách (pp. 35–36) như mô hình trong src/sim/toolchain.ts mã hóa nó. Hai ô nơi một kernel được gọi từ mã device chấp nhận cả hai đáp án và cho bạn biết backend nào làm cho nó đúng.',
      )}
    >
      <div data-who>
        <div className="panel" style={{ overflowX: 'auto', marginBottom: '0.8rem' }}>
          <table className="threads" style={{ whiteSpace: 'normal', minWidth: 700, tableLayout: 'fixed' }}>
            <thead>
              <tr>
                <th style={{ whiteSpace: 'normal', width: '15%' }}>
                  <Translated vi="Hàm gọi là">The caller is</Translated>
                </th>
                {CALLEES.map((q) => (
                  <th key={q} style={{ whiteSpace: 'normal', verticalAlign: 'bottom' }}>
                    <span className="muted" style={{ fontWeight: 400 }}>
                      <Translated vi="và hàm được gọi được khai báo">and the callee is declared</Translated>
                    </span>
                    <br />
                    {CALLEE_TEXT[q]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {CALLERS.map((caller) => (
                <tr key={caller}>
                  <td style={{ fontWeight: 600 }}>{CALLER_TEXT[caller]}</td>
                  {CALLEES.map((callee) => {
                    const key = cellKey(caller, callee);
                    const a = answers[key];
                    const mark = a ? grade(caller, callee, a) : null;
                    return (
                      <td key={callee} data-cell={key} data-mark={mark ?? 'open'} style={{ whiteSpace: 'normal', verticalAlign: 'top' }}>
                        <select value={a ?? ''} aria-label={tr(`${CALLER_TEXT[caller]} calls ${CALLEE_TEXT[callee]}`, `${CALLER_TEXT[caller]} gọi ${CALLEE_TEXT[callee]}`)} style={{ width: '100%', display: 'block', marginBottom: '0.25rem' }}
                          onChange={(e) => {
                            const v = e.target.value as Answer | '';
                            setAnswers((s) => ({ ...s, [key]: v === '' ? undefined : v }));
                            setLast(v === '' ? null : { caller, callee });
                          }}>
                          <option value="">{tr('choose', 'chọn')}</option>
                          {ANSWERS.map((x) => (
                            <option key={x} value={x}>{OPTION_TEXT[x]}</option>
                          ))}
                        </select>
                        {mark === 'right' ? (
                          <span className="chip ok">
                            <Translated vi="đúng">right</Translated>
                          </span>
                        ) : mark === 'wrong' ? (
                          <span className="chip fail">
                            <Translated vi="sai">no</Translated>
                          </span>
                        ) : mark === 'depends' ? (
                          <span className="chip">
                            <Translated vi="tùy">depends</Translated>
                          </span>
                        ) : null}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="row" style={{ marginBottom: '0.8rem', justifyContent: 'space-between' }}>
          <span className="muted" data-out="score">
            <Translated vi={<>đã xác định {settled} trên {TOTAL} ô</>}>
              {settled} of {TOTAL} cells settled
            </Translated>
          </span>
          <button className="quiet" onClick={() => { setAnswers({}); setLast(null); }}>
            <Translated vi="Làm lại">Start again</Translated>
          </button>
        </div>
        <div className={'narrate' + (bad ? ' bad' : '')} data-out="feedback" role="status">{feedback}</div>
        {settled === TOTAL ? (
          <div className="panel" style={{ marginTop: '0.8rem' }} data-out="rule">
            <Translated
              vi={
                <p style={{ margin: 0 }}>
                  <b>Đã xác định cả mười hai.</b> Bảng là một quy tắc và một mặc định. Quy tắc: mã chạy ở nơi hàm gọi nó chạy, do luồng gọi thực thi, trừ khi hàm được gọi là một kernel, và một kernel luôn có một lưới mới. Một lời gọi chỉ biên dịch được nếu hàm được gọi đã được biên dịch cho phía của hàm gọi. Mặc định là phần dễ vấp: một hàm không có qualifier là một hàm host, nên một kernel không thể gọi nó.
                </p>
              }
            >
              <p style={{ margin: 0 }}>
                <b>All twelve settled.</b> The table is one rule and one default. The rule: code runs where its caller
                runs, executed by the calling thread, unless the callee is a kernel, and a kernel always gets a new grid.
                A call compiles only if the callee was compiled for the caller's side. The default is the part that
                bites: a function with no qualifier is a host function, so a kernel cannot call it.
              </p>
            </Translated>
          </div>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the same question on Metal
function MetalCalls() {
  const word = (r: ReturnType<typeof metalCall>) =>
    r.result === 'call'
      ? tr(`yes: run by ${EXECUTOR_TEXT[r.executedBy!]}`, `có: chạy bởi ${EXECUTOR_TEXT[r.executedBy!]}`)
      : r.result === 'dispatch'
        ? tr(`not a call: dispatched, run by ${EXECUTOR_TEXT[r.executedBy!]}`, `không phải lời gọi: được dispatch, chạy bởi ${EXECUTOR_TEXT[r.executedBy!]}`)
        : tr('no', 'không');
  return (
    <table className="data" data-table="metal-calls" style={{ maxWidth: '58rem' }}>
      <thead>
        <tr>
          <th>
            <Translated vi="Nó có gọi được…">Can it call…</Translated>
          </th>
          {METAL_FNS.map((f) => <th key={f}>{METAL_FN_TEXT[f]}</th>)}
        </tr>
      </thead>
      <tbody>
        {METAL_FNS.map((caller) => (
          <tr key={caller}>
            <td>{METAL_FN_TEXT[caller]}</td>
            {METAL_FNS.map((callee) => {
              const r = metalCall(caller, callee);
              return (
                <td key={callee} data-metal={`${caller}:${callee}`} data-result={r.result}>
                  {word(r)}
                  {r.result !== 'call' || caller !== callee ? <div className="muted">{r.why}</div> : null}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------- listings
/** The functions of lab 2.3's solution the lesson shows, in file order, joined with a blank line. */
function pickFunctions(src: string | null, names: readonly string[]): string | null {
  if (!src) return null;
  const found = names.map((n) => functionSource(src, n));
  if (found.some((f) => f == null)) return null;
  return found
    .sort((a, b) => a!.firstLine - b!.firstLine)
    .map((f) => f!.text)
    .join('\n\n');
}

// The same words as the labs page (LabKit's solutionTabStatus), from this machine's runs.
function useLabStatus(lab: string) {
  const { reports } = useReports();
  return { cu: solutionTabStatus(reports, lab, 'kernel.cu'), metal: solutionTabStatus(reports, lab, 'kernel.metal') };
}

function HelperListing() {
  const cu = useLabFile(HELPER_LAB + '/solution/kernel.cu');
  const metal = useLabFile(HELPER_LAB + '/solution/kernel.metal');
  const status = useLabStatus(HELPER_LAB);
  const cuText = pickFunctions(cu, HELPER_CU);
  const metalText = pickFunctions(metal, HELPER_METAL);
  if (!cuText || !metalText)
    return (
      <Translated
        vi={
          <p className="muted">
            Đang nạp hàm phụ từ <code>labs/{HELPER_LAB}/solution/</code>.
          </p>
        }
      >
        <p className="muted">Loading the helper from <code>labs/{HELPER_LAB}/solution/</code>.</p>
      </Translated>
    );
  return (
    <Code
      tabs={[
        { label: tr('CUDA and HIP', 'CUDA và HIP'), code: cuText, tiers: HELPER_TIERS, status: status.cu },
        { label: 'Metal', code: metalText, tiers: HELPER_TIERS, status: status.metal },
      ]}
      caption={
        <Translated
          vi={
            <>
              Ba hàm của <code>kernel.cu</code> và hai hàm của <code>kernel.metal</code> trong{' '}<code>labs/{HELPER_LAB}/solution/</code>, lời giải tham chiếu (solution) của mốc dự án M0 (<a href={href('ch02', 'labs')}>lab 2.3</a>). Trên CUDA và HIP, một định nghĩa của <code>scaleShift</code> phục vụ cả kernel lẫn hàm host mà các bài kiểm thử của lab dùng để so sánh. Thẻ Metal là cùng phép tính được viết lần thứ hai, dưới tên <code>scale_shift</code>, vì lý do ở phần dưới.
            </>
          }
        >
          <>
            Three functions of <code>kernel.cu</code> and two of <code>kernel.metal</code> in{' '}
            <code>labs/{HELPER_LAB}/solution/</code>, the reference solution of milestone M0 (
            <a href={href('ch02', 'labs')}>lab 2.3</a>). On CUDA and HIP one definition of <code>scaleShift</code> serves
            the kernel and the host function the lab's tests compare it with. The Metal tab is the same arithmetic
            written a second time, as <code>scale_shift</code>, for the reason further down.
          </>
        </Translated>
      }
    />
  );
}

function PipelineListing() {
  const mm = useLabFile(LAB + '/solution/launch.mm');
  const { reports } = useReports();
  const f = mm ? functionSource(mm, 'makeVecAddPipeline') : null;
  if (!f)
    return (
      <Translated
        vi={
          <p className="muted">
            Đang nạp mã host Metal từ <code>labs/{LAB}/solution/launch.mm</code>.
          </p>
        }
      >
        <p className="muted">Loading the Metal host code from <code>labs/{LAB}/solution/launch.mm</code>.</p>
      </Translated>
    );
  const status = solutionTabStatus(reports, LAB, 'launch.mm');
  return (
    <Code
      tabs={[{ label: tr('Metal host', 'Phía host Metal'), code: f.text, status }]}
      firstLine={f.firstLine}
      caption={
        <Translated
          vi={
            <>
              <code>makeVecAddPipeline</code> từ <code>labs/{LAB}/solution/launch.mm</code>. Ba bước, và hai bước biên dịch báo lỗi qua một <code>NSError</code>. Một tên kernel bị thiếu chỉ là một <code>nil</code> đơn thuần.
            </>
          }
        >
          <>
            <code>makeVecAddPipeline</code> from <code>labs/{LAB}/solution/launch.mm</code>. Three steps, and the two
            that compile report through an <code>NSError</code>. A missing kernel name is a plain <code>nil</code>.
          </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- questions
const FUTURE = {
  cuda: () => canRunOnNewArchitecture('cuda', { ptx: true }),
  hip: () => canRunOnNewArchitecture('hip', { archs: ['a'], newArch: 'b' }),
  metal: () => canRunOnNewArchitecture('metal'),
};
const PLAIN = () => legalCall('kernel', 'none', 'cuda');
// The questions record is built once, at import; the model's sentence is read while rendering so it follows the language switch.
function Why({ of }: { of: () => { why: string } }) {
  useLocale();
  return of().why;
}
const BOTH = placement(fnOf('add1'), 'cuda').length;

export const questionsS27: Record<string, Question> = {
  'ch02/2.7/q-future': {
    kind: 'choice',
    id: 'ch02/2.7/q-future',
    prompt: (
      <Translated
        vi={
          <>
            Hôm nay bạn phát hành một tệp nhị phân. Năm sau xuất hiện một kiến trúc GPU chưa tồn tại khi bạn build. Theo tài liệu của các nhà cung cấp, trên những backend nào kernel của bạn vẫn có thể được biên dịch cho GPU mới mà không cần build lại?
          </>
        }
      >
        You ship a binary today. Next year a GPU architecture appears that did not exist when you built it. Going by
        the vendors' documentation, on which backends can your kernel still be compiled for the new GPU without a
        rebuild?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không backend nào. Mã device là mã máy, và mã máy được cố định khi bạn build.">
            On none. Device code is machine code, and machine code is fixed when you build.
          </Translated>
        ),
        why: (
          <Translated vi="Đúng với thứ mà một bản build HIP phát hành. Không đúng với hai backend còn lại.">
            True of what a HIP build ships. Not of the other two.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Trên CUDA, khi tệp nhị phân mang theo PTX, và trên Metal. Một tệp nhị phân HIP chạy trên các kiến trúc được chỉ định khi nó được build.">
            On CUDA, when the binary carries PTX, and on Metal. A HIP binary runs on the architectures named when it was built.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Trên cả ba. Mọi runtime GPU đều biên dịch kernel khi chương trình khởi động.">
            On all three. Every GPU runtime compiles kernels when the program starts.
          </Translated>
        ),
        why: (
          <Translated vi="HIP biên dịch trước cho các kiến trúc được chỉ định. Biên dịch lúc chạy là một API HIP riêng, không phải điều một bản build thông thường làm.">
            HIP compiles ahead of time for named architectures. Compiling at run time is a separate HIP API, not what an ordinary build does.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ trên HIP, vì trình biên dịch của nó là mã nguồn mở.">
            On HIP only, because its compiler is open source.
          </Translated>
        ),
        why: (
          <Translated vi="Ai đọc được trình biên dịch không thay đổi những gì nằm trong tệp nhị phân.">
            Who can read the compiler does not change what is inside the binary.
          </Translated>
        ),
      },
    ],
    explain: (
      <>
        <p>
          CUDA: <Why of={FUTURE.cuda} /> Metal: <Why of={FUTURE.metal} /> HIP: <Why of={FUTURE.hip} />{/* i18n-ignore: backend names before model text */}
        </p>
        <Translated vi={<p>Đi qua ba backend bên dưới và xem bên trong tệp nhị phân được phát hành của mỗi backend có gì.</p>}>
          <p>Step through the three backends below and look at what is inside the shipped binary in each.</p>
        </Translated>
      </>
    ),
    source: 'not in the book; notes/today/ch02-facts.md, A14, B6, C6',
    vi: { source: 'không có trong sách; notes/today/ch02-facts.md, A14, B6, C6' },
  },
  'ch02/2.7/q-first-call': {
    kind: 'choice',
    id: 'ch02/2.7/q-first-call',
    prompt: (
      <Translated
        vi={
          <>
            Trong một tiến trình, lần khởi chạy đầu tiên của một kernel mất nhiều thời gian hơn hẳn lần thứ hai, với đối số giống hệt. Giải thích nào có trong tài liệu?
          </>
        }
      >
        In one process, the first launch of a kernel takes far longer than the second, with identical arguments.
        Which explanation is documented?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Lần khởi chạy đầu tiên chạy kernel hai lần: một lần để đo và một lần cho kết quả.">
            The first launch runs the kernel twice: once to measure it and once for the result.
          </Translated>
        ),
        why: (
          <Translated vi="Không có gì như vậy trong tài liệu.">
            Nothing of the kind is documented.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Lời gọi đầu tiên vào runtime trả chi phí cho công việc khởi động: trên CUDA là khởi tạo trễ, một context và mọi lần biên dịch JIT; trên Metal là việc tạo pipeline state.">
            The first call into the runtime pays for start-up work: on CUDA, lazy initialisation, a context and any JIT compilation; on Metal, creating the pipeline state.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Lần sao chép đầu tiên sang device chậm hơn các lần sau, và lần khởi chạy chờ nó.">
            The first copy to the device is slower than later copies, and the launch waits for it.
          </Translated>
        ),
        why: (
          <Translated vi="Chi phí khởi động được trả bởi lời gọi runtime nào đến trước. Nếu đó là một lần sao chép, lần sao chép gánh nó, và lần khởi chạy sau đó nhanh hơn, không chậm hơn.">
            The start-up cost is paid by whichever runtime call comes first. If that is a copy, the copy absorbs it, and the launch after it gets faster, not slower.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Trình biên dịch tối ưu kernel lần thứ hai sau khi quan sát lần chạy đầu.">
            The compiler optimises the kernel a second time after watching the first run.
          </Translated>
        ),
        why: (
          <Translated vi="Không có gì như vậy trong tài liệu của cả ba backend.">
            Nothing of the kind is documented for any of the three backends.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hướng dẫn của NVIDIA nói việc khởi tạo runtime là trễ (lazy) và cảnh báo rằng điều đó quan trọng “khi đo thời gian các lời gọi hàm runtime”: lời gọi đầu tiên tạo context, và mã device được biên dịch JIT nếu cần rồi được nạp. Trên Metal, việc tạo pipeline state hoàn tất quá trình biên dịch cho GPU, và Apple nói việc biên dịch “có mất một lúc”. Hệ quả là một thói quen: chạy một lần không đo thời gian trước, và tạo pipeline bên ngoài mọi thứ bạn đo. Bộ khung chạy lab làm cả hai.
          </p>
        }
      >
        <p>
          NVIDIA's guide says runtime initialisation is lazy and warns that it matters "when timing runtime function
          calls": the first call creates the context, and device code is JIT-compiled if necessary and loaded. On Metal,
          creating the pipeline state finishes compilation for the GPU, and Apple says the compiling "does take a
          while". The consequence is a habit: do one untimed run first, and create pipelines outside anything you time.
          The lab harness does both.
        </p>
      </Translated>
    ),
    source: 'not in the book; notes/today/ch02-facts.md, A5, A14, C4',
    vi: { source: 'không có trong sách; notes/today/ch02-facts.md, A5, A14, C4' },
  },
  'ch02/2.7/q-plain-helper': {
    kind: 'choice',
    id: 'ch02/2.7/q-plain-helper',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel gọi <code>float sq(float x) {'{'} return x * x; {'}'}</code>, hàm được khai báo không có qualifier. Kết quả là gì?
          </>
        }
      >
        A kernel calls <code>float sq(float x) {'{'} return x * x; {'}'}</code>, which is declared with no qualifier.
        What is the result?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Nó chạy được. Các hàm nhỏ được inline vào kernel.">
            It works. Small functions are inlined into the kernel.
          </Translated>
        ),
        why: (
          <Translated vi="Inline xảy ra sau khi trình biên dịch đã chấp nhận lời gọi, và nó không chấp nhận lời gọi này.">
            Inlining happens after the compiler has accepted the call, and it does not accept this one.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một lỗi biên dịch, tại lời gọi trong kernel.">
            A compile error, at the call in the kernel.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nó biên dịch được, và mỗi lời gọi khởi chạy một lưới mới.">
            It compiles, and each call launches a new grid.
          </Translated>
        ),
        why: (
          <Translated vi="Chỉ gọi một kernel mới khởi chạy một lưới.">
            Only calling a kernel launches a grid.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một lỗi liên kết khi chương trình chạy.">
            A link error when the program runs.
          </Translated>
        ),
        why: (
          <Translated vi="Lời gọi bị từ chối khi tệp được biên dịch.">
            The call is rejected when the file is compiled.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <Why of={PLAIN} /> Đánh dấu một hàm phụ là <code>__device__</code> là cách sửa khi chỉ có kernel gọi nó.
          </p>
        }
      >
        <p>
          <Why of={PLAIN} /> Marking a helper <code>__device__</code> is the fix when only kernels call it.
        </p>
      </Translated>
    ),
    source: '§2.5, pp. 35–36 (Fig. 2.11)',
  },
  'ch02/2.7/q-declare-twice': {
    kind: 'choice',
    id: 'ch02/2.7/q-declare-twice',
    prompt: (
      <Translated
        vi={
          <>
            Một lập trình viên mới học CUDA viết mỗi hàm phụ hai lần, một bản host và một bản device với tên khác, và tự tay giữ hai bản đồng bộ. Họ nên làm gì thay vào đó?
          </>
        }
      >
        A programmer new to CUDA writes every helper twice, a host copy and a device copy with a different name, and
        keeps the two in step by hand. What should they do instead?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không gì cả. Hai bản sao là cách duy nhất để chạy cùng một đoạn mã ở cả hai phía.">
            Nothing. Two copies are the only way to run the same code on both sides.
          </Translated>
        ),
        why: (
          <Translated vi="Ngôn ngữ có một cách.">
            The language has a way.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Đánh dấu hàm phụ là __global__, để có thể gọi nó từ bất kỳ đâu.">
            Mark the helper __global__, so that it can be called from anywhere.
          </Translated>
        ),
        why: (
          <Translated vi="Như vậy nó thành một kernel: nó phải trả về void, và gọi nó sẽ khởi chạy một lưới.">
            That makes it a kernel: it must return void, and calling it launches a grid.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Đặt __host__ và __device__ trên cùng một định nghĩa.">
            Put __host__ and __device__ on one definition.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Bỏ qualifier đi. Trình biên dịch tự tìm ra hàm được dùng ở đâu.">
            Leave the qualifier off. The compiler works out where a function is used.
          </Translated>
        ),
        why: (
          <Translated vi="Không có qualifier nghĩa là một hàm host, thứ mã device không gọi được.">
            No qualifier means a host function, which device code cannot call.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Với cả hai qualifier trên một định nghĩa, trình biên dịch sinh ra {BOTH} phiên bản mã đối tượng, một cho hàm gọi phía host và một cho hàm gọi phía device, và chỉ có một mã nguồn cần giữ cho đúng. Sách dự kiến nhiều hàm thư viện được viết theo cách này.
          </p>
        }
      >
        <p>
          With both qualifiers on one definition the compiler generates {BOTH} versions of the object code, one for
          host callers and one for device callers, and there is one source to keep correct. The book expects many
          library functions to be written this way.
        </p>
      </Translated>
    ),
    source: 'book exercise 10, p. 42 (answer ours); §2.5, p. 36',
    vi: { source: 'bài tập 10 của sách, p. 42 (đáp án của khóa học này); §2.5, p. 36' },
  },
  'ch02/2.7/q-metal-helper': {
    kind: 'choice',
    id: 'ch02/2.7/q-metal-helper',
    prompt: (
      <Translated
        vi={
          <>
            Một hàm phụ <code>__host__ __device__</code> được dùng bởi một kernel và bởi bản tham chiếu CPU của nó. Bạn chuyển cả hai sang Metal. Metal cung cấp gì thay thế?
          </>
        }
      >
        A <code>__host__ __device__</code> helper is used by a kernel and by its CPU reference. You port both to
        Metal. What does Metal offer in its place?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Cùng hai qualifier đó, viết dưới dạng thuộc tính [[host]] và [[device]] trên một định nghĩa.">
            The same two qualifiers, spelled as the attributes [[host]] and [[device]] on one definition.
          </Translated>
        ),
        why: (
          <Translated vi="Metal Shading Language không có những thuộc tính như vậy.">
            Metal Shading Language has no such attributes.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không có cách nào trong tài liệu để build một hàm cho cả hai phía. Viết nó hai lần; dùng chung hằng số và struct trong một header.">
            No documented way to build one function for both. Write it twice; share constants and structs in a header.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Đánh dấu hàm là kernel, để host cũng gọi được nó.">
            Mark the function kernel, so that the host can call it as well.
          </Translated>
        ),
        why: (
          <Translated vi="Một hàm kernel không thể được ai gọi như một hàm. Host chỉ có thể dispatch nó như một lưới, và các hàm khác trong tệp .metal hoàn toàn không gọi được nó.">
            A kernel function cannot be called like a function by anyone. The host can only dispatch it as a grid, and other functions in the .metal file cannot call it at all.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Gọi hàm host từ kernel: trên Apple silicon CPU và GPU dùng chung bộ nhớ, nên đoạn mã đó với tới được.">
            Call the host function from the kernel: on Apple silicon the CPU and the GPU share memory, so the code is reachable.
          </Translated>
        ),
        why: (
          <Translated vi="Bộ nhớ dùng chung là chuyện dữ liệu. Mã kernel không thể gọi vào chương trình host.">
            Shared memory is about data. Kernel code cannot call into the host program.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mọi hàm trong một tệp <code>.metal</code> đều là mã GPU, và một hàm không có thuộc tính đóng vai trò của một hàm phụ{' '}<code>__device__</code>. Không gì trong tài liệu biên dịch một hàm cho cả CPU lẫn GPU. llama.cpp cho thấy những gì được dùng chung trong thực tế: một header đơn giản chứa các hằng số <code>#define</code> và các struct đối số, được include bởi mã host Metal và bởi các kernel của nó. Bản thân phép tính được viết ở mỗi phía, thêm một lý do để kiểm thử kernel với một bản tham chiếu CPU.
          </p>
        }
      >
        <p>
          Every function in a <code>.metal</code> file is GPU code, and an unattributed one plays the part of a{' '}
          <code>__device__</code> helper. Nothing in the documentation compiles one function for the CPU and the GPU.
          llama.cpp shows what is shared in practice: one plain header of <code>#define</code> constants and argument
          structs, included by its Metal host code and by its kernels. The arithmetic itself is written on each side,
          which is one more reason to test a kernel against a CPU reference.
        </p>
      </Translated>
    ),
    source: 'not in the book; notes/today/ch02-facts.md, C7, D2',
    vi: { source: 'không có trong sách; notes/today/ch02-facts.md, C7, D2' },
  },
};

export function S27() {
  return (
    <>
      <Translated
        vi={
          <p>
            Một tệp chứa <code>__global__</code> và <code>&lt;&lt;&lt; &gt;&gt;&gt;</code> không còn là C++, và một trình biên dịch C++ thông thường từ chối nó. Cần một trình biên dịch hiểu các phần mở rộng đó, và trình biên dịch của sách là NVCC, trình biên dịch CUDA của NVIDIA. NVCC dùng các từ khóa để tách tệp ra. Mã host là C++ thuần: nó đi đến trình biên dịch chuẩn của host và chạy như một tiến trình bình thường. Mã device, gồm các kernel cùng các hàm device và dữ liệu chúng dùng, được chuyển thành thứ sách gọi là tệp nhị phân ảo, hay tệp PTX, và theo sách thì một thành phần lúc chạy của NVCC biên dịch chúng thành mã đối tượng thật và chạy trên GPU.
          </p>
        }
      >
        <p>
          A file that contains <code>__global__</code> and <code>&lt;&lt;&lt; &gt;&gt;&gt;</code> is no longer C++, and
          an ordinary C++ compiler rejects it. It takes a compiler that understands the extensions, and the book's is
          NVCC, the NVIDIA CUDA compiler. NVCC uses the keywords to pull the file apart. Host code is plain C++: it goes
          to the host's standard compiler and runs as a normal process. Device code, the kernels together with the
          device functions and data they use, is turned into what the book calls virtual binary files, or PTX files, and
          the book has a run-time component of NVCC compile those into real object code and run it on the GPU.
        </p>
      </Translated>
      <Source section="2.7" pages="39–40" figs={tr('Fig. 2.14', 'Hình 2.14')} />
      <Translated
        vi={
          <p>
            Việc tách đúng như sách nói. Nửa sau của bức tranh ngày nay được NVIDIA vẽ khác. Mã device được biên dịch sang PTX rồi, bởi một công cụ tên là <code>ptxas</code>, sang cubin, tức mã máy cho một tập lệnh GPU thật, và tất cả điều đó xảy ra khi bạn build. PTX và cubin đi cùng nhau trong một fatbin bên trong tệp thực thi. Trình biên dịch làm việc lúc chạy nằm trong driver của device, và nó chỉ chạy khi PTX phải được biên dịch cho GPU hiện có. Các tệp chứa mã device có đuôi <code>.cu</code>, lệnh cơ bản là <code>nvcc file.cu -o out</code>, và nvcc cần một trình biên dịch host trên máy.
          </p>
        }
      >
        <p>
          The split is as the book says. The second half of the picture is drawn differently by NVIDIA today. Device
          code is compiled to PTX and then, by a tool called <code>ptxas</code>, to a cubin, which is machine code for a
          real GPU instruction set, and all of that happens when you build. PTX and cubin travel together in a fatbin
          inside the executable. The compiler that works at run time sits in the device driver, and it runs only when
          the PTX has to be compiled for the GPU at hand. Files that hold device code end in <code>.cu</code>, the basic
          command is <code>nvcc file.cu -o out</code>, and nvcc needs a host compiler on the machine.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A14)">{/* i18n-ignore: date and fact reference */}
        <Translated
          vi={
            <p>
              “Tệp nhị phân ảo” (virtual binary files) và “một thành phần lúc chạy của NVCC” (a run-time component of NVCC) là cách nói của sách. NVIDIA gọi PTX là “mã ISA ảo” (virtual ISA code) hoặc “mã hợp ngữ” (assembly code), và nói PTX được nạp lúc chạy “được driver của device biên dịch sang mã nhị phân”.
            </p>
          }
        >
          <p>
            "Virtual binary files" and "a run-time component of NVCC" are the book's phrases. NVIDIA calls PTX "virtual
            ISA code" or "assembly code", and says PTX loaded at run time "is compiled to binary code by the device
            driver".
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Một hệ quả xuất hiện trong mọi benchmark bạn sẽ chạy. Runtime khởi động trễ: lời gọi đầu tiên vào nó tạo một context, và trong lời gọi đó mã device được biên dịch tức thời (just in time) nếu cần và được nạp lên device. Vì vậy lần khởi chạy đầu tiên trong một tiến trình không phải là lần nên đo thời gian.
          </p>
        }
      >
        <p>
          One consequence shows up in every benchmark you will run. The runtime starts lazily: the first call into it
          creates a context, and during that call device code is compiled just in time if necessary and loaded onto the
          device. The first launch in a process is therefore not the one to time.
        </p>
      </Translated>
      <Aside>
        <Translated vi={<p>Hướng dẫn của NVIDIA, mục 2.1.6, như đã kiểm tra trong notes/today/ch02-facts.md, A5.</p>}>
          <p>NVIDIA's guide, section 2.1.6, as checked in notes/today/ch02-facts.md, A5.</p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Theo dấu một tệp qua ba bộ công cụ">Follow one file through three toolchains</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hai backend còn lại đi cùng hành trình với những điểm dừng khác, và sự khác biệt quyết định một điều thực tế: kernel của bạn còn chạy được ở đâu sau khi bạn đã phát hành nó. Hãy chốt một đáp án trước.
          </p>
        }
      >
        <p>
          The other two backends make the same journey with different stops, and the difference decides something
          practical: where your kernel can still run after you have shipped it. Commit to an answer first.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ nói về NVCC. Các giai đoạn của HIP và Metal lấy từ tài liệu của nhà cung cấp (notes/today/ch02-facts.md, B4, B6, C1, C4, C6, C7).
            </p>
          }
        >
          <p>
            The book covers NVCC only. The HIP and Metal stages are from vendor documentation
            (notes/today/ch02-facts.md, B4, B6, C1, C4, C6, C7).
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS27['ch02/2.7/q-future']} label={tr('Predict first', 'Dự đoán trước')} />
      <Translated
        vi={
          <p>
            Bây giờ hãy đi qua cả ba. Trên HIP, gõ một hoặc hai tên kiến trúc của riêng bạn; mô hình chỉ so sánh những tên bạn đưa vào.
          </p>
        }
      >
        <p>
          Now step through all three. On HIP, type an architecture name or two of your own; the model only compares the
          names you give it.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Các giai đoạn là những giai đoạn có trong tài liệu. Bị bỏ qua: liên kết, bộ nhớ đệm (cache) mà driver CUDA giữ cho những gì nó đã biên dịch, các tệp nhị phân build cho nhiều kiến trúc CUDA cùng lúc, và việc biên dịch mã nguồn qua một API lúc chạy trên CUDA và HIP.
            </p>
          }
        >
          <p>
            The stages are the documented ones. Left out: linking, the cache the CUDA driver keeps of what it has
            compiled, binaries built for several CUDA architectures at once, and compiling source through an API at run
            time on CUDA and HIP.
          </p>
        </Translated>
      </Simplification>
      <FollowFile />
      <Ask q={questionsS27['ch02/2.7/q-first-call']} />

      <h2>
        <Translated vi="Qualifier quyết định việc tách">Qualifiers decide the split</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Thứ cho trình biên dịch biết một hàm thuộc về phía nào là qualifier của nó, theo bảng ở cuối{' '}<a href={href('ch02', '2.5')}>mục 2.5</a>. Một hàm <code>__global__</code> là một kernel: nó chạy trên device, và gọi nó sẽ khởi chạy một lưới mới. Một hàm <code>__device__</code> chạy trên device và được thực thi bởi luồng device nào gọi nó. Một hàm <code>__host__</code> là một hàm C++ thông thường, và đó cũng là thứ bạn nhận được khi không viết qualifier nào. Bảng thì ngắn, và cách để nắm vững nó là tự điền vào.
          </p>
        }
      >
        <p>
          What tells the compiler which side a function belongs to is its qualifier, from the table at the end of{' '}
          <a href={href('ch02', '2.5')}>section 2.5</a>. A <code>__global__</code> function is a kernel: it runs on the
          device, and calling it launches a new grid. A <code>__device__</code> function runs on the device and is
          executed by whichever device thread calls it. A <code>__host__</code> function is an ordinary C++ function,
          and that is also what you get when you write no qualifier at all. The table is short, and the way to own it is
          to fill it in.
        </p>
      </Translated>
      <Source section="2.5" pages="35–36" figs={tr('Fig. 2.11', 'Hình 2.11')} />
      <WhoCalls />
      <Translated
        vi={
          <p>
            Một trong các ô đó là lỗi biên dịch đầu tiên thường gặp của bất kỳ ai viết kernel thứ hai: một hàm phụ nhỏ, viết như bạn sẽ viết trong C++, được gọi từ mã device.
          </p>
        }
      >
        <p>
          One of those cells is the usual first compile error of anyone who writes a second kernel: a small helper,
          written as you would write it in C++, called from device code.
        </p>
      </Translated>
      <Ask q={questionsS27['ch02/2.7/q-plain-helper']} />

      <h2>
        <Translated vi="Một hàm cho cả hai phía">One function for both sides</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một hàm có thể mang cả <code>__host__</code> và <code>__device__</code>. Khi đó trình biên dịch sinh ra hai phiên bản mã đối tượng của nó, một cho hàm gọi phía host và một cho kernel và các hàm device, và sách dự kiến nhiều hàm thư viện được viết theo cách này. Trong trình chạy từng bước ở trên, đây là hàm phụ{' '}<code>add1</code>, hàm duy nhất xuất hiện hai lần.
          </p>
        }
      >
        <p>
          A function may carry <code>__host__</code> and <code>__device__</code> together. The compiler then generates
          two versions of its object code, one for host callers and one for kernels and device functions, and the book
          expects many library functions to be written this way. In the stepper above this is the helper{' '}
          <code>add1</code>, the one function that appears twice.
        </p>
      </Translated>
      <Source section="2.5" pages="36" />
      <Translated
        vi={
          <p>
            Khi hai phiên bản phải khác nhau ở một dòng nào đó, có thể phân biệt các lượt biên dịch bằng một macro:{' '}<code>__CUDA_ARCH__</code> trên CUDA, và <code>__HIP_DEVICE_COMPILE__</code> trên HIP, được định nghĩa khi trình biên dịch đang biên dịch cho device.
          </p>
        }
      >
        <p>
          When the two versions must differ in some line, the passes can be told apart with a macro:{' '}
          <code>__CUDA_ARCH__</code> on CUDA, and <code>__HIP_DEVICE_COMPILE__</code> on HIP, which is defined while the
          compiler is compiling for the device.
        </p>
      </Translated>
      <Aside>
        <Translated vi={<p>Không có trong sách. notes/today/ch02-facts.md, A7 và B4.</p>}>
          <p>Not in the book. notes/today/ch02-facts.md, A7 and B4.</p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Đây là một cách dùng mà sách không nhắc đến và khóa học này dựa vào. Đặt phép tính của một kernel trong một hàm nhỏ <code>__host__ __device__</code>. Kernel gọi nó một lần cho mỗi luồng. Bản tham chiếu CPU dùng để kiểm thử kernel gọi chính mã nguồn đó trong một vòng lặp. Một lỗi gõ trong công thức không còn có thể ẩn trong một bản mà không có ở bản kia.
          </p>
        }
      >
        <p>
          Here is a use for it that the book does not mention and that this course leans on. Put the arithmetic of a
          kernel in a small <code>__host__ __device__</code> function. The kernel calls it once per thread. The CPU
          reference that tests the kernel calls the very same source in a loop. A typing mistake in the formula can no
          longer hide in one copy and not the other.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cùng mã nguồn không có nghĩa là cùng từng bit. Mỗi lần biên dịch có thể gộp <code>alpha * x + beta</code> thành một phép nhân-cộng hoặc không, nên lab 2.3 so sánh hai bản với một dung sai. Chúng tôi chưa kiểm tra nvcc và trình biên dịch HIP làm gì theo mặc định.
            </p>
          }
        >
          <p>
            Same source is not the same bits. Each compilation may fuse <code>alpha * x + beta</code> into one
            multiply-add or not, so lab 2.3 compares the two with a tolerance. What nvcc and the HIP compiler do by
            default we have not checked.
          </p>
        </Translated>
      </Beyond>
      <HelperListing />
      <Ask q={questionsS27['ch02/2.7/q-declare-twice']} />

      <h2>
        <Translated vi="HIP: một trình biên dịch, các kiến trúc được chỉ định">HIP: one compiler, named architectures</Translated>
      </h2>
      <Translated
        vi={
          <p>
            HIP không có trình biên dịch thứ hai. Một trình biên dịch dựa trên Clang, <code>amdclang++</code>, chạy một lượt qua tệp cho mã host rồi một lượt qua mã device cho mỗi kiến trúc GPU bạn chỉ định bằng{' '}<code>--offload-arch</code>. Kết quả của mỗi lượt device là mã máy cho kiến trúc đó, được nhúng vào tệp đối tượng của host. Mặc định không giữ lại giai đoạn nào kiểu PTX, nên không có gì để một driver biên dịch sau. Biên dịch lúc chạy có tồn tại, dưới dạng một API riêng tên là hiprtc nhận mã nguồn kernel dưới dạng chuỗi.
          </p>
        }
      >
        <p>
          HIP has no second compiler. One Clang-based compiler, <code>amdclang++</code>, makes a pass over the file for
          the host code and then one pass over the device code for each GPU architecture you name with{' '}
          <code>--offload-arch</code>. The result of each device pass is machine code for that architecture, embedded in
          the host object file. No stage like PTX is kept by default, so there is nothing for a driver to compile later.
          Compiling at run time exists, as a separate API called hiprtc that takes kernel source as a string.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Không có trong sách. notes/today/ch02-facts.md, B4 và B6. <code>--offload-arch=native</code> nhắm tới GPU của máy build. Chưa có gì ở đây được build bằng amdclang++: unverified.
            </p>
          }
        >
          <p>
            Not in the book. notes/today/ch02-facts.md, B4 and B6. <code>--offload-arch=native</code> targets the GPU of
            the build machine. Nothing here has been built with amdclang++: unverified.
          </p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Metal: một tệp thứ hai, hoàn tất lúc chạy">Metal: a second file, finished at run time</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trên Metal không có gì để tách, vì kernel chưa bao giờ nằm trong tệp host. Nó sống trong một tệp{' '}<code>.metal</code> viết bằng Metal Shading Language, dựa trên C++17 (C++14 trước Metal 4) và không dùng thư viện chuẩn C++. Tệp đó trở thành một thư viện theo một trong hai cách. Trình biên dịch <code>metal</code>, chạy dưới dạng{' '}<code>xcrun metal</code>, build một <code>.metallib</code> khi bạn build ứng dụng. Hoặc ứng dụng đưa văn bản mã nguồn cho <code>newLibraryWithSource</code> khi nó chạy, đó là điều các lab trong khóa học này làm. Trong cả hai trường hợp kernel vẫn chưa thực thi được: việc tạo pipeline state mới hoàn tất quá trình biên dịch cho GPU trong máy.
          </p>
        }
      >
        <p>
          On Metal there is nothing to split, because the kernel never was in the host file. It lives in a{' '}
          <code>.metal</code> file written in Metal Shading Language, which is based on C++17 (C++14 before Metal 4)
          and does not use the C++ standard library. That file becomes a library in one of two ways. The <code>metal</code> compiler, run as{' '}
          <code>xcrun metal</code>, builds a <code>.metallib</code> when you build the app. Or the app hands the source
          text to <code>newLibraryWithSource</code> when it runs, which is what the labs in this course do. In both
          cases the kernel is still not executable: creating the pipeline state is what finishes compilation for the
          GPU in the machine.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Không có trong sách. notes/today/ch02-facts.md, C6 và C7. Cả hai cách đều được dùng: llama.cpp nhúng mã nguồn kernel và mặc định biên dịch nó lúc chạy, còn MLX mặc định phát hành một thư viện biên dịch sẵn (ch05-facts.md, C10).
            </p>
          }
        >
          <p>
            Not in the book. notes/today/ch02-facts.md, C6 and C7. Both routes are in use: llama.cpp embeds its kernel
            source and compiles it at run time by default, and MLX ships a precompiled library by default
            (ch05-facts.md, C10).
          </p>
        </Translated>
      </Aside>
      <PipelineListing />
      <Translated
        vi={
          <p>
            Tài liệu của Apple nói việc biên dịch này “có mất một lúc” và không nên tạo pipeline state trong mã nhạy cảm về hiệu năng. Chương trình điều khiển (driver) của lab gọi hàm này một lần và giữ kết quả, nên không phần nào của nó rơi vào một vùng được đo thời gian.
          </p>
        }
      >
        <p>
          Apple's documentation says that this compiling "does take a while" and that pipeline states should not be
          created in performance-sensitive code. The lab's driver calls this function once and keeps the result, so none of it
          lands inside a timed region.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Metal cũng không có qualifier. Mọi hàm trong một tệp <code>.metal</code> đều là mã GPU; các hàm được đánh dấu{' '}<code>kernel</code> là những hàm ứng dụng nhìn thấy được, và một hàm không có thuộc tính đóng vai trò của một hàm phụ{' '}<code>__device__</code>. Bảng bạn đã điền ở trên thu lại còn thế này:
          </p>
        }
      >
        <p>
          Metal has no qualifiers either. Every function in a <code>.metal</code> file is GPU code; the ones marked{' '}
          <code>kernel</code> are the ones the app can see, and an unattributed function plays the part of a{' '}
          <code>__device__</code> helper. The table you filled in above shrinks to this:
        </p>
      </Translated>
      <MetalCalls />
      <Translated
        vi={
          <p>
            Thứ bị thiếu là hàng cho <code>__host__ __device__</code>. Không có cơ chế nào trong tài liệu biên dịch một hàm cho cả hai phía. llama.cpp cho thấy cách thay thế hiệu quả: mã host Metal và các kernel của nó include cùng một header đơn giản, <code>ggml-metal-impl.h</code>, chứa các hằng số <code>#define</code> và các struct mang đối số kernel. Bố cục dữ liệu được dùng chung; mã thì không. Thẻ Metal của lab 2.3 ở trên là phiên bản nhỏ của cùng điều đó: <code>scale_shift</code> là phép tính của <code>scaleShift</code>, được viết lại.
          </p>
        }
      >
        <p>
          What is missing is the row for <code>__host__ __device__</code>. No documented mechanism compiles one function
          for both sides. llama.cpp shows the working substitute: its Metal host code and its kernels include the same
          plain header, <code>ggml-metal-impl.h</code>, which holds <code>#define</code> constants and the structs that
          carry kernel arguments. Data layouts are shared; code is not. Lab 2.3's Metal tab above is the small version
          of the same thing: <code>scale_shift</code> is the arithmetic of <code>scaleShift</code>, written again.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Một điểm vướng nếu bạn biên dịch lúc chạy, như các lab làm: Apple ghi rằng mã nguồn đưa cho{' '}<code>newLibraryWithSource</code> chỉ được import thư viện riêng của Metal, nên nó không thể include một header của bạn. Bản build của llama.cpp dán mỗi tệp kernel cùng các header của nó lại với nhau trước khi nhúng văn bản.
          </p>
        }
      >
        <p>
          One catch if you compile at run time, as the labs do: Apple documents that source handed to{' '}
          <code>newLibraryWithSource</code> may import only Metal's own library, so it cannot include a header of
          yours. llama.cpp's build pastes each kernel file together with its headers before embedding the text.
        </p>
      </Translated>
      <Aside>
        <Translated vi={<p>Không có trong sách. notes/today/ch02-facts.md, D2; ch05-facts.md, C7 và C10.</p>}>
          <p>Not in the book. notes/today/ch02-facts.md, D2; ch05-facts.md, C7 and C10.</p>
        </Translated>
      </Aside>
      <Ask q={questionsS27['ch02/2.7/q-metal-helper']} />

      <h2>
        <Translated vi="Một kernel khởi chạy một kernel">A kernel that launches a kernel</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mục trong ngoặc trong bảng của sách, một kernel gọi được từ device, là song song động (dynamic parallelism). CUDA có nó, trong một phiên bản mà NVIDIA nay gọi là CDP2. HIP không hỗ trợ nó. Metal không mô tả điều gì tương tự. Sách xếp nó vào thời kiến trúc Kepler và không còn dành cho nó một chương, và không kernel nào trong khóa học này dùng nó.
          </p>
        }
      >
        <p>
          The bracketed entry in the book's table, a kernel callable from the device, is dynamic parallelism. CUDA has
          it, in a version NVIDIA now calls CDP2. HIP does not support it. Metal documents nothing like it. The book
          dates it to the Kepler architecture and no longer gives it a chapter, and no kernel in this course uses it.
        </p>
      </Translated>
      <Aside>
        <Translated vi={<p>notes/today/ch02-facts.md, A13 và C7; ch05-facts.md, B4.</p>}>
          <p>notes/today/ch02-facts.md, A13 and C7; ch05-facts.md, B4.</p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Đó là toàn bộ bộ máy của chương: một hàm host, bộ nhớ device, một kernel, một lần khởi chạy, và một trình biên dịch phân loại cái nào là cái nào. Các <a href={href('ch02', 'labs')}>lab</a> đưa nó lên GPU của chính bạn.
          </p>
        }
      >
        <p>
          That is the whole chapter's machinery: a host function, device memory, a kernel, a launch, and a compiler
          that sorts out which is which. The <a href={href('ch02', 'labs')}>labs</a> put it on your own GPU.
        </p>
      </Translated>
    </>
  );
}
