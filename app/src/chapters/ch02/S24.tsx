import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { benchRows, fmt, num, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr, useLocale } from '../../lib/i18n';
import {
  CLASS_LABEL,
  DEFECTS,
  DEFECT_LABEL,
  N,
  SENTINEL,
  expected,
  listing,
  outcome,
  phaseTimes,
  program,
  result,
  run,
  showCell,
  type ArrayId,
  type Backend,
  type Cell,
  type Defect,
  type Env,
  type Group,
  type Iface,
  type Outcome,
  type Program,
  type State,
} from '../../sim/offload';

const STUB = 'ch02/snippets/vecadd_stub.cu';
const SOLUTION = 'ch02/lab1_vecadd/solution/kernel.cu';
const LAB2 = 'ch02/lab2_timing';
const P_COPIES = 'ch02/2.4/p-copies';
const WANT = expected();
const IDS: ArrayId[] = ['A', 'B', 'C'];

const DEVICE_TIERS: Record<string, Tier> = { A_d: 'global', B_d: 'global', C_d: 'global' };
const METAL_TIERS: Record<string, Tier> = { A: 'global', B: 'global', C: 'global' };
const count = (v: number) => v.toLocaleString('en-US');

/** The CHECK macro of lab 2.1's solution, with the line of the file it starts on. */
export function checkMacro(src: string): { code: string; firstLine: number } | null {
  const lines = src.replace(/\n$/, '').split('\n');
  const at = lines.findIndex((l) => /^#define CHECK\(/.test(l));
  if (at < 0) return null;
  let last = at;
  while (last < lines.length - 1 && /\\\s*$/.test(lines[last])) last++;
  return { code: lines.slice(at, last + 1).join('\n'), firstLine: at + 1 };
}

/**
 * Where Parts 1 and 3 and the launch are in the stub file (1-based lines), so the
 * prose can point at them without hard-coding numbers that drift with the file.
 */
export function stubLines(src: string): { part1: [number, number]; launch: number; part3: [number, number] } | null {
  const lines = src.split('\n');
  const first = (re: RegExp) => lines.findIndex((l) => re.test(l)) + 1;
  const last = (re: RegExp) => lines.length - [...lines].reverse().findIndex((l) => re.test(l));
  const p1 = first(/int size = /);
  const p1End = last(/MemcpyHostToDevice/);
  const launch = first(/<<</);
  const p3 = first(/MemcpyDeviceToHost/);
  const p3End = last(/Free\(/);
  if (!(p1 > 0 && p1 < p1End && p1End < launch && launch < p3 && p3 < p3End && p3End <= lines.length)) return null;
  return { part1: [p1, p1End], launch, part3: [p3, p3End] };
}

// ---------------------------------------------------------------- your devices
const MEMORY_FIELDS = (): [string, string][] => [
  ['totalGlobalMem', tr('global memory on the device, in bytes', 'bộ nhớ toàn cục trên device, tính bằng byte')],
  ['hasUnifiedMemory', tr('the GPU shares all of its memory with the CPU', 'GPU dùng chung toàn bộ bộ nhớ của nó với CPU')],
  ['recommendedMaxWorkingSetSize', tr('as reported by the Metal device, in bytes', 'theo báo cáo của Metal device, tính bằng byte')],
];
const realBackends = (r: Reports) => Object.keys(r).filter((b) => b !== 'emu' && b !== 'race');

function YourDeviceMemory() {
  const { reports, loading } = useReports();
  const rows = realBackends(reports).flatMap((b) =>
    MEMORY_FIELDS().filter(([k]) => reports[b].device[k] !== undefined).map(([k, meaning]) => {
      const raw = String(reports[b].device[k]);
      const value = k === 'hasUnifiedMemory' ? (raw === '1' || raw === 'true' ? tr('yes', 'có') : tr('no', 'không')) : count(Number(raw));
      return { b, name: String(reports[b].device.name ?? b), k, meaning, value };
    }),
  );
  return (
    <div className="panel" data-your-memory={rows.length} style={{ marginBottom: '1.4rem' }}>
      <p style={{ margin: '0 0 0.4rem' }} className="ui">
        <b><Translated vi="Những gì các device của bạn đã báo cáo">What your own devices reported</Translated></b>
      </p>
      {rows.length ? (
        <table className="data" style={{ margin: 0 }}>
          <thead>
            <tr>
              <th><Translated vi="Backend">Backend</Translated></th>
              <th><Translated vi="Device">Device</Translated></th>
              <th><Translated vi="Thuộc tính">Property</Translated></th>
              <th><Translated vi="Ý nghĩa">Meaning</Translated></th>
              <th className="num"><Translated vi="Giá trị báo cáo">Reported</Translated></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.b + r.k}>
                <td>{r.b}</td>
                <td>{r.name}</td>
                <td><code>{r.k}</code></td>
                <td>{r.meaning}</td>
                <td className="num">{r.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p style={{ margin: 0 }} className="ui muted">
          {loading
            ? tr('Reading reports.', 'Đang đọc báo cáo.')
            : tr('No device report yet. python3 labs/pmpp.py doctor prints what each of your devices reports. Run any lab, for example python3 labs/pmpp.py run 2.1, and reload this page: the first run on a backend records its device, and this table fills in.', 'Chưa có báo cáo device nào. python3 labs/pmpp.py doctor in ra những gì mỗi device của bạn báo cáo. Hãy chạy một lab bất kỳ, chẳng hạn python3 labs/pmpp.py run 2.1, rồi tải lại trang này: lần chạy đầu tiên trên một backend ghi lại device của nó, và bảng này sẽ được điền.')}
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- the stub, one call at a time
const GROUP_LABEL: Record<Group, string> = {
  get allocate() {
    return tr('allocate', 'cấp phát');
  },
  get 'copy in'() {
    return tr('copy in', 'sao chép vào');
  },
  get fill() {
    return tr('fill through contents', 'điền qua contents');
  },
  get launch() {
    return tr('launch', 'khởi chạy');
  },
  get wait() {
    return tr('wait', 'chờ');
  },
  get 'copy back'() {
    return tr('copy back', 'sao chép về');
  },
  get read() {
    return tr('read through contents', 'đọc qua contents');
  },
  get free() {
    return tr('free', 'giải phóng');
  },
};

interface GroupSpan {
  group: Group;
  part: 1 | 2 | 3;
  first: number; // 1-based call numbers
  last: number;
}
function groupsOf(p: Program): GroupSpan[] {
  const out: GroupSpan[] = [];
  p.calls.forEach((c, i) => {
    const prev = out[out.length - 1];
    if (prev && prev.group === c.group && prev.part === c.part) prev.last = i + 1;
    else out.push({ group: c.group, part: c.part, first: i + 1, last: i + 1 });
  });
  return out;
}

interface Measured {
  device: string;
  n?: number;
  times: Partial<Record<Group, number>>;
}
/** The reader's own lab 2.2 phase times for one backend, or null when there are none. */
function measuredPhases(reports: Reports, backend: Backend, p: Program): Measured | null {
  const rep = reports[backend];
  const rows = benchRows(rep, LAB2).filter((r) => r.name === 'phases');
  if (!rep || !rows.length) return null;
  const row = rows[rows.length - 1];
  const times = phaseTimes(row);
  // A backend whose stub has no separate wait step shows the wait with the launch.
  if (times.wait != null && !p.calls.some((c) => c.group === 'wait')) {
    times.launch = (times.launch ?? 0) + times.wait;
    delete times.wait;
  }
  if (!Object.keys(times).length) return null;
  return { device: String(rep.device.name ?? backend), n: num(row.n), times };
}

function MemRow({ name, cells, prev, device, want, note }: {
  name: string;
  cells: Cell[];
  prev: Cell[] | null;
  /** Drawn in the colour of global memory. */
  device: boolean;
  /** For the caller's result: the right values, so right and wrong elements can be marked. */
  want?: number[];
  note?: string;
}) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.3rem', flexWrap: 'wrap' }}>
      <code style={{ width: '5.2rem', fontSize: '0.78rem' }} className={device ? 't-global' : undefined}>{name}</code>
      <div className={'mx' + (device ? ' global' : '')} data-row={name} style={{ gridTemplateColumns: `repeat(${cells.length}, auto)` }}>
        {cells.map((v, i) => {
          const changed = prev != null && prev[i] !== v;
          let cls = '';
          if (v === 'uncleared') cls = 'void';
          else if (v === 'undetermined') cls = 'bad';
          else if (want) cls = v === want[i] ? 'good' : v === SENTINEL ? '' : 'bad';
          else if (device) cls = changed ? 'hot' : 'on';
          return (
            <span key={i} className={cls} style={!device && changed && !want ? { borderColor: 'var(--ink)', fontWeight: 650 } : undefined}>
              {showCell(v)}
            </span>
          );
        })}
      </div>
      {note ? <span className="muted">{note}</span> : null}
    </div>
  );
}

function ReadNow({ s, shared }: { s: State; shared: boolean }) {
  const [asked, setAsked] = useState(false);
  useEffect(() => setAsked(false), [s.step, shared]);
  const r = s.hostRead;
  return (
    <div style={{ marginTop: '0.8rem' }}>
      <button onClick={() => setAsked(true)} disabled={asked}><Translated vi="Đọc C ngay">Read C now</Translated></button>
      {asked ? (
        <div className="panel" data-out="read-now" data-status={r.status} style={{ marginTop: '0.5rem' }}>
          <p style={{ margin: '0 0 0.4rem' }}>
            {r.status === 'valid' ? (
              <span className="chip ok"><Translated vi="kết quả">the result</Translated></span>
            ) : r.status === 'stale' ? (
              <span className="chip fail"><Translated vi="đọc được, nhưng không phải kết quả">readable, and not the result</Translated></span>
            ) : (
              <span className="chip fail"><Translated vi="không thể đọc kết quả từ đây">the result cannot be read from here</Translated></span>
            )}{' '}
            {r.why}
          </p>
          {r.values ? (
            <MemRow name={shared ? 'C.contents' : 'C_h'} cells={r.values} prev={null} device={shared} want={shared && r.status !== 'valid' ? undefined : WANT} />
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function StubView({ prog, env, step, setStep, measured, status, label }: {
  prog: Program;
  env: Env;
  step: number;
  setStep: (n: number) => void;
  measured?: Measured | null;
  status?: string;
  label: string;
}) {
  const { lines, range } = useMemo(() => listing(prog), [prog]);
  const locale = useLocale();
  const end = useMemo(() => run(prog, prog.calls.length, env), [prog, env, locale]);
  const total = end.stoppedAt ?? prog.calls.length;
  const at = Math.min(step, total);
  const s = useMemo(() => run(prog, at, env), [prog, at, env, locale]);
  const before = useMemo(() => (at > 0 ? run(prog, at - 1, env) : null), [prog, at, env, locale]);
  const call = at > 0 ? prog.calls[at - 1] : null;
  const shared = prog.memory === 'shared';
  const groups = groupsOf(prog);
  const hl = call ? Array.from({ length: range[at - 1][1] - range[at - 1][0] + 1 }, (_, i) => range[at - 1][0] + i) : [];
  const stopped = s.stoppedAt != null;
  const got = result(s);
  const holdsSum = got.length === WANT.length && got.every((v, i) => v === WANT[i]);
  const findingHere = s.finding && !stopped;
  const dev = (a: ArrayId) => s.device[a];
  const devName = (a: ArrayId) => (prog.backend === 'metal' ? tr(`buffer ${a}`, `buffer ${a}`) : `${a}_d`);
  const devRow = (a: ArrayId) => {
    const d = dev(a);
    if (!d)
      return (
        <div key={a} style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.3rem', minHeight: '1.9rem', alignItems: 'center' }}>
          <code style={{ width: '5.2rem', fontSize: '0.78rem' }} className="t-global">{devName(a)}</code>
          <span className="muted"><Translated vi="chưa được cấp phát">not allocated</Translated></span>
        </div>
      );
    if (d.freed)
      return (
        <div key={a} style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.3rem', minHeight: '1.9rem', alignItems: 'center' }}>
          <code style={{ width: '5.2rem', fontSize: '0.78rem' }} className="t-global">{devName(a)}</code>
          <span className="muted"><Translated vi="đã giải phóng: trả về vùng nhớ chung của device">freed: returned to the device's pool</Translated></span>
        </div>
      );
    const short = d.cells.length < prog.n ? tr(`${d.bytes} bytes: room for ${d.cells.length} of the ${prog.n} floats`, `${d.bytes} byte: đủ chỗ cho ${d.cells.length} trong ${prog.n} số float`) : undefined;
    return (
      <MemRow key={a} name={devName(a)} cells={d.cells} prev={before?.device[a]?.cells ?? null} device
        note={short ?? (d.inFlight ? tr('a kernel that writes this has been issued', 'một kernel ghi vào đây đã được phát lệnh') : undefined)} />
    );
  };
  const hostRows = s.host
    ? IDS.map((a) => (
        <MemRow key={a} name={`${a}_h`} cells={s.host![a]} prev={before?.host?.[a] ?? null} device={false} want={a === 'C' ? WANT : undefined} />
      ))
    : null;

  return (
    <div data-stub={label} data-step={at}>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <button onClick={() => setStep(0)} disabled={at === 0}><Translated vi="Làm lại">Restart</Translated></button>
        <button onClick={() => setStep(at - 1)} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => setStep(at + 1)} disabled={at >= total}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => setStep(total)} disabled={at >= total}><Translated vi="Chạy đến cuối">Run to the end</Translated></button>
        <span className="muted" aria-live="polite">
          <Translated vi={<>lời gọi {at} trên {prog.calls.length}</>}>
            call {at} of {prog.calls.length}
          </Translated>
        </span>
      </div>

      <div className="row" style={{ marginBottom: '0.8rem', gap: '0.4rem 0.5rem', alignItems: 'stretch' }} role="group" aria-label={tr('Step groups', 'Các nhóm bước')}>
        {groups.map((g) => {
          const here = at >= g.first && at <= g.last;
          const t = measured?.times[g.group];
          return (
            <button key={g.first} aria-pressed={here} data-group={g.group} onClick={() => setStep(Math.min(g.last, total))}
              disabled={g.first > total} style={{ textAlign: 'left', lineHeight: 1.3 }}>
              <span style={{ fontSize: '0.7rem', opacity: 0.72 }}>
                <Translated vi={<>Phần {g.part}</>}>Part {g.part}</Translated>
              </span>
              <br />
              {prog.backend === 'metal' && g.group === 'launch' ? tr('encode and commit', 'mã hóa và commit') : GROUP_LABEL[g.group]}
              {t != null ? (
                <>
                  <br />
                  <span data-measured={g.group} style={{ fontVariantNumeric: 'tabular-nums' }}>{fmt(t * 1e3)} ms</span>
                </>
              ) : null}
            </button>
          );
        })}
      </div>
      {measured ? (
        <p className="muted" data-out="phases-note" style={{ margin: '-0.3rem 0 0.8rem' }}>
          <Translated
            vi={
              <>
                Thời gian là số đo lab 2.2 của bạn trên {measured.device}
                {measured.n ? `, n = ${count(measured.n)}` : ''}: đo được, không phải mô hình, và cho n đó, không phải cho 8 phần tử này.
              </>
            }
          >
            Times are your lab 2.2 measurement on {measured.device}
            {measured.n ? `, n = ${count(measured.n)}` : ''}: measured, not modelled, and for that n, not for these 8 elements.
          </Translated>
        </p>
      ) : null}

      <div className={'narrate' + (stopped || findingHere ? ' bad' : '')} role="status" data-out="says" style={{ marginBottom: '0.9rem' }}>
        {call ? (
          <>
            {call.changed ? <b><Translated vi="Đây là một dòng đã bị thay đổi. ">This is a changed line. </Translated></b> : null}
            {s.says}
            {call.note && !stopped ? <> <span className="muted">{call.note}</span></> : null}
            {findingHere ? (
              <>
                <br />
                <b className="t-fail">
                  <Translated vi={<>Từ lời gọi {s.finding!.step}: {s.finding!.detail}. Không lời gọi nào báo lỗi này.</>}>
                    Since call {s.finding!.step}: {s.finding!.detail}. No call reported it.
                  </Translated>
                </b>
              </>
            ) : null}
          </>
        ) : (
          <Translated
            vi={
              <>
                Chưa có gì chạy.{' '}
                {s.host
                  ? <><code>A_h</code> và <code>B_h</code> của hàm gọi giữ dữ liệu đầu vào, và nó đã điền <code>C_h</code> bằng {SENTINEL} để một phần tử không ai ghi sẽ nổi bật.</>
                  : 'Trong phiên bản này hàm gọi không có mảng riêng: dữ liệu của nó sẽ được tạo trong các buffer.'}{' '}
                Hãy nhấn Bước.
              </>
            }
          >
            Nothing has run yet.{' '}
            {s.host
              ? <>The caller's <code>A_h</code> and <code>B_h</code> hold the inputs, and it filled <code>C_h</code> with {SENTINEL} so that an element nobody writes stands out.</>
              : 'In this version the caller has no arrays of its own: its data will be created in the buffers.'}{' '}
            Press Step.
          </Translated>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 11fr) minmax(0, 9fr)' }}>
        <div>
          <Code
            tabs={[{ label, code: lines.join('\n'), tiers: prog.backend === 'metal' ? METAL_TIERS : DEVICE_TIERS, status }]}
            highlight={hl}
          />
        </div>
        <div>
          {shared ? (
            <div className="panel" data-memory="shared" style={{ marginBottom: '0.8rem' }}>
              <div className="mx-label"><Translated vi="Một bộ nhớ: vùng lưu trữ dùng chung, CPU và GPU cùng sử dụng">One memory: shared storage, used by the CPU and the GPU</Translated></div>
              {IDS.map(devRow)}
              {hostRows ? (
                <>
                  <div className="mx-label" style={{ marginTop: '0.7rem' }}><Translated vi="Các mảng riêng của hàm gọi, trong cùng bộ nhớ nhưng không nằm trong buffer">The caller's own arrays, in the same memory but not in buffers</Translated></div>
                  {hostRows}
                </>
              ) : null}
            </div>
          ) : (
            <>
              <div className="panel" data-memory="host" style={{ marginBottom: '0.6rem' }}>
                <div className="mx-label"><Translated vi="Bộ nhớ host">Host memory</Translated></div>
                {hostRows}
              </div>
              <div className="panel" data-memory="device" style={{ marginBottom: '0.8rem' }}>
                <div className="mx-label t-global"><Translated vi="Bộ nhớ toàn cục của device">Device global memory</Translated></div>
                {IDS.map(devRow)}
              </div>
            </>
          )}
          <table className="threads" aria-label={tr('Pointer variables', 'Các biến con trỏ')} style={{ marginBottom: '0.8rem' }}>
            <thead>
              <tr><th><Translated vi="con trỏ">pointer</Translated></th><th><Translated vi="trỏ tới">refers to</Translated></th><th><Translated vi="host có được đọc qua nó">host may read through it</Translated></th></tr>
            </thead>
            <tbody>
              {s.pointers.map((ptr) => (
                <tr key={ptr.name} data-pointer={ptr.name}>
                  <td><code>{ptr.name}</code></td>
                  <td>{tr(ptr.refersTo, ({ 'the one shared memory': 'bộ nhớ dùng chung duy nhất', 'host memory': 'bộ nhớ host', 'nothing yet': 'chưa trỏ tới đâu', 'freed device memory': 'bộ nhớ device đã giải phóng', 'device memory': 'bộ nhớ device' } as Record<string, string>)[ptr.refersTo] ?? ptr.refersTo)}</td>
                  <td>{ptr.hostMayUse ? tr('yes', 'có') : tr('no', 'không')}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="threads" aria-label={tr('Host thread, device and counters', 'Luồng host, device và các bộ đếm')}>
            <tbody>
              <tr data-out="host-thread">
                <td><Translated vi="luồng host, trong lời gọi này">host thread, during this call</Translated></td>
                <td>{at === 0 ? tr('running', 'đang chạy') : s.hostThread === 'blocked' ? tr('blocked until the call returns', 'bị chặn cho đến khi lời gọi trả về') : tr('running', 'đang chạy')}</td>
              </tr>
              <tr data-out="device-ran">
                <td><Translated vi="đã hoàn thành trên device trong lời gọi này">finished on the device during this call</Translated></td>
                <td>{s.ran.length ? s.ran.join(', ') : tr('nothing', 'không có')}</td>
              </tr>
              <tr data-out="device-queue">
                <td><Translated vi="đã phát lệnh và chưa được chờ">issued and not yet waited for</Translated></td>
                <td>{s.queue.length ? s.queue.join(', ') : tr('nothing', 'không có')}</td>
              </tr>
              <tr data-out="bytes">
                <td>{shared ? tr('bytes copied between allocations', 'byte được sao chép giữa các vùng cấp phát') : tr('bytes across the boundary', 'byte đi qua ranh giới')}</td>
                <td>{s.bytesMoved}</td>
              </tr>
              <tr data-out="copies"><td><Translated vi="các lần sao chép cỡ cả mảng">array-sized copies</Translated></td><td>{s.copies}</td></tr>
              <tr data-out="frees"><td><Translated vi="lời gọi free">free calls</Translated></td><td>{s.frees}</td></tr>
              <tr data-out="api-calls"><td><Translated vi="số lời gọi runtime hoặc framework đã thực hiện">runtime or framework calls made</Translated></td><td>{s.apiCalls}</td></tr>
            </tbody>
          </table>
          {stopped ? null : <ReadNow s={s} shared={shared} />}
          {at >= total && !stopped ? (
            <p style={{ margin: '0.8rem 0 0' }} data-out="verdict">
              {holdsSum && !s.finding ? (
                <span className="chip ok"><Translated vi="Hàm gọi giữ A + B.">The caller holds A + B.</Translated></span>
              ) : holdsSum ? (
                <span className="chip fail"><Translated vi="Hàm gọi giữ A + B, và stub đã phạm một quy tắc trên đường đi.">The caller holds A + B, and the stub broke a rule on the way.</Translated></span>
              ) : (
                <span className="chip fail"><Translated vi="Hàm gọi không giữ A + B.">The caller does not hold A + B.</Translated></span>
              )}
            </p>
          ) : null}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- interactive A: the round trip, call by call
const BACKEND_LABEL: Record<Backend, string> = {
  cuda: 'CUDA',
  hip: 'HIP',
  get metal() {
    return tr('Metal, shared buffers', 'Metal, buffer dùng chung');
  },
};
const IFACE_LABEL: Record<Iface, string> = {
  get buffers() {
    return tr('data created in the buffers', 'dữ liệu được tạo trong các buffer');
  },
  get 'host-arrays'() {
    return tr('caller hands me host arrays', 'hàm gọi đưa cho tôi các mảng host');
  },
};
const STATUS: Record<Backend, string> = {
  cuda: 'front-end-checked; unverified on CUDA hardware',
  hip: 'unverified',
  metal: 'verified on Metal',
};

function RoundTripStepper() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const predicted = progress.answers[P_COPIES] != null;
  const [backend, setBackend] = useState<Backend>('cuda');
  const [iface, setIface] = useState<Iface>('buffers');
  const [step, setStep] = useState(0);
  const locale = useLocale();
  const prog = useMemo(() => program(backend, iface), [backend, iface, locale]);
  useEffect(() => setStep(0), [backend, iface]);
  const measured = measuredPhases(reports, backend, prog);
  return (
    <Bench
      title={tr('The round trip, call by call', 'Chuyến khứ hồi, từng lời gọi một')}
      task={tr('Step through on CUDA and press Read C now after the launch. Then do the same on Metal.', 'Chạy từng bước trên CUDA và nhấn Đọc C ngay sau lệnh khởi chạy. Rồi làm như vậy trên Metal.')}
      foot={
        <Translated
          vi={
            <>
              8 số nguyên nhỏ cho mỗi mảng. <b>?</b> là bộ nhớ đã được cấp phát và chưa bao giờ được ghi. Một ô trên nền
              tối đã thay đổi trong lời gọi này. Các tên có màu của <span className="chip global">bộ nhớ toàn cục</span>{' '}
              nằm trên device.
            </>
          }
        >
          <>
            8 small integers per array. <b>?</b> is memory that was allocated and never written. A cell on a dark
            ground changed in this call. Names in the colour of <span className="chip global">global memory</span> are
            on the device.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Backend', 'Backend')}>
          {(['cuda', 'hip', 'metal'] as Backend[]).map((b) => (
            <button key={b} aria-pressed={backend === b} disabled={b === 'metal' && !predicted} onClick={() => setBackend(b)}>
              {BACKEND_LABEL[b]}
            </button>
          ))}
        </span>
        {backend === 'metal' ? (
          <span className="seg" role="group" aria-label={tr('Interface', 'Giao diện')}>
            {(['buffers', 'host-arrays'] as Iface[]).map((f) => (
              <button key={f} aria-pressed={iface === f} onClick={() => setIface(f)}>{IFACE_LABEL[f]}</button>
            ))}
          </span>
        ) : null}
        {!predicted ? <span className="muted" data-out="locked"><Translated vi="Metal mở khóa khi bạn đã chốt dự đoán ở trên.">Metal unlocks when you have committed the prediction above.</Translated></span> : null}
      </div>
      <StubView prog={prog} env={backend} step={step} setStep={setStep} measured={measured} status={STATUS[backend]}
        label={backend === 'metal' ? tr(`Metal stub, ${IFACE_LABEL[iface]}`, `Stub Metal, ${IFACE_LABEL[iface]}`) : tr(`${BACKEND_LABEL[backend]} stub`, `Stub ${BACKEND_LABEL[backend]}`)} />
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive B: break the stub
export const HUNT_ROUNDS = 6;
function shuffled<T>(xs: readonly T[]): T[] {
  const a = [...xs];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
const ENV_LABEL: Record<Env, string> = {
  get cuda() {
    return tr('CUDA device, per NVIDIA\'s documentation', 'device CUDA, theo tài liệu của NVIDIA');
  },
  get hip() {
    return tr('HIP device, per AMD\'s documentation', 'device HIP, theo tài liệu của AMD');
  },
  get emu() {
    return tr('the course emulator', 'trình giả lập của khóa học');
  },
  get metal() {
    return tr('Metal, data in shared buffers', 'Metal, dữ liệu trong buffer dùng chung');
  },
};

function ResultRow({ label, cells }: { label: string; cells: Cell[] }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.35rem', flexWrap: 'wrap' }}>
      <span style={{ width: '13.5rem' }}>{label}</span>
      <div className="mx" style={{ gridTemplateColumns: `repeat(${cells.length}, auto)` }}>
        {cells.map((v, i) => (
          <span key={i} className={v === 'uncleared' ? 'void' : v === WANT[i] ? 'good' : 'bad'}>{showCell(v)}</span>
        ))}
      </div>
    </div>
  );
}

/** What the emulator run looks like from outside, as one string (also what the browser check reads). */
function emuSeen(o: Outcome): string {
  if (o.stoppedAt != null) return `stopped: ${o.finding!.kind}`; // i18n-ignore: data attribute read by the browser check
  return o.result.map(showCell).join(' ');
}

function BreakTheStub() {
  const [order, setOrder] = useState<Defect[]>(() => shuffled(DEFECTS).slice(0, HUNT_ROUNDS));
  const [round, setRound] = useState(0);
  const [pick, setPick] = useState<Defect | null>(null);
  const [checked, setChecked] = useState(false);
  const [score, setScore] = useState(0);
  const [stepEnv, setStepEnv] = useState<Env | null>(null);
  const [step, setStep] = useState(0);
  const over = round >= HUNT_ROUNDS;
  const defect = order[Math.min(round, HUNT_ROUNDS - 1)];
  const locale = useLocale();
  const dev = useMemo(() => outcome(defect, 'cuda'), [defect, locale]);
  const emu = useMemo(() => outcome(defect, 'emu'), [defect, locale]);
  const all = useMemo(() => (['cuda', 'hip', 'emu', 'metal'] as Env[]).map((e) => outcome(defect, e)), [defect, locale]);
  const stopCall = emu.stoppedAt != null ? emu.program!.calls[emu.stoppedAt - 1] : null;
  const right = pick === defect;
  const next = () => {
    setRound(round + 1);
    setPick(null);
    setChecked(false);
    setStepEnv(null);
    setStep(0);
  };
  const again = () => {
    setOrder(shuffled(DEFECTS).slice(0, HUNT_ROUNDS));
    setRound(0);
    setScore(0);
    setPick(null);
    setChecked(false);
    setStepEnv(null);
    setStep(0);
  };
  if (over)
    return (
      <Bench title={tr('Break the stub', 'Phá stub')} task={tr('Six rounds: from what a test sees to the one line that is wrong.', 'Sáu vòng: từ những gì một bài kiểm tra thấy đến đúng một dòng bị sai.')}>
        <div className="verdict" data-out="hunt-final" data-score={score}>
          <Translated
            vi={
              <p className="ui">
                <b>
                  Bạn đã gọi đúng nguyên nhân trong {score} trên {HUNT_ROUNDS} vòng.
                </b>{' '}
                Trong cả bảy lỗi, không lỗi nào tạo ra một mã lỗi mà tài liệu bảo đảm, và AddressSanitizer của trình giả
                lập dừng đúng hai lỗi: vùng cấp phát quá nhỏ và việc dùng sau khi giải phóng.
              </p>
            }
          >
            <p className="ui">
              <b>
                You named the cause in {score} of {HUNT_ROUNDS} rounds.
              </b>{' '}
              Across all seven defects, not one produced an error code the documentation guarantees, and the emulator's
              AddressSanitizer stopped exactly two: the allocation that was too small and the use after free.
            </p>
          </Translated>
          <button className="primary" onClick={again}><Translated vi="Chơi thêm sáu vòng">Play six more</Translated></button>
        </div>
      </Bench>
    );
  return (
    <Bench
      title={tr('Break the stub', 'Phá stub')}
      task={tr('One line of the CUDA stub is wrong. From what the test sees, say which.', 'Một dòng của stub CUDA bị sai. Từ những gì bài kiểm tra thấy, hãy nói đó là dòng nào.')}
      foot={
        <Translated
          vi={
            <>
              <b>?</b> chưa bao giờ được ghi: vùng cấp phát không được xóa. <b>~</b> tài liệu không xác định.{' '}
              <b>{SENTINEL}</b> là giá trị hàm gọi đặt vào <code>C_h</code> trước lời gọi. <b>A5</b> là mẫu điền của
              trình giả lập cho bộ nhớ device mới, một số float khoảng -2.9e-16.
            </>
          }
        >
          <>
            <b>?</b> never written: the allocation was not cleared. <b>~</b> not determined by the documentation.{' '}
            <b>{SENTINEL}</b> what the caller put in <code>C_h</code> before the call. <b>A5</b> the emulator's fill
            pattern for fresh device memory, a float of about -2.9e-16.
          </>
        </Translated>
      }
    >
      <p className="ui" style={{ margin: '0 0 0.7rem' }}>
        <b data-out="hunt-round"><Translated vi={<>Vòng {round + 1} trên {HUNT_ROUNDS}.</>}>Round {round + 1} of {HUNT_ROUNDS}.</Translated></b>{' '}
        <Translated vi="Điểm hiện tại:">Score so far:</Translated> <span data-out="hunt-score">{score}</span>.
      </p>
      <div className="panel" data-out="hunt-symptom" data-emu={emuSeen(emu)} data-device={dev.result.map(showCell).join(' ')} style={{ marginBottom: '0.9rem' }}>
        <ResultRow label={tr('expected: A + B', 'mong đợi: A + B')} cells={WANT} />
        <ResultRow label={tr('C_h on a CUDA device, as documented', 'C_h trên device CUDA, theo tài liệu')} cells={dev.result} />
        {emu.stoppedAt != null ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.35rem', flexWrap: 'wrap' }}>
            <span style={{ width: '13.5rem' }}><Translated vi="C_h trong trình giả lập của khóa học">C_h in the course emulator</Translated></span>
            <span className="chip fail">
              <Translated
                vi={
                  <>
                    AddressSanitizer dừng lần chạy: {emu.finding!.kind === 'use-after-free' ? 'dùng sau khi giải phóng' : 'truy cập ngoài biên'} trong{' '}
                    {stopCall!.group === 'launch' ? 'kernel' : `một bước ${GROUP_LABEL[stopCall!.group]}`}
                  </>
                }
              >
                AddressSanitizer stops the run: {emu.finding!.kind === 'use-after-free' ? 'use after free' : 'out of bounds'} in{' '}
                {stopCall!.group === 'launch' ? 'the kernel' : `a ${stopCall!.group}`}
              </Translated>
            </span>
          </div>
        ) : (
          <ResultRow label={tr('C_h in the course emulator', 'C_h trong trình giả lập của khóa học')} cells={emu.result} />
        )}
        <Translated
          vi={
            <p style={{ margin: '0.5rem 0 0' }}>
              Mã lỗi các lời gọi trả về: <b>{dev.errorCode ?? 'không có'}</b>.
            </p>
          }
        >
          <p style={{ margin: '0.5rem 0 0' }}>
            Error codes returned by the calls: <b>{dev.errorCode ?? 'none'}</b>.
          </p>
        </Translated>
      </div>
      <div className="q" style={{ margin: 0, border: 0, padding: 0 }}>
        <div className="opts" role="radiogroup" aria-label={tr('Which defect is it?', 'Đó là lỗi nào?')} style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(19rem, 1fr))' }}>
          {DEFECTS.map((d) => (
            <label key={d} data-defect={d} className={checked ? (d === defect ? 'right' : d === pick ? 'wrong' : '') : ''}>
              <input type="radio" name={`hunt-${round}`} checked={pick === d} disabled={checked} onChange={() => setPick(d)} />{/* i18n-ignore: radio group name */}
              <span>{tr(DEFECT_LABEL[d], ({
  'bytes-as-elements': 'Lệnh sao chép về được cho n làm kích thước, chứ không phải n * sizeof(float).',
  'alloc-in-elements': 'Ba vùng cấp phát xin n byte; các lệnh sao chép chuyển n * sizeof(float).',
  'missing-copy-in': 'B không bao giờ được sao chép sang device.',
  'copy-back-early': 'Lệnh sao chép về đứng trước lệnh khởi chạy.',
  'free-before-copy-back': 'C_d bị giải phóng trước lệnh sao chép về.',
  'swapped-pointers': 'Lệnh sao chép về bị đổi chỗ hai đối số đầu, với loại sao chép giữ nguyên.',
  'host-dereference': 'Mã host đọc trực tiếp C_d[0].',
} as Record<string, string>)[d])}</span>
            </label>
          ))}
        </div>
        <div className="row">
          {!checked ? (
            <button className="primary" disabled={pick == null}
              onClick={() => {
                setChecked(true);
                if (right) setScore(score + 1);
              }}>
              <Translated vi="Kiểm tra">Check</Translated>
            </button>
          ) : (
            <button className="primary" onClick={next}>{round + 1 < HUNT_ROUNDS ? tr('Next round', 'Vòng tiếp theo') : tr('See the score', 'Xem điểm')}</button>
          )}
        </div>
        {checked ? (
          <div className={'verdict ' + (right ? 'right' : 'wrong')} data-out="hunt-verdict" data-right={right ? 'yes' : 'no'}>
            <p>
              <b>{right ? tr('Correct.', 'Đúng.') : tr('Not this one.', 'Không phải lỗi này.')}</b> {tr(DEFECT_LABEL[defect], ({
  'bytes-as-elements': 'Lệnh sao chép về được cho n làm kích thước, chứ không phải n * sizeof(float).',
  'alloc-in-elements': 'Ba vùng cấp phát xin n byte; các lệnh sao chép chuyển n * sizeof(float).',
  'missing-copy-in': 'B không bao giờ được sao chép sang device.',
  'copy-back-early': 'Lệnh sao chép về đứng trước lệnh khởi chạy.',
  'free-before-copy-back': 'C_d bị giải phóng trước lệnh sao chép về.',
  'swapped-pointers': 'Lệnh sao chép về bị đổi chỗ hai đối số đầu, với loại sao chép giữ nguyên.',
  'host-dereference': 'Mã host đọc trực tiếp C_d[0].',
} as Record<string, string>)[defect])}
            </p>
            <table className="data" style={{ margin: '0.4rem 0 0.6rem', background: 'transparent' }}>
              <thead>
                <tr><th><Translated vi="Ở đâu">Where</Translated></th><th><Translated vi="Loại kết quả">Class of outcome</Translated></th><th><Translated vi="Điều gì xảy ra">What happens</Translated></th></tr>
              </thead>
              <tbody>
                {all.map((o) => (
                  <tr key={o.env} data-env={o.env} data-class={o.klass}>
                    <td>{ENV_LABEL[o.env]}</td>
                    <td>
                      {tr(CLASS_LABEL[o.klass], ({
  'no-counterpart': 'không thể viết ở đây',
  passes: 'đáp án đúng, không có gì được báo',
  'wrong-values': 'giá trị sai',
  'out-of-bounds': 'truy cập ngoài biên',
  'use-after-free': 'dùng sau khi giải phóng',
  'undefined-per-docs': 'hành vi không xác định theo tài liệu',
  'not-documented': 'không được phép, lỗi không được ghi trong tài liệu',
  'slow-zero-copy': 'truy cập zero-copy chậm theo tài liệu của AMD',
} as Record<string, string>)[o.klass])}
                      {o.klass === 'wrong-values' ? tr(` (${o.wrong.length === N ? 'every element' : 'elements ' + o.wrong.join(', ')})`, ` (${o.wrong.length === N ? 'mọi phần tử' : 'các phần tử ' + o.wrong.join(', ')})`) : ''}
                    </td>
                    <td>
                      {o.summary} <span className="muted"><Translated vi={<>(dữ kiện {o.facts})</>}>(facts {o.facts})</Translated></span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="row">
              <span><Translated vi="Chạy từng bước stub bị hỏng để xác nhận:">Step through the broken stub to confirm:</Translated></span>
              <span className="seg" role="group" aria-label={tr('Step the broken stub', 'Chạy từng bước stub bị hỏng')}>
                <button aria-pressed={stepEnv === 'cuda'} onClick={() => { setStepEnv('cuda'); setStep(0); }}><Translated vi="theo cách device CUDA được mô tả trong tài liệu">as a CUDA device is documented</Translated></button>
                <button aria-pressed={stepEnv === 'emu'} onClick={() => { setStepEnv('emu'); setStep(0); }}><Translated vi="trong trình giả lập của khóa học">in the course emulator</Translated></button>
              </span>
            </div>
          </div>
        ) : null}
      </div>
      {checked && stepEnv ? (
        <div style={{ marginTop: '1rem' }}>
          <StubView prog={(stepEnv === 'emu' ? emu : dev).program!} env={stepEnv} step={step} setStep={setStep} label={tr('CUDA stub with one defect', 'Stub CUDA có một lỗi')} />
        </div>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive C: two listings that would not compile
interface Tok {
  t: string;
  /** Present on tokens the reader can click. */
  key?: string;
  why?: string;
}
interface Listing {
  id: string;
  title: string;
  lines: Tok[][];
  answer: string;
  found: ReactNode;
  corrected: string;
  message: string;
}
const fine = () => tr('This token is fine where it stands.', 'Token này ổn ở đúng vị trí của nó.');
// The two printed lines are quoted from the book because they are wrong (pp. 27 and 32); the rest is ours.
const LISTINGS: Listing[] = [
  {
    id: 'fig25',
    get title() {
      return tr('Fig. 2.5, line 03, as printed (p. 27)', 'Hình 2.5, dòng 03, như bản in (p. 27)');
    },
    lines: [
      [
        {
          t: 'float',
          key: 'float',
          get why() {
            return tr('A fine type for the three pointers.', 'Một kiểu hợp lệ cho ba con trỏ.');
          },
        },
        { t: ' ' },
        {
          t: '*d_A',
          key: 'dA',
          get why() {
            return tr('Nothing is wrong with this declarator. Look at what has to come after one.', 'Declarator này không có gì sai. Hãy nhìn xem cái gì phải đứng sau một declarator.');
          },
        },
        { t: ' ' },
        { t: '*d_B', key: 'dB' },
        {
          t: ',',
          key: 'comma',
          get why() {
            return tr('This comma is where it belongs. Is there another place that needs one?', 'Dấu phẩy này nằm đúng chỗ. Có chỗ nào khác cũng cần một dấu phẩy không?');
          },
        },
        { t: ' ' },
        {
          t: '*d_C',
          key: 'dC',
          get why() {
            return fine();
          },
        },
        {
          t: ';',
          key: 'semi',
          get why() {
            return fine();
          },
        },
      ],
    ],
    answer: 'dB',
    found: (
      <Translated
        vi={
          <>
            Trình biên dịch vừa đọc xong một declarator hoàn chỉnh, <code>*d_A</code>, và giờ cần một dấu phẩy hoặc dấu
            chấm phẩy. Nó lại gặp <code>*</code>. Dấu phẩy giữa hai tên đầu tiên bị thiếu. Đây cũng là những tên duy nhất
            trong chương có tiền tố <code>d_</code>; ở mọi chỗ khác sách viết <code>A_d</code>.
          </>
        }
      >
        <>
          The compiler has just read a complete declarator, <code>*d_A</code>, and now needs a comma or a semicolon. It
          finds a <code>*</code>. The comma between the first two names is missing. The names are also the only ones in
          the chapter with a <code>d_</code> prefix; everywhere else the book writes <code>A_d</code>.
        </>
      </Translated>
    ),
    corrected: program('cuda').head[2].trim(),
    message: "error: expected ';' at end of declaration", // i18n-ignore: compiler output, quoted verbatim
  },
  {
    id: 'check',
    get title() {
      return tr('The error check of the sidebar, as printed (p. 32)', 'Đoạn kiểm tra lỗi trong khung bên lề, như bản in (p. 32)');
    },
    lines: [
      [
        {
          t: 'cudaError_t',
          key: 'type',
          get why() {
            return tr('The right type for the return value of a CUDA API call.', 'Đúng kiểu cho giá trị trả về của một lời gọi CUDA API.');
          },
        },
        { t: ' ' },
        {
          t: 'err',
          key: 'err',
          get why() {
            return tr('This is the declaration. Find the use that does not match it.', 'Đây là phần khai báo. Hãy tìm chỗ sử dụng không khớp với nó.');
          },
        },
        { t: ' = ' },
        {
          t: 'cudaMalloc',
          key: 'malloc',
          get why() {
            return fine();
          },
        },
        { t: '(' },
        {
          t: '(void **)', // i18n-ignore: code token
          key: 'cast',
          get why() {
            return tr('Legal. It is also no longer needed in current CUDA C++, as you read above.', 'Hợp lệ. Nó cũng không còn cần thiết trong CUDA C++ hiện nay, như bạn đã đọc ở trên.');
          },
        },
        { t: ' ' },
        {
          t: '&A_d',
          key: 'addr',
          get why() {
            return tr('The address of the pointer is what cudaMalloc needs.', 'Địa chỉ của con trỏ là thứ cudaMalloc cần.');
          },
        },
        { t: ', ' },
        {
          t: 'size',
          key: 'size',
          get why() {
            return fine();
          },
        },
        { t: ');' },
      ],
      [
        { t: 'if (' }, // i18n-ignore: code token
        { t: 'error', key: 'error' },
        { t: ' != ' },
        {
          t: 'cudaSuccess',
          key: 'success',
          get why() {
            return tr('The right constant to compare with.', 'Đúng hằng số để so sánh.');
          },
        },
        { t: ') {' },
      ],
      [{ t: '    // print cudaGetErrorString(err), the file and the line, then exit' }],
      [{ t: '}' }],
    ],
    answer: 'error',
    found: (
      <Translated
        vi={
          <>
            Dòng ở trên khai báo <code>err</code>. Không có gì tên là <code>error</code>, nên điều kiện không biên dịch
            được. Thân của <code>if</code> dùng <code>err</code>, điều đó cho thấy ý định thật.
          </>
        }
      >
        <>
          The line above declares <code>err</code>. Nothing named <code>error</code> exists, so the condition does not
          compile. The body of the <code>if</code> uses <code>err</code>, which shows what was meant.
        </>
      </Translated>
    ),
    corrected: 'if (err != cudaSuccess) {',
    message: "error: use of undeclared identifier 'error'", // i18n-ignore: compiler output, quoted verbatim
  },
];

function TwoListings({ onDone }: { onDone: () => void }) {
  const [clicked, setClicked] = useState<Record<string, string>>({});
  const found = (l: Listing) => clicked[l.id] === l.answer;
  const done = LISTINGS.every(found);
  useEffect(() => {
    if (done) onDone();
  }, [done, onDone]);
  return (
    <Bench
      title={tr('Two listings that would not compile', 'Hai đoạn mã không biên dịch được')}
      task={tr('In each listing, click the first token a compiler cannot accept.', 'Trong mỗi đoạn mã, hãy nhấn vào token đầu tiên mà trình biên dịch không thể chấp nhận.')}
      foot={tr("The clickable lines are the book's, as printed. They are shown because they are wrong; do not copy them. The comment line stands for the body of the if, in our words. The messages are what clang printed when each line was compiled here.", 'Các dòng có thể nhấn là của sách, như bản in. Chúng được hiển thị vì chúng sai; đừng sao chép chúng. Dòng chú thích đại diện cho thân của if, theo lời của chúng tôi. Các thông báo là những gì clang in ra khi mỗi dòng được biên dịch ở đây.')}
    >
      <div className="cols">
        {LISTINGS.map((l) => {
          const key = clicked[l.id];
          const tok = l.lines.flat().find((x) => x.key === key);
          const ok = found(l);
          return (
            <div className="panel" key={l.id} data-listing={l.id} data-found={ok ? 'yes' : 'no'}>
              <p className="muted" style={{ margin: '0 0 0.5rem' }}>{l.title}</p>
              <pre style={{ margin: 0, overflowX: 'auto', lineHeight: 1.9, fontSize: '0.84rem' }}>
                <code>
                  {l.lines.map((line, i) => (
                    <span key={i}>
                      {line.map((x, j) =>
                        x.key ? (
                          <button key={j} data-token={x.key} aria-pressed={key === x.key}
                            onClick={() => setClicked({ ...clicked, [l.id]: x.key! })}
                            style={{
                              font: 'inherit', padding: '0.05rem 0.25rem', margin: 0, borderRadius: 3, cursor: 'pointer',
                              borderColor: key === x.key ? (x.key === l.answer ? 'var(--ok)' : 'var(--fail)') : 'var(--rule)',
                              background: key === x.key ? (x.key === l.answer ? 'var(--ok-wash)' : 'var(--fail-wash)') : 'var(--cell)',
                              color: 'var(--ink)',
                            }}>
                            {x.t}
                          </button>
                        ) : (
                          <span key={j}>{x.t}</span>
                        ),
                      )}
                      {'\n'}
                    </span>
                  ))}
                </code>
              </pre>
              <div className="narrate" role="status" data-out={`listing-${l.id}`} style={{ marginTop: '0.7rem', minHeight: '4.4em' }}>
                {!tok ? (
                  tr('Nothing clicked yet.', 'Chưa nhấn gì.')
                ) : ok ? (
                  <Translated
                    vi={
                      <>
                        <b className="t-ok">Chính là nó.</b> {l.found}
                        <br />
                        Clang báo: <code>{l.message}</code>
                        <br />
                        Bản sửa, như trong các tệp đã kiểm thử của khóa học: <code>{l.corrected}</code>
                      </>
                    }
                  >
                    <>
                      <b className="t-ok">That is the one.</b> {l.found}
                      <br />
                      Clang says: <code>{l.message}</code>
                      <br />
                      Corrected, as the course's tested files have it: <code>{l.corrected}</code>
                    </>
                  </Translated>
                ) : (
                  <>
                    <b className="t-fail"><Translated vi="Không phải token này.">Not this token.</Translated></b> {tok.why}
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </Bench>
  );
}

function HuntThenMacro() {
  const [show, setShow] = useState(false);
  const file = useLabFile(SOLUTION);
  const { reports } = useReports();
  const macro = file ? checkMacro(file) : null;
  return (
    <>
      <TwoListings onDone={() => setShow(true)} />
      {show ? (
        <Code
          tabs={[{
            label: tr('CHECK, from lab 2.1', 'CHECK, từ lab 2.1'),
            code: macro?.code ?? file ?? '// loading labs/' + SOLUTION,
            status: solutionTabStatus(reports, 'ch02/lab1_vecadd', 'kernel.cu'),
          }]}
          firstLine={macro?.firstLine ?? 1}
          caption={
            <Translated
              vi={
                <>
                  Mẫu của khung bên lề dưới dạng macro, từ <code>labs/{SOLUTION}</code>. Nó khác trang sách ở hai điểm:
                  nó kiểm tra <code>err</code>, và nó khiến hàm bao quanh trả về 1 ở chỗ sách gọi{' '}
                  <code>exit</code>, để một bài kiểm tra có thể thấy một lời gọi thất bại. Lab 2.1 yêu cầu bạn tự viết
                  macro này.
                </>
              }
            >
              <>
                The sidebar's pattern as a macro, from <code>labs/{SOLUTION}</code>. It differs from the page in two
                ways: it tests <code>err</code>, and it makes the enclosing function return 1 where the book calls{' '}
                <code>exit</code>, so that a test can see a failed call. Lab 2.1 asks you to write this macro yourself.
              </>
            </Translated>
          }
        />
      ) : (
        <Translated
          vi={
            <p className="ui">
              Bản kiểm tra đã sửa, dưới dạng macro, xuất hiện ở đây khi bạn đã tìm ra cả hai token.{' '}
              <button className="quiet" onClick={() => setShow(true)}>Hiện ngay</button>
            </p>
          }
        >
          <p className="ui">
            The corrected check, as a macro, appears here when you have found both tokens.{' '}
            <button className="quiet" onClick={() => setShow(true)}>Show it now</button>
          </p>
        </Translated>
      )}
    </>
  );
}

// ---------------------------------------------------------------- questions
const METAL_END = run(program('metal', 'buffers'));
const CUDA_END = run(program('cuda'));

export const questionsS24: Record<string, Question> = {
  'ch02/2.4/q-malloc-size': {
    kind: 'choice',
    id: 'ch02/2.4/q-malloc-size',
    prompt: (
      <Translated
        vi={
          <>
            Bạn muốn một mảng gồm <code>v</code> phần tử kiểu <code>int</code> trong bộ nhớ toàn cục của device. Đối số
            thứ hai của <code>cudaMalloc</code> là gì?
          </>
        }
      >
        <>
          You want an array of <code>v</code> elements of type <code>int</code> in device global memory. What is the
          second argument of <code>cudaMalloc</code>?
        </>
      </Translated>
    ),
    options: [
      {
        label: <code>n</code>,
        why: (
          <Translated vi="Trong câu hỏi không có gì tên là n, và dù sao một số đếm trần cũng là sai đơn vị.">
            Nothing in the question is called n, and a bare count would be the wrong unit anyway.
          </Translated>
        ),
      },
      {
        label: <code>v</code>,
        why: (
          <Translated vi="Đó là số phần tử. Hàm cần số byte, nên lệnh này chỉ cấp phát một phần tư lượng bạn cần.">
            That is a count of elements. The function wants bytes, so this allocates a quarter of what you need.
          </Translated>
        ),
      },
      {
        label: <code>n * sizeof(int)</code>,
        why: (
          <Translated vi="Đúng đơn vị nhưng sai số lượng: mảng có v phần tử.">
            The right unit and the wrong count: the array has v elements.
          </Translated>
        ),
      },
      { label: <code>v * sizeof(int)</code>, correct: true },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Kích thước tính bằng byte. <code>cudaMalloc</code> không biết gì về kiểu dữ liệu bạn sẽ lưu, nên số phần tử
            phải được nhân với kích thước một phần tử, đúng như với <code>malloc</code>.
          </p>
        }
      >
        <p>
          The size is in bytes. <code>cudaMalloc</code> knows nothing about the type you will store, so the element
          count has to be multiplied by the element size, exactly as with <code>malloc</code>.
        </p>
      </Translated>
    ),
    source: 'ch. 2 exercise 5, p. 41; §2.4, pp. 29–30',
    vi: { source: 'ch. 2 bài tập 5, p. 41; §2.4, pp. 29–30' },
  },
  'ch02/2.4/q-malloc-ptr': {
    kind: 'choice',
    id: 'ch02/2.4/q-malloc-ptr',
    prompt: (
      <Translated
        vi={
          <>
            <code>float* A_d;</code> cần trỏ tới <code>n</code> số float trong bộ nhớ toàn cục của device. Đối số đầu
            tiên của <code>cudaMalloc</code> là gì?
          </>
        }
      >
        <>
          <code>float* A_d;</code> is to point at <code>n</code> floats in device global memory. What is the first
          argument of <code>cudaMalloc</code>?
        </>
      </Translated>
    ),
    options: [
      {
        label: <code>n</code>,
        why: (
          <Translated vi="Số lượng thuộc về kích thước, tính bằng byte. Đối số đầu tiên cho biết đặt địa chỉ mới vào đâu.">
            The count belongs in the size, as bytes. The first argument says where to put the new address.
          </Translated>
        ),
      },
      {
        label: <code>(void *) A_d</code>,
        why: (
          <Translated vi="Cách này truyền giá trị hiện tại của con trỏ, vốn là rác. Hàm cần biết biến nằm ở đâu để có thể thay đổi nó.">
            This passes the pointer's current value, which is garbage. The function needs to know where the variable is so that it can change it.
          </Translated>
        ),
      },
      {
        label: <code>*A_d</code>,
        why: (
          <Translated vi="Cách này đọc qua một con trỏ chưa khởi tạo, trên host, để lấy một số float.">
            This reads through an uninitialised pointer, on the host, to get a float.
          </Translated>
        ),
      },
      { label: <code>(void **) &amp;A_d</code>, correct: true },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              <code>cudaMalloc</code> ghi địa chỉ của vùng nhớ mới vào biến của bạn, nên nó cần địa chỉ của biến đó:
              một con trỏ tới con trỏ. Phép ép kiểu sang <code>void**</code> có mặt vì hàm này là hàm tổng quát và không
              biết về <code>float</code>.
            </p>
            <p>
              Trong CUDA C++ hiện nay, chỉ <code>&amp;A_d</code> cũng biên dịch được: API có một overload có kiểu và
              chính tutorial của NVIDIA bỏ phép ép kiểu. Các lab giữ phép ép kiểu vì nó biên dịch được ở mọi nơi, kể cả
              trình giả lập của khóa học.
            </p>
          </>
        }
      >
        <>
          <p>
            <code>cudaMalloc</code> writes the address of the new memory into your variable, so it needs the address of
            that variable: a pointer to a pointer. The cast to <code>void**</code> is there because the function is
            generic and does not know about <code>float</code>.
          </p>
          <p>
            In current CUDA C++ plain <code>&amp;A_d</code> compiles too: the API has a typed overload and NVIDIA's own
            tutorial leaves the cast out. The labs keep the cast because it compiles everywhere, including the course's
            emulator.
          </p>
        </>
      </Translated>
    ),
    source: 'ch. 2 exercise 6, pp. 41–42; §2.4, p. 29; the overload from notes/today/ch02-facts.md, A2',
    vi: { source: 'ch. 2 bài tập 6, pp. 41–42; §2.4, p. 29; overload lấy từ notes/today/ch02-facts.md, A2' },
  },
  'ch02/2.4/q-memcpy': {
    kind: 'choice',
    id: 'ch02/2.4/q-memcpy',
    prompt: (
      <Translated
        vi={
          <>
            Sao chép 3000 byte từ mảng host <code>A_h</code> sang mảng device <code>A_d</code>. Lời gọi nào làm việc đó?
          </>
        }
      >
        <>
          Copy 3000 bytes from the host array <code>A_h</code> to the device array <code>A_d</code>. Which call does it?
        </>
      </Translated>
    ),
    options: [
      {
        label: <code>cudaMemcpy(A_h, A_d, 3000, cudaMemcpyHostToDevice);</code>,
        why: (
          <Translated vi="Đích đứng trước. Ở đây một con trỏ host là đích của một phép sao chép từ host sang device: các con trỏ không khớp với loại sao chép, điều mà tài liệu gọi là hành vi không xác định, chứ không phải một lỗi bạn có thể trông cậy.">
            The destination comes first. Here a host pointer is the destination of a host-to-device copy: pointers that do not match the kind, which the documentation calls undefined behaviour, not an error you can rely on.
          </Translated>
        ),
      },
      { label: <code>cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);</code>, correct: true },
      {
        label: <code>cudaMemcpy(A_d, A_h, 3000, cudaMemcpyDeviceToHost);</code>,
        why: (
          <Translated vi="Các con trỏ đúng thứ tự nhưng loại sao chép mâu thuẫn với chúng. Lại là hành vi không xác định, không có mã lỗi nào được hứa hẹn.">
            The pointers are in the right order and the kind contradicts them. Undefined behaviour again, with no promised error code.
          </Translated>
        ),
      },
      {
        label: <code>cudaMemcpy(A_h, A_d, 3000, cudaMemcpyDeviceToHost);</code>,
        why: (
          <Translated vi="Một lời gọi hợp lệ nhưng sao chép theo chiều ngược lại. Nó ghi đè dữ liệu host của bạn bằng bất cứ thứ gì A_d đang giữ.">
            A well-formed call that copies the other way. It overwrites your host data with whatever A_d holds.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đích, nguồn, số byte, loại. Thứ tự hai con trỏ là thứ tự của một phép gán:{' '}
            <code>A_d = A_h</code>. Loại sao chép phải khớp với chúng, và phương án cuối cho thấy vì sao bạn không thể
            trông cậy vào việc được báo khi nó không khớp: một lời gọi có thể hoàn toàn hợp lệ mà vẫn là lời gọi sai.
          </p>
        }
      >
        <p>
          Destination, source, bytes, kind. The order of the two pointers is the order of an assignment:{' '}
          <code>A_d = A_h</code>. The kind has to agree with them, and the last option shows why you cannot count on
          being told when it does not: a call can be perfectly legal and still be the wrong one.
        </p>
      </Translated>
    ),
    source: 'after ch. 2 exercise 7, p. 42 (the four calls are ours); §2.4, pp. 30–31; notes/today/ch02-facts.md, A4',
    vi: { source: 'theo sau ch. 2 bài tập 7, p. 42 (bốn lời gọi là của khóa học này); §2.4, pp. 30–31; notes/today/ch02-facts.md, A4' },
  },
  'ch02/2.4/q-out-param': {
    kind: 'choice',
    id: 'ch02/2.4/q-out-param',
    prompt: (
      <Translated
        vi={
          <>
            Vì sao <code>cudaMalloc</code> nhận <code>&amp;A_d</code> trong khi <code>cudaFree</code> nhận{' '}
            <code>A_d</code>?
          </>
        }
      >
        <>
          Why does <code>cudaMalloc</code> take <code>&amp;A_d</code> while <code>cudaFree</code> takes{' '}
          <code>A_d</code>?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Việc cấp phát ghi một địa chỉ mới vào biến của bạn và trả về một mã lỗi; việc giải phóng chỉ cần địa chỉ.">
            Allocation writes a new address into your variable and returns an error code; freeing only needs the address.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="cudaFree sao chép dữ liệu về host trước khi giải phóng bộ nhớ, nên nó cần chính giá trị con trỏ.">
            cudaFree copies the data back to the host before it releases the memory, so it needs the pointer value itself.
          </Translated>
        ),
        why: (
          <Translated vi="Nó không sao chép gì cả. Sau cudaFree dữ liệu biến mất, đó là lý do stub sao chép C về trước khi giải phóng.">
            It copies nothing. After cudaFree the data is gone, which is why the stub copies C back before it frees.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Con trỏ device rộng hơn con trỏ host, nên phải được truyền theo địa chỉ.">
            Device pointers are wider than host pointers, so they have to be passed by address.
          </Translated>
        ),
        why: (
          <Translated vi="Với host, một con trỏ device là một giá trị con trỏ như mọi giá trị khác. cudaFree, cudaMemcpy và lệnh khởi chạy đều nhận nó theo giá trị.">
            To the host a device pointer is a pointer value like any other. cudaFree, cudaMemcpy and the launch all take it by value.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>malloc</code> của C trả về con trỏ. <code>cudaMalloc</code> không thể làm vậy, vì như mọi hàm của
            CUDA API nó trả về một mã lỗi, nên địa chỉ quay về qua tham số đầu tiên. Kiểu nhận giá trị trả về đó là{' '}
            <code>cudaError_t</code>, cũng là đáp án cho bài tập 8 của sách.
          </p>
        }
      >
        <p>
          C's <code>malloc</code> returns the pointer. <code>cudaMalloc</code> cannot, because like every function of the
          CUDA API it returns an error code, so the address comes back through the first parameter. The type that
          receives that return value is <code>cudaError_t</code>, which is the answer to the book's exercise 8.
        </p>
      </Translated>
    ),
    source: 'ch. 2 §2.4, pp. 28–30; exercise 8, p. 42',
    vi: { source: 'ch. 2 §2.4, pp. 28–30; bài tập 8, p. 42' },
  },
  'ch02/2.4/q-not-cleared': {
    kind: 'choice',
    id: 'ch02/2.4/q-not-cleared',
    prompt: (
      <Translated
        vi={
          <>
            Một lần khởi chạy chỉ phủ 768 phần tử đầu trong số 1000: ba khối 256 trong khi cần bốn. Không có gì khác sai.
            Sau khi sao chép về, các phần tử 768 đến 999 của <code>C_h</code> chứa gì trên CUDA, và trên Metal khi C được
            tạo bằng <code>newBufferWithLength</code>?
          </>
        }
      >
        <>
          A launch covers only the first 768 of 1000 elements: three blocks of 256 where four were needed. Nothing else
          is wrong. After the copy back, what do elements 768 to 999 of <code>C_h</code> hold on CUDA, and on Metal when
          C was made with <code>newBufferWithLength</code>?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Trên CUDA là bất cứ thứ gì vùng nhớ device đó chứa trước đó. Trên Metal là các số 0.">
            On CUDA whatever that device memory held before. On Metal zeros.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Các số 0 trên cả hai: bộ nhớ device mới được xóa trước khi giao ra.">
            Zeros on both: fresh device memory is cleared before it is handed out.
          </Translated>
        ),
        why: (
          <Translated vi="Tài liệu tham chiếu của NVIDIA nói rõ ràng về cudaMalloc rằng bộ nhớ không được xóa.">
            NVIDIA's reference says of cudaMalloc, in so many words, that the memory is not cleared.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Trên cả hai, là thứ C_h chứa trước lời gọi, vì các phần tử cũ không bao giờ được sao chép về.">
            On both, what C_h held before the call, since the stale elements are never copied back.
          </Translated>
        ),
        why: (
          <Translated vi="Lệnh sao chép về ghi cả 1000 phần tử của C_h. Phần cũ đến từ bản sao trên device, không phải từ mảng host.">
            The copy back writes all 1000 elements of C_h. The stale part comes from the device's copy, not from the host array.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không đọc được cái nào: lần chạy dừng với một mã lỗi.">
            Neither can be read: the run stops with an error code.
          </Translated>
        ),
        why: (
          <Translated vi="Không lời gọi nào nhận đối số sai và không có gì vượt ra ngoài biên. Mọi lời gọi đều thành công.">
            No call was given a bad argument and nothing went out of bounds. Every call succeeds.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>cudaMalloc</code> không xóa vùng nhớ nó giao cho bạn; <code>newBufferWithLength:options:</code> trả
            về một buffer chứa toàn số 0. Không kết quả nào đúng, và kết quả của Metal nguy hiểm hơn, vì một số 0 có thể
            trông giống một đáp án hợp lý. Đây là lý do các bài kiểm tra của khóa học dùng dữ liệu có kết quả đúng không
            bao giờ bằng 0, và lý do hàm gọi trong trình chạy từng bước điền <code>C_h</code> bằng {SENTINEL} trước.
          </p>
        }
      >
        <p>
          <code>cudaMalloc</code> does not clear what it hands you; <code>newBufferWithLength:options:</code> returns a
          buffer filled with zeros. Neither result is right, and the Metal one is the more dangerous of the two, because
          a zero can look like a plausible answer. This is why the course's tests use data whose correct result is never
          0, and why the stepper's caller fills <code>C_h</code> with {SENTINEL} first.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch02-facts.md, A2 and C2; the 768 of three blocks of 256 is the book\'s, §2.5, p. 34',
    vi: { source: 'notes/today/ch02-facts.md, A2 và C2; con số 768 của ba khối 256 là của sách, §2.5, p. 34' },
  },
  'ch02/2.4/q-shared-wait': {
    kind: 'choice',
    id: 'ch02/2.4/q-shared-wait',
    prompt: (
      <Translated
        vi={
          <>
            Metal, buffer dùng chung. Host commit command buffer và, ngay ở dòng kế tiếp, đọc{' '}
            <code>result.contents</code>. Nó đọc được gì?
          </>
        }
      >
        <>
          Metal, shared buffers. The host commits the command buffer and, on the very next line, reads{' '}
          <code>result.contents</code>. What does it read?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Kết quả hoàn chỉnh: CPU và GPU dùng chung một bộ nhớ, nên nó là dữ liệu mới nhất.">
            The finished result: the CPU and the GPU share one memory, so it is current.
          </Translated>
        ),
        why: (
          <Translated vi="Bộ nhớ là dùng chung. Công việc không xong chỉ vì nó đã được gửi đi.">
            The memory is shared. The work is not finished just because it was submitted.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Có thể là dữ liệu cũ hoặc mới được ghi một phần: nó phải chờ command buffer hoàn thành.">
            Possibly old or partly written data: it must wait for the command buffer to complete.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Một con trỏ null, cho đến khi GPU xong việc với buffer.">
            A null pointer, until the GPU has finished with the buffer.
          </Translated>
        ),
        why: (
          <Translated vi="contents chỉ là null với các buffer có private storage. Với một buffer dùng chung nó luôn là một địa chỉ dùng được.">
            contents is null only for buffers with private storage. For a shared buffer it is always a usable address.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không gì cả: Metal dừng chương trình vì chạm vào một buffer đang bận.">
            Nothing: Metal stops the program for touching a busy buffer.
          </Translated>
        ),
        why: (
          <Translated vi="Không có gì kiểm tra. Header của Apple nói việc truy cập buffer không được đồng bộ hóa và bên gọi phải chịu trách nhiệm.">
            Nothing checks. Apple's header says access to buffers is not synchronised and that the caller is responsible.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>commit</code> trả về trong khi Metal vẫn đang chuẩn bị và lên lịch công việc. Với shared storage không
            có lần sao chép nào để nấp sau, nên việc giữ đúng thứ tự là trách nhiệm của bạn: hoàn tất công việc của GPU trên
            một buffer trước khi CPU chạm vào nó, mà trên thực tế nghĩa là <code>waitUntilCompleted</code> hoặc một
            completion handler. Đây là nhiệm vụ duy nhất của stub CUDA còn tồn tại trên Apple silicon.
          </p>
        }
      >
        <p>
          <code>commit</code> returns while Metal is still preparing and scheduling the work. With shared storage there
          is no copy to hide behind, so the ordering is yours to keep: finish the GPU's work on a buffer before the CPU
          touches it, which in practice means <code>waitUntilCompleted</code> or a completion handler. This is the one
          duty of the CUDA stub that survives on Apple silicon.
        </p>
      </Translated>
    ),
    source: 'not in the book; notes/today/ch05-facts.md, C6, and ch02-facts.md, C3 and C4',
    vi: { source: 'không có trong sách; notes/today/ch05-facts.md, C6, và ch02-facts.md, C3 và C4' },
  },
};

// ---------------------------------------------------------------- the section
export function S24() {
  const stub = useLabFile(STUB);
  const where = stub ? stubLines(stub) : null;
  return (
    <>
      <Translated
        vi={
          <p>
            Trong hình dung của sách, một device là một card phần cứng đi kèm bộ nhớ riêng. Bộ nhớ đó được gọi là bộ nhớ
            toàn cục của device, hay gọn là bộ nhớ toàn cục (global memory). Chữ "toàn cục" phân biệt nó với các loại bộ
            nhớ khác mà một kernel có thể dùng, là chủ đề của <a href={href('ch05', '5.2')}>chương 5</a>. Và từ đây trở
            đi, "truyền từ host sang device" là cách nói gọn cho một việc duy nhất: sao chép byte từ bộ nhớ host sang bộ
            nhớ toàn cục của device.
          </p>
        }
      >
        <p>
          In the book's picture a device is a hardware card that comes with its own memory. That memory is called device
          global memory, or just global memory. "Global" sets it apart from the other kinds of memory a kernel can use,
          which are the subject of <a href={href('ch05', '5.2')}>chapter 5</a>. And from here on "transfer from host to
          device" is shorthand for one thing: copy bytes from host memory to device global memory.
        </p>
      </Translated>
      <Source section="2.4" pages="28" />
      <Translated
        vi={
          <p>
            Ví dụ của sách về dung lượng là NVIDIA H100, với 80 GB hoặc 94 GB. Bạn có bao nhiêu bộ nhớ toàn cục là câu
            hỏi dành cho device của bạn, không phải cho một cuốn sách, và bảng bên dưới cho thấy device của bạn đã trả lời
            gì. Một GPU của Apple trả lời một câu hỏi khác, vì nó dùng chung bộ nhớ hệ thống với CPU và không có bộ nhớ
            riêng nào để đếm.
          </p>
        }
      >
        <p>
          The book's example of a size is the NVIDIA H100, with 80 GB or 94 GB. How much global memory you have is a
          question for your device, not for a book, and the table below shows what yours answered. An Apple GPU answers
          a different question, because it shares the system's memory with the CPU and has none of its own to count.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md, A12)', '2026-11-20 (notes/today/ch02-facts.md, A12)')}>
        <Translated
          vi={
            <p>
              Hai con số là hai sản phẩm, không phải hai tùy chọn của một card: NVIDIA ghi 80 GB cho H100 SXM và 94 GB
              cho H100 NVL. Chúng xuất hiện ở đây chỉ như ví dụ của sách.
            </p>
          }
        >
          <p>
            The two figures are two products, not two options of one card: NVIDIA lists 80 GB for the H100 SXM and 94 GB
            for the H100 NVL. They appear here only as the book's example.
          </p>
        </Translated>
      </Today>
      <YourDeviceMemory />

      <h2><Translated vi="Stub mà phần này mổ xẻ">The stub this section takes apart</Translated></h2>
      <Translated
        vi={
          <p>
            <a href={href('ch02', '2.3')}>Phần 2.3</a> để ba phần của hàm host dưới dạng chú thích. Đây là hàm đó với
            Phần 1 và Phần 3 được viết đầy đủ.{' '}
            {where ? (
              <span data-out="stub-lines">
                Mọi thứ trong phần này nói về các dòng {where.part1[0]} đến {where.part1[1]} và {where.part3[0]} đến{' '}
                {where.part3[1]}; kernel ở trên cùng và lệnh khởi chạy ở dòng {where.launch} thuộc về hai phần tiếp
                theo.
              </span>
            ) : (
              'Mọi thứ trong phần này nói về Phần 1 và Phần 3; kernel ở trên cùng và lệnh khởi chạy nằm giữa chúng thuộc về hai phần tiếp theo.'
            )}
          </p>
        }
      >
        <p>
          <a href={href('ch02', '2.3')}>Section 2.3</a> left the three parts of the host function as comments. Here is
          the function with Parts 1 and 3 written out.{' '}
          {where ? (
            <span data-out="stub-lines">
              Everything in this section is about lines {where.part1[0]} to {where.part1[1]} and {where.part3[0]} to{' '}
              {where.part3[1]}; the kernel at the top and the launch on line {where.launch} belong to the next two
              sections.
            </span>
          ) : (
            'Everything in this section is about Parts 1 and 3; the kernel at the top and the launch between them belong to the next two sections.'
          )}
        </p>
      </Translated>
      <Source section="2.4" pages="31" figs={tr('Fig. 2.8; complete in Fig. 2.13, p. 38', 'Fig. 2.8; bản đầy đủ ở Fig. 2.13, p. 38')} />
      <Code
        tabs={[{ label: 'CUDA', code: stub ?? '// loading labs/' + STUB, tiers: DEVICE_TIERS, status: 'front-end-checked; unverified on CUDA hardware' }]}
        caption={
          <Translated
            vi={
              <>
                <code>labs/{STUB}</code>: Fig. 2.13 của sách với các tên <code>_h</code> và <code>_d</code>{' '}
                của khóa học này, đầu vào <code>const</code>, và số khối nguyên của Fig. 2.15 trong sách ở lệnh khởi
                chạy. Giống bản in của sách, nó không kiểm tra giá trị trả về nào; cuối phần này sẽ sửa điều đó. Các tên
                có màu của bộ nhớ toàn cục là địa chỉ trên device.
              </>
            }
          >
            <>
              <code>labs/{STUB}</code>: the book's Fig. 2.13 with this course's <code>_h</code> and <code>_d</code>{' '}
              names, <code>const</code> inputs, and the integer block count of the book's Fig. 2.15 in the launch. Like
              the book's listing it checks no return value; the end of this section fixes that. Names in the colour of
              global memory are addresses on the device.
            </>
          </Translated>
        }
      />

      <h2><Translated vi="Cấp phát và giải phóng: cudaMalloc và cudaFree">Allocating and freeing: cudaMalloc and cudaFree</Translated></h2>
      <Translated
        vi={
          <p>
            <code>cudaMalloc</code> nhận hai đối số: địa chỉ của một biến con trỏ, và một kích thước tính bằng byte. Nó
            không trả con trỏ về theo cách <code>malloc</code> của C làm. Nó ghi địa chỉ mới vào biến của bạn qua tham số
            đầu tiên, để giá trị trả về rảnh mang một mã lỗi, như mọi hàm khác của CUDA API. <code>cudaFree</code> nhận
            giá trị của con trỏ, không phải địa chỉ của nó. Nó chỉ cần biết vùng cấp phát nào được trả về vùng nhớ chung.
          </p>
        }
      >
        <p>
          <code>cudaMalloc</code> takes two arguments: the address of a pointer variable, and a size in bytes. It does
          not hand the pointer back the way C's <code>malloc</code> does. It writes the new address into your variable
          through the first parameter, which leaves its return value free to carry an error code, as with every other
          function of the CUDA API. <code>cudaFree</code> takes the pointer's value, not its address. It only has to know
          which allocation goes back to the pool.
        </p>
      </Translated>
      <Source section="2.4" pages="28–30" figs="Fig. 2.6" />
      <Translated
        vi={
          <p>
            Kích thước tính bằng byte, nên <code>n</code> số float cần <code>n * sizeof(float)</code>. Chỉ viết{' '}
            <code>n</code>, ở đây hoặc trong một lệnh sao chép, là lỗi phổ biến nhất với hàm này, và bạn sẽ gặp cả hai
            dạng của nó trong phần săn lỗi bên dưới. Một <code>float</code> là 4 byte trên cả ba backend của khóa học, nên
            biểu thức này dùng được nguyên vẹn.
          </p>
        }
      >
        <p>
          The size is in bytes, so <code>n</code> floats need <code>n * sizeof(float)</code>. Writing <code>n</code>{' '}
          alone, here or in a copy, is the most common mistake in this function, and you will meet both forms of it in
          the bug hunt below. A <code>float</code> is 4 bytes on all three backends of this course, so the expression
          ports unchanged.
        </p>
      </Translated>
      <Source section="2.4" pages="30" />
      <Translated
        vi={
          <p>
            Sách nói đối số đầu tiên phải được ép kiểu sang <code>(void**)</code>, vì hàm này là hàm tổng quát và không
            biết gì về kiểu bạn sẽ lưu. Đó là toàn bộ câu chuyện khi thành ngữ này được hình thành. Giờ thì không còn như
            vậy nữa.
          </p>
        }
      >
        <p>
          The book says the first argument has to be cast to <code>(void**)</code>, because the function is generic and
          knows nothing of the type you will store. That was the whole story when the idiom was established. It is not
          any more.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md, A2 and B1)', '2026-11-20 (notes/today/ch02-facts.md, A2 và B1)')}>
        <Translated
          vi={
            <p>
              Phép ép kiểu là tùy chọn trong CUDA C++ hiện nay. Tài liệu tham chiếu API liệt kê một overload có kiểu, và
              chính tutorial của NVIDIA gọi <code>cudaMalloc(&amp;devA, ...)</code> trên một <code>float*</code>. HIP có
              cùng wrapper đó. Các lab giữ phép ép kiểu: nó biên dịch được ở mọi nơi, kể cả trình giả lập của khóa học.
            </p>
          }
        >
          <p>
            The cast is optional in current CUDA C++. The API reference lists a typed overload, and NVIDIA's own
            tutorial calls <code>cudaMalloc(&amp;devA, ...)</code> on a <code>float*</code>. HIP has the same wrapper.
            The labs keep the cast: it compiles everywhere, including this course's emulator.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Ba thuộc tính nữa có trong tài liệu tham chiếu của NVIDIA mà không có trong chương, và mỗi thuộc tính quyết
            định một lỗi trông ra sao. Bộ nhớ bạn nhận được không được xóa. Giải phóng cùng một con trỏ hai lần là lỗi,
            còn giải phóng một con trỏ null thì không làm gì. Và <code>cudaFree</code> có thể chờ công việc còn dở trên
            device trước khi trả về, nên một bộ đếm thời gian bao quanh nó có thể đo nhiều hơn một lần free.
          </p>
        }
      >
        <p>
          Three more properties are in NVIDIA's reference and not in the chapter, and each one decides what a bug looks
          like. The memory you get is not cleared. Freeing the same pointer twice is an error, while freeing a null
          pointer does nothing. And <code>cudaFree</code> may wait for outstanding work on the device before it returns,
          so a timer around it can measure more than a free.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md, A2 and A3)', '2026-11-20 (notes/today/ch02-facts.md, A2 và A3)')}>
        <Translated
          vi={
            <p>
              Theo lời NVIDIA: "The memory is not cleared" (bộ nhớ không được xóa); "If <code>cudaFree(devPtr)</code> has
              already been called before, an error is returned. If <code>devPtr</code> is 0, no operation is performed"
              (nếu đã gọi trước đó thì trả về lỗi; nếu bằng 0 thì không làm gì); "this API may perform implicit
              synchronization" (API này có thể đồng bộ hóa ngầm).
            </p>
          }
        >
          <p>
            In NVIDIA's words: "The memory is not cleared"; "If <code>cudaFree(devPtr)</code> has already been called
            before, an error is returned. If <code>devPtr</code> is 0, no operation is performed"; "this API may perform
            implicit synchronization".
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS24['ch02/2.4/q-malloc-size']} label={tr('Book exercise 5', 'Bài tập 5 của sách')} />
      <Ask q={questionsS24['ch02/2.4/q-malloc-ptr']} label={tr('Book exercise 6', 'Bài tập 6 của sách')} />
      <Ask q={questionsS24['ch02/2.4/q-out-param']} />

      <h2><Translated vi="Sao chép: cudaMemcpy">Copying: cudaMemcpy</Translated></h2>
      <Translated
        vi={
          <p>
            Một hàm duy nhất di chuyển dữ liệu theo cả hai chiều. Bốn đối số của nó là đích, nguồn, số byte, và một hằng
            số cho biết loại sao chép. Sách liệt kê bốn loại: host sang host, host sang device, device sang host, device
            sang device. Stub dùng hai loại, <code>cudaMemcpyHostToDevice</code> cho A và B và{' '}
            <code>cudaMemcpyDeviceToHost</code> cho C. Bạn chọn chiều hai lần trong mỗi lời gọi, một lần bằng thứ tự các
            con trỏ và một lần bằng hằng số, và hai lựa chọn phải khớp nhau.
          </p>
        }
      >
        <p>
          One function moves data in both directions. Its four arguments are the destination, the source, the number of
          bytes, and a constant naming the kind of copy. The book lists four kinds: host to host, host to device, device
          to host, device to device. The stub uses two, <code>cudaMemcpyHostToDevice</code> for A and B and{' '}
          <code>cudaMemcpyDeviceToHost</code> for C. You choose the direction twice in every call, once by the order of
          the pointers and once by the constant, and the two have to agree.
        </p>
      </Translated>
      <Source section="2.4" pages="30–31" figs="Fig. 2.7" />
      <Translated
        vi={
          <p>
            Điều xảy ra khi chúng không khớp không phải điều bạn mong đợi. Tài liệu không hứa hẹn một lỗi. Nó nói hành vi
            là không xác định, đó là lý do một cặp con trỏ bị đảo xuất hiện trong phần săn lỗi mà không có mã lỗi nào đi
            kèm.
          </p>
        }
      >
        <p>
          What happens when they disagree is not what you might hope for. The documentation does not promise an error.
          It says the behaviour is undefined, which is the reason a swapped pair of pointers appears in the bug hunt
          with no error code beside it.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md, A4)', '2026-11-20 (notes/today/ch02-facts.md, A4)')}>
        <Translated
          vi={
            <p>
              Giờ có thêm loại thứ năm, <code>cudaMemcpyDefault</code>, suy ra chiều từ các con trỏ. NVIDIA khuyến nghị
              nó và dùng nó trong tutorial của mình. Các con trỏ không khớp với loại sao chép được ghi trong tài liệu là
              hành vi không xác định. Lời gọi là đồng bộ: nó trả về khi việc sao chép đã hoàn tất.
            </p>
          }
        >
          <p>
            There is now a fifth kind, <code>cudaMemcpyDefault</code>, which infers the direction from the pointers.
            NVIDIA recommends it and uses it in its tutorial. Pointers that do not match the kind are documented as
            undefined behaviour. The call is synchronous: it returns when the copy has completed.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS24['ch02/2.4/q-memcpy']} label={tr('After book exercise 7', 'Theo sau bài tập 7 của sách')} />

      <h2><Translated vi="Chuyến khứ hồi, từng lời gọi một">The round trip, call by call</Translated></h2>
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy stub từng lời gọi một trên 8 số nhỏ và cho thấy cả hai bộ nhớ sau mỗi lời gọi: mỗi
            mảng chứa gì, mỗi biến con trỏ trỏ tới đâu, luồng host có phải chờ không, và bao nhiêu byte đã đi qua cho đến
            lúc đó. Hãy chạy từng bước phiên bản CUDA trước. Ngay sau lệnh khởi chạy, nhấn{' '}
            <i>Đọc C ngay</i> và xem host lấy được gì vào lúc đó. Trước khi bạn đổi backend, hãy chốt một câu trả lời ở
            đây.
          </p>
        }
      >
        <p>
          The stepper runs the stub one call at a time on 8 small numbers and shows both memories after each call: what
          each array holds, what each pointer variable refers to, whether the host thread had to wait, and how many
          bytes have crossed so far. Step through the CUDA version first. Right after the launch, press{' '}
          <i>Read C now</i> and see what the host can get at that moment. Before you switch the backend, commit to an
          answer here.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Bộ nhớ là 8 số nguyên nhỏ cho mỗi mảng, và bộ nhớ chưa được xóa được vẽ là <b>?</b>, không phải một giá trị
              bịa ra nào đó. Mô hình báo cáo những gì các nhà cung cấp ghi trong tài liệu và nói "không xác định" ở chỗ họ
              không nói gì. Nó không có đồng hồ, và không mô hình hóa không gian địa chỉ hay một lỗi thật trông ra sao.
            </p>
          }
        >
          <p>
            Memory is 8 small integers per array, and memory that was not cleared is drawn as <b>?</b>, not as some
            invented value. The model reports what the vendors document and says "not determined" where they say
            nothing. It has no clock, and it does not model address spaces or what a real fault looks like.
          </p>
        </Translated>
      </Simplification>
      <Predict
        id={P_COPIES}
        prompt={
          <Translated
            vi={
              <>
                Stub CUDA sao chép dữ liệu ba lần: A vào, B vào, C ra. Trên Apple silicon, với cả ba mảng được tạo dưới
                dạng buffer shared storage, phiên bản Metal thực hiện bao nhiêu lần sao chép?
              </>
            }
          >
            <>
              The CUDA stub copies data three times: A in, B in, C out. On Apple silicon, with all three arrays created
              as shared-storage buffers, how many copies does the Metal version make?
            </>
          </Translated>
        }
        unit={tr('copies', 'lần sao chép')}
      >
        {(v) => (
          <Translated
            vi={
              <>
                <p>
                  Nó thực hiện {METAL_END.copies}, so với {CUDA_END.copies} của CUDA.{' '}
                  {v === METAL_END.copies ? 'Đúng như bạn nói.' : 'Shared storage là một vùng nhớ duy nhất mà cả hai bên cùng dùng, nên không có nơi nào để sao chép tới.'}{' '}
                  Nó cũng thực hiện {METAL_END.frees} lời gọi free, so với {CUDA_END.frees}.
                </p>
                <p>
                  Thứ còn lại là việc chờ trước khi đọc kết quả. Hãy chuyển trình chạy từng bước sang Metal, và nhấn{' '}
                  <i>Đọc C ngay</i> ngay sau lệnh commit.
                </p>
              </>
            }
          >
            <>
              <p>
                It makes {METAL_END.copies}, against {CUDA_END.copies} for CUDA.{' '}
                {v === METAL_END.copies ? 'As you said.' : 'Shared storage is one piece of memory that both sides use, so there is nowhere to copy to.'}{' '}
                It also makes {METAL_END.frees} free calls, against {CUDA_END.frees}.
              </p>
              <p>
                What is left is the wait before the result is read. Switch the stepper to Metal, and press{' '}
                <i>Read C now</i> right after the commit.
              </p>
            </>
          </Translated>
        )}
      </Predict>
      <RoundTripStepper />
      <Translated
        vi={
          <p>
            Trên CUDA và HIP, mười lời gọi giống nhau chỉ khác tiền tố. Host không thể thấy kết quả cho đến khi lệnh sao
            chép về đưa nó về nhà, nên đọc C quá sớm không phải là lỗi bạn có thể vô tình mắc: không có gì ở đó để đọc.
            Trên Metal, với dữ liệu được tạo trong các buffer, ba lần sao chép và ba lần free hoàn toàn không có thứ tương
            ứng. Giữ giao diện của sách, mảng host vào và ra, thì một lần sao chép cho mỗi đầu vào và một lần sao chép ra
            quay trở lại ngay. Và trong cả hai phiên bản Metal, đọc C quá sớm <em>là</em> có thể: bộ nhớ ở ngay đó, và nó
            chưa chứa kết quả. Việc chờ vẫn còn ngay cả ở nơi việc sao chép không còn.
          </p>
        }
      >
        <p>
          On CUDA and HIP the ten calls are the same with different prefixes. The host cannot see the result until the
          copy back has brought it home, so reading C early is not a mistake you can make by accident: there is nothing
          there to read. On Metal, with the data created in the buffers, the three copies and the three frees have no
          counterpart at all. Keep the book's interface, host arrays in and out, and one copy per input and one copy out
          come straight back. And in both Metal versions reading C early <em>is</em> possible: the memory is right
          there, and it does not hold the result yet. The wait survives even where the copy does not.
        </p>
      </Translated>

      <h2><Translated vi="Con trỏ device trên host">Device pointers on the host</Translated></h2>
      <Translated
        vi={
          <p>
            Sau Phần 1, <code>A_d</code>, <code>B_d</code> và <code>C_d</code> là các biến con trỏ bình thường trong mã
            host, giữ các địa chỉ chỉ có ý nghĩa trên device. Quy tắc của sách: trong mã host, hãy truyền chúng cho các
            hàm API và cho kernel và không làm gì khác với chúng. Theo sách, giải tham chiếu một con trỏ như vậy trên host
            có thể gây ra ngoại lệ hoặc các lỗi thời gian chạy khác. Mã host có dạng này, tức cấp phát, truyền dữ liệu và
            gọi một kernel, có một cái tên bạn sẽ còn gặp lại: nó là một <b>stub</b> để gọi kernel.
          </p>
        }
      >
        <p>
          After Part 1, <code>A_d</code>, <code>B_d</code> and <code>C_d</code> are ordinary pointer variables in host
          code, holding addresses that mean something only on the device. The book's rule: in host code, pass them to
          API functions and to kernels and do nothing else with them. Dereferencing one on the host, it says, can cause
          exceptions or other run-time errors. Host code of this shape, which allocates, transfers and calls a kernel,
          has a name you will see again: it is a <b>stub</b> for calling the kernel.
        </p>
      </Translated>
      <Source section="2.4" pages="30–31" />
      <Translated
        vi={
          <p>
            Ba backend ghi trong tài liệu ba điều khác nhau ở đây. NVIDIA nêu quy tắc, rằng bộ nhớ device chỉ kernel mới
            truy cập được, và không nói vi phạm nó thì sao; "ngoại lệ hoặc các lỗi thời gian chạy khác" là cách diễn đạt
            của sách. AMD ghi trong tài liệu việc host truy cập bộ nhớ <code>hipMalloc</code> là truy cập "zero copy" qua
            một giao tiếp chậm hơn nhiều, phụ thuộc cấu hình và chưa được kiểm tra ở đây, nên thói quen của sách vẫn là
            đúng. Trên Metal, đọc và ghi một buffer dùng chung qua <code>contents</code> là con đường bình thường.
          </p>
        }
      >
        <p>
          The three backends document three different things here. NVIDIA states the rule, that device memory is
          accessible only to kernels, and does not say what breaking it does; "exceptions or other run-time errors" is
          the book's wording. AMD documents host access to <code>hipMalloc</code> memory as "zero copy" access over a
          much slower interface, dependent on configuration and not tested here, so the book's habit is still the right
          one. On Metal, reading and writing a shared buffer through <code>contents</code> is the normal path.
        </p>
      </Translated>
      <Beyond title={tr('Checked, not in the book', 'Đã kiểm tra, không có trong sách')}>
        <Translated
          vi={
            <p>
              notes/today/ch02-facts.md: A17 cho CUDA, B7 cho HIP, C2 và C3 cho Metal, đọc ngày 2026-11-20. Không điều gì
              ở đây được xác lập bằng cách chạy mã trên phần cứng CUDA hay HIP.
            </p>
          }
        >
          <p>
            notes/today/ch02-facts.md: A17 for CUDA, B7 for HIP, C2 and C3 for Metal, read on 2026-11-20. Nothing here
            was established by running code on CUDA or HIP hardware.
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Thứ thay thế mỗi lời gọi trên Metal">What replaces each call on Metal</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn đã thấy nó chuyển động trong trình chạy từng bước. Đây là nó dưới dạng bảng, cùng với các trường hợp mà
            việc sao chép quay trở lại.
          </p>
        }
      >
        <p>
          You have seen it move in the stepper. Here it is as a table, with the cases in which a copy comes back.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ dùng CUDA. Cột Metal lấy từ tài liệu và header SDK của Apple, còn cột HIP từ header runtime của
              AMD (notes/today/ch02-facts.md, B1, C2, C3; bảng "Where the backends really differ").
            </p>
          }
        >
          <p>
            The book uses CUDA only. The Metal column is from Apple's documentation and SDK headers, and the HIP column
            from AMD's runtime header (notes/today/ch02-facts.md, B1, C2, C3; the table "Where the backends really
            differ").
          </p>
        </Translated>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '62rem' }}>
        <thead>
          <tr>
            <th><Translated vi="Bước của stub">The stub's step</Translated></th>
            <th>CUDA</th>
            <th>HIP</th>
            <th><Translated vi="Metal, shared storage">Metal, shared storage</Translated></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="Cấp phát">Allocate</Translated></td>
            <td><Translated vi={<><code>cudaMalloc(&amp;p, bytes)</code>: không được xóa; thất bại là một mã lỗi</>}><code>cudaMalloc(&amp;p, bytes)</code>: not cleared; failure is an error code</Translated></td>
            <td><Translated vi={<><code>hipMalloc(&amp;p, bytes)</code>: header của AMD không nói nó có được xóa hay không, nên hãy giả định là không</>}><code>hipMalloc(&amp;p, bytes)</code>: AMD's header does not say whether it is cleared, so assume not</Translated></td>
            <td><Translated vi={<><code>[device newBufferWithLength:options:]</code>: chứa toàn số 0; thất bại là giá trị trả về nil</>}><code>[device newBufferWithLength:options:]</code>: filled with zeros; failure is a nil return</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Giải phóng">Free</Translated></td>
            <td><Translated vi={<><code>cudaFree(p)</code>: hai lần là lỗi; có thể đồng bộ hóa</>}><code>cudaFree(p)</code>: twice is an error; may synchronise</Translated></td>
            <td><Translated vi={<><code>hipFree(p)</code>: luôn đồng bộ hóa trước</>}><code>hipFree(p)</code>: always synchronises first</Translated></td>
            <td><Translated vi="không có lời gọi: buffer là một đối tượng và biến mất cùng tham chiếu cuối cùng của nó">no call: the buffer is an object and goes away with its last reference</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Sao chép vào, sao chép về">Copy in, copy back</Translated></td>
            <td><code>cudaMemcpy(dst, src, bytes, kind)</code></td>
            <td><code>hipMemcpy(dst, src, bytes, kind)</code></td>
            <td><Translated vi={<>không có: đọc và ghi qua <code>contents</code></>}>none: read and write through <code>contents</code></Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Nơi việc sao chép quay lại">Where a copy returns</Translated></td>
            <td colSpan={2}><Translated vi="luôn luôn, trong stub này">always, in this stub</Translated></td>
            <td>
              <Translated
                vi={
                  <>
                    <code>newBufferWithBytes</code> sao chép một lần khi tạo; private storage cần một lệnh blit; và bất kỳ
                    giao diện nào đòi mảng host đều cần một lần sao chép mỗi chiều
                  </>
                }
              >
                <code>newBufferWithBytes</code> copies once at creation; private storage needs a blit command; and any
                interface that insists on host arrays needs one copy each way
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="Bọc vùng nhớ bạn đã có">Wrapping memory you already have</Translated></td>
            <td colSpan={2}><Translated vi="không có trong chương này">not in this chapter</Translated></td>
            <td><Translated vi={<><code>newBufferWithBytesNoCopy</code>: con trỏ và độ dài phải căn theo trang</>}><code>newBufferWithBytesNoCopy</code>: the pointer and the length must be page-aligned</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Việc bạn phải làm trước khi đọc kết quả">What you owe before reading the result</Translated></td>
            <td><Translated vi="lệnh sao chép về kiểu chặn lo việc chờ">the blocking copy back does the waiting</Translated></td>
            <td><Translated vi={'như vậy, theo những gì tài liệu của AMD nói: họ gọi các lệnh sao chép tường minh là "long blocking operations" trong một mẹo hiệu năng'}>the same, as far as AMD's docs go: they call explicit copies "long blocking operations" in a performance tip</Translated></td>
            <td><Translated vi={<>chờ command buffer: <code>waitUntilCompleted</code> hoặc một completion handler</>}>wait for the command buffer: <code>waitUntilCompleted</code> or a completion handler</Translated></td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Hàng cuối là hàng cần nhớ. Không có sao chép, không có gì đứng giữa host và một buffer mà GPU vẫn đang ghi.
            Quy tắc của Apple là các thay đổi được lên lịch ở một bên phải hoàn tất trước khi bên kia chạm vào tài
            nguyên, và không có gì thực thi quy tắc đó thay bạn.
          </p>
        }
      >
        <p>
          The last row is the one to remember. With no copy, nothing stands between the host and a buffer the GPU is
          still writing. Apple's rule is that changes scheduled on one side must be complete before the other side
          touches the resource, and nothing enforces it for you.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Chú thích trong header của Apple về <code>MTLBuffer</code>: việc truy cập buffer "is not synchronized"
              (không được đồng bộ hóa), và bên gọi "responsible for ensuring synchronization and coherency" (chịu trách
              nhiệm bảo đảm đồng bộ hóa và tính nhất quán) (ch02-facts.md, C3; ch05-facts.md, C6). Việc chờ command buffer
              là cách đáp ứng quy tắc đó là cách đọc của chúng tôi.
            </p>
          }
        >
          <p>
            Apple's header comment on <code>MTLBuffer</code>: access to buffers "is not synchronized", and the caller is
            "responsible for ensuring synchronization and coherency" (ch02-facts.md, C3; ch05-facts.md, C6). That waiting
            for the command buffer is how you meet the rule is our reading.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS24['ch02/2.4/q-not-cleared']} />
      <Ask q={questionsS24['ch02/2.4/q-shared-wait']} />
      <Translated
        vi={
          <p>
            Ba backend mà bạn có thể sẽ đọc làm "Phần 1" đúng theo những cách này. Trong llama.cpp, hàm đặt dữ liệu cho
            một tensor trên Metal bắt đầu bằng <code>{'if (buf->is_shared) { memcpy(...); return; }'}</code>, trong khi
            hàm tương ứng cho CUDA gọi <code>cudaMemcpyAsync</code> rồi <code>cudaStreamSynchronize</code>. MLX cấp phát
            mọi mảng dưới dạng một buffer dùng chung, và con trỏ CPU của một mảng chính là <code>contents</code> của
            buffer. Backend MPS của PyTorch từ chối chạy nếu không có unified memory, và <code>tensor.to('mps')</code>{' '}
            vẫn sao chép, vì tensor CPU nó bắt đầu từ đó không được tạo trong một buffer.
          </p>
        }
      >
        <p>
          Three backends you may end up reading do "Part 1" in exactly these ways. In llama.cpp, the function that sets
          a tensor's data on Metal begins <code>{'if (buf->is_shared) { memcpy(...); return; }'}</code>, while its CUDA
          counterpart calls <code>cudaMemcpyAsync</code> and then <code>cudaStreamSynchronize</code>. MLX allocates
          every array as a shared buffer, and an array's CPU pointer is the buffer's <code>contents</code>. PyTorch's
          MPS backend refuses to run without unified memory, and <code>tensor.to('mps')</code> still copies, because the
          CPU tensor it starts from was not created in a buffer.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Đọc từ nhánh main của từng dự án ngày 2026-11-20 (notes/today/ch02-facts.md, D1 đến D4). Số dòng ở đó sẽ
              thay đổi theo thời gian.
            </p>
          }
        >
          <p>
            Read from each project's main branch on 2026-11-20 (notes/today/ch02-facts.md, D1 to D4). Line numbers
            there will drift.
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Phá stub">Break the stub</Translated></h2>
      <Translated
        vi={
          <p>
            Mọi lời gọi trong stub đều có thể bị viết sai theo cách vẫn biên dịch được. Trò săn lỗi bên dưới áp một lỗi ẩn
            vào stub CUDA và chỉ cho bạn thấy những gì một bài kiểm tra sẽ thấy: <code>C_h</code> của hàm gọi bên cạnh đáp
            án đúng, trên một device theo những gì tài liệu nói và trong trình giả lập của khóa học, kèm mã lỗi nếu có. Hãy
            gọi tên nguyên nhân. Rồi chạy từng bước stub bị hỏng để thấy khoảnh khắc nó đi sai.
          </p>
        }
      >
        <p>
          Every call in the stub can be written wrongly in a way that compiles. The hunt below applies one hidden defect
          to the CUDA stub and shows you only what a test would see: the caller's <code>C_h</code> next to the right
          answer, on a device as far as the documentation goes and in the course emulator, with any error code. Name the
          cause. Then step through the broken stub to see the moment it goes wrong.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Ở chỗ tài liệu không còn hứa hẹn một kết quả, hàng device hiển thị <b>~</b>. Một device thật sẽ hiển thị
              một con số nào đó ở đó, có thể là con số đúng. Hàng trình giả lập là những gì trình giả lập của khóa học thực
              sự làm, với mẫu điền thật và AddressSanitizer của nó.
            </p>
          }
        >
          <p>
            Where the documentation stops promising a result, the device row shows <b>~</b>. A real device would show
            some number there, possibly the right one. The emulator row is what the course's emulator does, with its
            real fill pattern and AddressSanitizer.
          </p>
        </Translated>
      </Simplification>
      <BreakTheStub />
      <Translated
        vi={
          <p>
            Có ba mẫu đáng ghi nhớ. Một kết quả đúng với chính xác một phần tư đầu nghĩa là một số đếm theo phần tử ở chỗ
            cần byte: 4 byte mỗi float, nên một phần tư số float. Một kết quả hợp lý nhưng cũ nghĩa là có gì đó xảy ra sai
            thứ tự. Và gần như không có gì ở đây tạo ra mã lỗi. Những lỗi ghi hoặc đọc ra ngoài một vùng cấp phát là những
            lỗi mà trình giả lập dừng lại, đó là lý do mọi lab đều chạy ở đó trước khi chạy trên GPU.
          </p>
        }
      >
        <p>
          Three patterns are worth keeping. A result that is right for exactly the first quarter means a count in
          elements where bytes were needed: 4 bytes per float, so a quarter of the floats. A result that is plausible
          and stale means something happened in the wrong order. And almost nothing here produces an error code. The
          mistakes that write or read outside an allocation are the ones the emulator stops, which is why every lab runs
          there before it runs on a GPU.
        </p>
      </Translated>

      <h2><Translated vi="Kiểm tra lỗi">Error checking</Translated></h2>
      <Translated
        vi={
          <p>
            Mọi hàm của CUDA API đều trả về một giá trị cho biết nó có thành công không, và các đoạn mã của sách bỏ qua tất
            cả để mã ngắn gọn. Một khung bên lề đưa ra mẫu mà chúng bỏ đi: giữ kết quả trong một{' '}
            <code>cudaError_t</code>, so sánh nó với <code>cudaSuccess</code>, và khi thất bại thì in{' '}
            <code>cudaGetErrorString</code> của nó cùng tệp và dòng, rồi thoát. Ví dụ của nó về một thất bại đáng bắt:
            device hết bộ nhớ. Nó gợi ý bọc phép kiểm tra trong một macro. Đoạn mã có một lỗi nhỏ, và dòng khai báo của
            Fig. 2.5 ở phần trước cũng vậy. Hãy tìm cả hai.
          </p>
        }
      >
        <p>
          Every function of the CUDA API returns a value that says whether it worked, and the book's listings ignore all
          of them to keep the code short. A sidebar gives the pattern they leave out: keep the result in a{' '}
          <code>cudaError_t</code>, compare it with <code>cudaSuccess</code>, and on failure print{' '}
          <code>cudaGetErrorString</code> of it with the file and line, then exit. Its example of a failure worth
          catching: the device is out of memory. It suggests wrapping the check in a macro. The snippet has a slip in
          it, and so has the declaration line of Fig. 2.5 from the last section. Find both.
        </p>
      </Translated>
      <Source section="2.4" pages="32" figs={tr('sidebar; Fig. 2.5 is p. 27', 'khung bên lề; Fig. 2.5 ở p. 27')} />
      <HuntThenMacro />
      <Translated
        vi={
          <p>
            Khung bên lề bao quát các lời gọi có trả về giá trị. Câu lệnh khởi chạy không trả về gì, nên một kernel không
            khởi chạy được, hoặc thất bại trong lúc chạy, đi thẳng qua mã kiểm tra mọi lời gọi API. Mẫu của NVIDIA là thêm
            hai phép kiểm tra: <code>cudaGetLastError()</code> ngay sau lệnh khởi chạy, và một{' '}
            <code>cudaDeviceSynchronize()</code> có kiểm tra sau đó. <a href={href('ch02', '2.6')}>Phần 2.6</a> đi qua
            cả hai, và lab 2.1 yêu cầu chúng.
          </p>
        }
      >
        <p>
          The sidebar covers calls that return a value. The launch statement returns nothing, so a kernel that could not
          be launched, or that failed while it ran, passes straight through code that checks every API call. NVIDIA's
          pattern is two more checks: <code>cudaGetLastError()</code> right after the launch, and a checked{' '}
          <code>cudaDeviceSynchronize()</code> after that. <a href={href('ch02', '2.6')}>Section 2.6</a> works through
          both, and lab 2.1 requires them.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md, A6)', '2026-11-20 (notes/today/ch02-facts.md, A6)')}>
        <Translated
          vi={
            <p>
              Một lỗi từ công việc bất đồng bộ vẫn được giữ, và được trả về bởi mọi lời gọi runtime sau đó, cho đến khi{' '}
              <code>cudaGetLastError</code> đọc và xóa nó. Một kết quả thành công ngay sau lệnh khởi chạy chỉ nói rằng
              lệnh khởi chạy đã được chấp nhận, không phải kernel đã chạy.
            </p>
          }
        >
          <p>
            An error from asynchronous work stays set, and is returned by every later runtime call, until{' '}
            <code>cudaGetLastError</code> reads and clears it. A success right after a launch says only that the launch
            was accepted, not that the kernel ran.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            HIP có cùng mẫu với tiền tố <code>hip</code>: <code>hipError_t</code>, <code>hipSuccess</code>,{' '}
            <code>hipGetErrorString</code>. Có hai khác biệt đáng biết. Dưới C++17, bỏ qua giá trị trả về của HIP là một
            cảnh báo của trình biên dịch, nên các đoạn mã không kiểm tra của sách không im lặng ở đó. Và mã hết bộ nhớ có
            tên là <code>hipErrorOutOfMemory</code>.
          </p>
        }
      >
        <p>
          HIP has the same pattern with the <code>hip</code> prefix: <code>hipError_t</code>, <code>hipSuccess</code>,{' '}
          <code>hipGetErrorString</code>. Two differences are worth knowing. Under C++17, ignoring a HIP return value is
          a compiler warning, so the book's unchecked listings are not silent there. And the out-of-memory code is named{' '}
          <code>hipErrorOutOfMemory</code>.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Header của AMD đánh dấu <code>hipError_t</code> <code>[[nodiscard]]</code> dưới C++17, và giữ tên kiểu CUDA{' '}
              <code>hipErrorMemoryAllocation</code> làm bí danh đã lỗi thời (notes/today/ch02-facts.md, B1 và B2).
            </p>
          }
        >
          <p>
            AMD's header marks <code>hipError_t</code> <code>[[nodiscard]]</code> under C++17, and keeps the CUDA-style
            name <code>hipErrorMemoryAllocation</code> as a deprecated alias (notes/today/ch02-facts.md, B1 and B2).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Metal không có mã lỗi và không có trạng thái lỗi cuối. Một phương thức tạo bị thất bại trả về nil. Các bước
            biên dịch nhận một <code>NSError**</code>. Và việc kernel có chạy hay không nằm trên command buffer sau khi
            chờ, trong các thuộc tính <code>status</code> và <code>error</code> của nó. Nhiệm vụ giống như trong CUDA,
            nhưng ở ba chỗ trong khi CUDA chỉ có một.
          </p>
        }
      >
        <p>
          Metal has no error codes and no last-error state. A creation method that fails returns nil. The steps that
          compile take an <code>NSError**</code>. And whether the kernel ran is on the command buffer after the wait, in
          its <code>status</code> and <code>error</code> properties. The duty is the same as in CUDA, in three places
          where CUDA has one.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Từ tài liệu và header SDK của Apple (notes/today/ch02-facts.md, C5). Nửa Metal của lab 2.1 yêu cầu bạn kiểm
              tra cả ba.
            </p>
          }
        >
          <p>
            From Apple's documentation and SDK headers (notes/today/ch02-facts.md, C5). The Metal half of lab 2.1 has
            you check each of the three.
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Các hàm cấp phát mà sách để lại sau">The allocation functions the book defers</Translated></h2>
      <Translated
        vi={
          <p>
            Sách nhắc đến, mà không nêu tên, các hàm cấp phát nâng cao hơn giúp sao chép nhanh hơn hoặc bỏ hẳn việc sao chép
            tường minh, và để chúng cho <a href={href('ch23')}>chương 23</a> và{' '}
            <a href={href('appC')}>phụ lục C</a>. Một chú thích cuối trang ở cùng trang để lại thêm một điều,
            mảng nhiều chiều được cấp phát bằng <code>cudaMalloc</code>, cho{' '}
            <a href={href('ch03')}>chương 3</a>.
          </p>
        }
      >
        <p>
          The book mentions, without naming them, more advanced allocation functions that make copies faster or remove
          the explicit copy, and leaves them to <a href={href('ch23')}>chapter 23</a> and{' '}
          <a href={href('appC')}>appendix C</a>. A footnote on the same page defers one more
          thing, multidimensional arrays allocated with <code>cudaMalloc</code>, to{' '}
          <a href={href('ch03')}>chapter 3</a>.
        </p>
      </Translated>
      <Source section="2.4" pages="29" figs={tr('footnote 2', 'chú thích cuối trang 2')} />
      <Translated
        vi={
          <p>
            Hai cái tên là đủ cho lúc này. <code>cudaMallocHost</code> cấp phát bộ nhớ host được khóa trang, truyền dữ
            liệu với băng thông cao hơn bộ nhớ host thông thường. <code>cudaMallocManaged</code> cấp phát CUDA unified
            memory: một con trỏ hợp lệ ở cả hai bên, với driver di chuyển dữ liệu giữa hai bộ nhớ vật lý khi cần. Đó không
            phải điều Apple muốn nói khi dùng từ unified memory, nơi chỉ có một bộ nhớ vật lý và không có gì di chuyển.
          </p>
        }
      >
        <p>
          Two names are enough for now. <code>cudaMallocHost</code> allocates page-locked host memory, which transfers
          at a higher bandwidth than ordinary host memory. <code>cudaMallocManaged</code> allocates CUDA unified memory:
          one pointer valid on both sides, with the driver moving the data between two physical memories as needed. That
          is not what Apple means by unified memory, where there is one physical memory and nothing migrates.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch02-facts.md, A16; ch05-facts.md, C6)', '2026-11-20 (notes/today/ch02-facts.md, A16; và ch05-facts.md, C6)')}>
        <Translated
          vi={
            <p>
              Cả hai hàm đều còn hiện hành. Hướng dẫn của NVIDIA giờ trình bày phiên bản unified memory của phép cộng
              vector trước, và phiên bản tường minh, phiên bản trong chương này, sau.
            </p>
          }
        >
          <p>
            Both functions are current. NVIDIA's guide now presents the unified-memory version of vector addition first
            and the explicit version, the one in this chapter, second.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Stub giờ đã hoàn chỉnh, trừ hai thứ mà nó tồn tại để đạt tới: kernel, là{' '}
            <a href={href('ch02', '2.5')}>phần 2.5</a>, và lệnh khởi chạy, là{' '}
            <a href={href('ch02', '2.6')}>phần 2.6</a>.
          </p>
        }
      >
        <p>
          The stub is now complete except for the two things it exists to reach: the kernel, which is{' '}
          <a href={href('ch02', '2.5')}>section 2.5</a>, and the launch, which is{' '}
          <a href={href('ch02', '2.6')}>section 2.6</a>.
        </p>
      </Translated>
    </>
  );
}
