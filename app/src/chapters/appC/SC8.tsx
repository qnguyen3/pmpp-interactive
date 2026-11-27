import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, NoteGroup, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr as trText } from '../../lib/i18n';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  DECODE_TOKENS,
  EXPLORE_PAGES,
  EXPLORE_POLICIES,
  POLICY_NAME,
  PRESETS,
  cost,
  explore,
  moved,
  scriptLabel,
  warmTraffic,
  type Counters,
  type Ev,
  type Policy,
  type PresetId,
  type ScriptKind,
  type ScriptOp,
  type SweepPolicy,
  type View,
} from '../../sim/managedmem';
import { PageFaultScene } from './PageFaultScene';

// Section C.8, Page fault handling (book pp. 628–629). Both explorers run on
// src/sim/managedmem.ts: `explore` for "Coherence by invalidation" and `warmTraffic`
// (decodeSweep) for "When the model does not fit".

const POLICY_HOW: Record<Policy, string> = {
  get bulk() {
    return trText('limited: every CPU-held page moves at each launch', 'hạn chế: mọi trang do CPU giữ đều di chuyển ở mỗi lần khởi chạy');
  },
  get migrate() {
    return trText('the default: fault, then move the page', 'mặc định: lỗi trang, rồi di chuyển trang');
  },
  get map() {
    return trText('SetPreferredLocation(host) + SetAccessedBy(GPU)', 'SetPreferredLocation(host) + SetAccessedBy(GPU)');
  },
  get replicate() {
    return trText('SetReadMostly: copies on read, invalidation on write', 'SetReadMostly: sao chép khi đọc, vô hiệu hóa khi ghi');
  },
  one: '',
};
const ADD: { kind: ScriptKind; label: string }[] = [
  {
    kind: 'cpu-write',
    get label() {
      return trText('CPU writes', 'CPU ghi');
    },
  },
  {
    kind: 'cpu-read',
    get label() {
      return trText('CPU reads', 'CPU đọc');
    },
  },
  {
    kind: 'kernel-read',
    get label() {
      return trText('kernel reads', 'kernel đọc');
    },
  },
  {
    kind: 'kernel-write',
    get label() {
      return trText('kernel writes', 'kernel ghi');
    },
  },
];
const MAX_SCRIPT = 40;
const show = (v: number) => String(Number(v.toPrecision(6)));

/** A sixteen-page strip per memory, for one policy at one step. */
function Strips({ view, events, policy }: { view: View; events: Ev[]; policy: Policy }) {
  const pages = view.m?.pages ?? [];
  const cw = 15;
  const gap = 2.5;
  const x0 = 34;
  const touched = new Set(events.filter((e) => e.page >= 0 && e.e !== 'error').map((e) => e.page));
  const invalidated = new Set(events.filter((e) => e.e === 'invalidate').map((e) => `${e.proc}:${e.page}`));
  const remote = new Set(events.filter((e) => e.e === 'remote').map((e) => e.page));
  const W = x0 + EXPLORE_PAGES * (cw + gap);
  return (
    <svg
      viewBox={`0 0 ${W} 62`}
      style={{ width: '100%', maxWidth: 360, display: 'block' }}
      role="img"
      aria-label={trText(
        `Pages under ${POLICY_NAME[policy]}`,
        `Các trang theo chính sách ${{ bulk: 'chuyển hàng loạt lúc khởi chạy', migrate: 'lỗi trang rồi di chuyển', map: 'ánh xạ khi truy cập', replicate: 'read-mostly', one: 'một bộ nhớ' }[policy]}`,
      )}
    >
      {(['cpu', 'gpu'] as const).map((side, r) => (
        <g key={side}>
          <text x={0} y={r * 26 + 19} fontSize="10" fontWeight="600" style={{ fill: side === 'gpu' ? 'var(--global)' : 'var(--ink)' }}>
            {side.toUpperCase()}
          </text>
          {pages.map((p, i) => {
            const here = p.loc === side || p.loc === 'both';
            const mappedRemote = side === 'gpu' && policy === 'map' && p.loc === 'cpu';
            const gone = invalidated.has(`${side}:${i}`);
            const x = x0 + i * (cw + gap);
            const y = r * 26 + 6;
            return (
              <g key={i} data-side={side} data-page={i} data-present={here ? 'yes' : mappedRemote ? 'mapped' : 'no'}>
                <rect x={x} y={y} width={cw} height={18} rx="2"
                  fill={here ? (side === 'gpu' ? 'var(--global)' : 'var(--ink-2)') : 'transparent'}
                  fillOpacity={here ? (side === 'gpu' ? 0.85 : 0.55) : 1}
                  stroke={gone ? 'var(--fail)' : mappedRemote ? 'var(--global)' : here ? 'none' : 'var(--rule)'}
                  strokeDasharray={mappedRemote || gone ? '3 2' : here ? undefined : '2 2'} strokeWidth={gone ? 1.6 : 1} />
                {touched.has(i) && (here || (side === 'gpu' && remote.has(i))) ? (
                  <rect x={x - 1.5} y={y - 1.5} width={cw + 3} height={21} rx="3" fill="none" stroke="var(--ink)" strokeWidth="1.4" />
                ) : null}
              </g>
            );
          })}
        </g>
      ))}
    </svg>
  );
}

function stepSummary(events: Ev[]): string {
  const n = (f: (e: Ev) => boolean) => events.filter(f).length;
  const parts: string[] = [];
  const h = n((e) => e.e === 'move' && e.dir === 'h2d');
  const d = n((e) => e.e === 'move' && e.dir === 'd2h');
  const gf = n((e) => e.e === 'fault' && e.proc === 'gpu');
  const cf = n((e) => e.e === 'fault' && e.proc === 'cpu');
  const r = n((e) => e.e === 'remote');
  const iv = n((e) => e.e === 'invalidate');
  if (h) parts.push(trText(`${h} to GPU`, `${h} lên GPU`));
  if (d) parts.push(trText(`${d} to CPU`, `${d} về CPU`));
  if (gf) parts.push(trText(`${gf} GPU ${gf === 1 ? 'fault' : 'faults'}`, `${gf} lỗi trang GPU`));
  if (cf) parts.push(trText(`${cf} CPU ${cf === 1 ? 'fault' : 'faults'}`, `${cf} lỗi trang CPU`));
  if (r) parts.push(trText(`${r} remote`, `${r} truy cập từ xa`));
  if (iv) parts.push(trText(`${iv} ${iv === 1 ? 'invalidation' : 'invalidations'}`, `${iv} lần vô hiệu hóa`));
  return parts.length ? parts.join(', ') : trText('nothing crossed', 'không có gì đi qua');
}

function Coherence() {
  const [presetId, setPresetId] = useState<PresetId | 'own'>('figC1');
  const [script, setScript] = useState<ScriptOp[]>(PRESETS[0].script);
  const [n, setN] = useState(0);
  const [from, setFrom] = useState(0);
  const [to, setTo] = useState(15);
  const [perMove, setPerMove] = useState(3);
  const [perRemote, setPerRemote] = useState(1);
  const runs = useMemo(() => Object.fromEntries(EXPLORE_POLICIES.map((p) => [p, explore(p, script)])) as Record<Policy, ReturnType<typeof explore>>, [script]);
  const total = script.length;
  const at = Math.min(n, total);

  const load = (id: PresetId) => {
    setPresetId(id);
    setScript(PRESETS.find((p) => p.id === id)!.script);
    setN(0);
  };
  // A script the reader builds: keep them at the end of what they are building.
  const append = (op: ScriptOp) => {
    if (script.length >= MAX_SCRIPT) return;
    setPresetId('own');
    setScript([...script, op]);
    setN(script.length + 1);
  };
  const add = (kind: ScriptKind) => append({ kind, from: Math.min(from, to), to: Math.max(from, to) });
  const costs = EXPLORE_POLICIES.map((p) => cost(runs[p].steps[at].counters, perMove, perRemote));
  const best = Math.min(...costs);

  return (
    <Bench
      title={trText('Coherence by invalidation', 'Nhất quán bằng vô hiệu hóa')}
      task={trText(
        'Load a preset, step it, and compare the four policies. Then build your own script.',
        'Nạp một kịch bản có sẵn, chạy từng bước, và so sánh bốn chính sách. Sau đó tự dựng kịch bản của bạn.',
      )}
      foot={
        <Translated
          vi={
            <>
              Tô đặc: trang có một bản sao hợp lệ trong bộ nhớ đó. Nét đứt màu hổ phách ở hàng GPU: được ánh xạ, đọc qua đường
              truyền. Nét đứt màu đỏ: một bản sao bị vô hiệu hóa ở bước này. Viền đậm: được chạm tới ở bước này. Chi phí đếm
              số trang đi qua đường truyền (theo cả hai chiều) và số lần truy cập trang từ xa; các lần vô hiệu hóa và lỗi
              trang được hiển thị nhưng không tính giá.
            </>
          }
        >
          <>
            Filled: the page has a valid copy in that memory. Dashed amber in the GPU row: mapped, read over the link. Red dashes:
            a copy invalidated by this step. A dark outline: touched by this step. Costs count pages crossing the link (either
            way) and remote page accesses; invalidations and faults are shown but not priced.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi="Kịch bản có sẵn">Preset</Translated>
          <select value={presetId} onChange={(e) => e.target.value !== 'own' && load(e.target.value as PresetId)} aria-label={trText('Preset', 'Kịch bản có sẵn')}>
            {PRESETS.map((p) => (
              <option key={p.id} value={p.id}>
                {trText(
                  p.label,
                  {
                    figC1: 'Fig. C.1',
                    'one-page': 'kernel chạm tới 1 trang trong 16',
                    'ping-pong': 'CPU và GPU lần lượt ghi vào một trang bộ đếm',
                    decode: 'vòng lặp decode đọc lại mọi trang',
                  }[p.id],
                )}
              </option>
            ))}
            <option value="own" disabled={presetId !== 'own'}>{trText('your script', 'kịch bản của bạn')}</option>
          </select>
        </label>
        <label className="field">
          <Translated vi="Các trang từ">Pages from</Translated>
          <select value={from} onChange={(e) => setFrom(+e.target.value)} aria-label={trText('First page', 'Trang đầu')}>
            {Array.from({ length: EXPLORE_PAGES }, (_, i) => <option key={i} value={i}>{i}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="đến">to</Translated>
          <select value={to} onChange={(e) => setTo(+e.target.value)} aria-label={trText('Last page', 'Trang cuối')}>
            {Array.from({ length: EXPLORE_PAGES }, (_, i) => <option key={i} value={i}>{i}</option>)}
          </select>
        </label>
        <span className="row" style={{ gap: '0.4rem' }} role="group" aria-label={trText('Add to the script', 'Thêm vào kịch bản')}>
          {ADD.map((a) => (
            <button key={a.kind} className="quiet" onClick={() => add(a.kind)} disabled={script.length >= MAX_SCRIPT}>+ {a.label}</button>
          ))}
          <button className="quiet" onClick={() => append({ kind: 'sync', from: 0, to: 0 })} disabled={script.length >= MAX_SCRIPT}>
            <Translated vi="+ đồng bộ hóa">+ synchronise</Translated>
          </button>
          <button className="quiet" onClick={() => { setPresetId('own'); setScript([]); setN(0); }} disabled={!script.length}>
            <Translated vi="Xóa hết">Clear</Translated>
          </button>
        </span>
      </div>

      <ol aria-label={trText('Script', 'Kịch bản')} style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem 0.4rem', listStyle: 'none', padding: 0, margin: '0 0 0.8rem' }}>
        {script.length === 0 ? (
          <li className="muted">
            <Translated vi="Kịch bản đang trống. Hãy thêm một thao tác.">The script is empty. Add an operation.</Translated>
          </li>
        ) : null}
        {script.map((s, i) => (
          <li key={i}>
            <button className="quiet" aria-pressed={i === at - 1} onClick={() => setN(i + 1)} data-op={i + 1}
              style={{ padding: '0.15rem 0.45rem', fontSize: '0.76rem', opacity: i < at ? 1 : 0.6 }}>
              {i + 1}.{' '}
              {trText(
                scriptLabel(s),
                s.kind === 'sync'
                  ? 'đồng bộ hóa'
                  : `${{ 'cpu-write': 'CPU ghi', 'cpu-read': 'CPU đọc', 'kernel-read': 'kernel đọc', 'kernel-write': 'kernel ghi', sync: 'đồng bộ hóa' }[s.kind]} ${s.from === s.to ? `trang ${s.from}` : `các trang ${s.from}–${s.to}`}`,
              )}
            </button>
          </li>
        ))}
      </ol>

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setN(0)} disabled={at === 0}>
          <Translated vi="Bắt đầu lại">Restart</Translated>
        </button>
        <button onClick={() => setN(at - 1)} disabled={at === 0}>
          <Translated vi="Lùi">Back</Translated>
        </button>
        <button className="primary" onClick={() => setN(at + 1)} disabled={at >= total}>
          <Translated vi="Bước">Step</Translated>
        </button>
        <button onClick={() => setN(total)} disabled={at >= total}>
          <Translated vi="Chạy đến cuối">Run to the end</Translated>
        </button>
        <span className="muted" aria-live="polite">
          <Translated vi={<>bước {at} trên {total}</>}>
            <>step {at} of {total}</>
          </Translated>
        </span>
        <label className="field" style={{ marginLeft: 'auto' }}>
          <Translated vi="chi phí mỗi trang được chuyển">cost per page moved</Translated>
          <input type="number" min={0} step="any" value={perMove} onChange={(e) => setPerMove(Math.max(0, Number(e.target.value) || 0))} aria-label={trText('cost per page moved', 'chi phí mỗi trang được chuyển')} />
        </label>
        <label className="field">
          <Translated vi="chi phí mỗi lần truy cập trang từ xa">cost per remote page access</Translated>
          <input type="number" min={0} step="any" value={perRemote} onChange={(e) => setPerRemote(Math.max(0, Number(e.target.value) || 0))} aria-label={trText('cost per remote page access', 'chi phí mỗi lần truy cập trang từ xa')} />
        </label>
      </div>

      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}>
        {at === 0 ? (
          <Translated vi={<>Mười sáu trang managed, vừa được cấp phát: chưa trang nào được đặt chỗ. Bấm Bước.</>}>
            <>Sixteen managed pages, just allocated: none placed yet. Press Step.</>
          </Translated>
        ) : (
          <Translated
            vi={
              <>
                Bước {at}:{' '}
                <b>
                  {script[at - 1].kind === 'sync'
                    ? 'đồng bộ hóa'
                    : `${{ 'cpu-write': 'CPU ghi', 'cpu-read': 'CPU đọc', 'kernel-read': 'kernel đọc', 'kernel-write': 'kernel ghi', sync: 'đồng bộ hóa' }[script[at - 1].kind]} ${script[at - 1].from === script[at - 1].to ? `trang ${script[at - 1].from}` : `các trang ${script[at - 1].from}–${script[at - 1].to}`}`}
                </b>
                .
              </>
            }
          >
            <>Step {at}: <b>{scriptLabel(script[at - 1])}</b>.</>
          </Translated>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(19rem, 1fr))' }}>
        {EXPLORE_POLICIES.map((p, k) => {
          const st = runs[p].steps[at];
          const c: Counters = st.counters;
          const errs = runs[p].errors.filter((e) => e.step <= at);
          const crashed = st.crashed;
          return (
            <div key={p} className="panel" data-policy={p} data-cost={costs[k]} data-moved={moved(c)}>
              <div className="row" style={{ justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                <b>
                  {trText(
                    POLICY_NAME[p],
                    { bulk: 'chuyển hàng loạt lúc khởi chạy', migrate: 'lỗi trang rồi di chuyển', map: 'ánh xạ khi truy cập', replicate: 'read-mostly', one: 'một bộ nhớ' }[p],
                  )}
                </b>
                {at > 0 && costs[k] === best && !crashed ? <span className="chip ok" data-best="">{trText('cheapest so far', 'rẻ nhất đến lúc này')}</span> : null}
                {crashed ? <span className="chip fail">{trText('segmentation fault', 'segmentation fault')}</span> : null}
              </div>
              <div className="muted" style={{ fontSize: '0.74rem', marginBottom: '0.4rem' }}>{POLICY_HOW[p]}</div>
              <Strips view={st.view} events={st.events} policy={p} />
              <div style={{ margin: '0.35rem 0', fontSize: '0.78rem' }} data-step-summary="">
                {at === 0 ? '' : st.skipped ? trText('did not run: the program had stopped', 'không chạy: chương trình đã dừng') : trText(`this step: ${stepSummary(st.events)}`, `bước này: ${stepSummary(st.events)}`)}
              </div>
              {errs.map((e, i) => (
                <div key={i} className="t-fail" style={{ fontSize: '0.76rem', marginBottom: '0.3rem' }} data-error={e.kind}>
                  <Translated vi={<>bước {e.step}: {e.message}.</>}>
                    <>step {e.step}: {e.message}.</>
                  </Translated>
                </div>
              ))}
              <table
                className="threads"
                aria-label={trText(
                  `Totals under ${POLICY_NAME[p]}`,
                  `Tổng theo chính sách ${{ bulk: 'chuyển hàng loạt lúc khởi chạy', migrate: 'lỗi trang rồi di chuyển', map: 'ánh xạ khi truy cập', replicate: 'read-mostly', one: 'một bộ nhớ' }[p]}`,
                )}
              >
                <tbody>
                  <tr>
                    <td>
                      <Translated vi="trang lên GPU">pages to GPU</Translated>
                    </td>
                    <td data-total="h2d">{c.h2d}</td>
                    <td>
                      <Translated vi="về CPU">to CPU</Translated>
                    </td>
                    <td data-total="d2h">{c.d2h}</td>
                  </tr>
                  <tr>
                    <td>
                      <Translated vi="lỗi trang GPU">GPU faults</Translated>
                    </td>
                    <td data-total="gpuFaults">{c.gpuFaults}</td>
                    <td>
                      <Translated vi="lỗi trang CPU">CPU faults</Translated>
                    </td>
                    <td data-total="cpuFaults">{c.cpuFaults}</td>
                  </tr>
                  <tr>
                    <td>
                      <Translated vi="truy cập từ xa">remote accesses</Translated>
                    </td>
                    <td data-total="remote">{c.remote}</td>
                    <td>
                      <Translated vi="lần vô hiệu hóa">invalidations</Translated>
                    </td>
                    <td data-total="invalidations">{c.invalidations}</td>
                  </tr>
                  <tr>
                    <td>
                      <b>
                        <Translated vi="chi phí">cost</Translated>
                      </b>
                    </td>
                    <td data-total="cost"><b>{show(costs[k])}</b></td>
                    <td colSpan={2} className="muted">{moved(c)} × {perMove} + {c.remote} × {perRemote}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          );
        })}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- oversubscription
const CAP = 100; // device memory in the model, in pages
const FRACS = Array.from({ length: 151 }, (_, i) => +(0.5 + i * 0.01).toFixed(2));
const SWEEP_LABEL: Record<SweepPolicy, string> = {
  get lru() {
    return trText('fault and migrate, least-recently-used eviction', 'lỗi trang rồi di chuyển, loại bỏ trang ít dùng gần đây nhất (LRU)');
  },
  get prefix() {
    return trText('keep a prefix resident, map the rest from host memory', 'giữ một phần đầu thường trú, ánh xạ phần còn lại từ bộ nhớ host');
  },
};

function useDeviceMemory() {
  const { reports } = useReports();
  const gpus: { backend: string; name: string; bytes: number }[] = [];
  let metal: { name: string; bytes: number } | null = null;
  for (const [b, r] of Object.entries(reports)) {
    const d = r.device ?? {};
    const g = Number(d.totalGlobalMem);
    if (b !== 'emu' && b !== 'race' && g > 0) gpus.push({ backend: b, name: String(d.name ?? b), bytes: g });
    const w = Number(d.recommendedMaxWorkingSetSize);
    if (b === 'metal' && w > 0) metal = { name: String(d.name ?? 'Metal device'), bytes: w }; // i18n-ignore: fallback device name from a report
  }
  return { gpus, metal };
}
const gb = (bytes: number) => (bytes / 1e9).toLocaleString('en-US', { maximumFractionDigits: 1 });

function Overflow() {
  const [frac, setFrac] = useState(0.8);
  const [policy, setPolicy] = useState<SweepPolicy>('lru');
  const { gpus, metal } = useDeviceMemory();
  const curves = useMemo(
    () => ({
      lru: FRACS.map((f) => warmTraffic(Math.round(f * CAP), CAP, 'lru').h2d / CAP),
      prefix: FRACS.map((f) => warmTraffic(Math.round(f * CAP), CAP, 'prefix').remote / CAP),
    }),
    [],
  );
  const W = Math.round(frac * CAP);
  const tr = warmTraffic(W, CAP, policy);
  const crossing = policy === 'lru' ? tr.h2d : tr.remote;
  // Plot geometry.
  const PW = 560, PH = 250, pad = { l: 52, r: 16, t: 14, b: 40 };
  const X = (f: number) => pad.l + ((f - 0.5) / 1.5) * (PW - pad.l - pad.r);
  const Y = (v: number) => PH - pad.b - (v / 2) * (PH - pad.t - pad.b);
  const path = (vs: number[]) => FRACS.map((f, i) => `${X(f).toFixed(1)},${Y(vs[i]).toFixed(1)}`).join(' ');
  return (
    <Bench
      title={trText('When the model does not fit', 'Khi mô hình không vừa')}
      task={trText(
        'Drag the weights past 100% of device memory, once with each placement.',
        'Kéo kích thước trọng số vượt quá 100% bộ nhớ device, mỗi cách đặt chỗ một lần.',
      )}
      foot={
        <Translated
          vi={
            <>
              Bộ nhớ device của mô hình là {CAP} trang. Mỗi token đọc mọi trang trọng số một lần, theo cùng một thứ tự; các con
              số là cho một token sau khi vài token đầu đã làm nóng hệ thống. Màu hổ phách: số trang được chuyển vào mỗi token
              (và cũng ngần ấy trang bị đẩy ra). Màu mực: số trang được đọc qua đường truyền mỗi token.
            </>
          }
        >
          <>
            The model's device memory is {CAP} pages. Each token reads every weight page once, in the same order; the numbers are
            for a token once the first few have warmed things up. Amber: pages migrated in per token (and as many evicted back
            out). Ink: pages read over the link per token.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field" style={{ flex: '1 1 16rem' }}>
          <Translated vi={<>Trọng số: {Math.round(frac * 100)}% bộ nhớ device</>}>
            <>Weights: {Math.round(frac * 100)}% of device memory</>
          </Translated>
          <input type="range" min={0.5} max={2} step={0.01} value={frac} onChange={(e) => setFrac(+e.target.value)} aria-label={trText('Weights as a fraction of device memory', 'Trọng số tính theo tỷ lệ bộ nhớ device')} style={{ width: '100%' }} />
        </label>
        <span className="seg" role="group" aria-label={trText('Placement', 'Cách đặt chỗ')}>
          {(['lru', 'prefix'] as SweepPolicy[]).map((p) => (
            <button key={p} aria-pressed={p === policy} onClick={() => setPolicy(p)}>{SWEEP_LABEL[p]}</button>
          ))}
        </span>
      </div>
      <div className="cols plot">
        <svg viewBox={`0 0 ${PW} ${PH}`} role="img" aria-label={trText('Pages crossing the link per token against weight size', 'Số trang đi qua đường truyền mỗi token theo kích thước trọng số')} style={{ width: '100%', maxWidth: PW, display: 'block' }}>
          <rect x={pad.l} y={pad.t} width={PW - pad.l - pad.r} height={PH - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
          {[0.5, 1, 1.5, 2].map((f) => (
            <g key={f}>
              <line x1={X(f)} x2={X(f)} y1={pad.t} y2={PH - pad.b} stroke="var(--rule-soft)" />
              <text x={X(f)} y={PH - pad.b + 15} fontSize="11" textAnchor="middle">{f * 100}%</text>
            </g>
          ))}
          {[0, 0.5, 1, 1.5, 2].map((v) => (
            <g key={v}>
              <line x1={pad.l} x2={PW - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
              <text x={pad.l - 6} y={Y(v) + 4} fontSize="11" textAnchor="end">{v}×</text>
            </g>
          ))}
          <text x={(PW + pad.l) / 2} y={PH - 6} fontSize="11.5" textAnchor="middle">
            {trText('weights, as a share of device memory', 'trọng số, tính theo tỷ lệ bộ nhớ device')}
          </text>
          <text x={13} y={(PH - pad.b + pad.t) / 2} fontSize="11" textAnchor="middle" transform={`rotate(-90 13 ${(PH - pad.b + pad.t) / 2})`}>
            {trText('crossing per token (× device memory)', 'lượng đi qua mỗi token (× bộ nhớ device)')}
          </text>
          <line x1={X(1)} x2={X(1)} y1={pad.t} y2={PH - pad.b} stroke="var(--ink-3)" strokeDasharray="4 3" />
          <polyline points={path(curves.lru)} fill="none" stroke="var(--global)" strokeWidth={policy === 'lru' ? 2.6 : 1.2} opacity={policy === 'lru' ? 1 : 0.45} data-curve="lru" />
          <polyline points={path(curves.prefix)} fill="none" stroke="var(--ink)" strokeWidth={policy === 'prefix' ? 2.6 : 1.2} opacity={policy === 'prefix' ? 1 : 0.45} data-curve="prefix" />
          <line x1={X(frac)} x2={X(frac)} y1={pad.t} y2={PH - pad.b} stroke="var(--ink)" strokeWidth="1" />
          <circle cx={X(frac)} cy={Y(crossing / CAP)} r={4.5} fill={policy === 'lru' ? 'var(--global)' : 'var(--ink)'} data-marker="" />
        </svg>
        <div className="panel" data-readout="" data-weights={W} data-crossing={crossing} data-policy={policy}>
          <p style={{ marginTop: 0 }}>
            <Translated vi={<><b>{W}</b> trang trọng số, <b>{CAP}</b> trang bộ nhớ device.</>}>
              <><b>{W}</b> weight pages, <b>{CAP}</b> pages of device memory.</>
            </Translated>
          </p>
          {policy === 'lru' ? (
            <p data-line="lru">
              <Translated
                vi={
                  <>
                    Mỗi token, khi đã nóng: <b>{tr.h2d}</b> trang được chuyển vào và <b>{tr.d2h}</b> trang bị đẩy ra, với{' '}
                    {tr.faults} lỗi trang GPU.
                    {tr.h2d === W && W > CAP ? ' Mọi trang, mọi token.' : tr.h2d === 0 ? ' Không có gì: tất cả đều vừa.' : ''}
                  </>
                }
              >
                <>
                  Per token, once warm: <b>{tr.h2d}</b> pages migrated in and <b>{tr.d2h}</b> evicted out, with {tr.faults} GPU faults.
                  {tr.h2d === W && W > CAP ? ' Every page, every token.' : tr.h2d === 0 ? ' Nothing: it all fits.' : ''}
                </>
              </Translated>
            </p>
          ) : (
            <p data-line="prefix">
              <Translated
                vi={
                  <>
                    Mỗi token, khi đã nóng: <b>{tr.remote}</b> trang được đọc qua đường truyền, không có gì được chuyển.
                    {tr.remote ? ` Đó là phần dư, ${Math.round((tr.remote / W) * 100)}% số trọng số.` : ' Không có gì: tất cả đều vừa.'}
                  </>
                }
              >
                <>
                  Per token, once warm: <b>{tr.remote}</b> pages read over the link, nothing migrated.
                  {tr.remote ? ` That is the excess, ${Math.round((tr.remote / W) * 100)}% of the weights.` : ' Nothing: it all fits.'}
                </>
              </Translated>
            </p>
          )}
          {gpus.length ? (
            gpus.map((g) => (
              <p key={g.backend} className="muted" data-device={g.backend}>
                <Translated
                  vi={
                    <>
                      Trên {g.name} của bạn ({g.backend}, {gb(g.bytes)} GB bộ nhớ device), điều này tương ứng khoảng{' '}
                      <b>{gb((crossing / CAP) * g.bytes)} GB</b> đi qua đường truyền cho mỗi token.
                    </>
                  }
                >
                  <>
                    On your {g.name} ({g.backend}, {gb(g.bytes)} GB of device memory) this would be about{' '}
                    <b>{gb((crossing / CAP) * g.bytes)} GB</b> crossing the link for every token.
                  </>
                </Translated>
              </p>
            ))
          ) : (
            <p className="muted" data-device="none">
              <Translated
                vi={
                  <>
                    Chưa có báo cáo device CUDA hay HIP nào, nên chưa có số gigabyte: hãy chạy{' '}
                    <code>python3 labs/pmpp.py run appC/lab1 --bench</code> trên một máy như vậy để thấy tỷ lệ này tính bằng
                    byte của chính device của bạn.
                  </>
                }
              >
                <>
                  No CUDA or HIP device report yet, so no gigabytes: run{' '}
                  <code>python3 labs/pmpp.py run appC/lab1 --bench</code> on such a machine to see this ratio in your own
                  device's bytes.
                </>
              </Translated>
            </p>
          )}
          {metal ? (
            <p className="muted" data-device="metal">
              <Translated
                vi={
                  <>
                    {metal.name} của bạn báo <code>recommendedMaxWorkingSetSize</code> là {gb(metal.bytes)} GB. Apple gọi đó là
                    một giá trị xấp xỉ cho lượng bộ nhớ GPU có thể cấp phát mà không ảnh hưởng hiệu năng, và không nói điều gì
                    xảy ra khi vượt quá. Metal không ghi nhận cơ chế di chuyển nào, nên mô hình này không mô tả nó.
                  </>
                }
              >
                <>
                  Your {metal.name} reports a <code>recommendedMaxWorkingSetSize</code> of {gb(metal.bytes)} GB. Apple calls it an
                  approximation of how much the GPU can allocate without affecting performance, and does not say what happens above
                  it. Metal documents no migration, so this model does not describe it.
                </>
              </Translated>
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

export const questionsSC8: Record<string, Question> = {
  'appC/C.8/q-pages-moved': {
    kind: 'numeric',
    id: 'appC/C.8/q-pages-moved',
    prompt: (
      <Translated
        vi={
          <>
            Sáu mươi tư trang managed, tất cả đều do CPU ghi. Kernel tiếp theo đọc 4 trang trong số đó. Với di chuyển theo lỗi
            trang, số trang chuyển sang GPU ít hơn bao nhiêu so với lần chuyển hàng loạt lúc khởi chạy của chế độ hạn chế?
          </>
        }
      >
        <>
          Sixty-four managed pages, all written by the CPU. The next kernel reads 4 of them. How many fewer pages move to the GPU
          under fault-driven migration than under the limited mode's bulk transfer at the launch?
        </>
      </Translated>
    ),
    answer: 60,
    unit: 'pages',
    explain: (
      <Translated
        vi={
          <p>
            Chuyển hàng loạt di chuyển mọi trang managed mà CPU đã sửa trước khi kernel bắt đầu: 64. Với lỗi trang GPU, không
            có gì di chuyển lúc khởi chạy, và chỉ 4 trang kernel chạm tới gây lỗi trang và di chuyển. 64 − 4 = 60. Hãy nạp kịch
            bản "kernel chạm tới 1 trang trong 16" trong trình khám phá bên dưới để thấy cùng điều đó ở quy mô nhỏ hơn.
          </p>
        }
      >
        <p>
          Bulk transfer moves every managed page the CPU has modified before the kernel starts: 64. With GPU page faults nothing
          moves at the launch, and only the 4 pages the kernel touches fault and migrate. 64 − 4 = 60. Load the preset "kernel
          touches 1 page of 16" in the explorer below to see the same thing at a smaller scale.
        </p>
      </Translated>
    ),
    source: 'book §C.6, p. 627 and §C.8, p. 628; notes/today/appC-facts.md A10, A11',
    vi: { unit: 'trang', source: 'sách §C.6, p. 627 và §C.8, p. 628; notes/today/appC-facts.md A10, A11' },
  },
  'appC/C.8/q-pingpong': {
    kind: 'choice',
    id: 'appC/C.8/q-pingpong',
    prompt: (
      <Translated
        vi={<>CPU và GPU lần lượt cập nhật cùng một trang bộ đếm nhỏ, ở mỗi bước. Cách đặt chỗ nào giữ lưu lượng thấp nhất?</>}
      >
        <>The CPU and the GPU take turns updating the same small counter page, every step. Which placement keeps traffic lowest?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mặc định, lỗi trang rồi di chuyển, đặt trang ở bất cứ nơi nào có truy cập gần nhất.">
            The default, fault and migrate, which puts the page wherever the latest access was.
          </Translated>
        ),
        why: (
          <Translated vi="Trang di chuyển ở mỗi lượt, mỗi chiều một lần: kịch bản ping-pong của trình khám phá chuyển nó 8 lần trong 4 vòng.">
            {"The page moves on every turn, once each way: the explorer's ping-pong preset moves it 8 times in 4 rounds."}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Sao chép read-mostly, để mỗi bên giữ bản sao riêng của trang và đọc nó cục bộ.">
            Read-mostly replication, so that each side keeps its own copy of the page and reads it locally.
          </Translated>
        ),
        why: (
          <Translated vi="Lượt nào cũng ghi, và mỗi lần ghi vô hiệu hóa bản sao của bên kia, bản sao đó sau đó phải quay về.">
            {"Every turn writes, and every write invalidates the other side's copy, which then has to come back."}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Giữ nó trong một bộ nhớ và ánh xạ nó cho bộ xử lý kia (SetPreferredLocation cộng SetAccessedBy).">
            Keep it in one memory and map it for the other processor (SetPreferredLocation plus SetAccessedBy).
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Prefetch nó sang GPU trước mỗi lần khởi chạy, và về CPU trước mỗi lượt của host, để không truy cập nào gây lỗi trang.">
            Prefetch it to the GPU before every launch, and back to the CPU before each host turn, so no access faults.
          </Translated>
        ),
        why: (
          <Translated vi="Đó vẫn là cùng một lần di chuyển ở mỗi lượt, chỉ bắt đầu sớm hơn. Nó không loại bỏ các chuyến đi.">
            That is the same migration on every turn, started earlier. It does not remove the trips.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một trang mà cả hai bên lần lượt chạm tới sẽ di chuyển ở mỗi lượt. Ghim nó vào một bộ nhớ và ánh xạ nó cho bộ xử
            lý kia, thì không có gì di chuyển: thay vào đó mỗi lần GPU truy cập đi qua đường truyền. Sách viết: ánh xạ "đôi khi
            có thể nhanh hơn di chuyển". NVIDIA ghi nhận tổ hợp này: ghim các vùng vào bộ nhớ CPU "và thiết lập một ánh xạ trực
            tiếp từ GPU" bằng <code>cudaMemAdviseSetPreferredLocation</code> và <code>cudaMemAdviseSetAccessedBy</code>.
          </p>
        }
      >
        <p>
          A page that both sides touch in turn migrates on every turn. Pin it to one memory and map it for the other processor,
          and nothing migrates: each GPU access goes over the link instead. The book: mapping "can sometimes be faster than
          migration". NVIDIA documents the combination: pin regions to CPU memory "and establish a direct mapping from the GPU" with{' '}
          <code>cudaMemAdviseSetPreferredLocation</code> and <code>cudaMemAdviseSetAccessedBy</code>.
        </p>
      </Translated>
    ),
    source: 'book §C.8, p. 628; notes/today/appC-facts.md A13; the explorer\'s model',
    vi: { source: 'sách §C.8, p. 628; notes/today/appC-facts.md A13; mô hình của trình khám phá' },
  },
  'appC/C.8/q-cliff': {
    kind: 'choice',
    id: 'appC/C.8/q-cliff',
    prompt: (
      <Translated
        vi={
          <>
            Trọng số bằng 105% bộ nhớ device. Decode đọc toàn bộ trọng số theo cùng một thứ tự ở mỗi token, và chính sách loại
            bỏ là ít dùng gần đây nhất. Khi đã nóng, khoảng bao nhiêu phần các trang trọng số di chuyển mỗi token?
          </>
        }
      >
        <>
          Weights are 105% of device memory. Decoding reads all the weights in the same order every token, and eviction is least
          recently used. About what fraction of the weight pages migrate per token, once warm?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Khoảng 5%: chỉ phần không vừa.">About 5%: just the part that does not fit.</Translated>,
        why: (
          <Translated vi="Đó là điều mà giữ một phần đầu thường trú mang lại, qua đường truyền. Loại bỏ theo ít dùng gần đây nhất còn tệ hơn.">
            That is what keeping a prefix resident gives you, over the link. Least-recently-used eviction does worse.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Khoảng 50%, vì một nửa mỗi lượt quét vẫn thấy các trang của nó đang thường trú.">
            About 50%, since half of each sweep still finds its pages resident.
          </Translated>
        ),
        why: (
          <Translated vi="Không có tỷ lệ trúng một phần nào trong một lượt quét vòng lớn hơn bộ nhớ dưới quy tắc loại bỏ này.">
            There is no partial hit rate in a cyclic sweep larger than memory under this eviction rule.
          </Translated>
        ),
      },
      { label: <Translated vi="Tất cả.">All of them.</Translated>, correct: true },
      {
        label: <Translated vi="Không trang nào: khi đã nóng, các trang ở yên.">None: once warm, the pages stay.</Translated>,
        why: <Translated vi="Chỉ khi tất cả đều vừa.">Only if they all fit.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Khi bộ nhớ đã đầy, trang ít dùng gần đây nhất là trang được đọc lâu nhất trước đó, và trong một lượt quét vòng đó
              lại là trang cần sớm nhất. Trang 100 đẩy trang 0 ra, trang đầu tiên mà token tiếp theo đọc; đưa trang 0 trở lại
              đẩy trang 5 ra, trang mà token đó cần sau năm trang nữa; cứ thế. Mọi truy cập đều trượt. Thay vào đó hãy giữ một
              phần đầu thường trú và ánh xạ phần còn lại, khi đó chỉ phần dư, khoảng 5% trọng số, đi qua đường truyền mỗi token.
            </p>
            <p>
              Đây là một tính chất của loại bỏ theo ít dùng gần đây nhất trên một lượt quét vòng, không phải một sự thật được
              ghi nhận về bất kỳ driver nào: các nguồn của chúng tôi không nói driver của NVIDIA chọn trang nào để loại bỏ. Hãy
              xem vách đứng này là một dự đoán cần kiểm chứng.
            </p>
          </>
        }
      >
        <>
          <p>
            Once memory is full, the least recently used page is the one read longest ago, and in a cyclic sweep that is the
            one needed soonest. Page 100 evicts page 0, the first page the next token reads; bringing page 0 back evicts page 5,
            which that token needs five pages later; and so on. Every access misses. Keep a prefix resident and
            map the rest instead, and only the excess, about 5% of the weights, crosses the link per token.
          </p>
          <p>
            This is a property of least-recently-used eviction on a cyclic sweep, not a documented fact about any driver: our
            sources do not say how NVIDIA's driver chooses what to evict. Treat the cliff as a prediction to test.
          </p>
        </>
      </Translated>
    ),
    source: 'the explorer\'s model (src/sim/managedmem.ts, decodeSweep); eviction policy not documented (notes/today/appC-facts.md A11, A13)',
    vi: {
      source:
        'mô hình của trình khám phá (src/sim/managedmem.ts, decodeSweep); chính sách loại bỏ không được ghi nhận (notes/today/appC-facts.md A11, A13)',
    },
  },
  'appC/C.8/q-hints': {
    kind: 'choice',
    id: 'appC/C.8/q-hints',
    prompt: (
      <Translated
        vi={
          <>
            Một vòng lặp decode giữ logits của nó trong bộ nhớ managed: một kernel ghi chúng, rồi mã host lấy mẫu token tiếp
            theo. Với một lời gọi <code>cudaMemAdvise(..., cudaMemAdviseSetPreferredLocation, ...)</code> trên logits, mọi đầu ra
            đều khớp với tham chiếu. Xóa đúng lời gọi đó và một số token ra khác đi. Nguyên nhân khả dĩ nhất là gì?
          </>
        }
      >
        <>
          A decode loop keeps its logits in managed memory: a kernel writes them, then host code samples the next token.
          With a <code>cudaMemAdvise(..., cudaMemAdviseSetPreferredLocation, ...)</code> call on the logits, every output
          matches the reference. Delete that one call and some tokens come out different. What is the most likely cause?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Thiếu một lần đồng bộ: host đọc logits mà kernel có thể vẫn đang ghi, và gợi ý kia chỉ làm lệch thời điểm.">
            {'A missing synchronise: the host reads logits the kernel may still be writing, and the hint only shifted the timing.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Gợi ý giữ các trang nhất quán: không có vị trí ưu tiên, driver có thể di chuyển một bản sao cũ.">
            The hint kept the pages coherent: without a preferred location the driver can migrate a stale copy.
          </Translated>
        ),
        why: (
          <Translated vi="Lời khuyên thay đổi nơi trang nằm và chi phí di chuyển chúng, không bao giờ thay đổi giá trị một lần đọc trả về.">
            Advice changes where pages live and what moving them costs, never what a load returns.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Lời khuyên có tính ràng buộc trên một số hệ thống, nên bỏ nó đi sẽ thay đổi ngữ nghĩa ở đó.">
            Advice is binding on some systems, so removing it changes the semantics there.
          </Translated>
        ),
        why: (
          <Translated vi="Lời khuyên là gợi ý ở mọi nơi nó được chấp nhận. NVIDIA nói gợi ý chỉ ảnh hưởng đến hiệu năng.">
            Advice is a hint wherever it is accepted. NVIDIA says hints only affect performance.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Lấy mẫu là ngẫu nhiên, nên token khác đi là điều được dự kiến.">
            Sampling is random, so different tokens are expected.
          </Translated>
        ),
        why: (
          <Translated vi="Vậy thì bản có gợi ý cũng sẽ khác. Chỉ bản không có gợi ý mới khác, nên có thứ gì đó trong chương trình phụ thuộc vào thời điểm.">
            {'Then the version with the hint would differ too. Only the version without it does, so something in the program depends on timing.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            NVIDIA: "Gợi ý hiệu năng không bao giờ ảnh hưởng đến tính đúng đắn của ứng dụng. Gợi ý hiệu năng chỉ ảnh hưởng đến
            hiệu năng." Vì vậy khi bỏ một gợi ý mà đầu ra thay đổi, gợi ý đó đang che giấu một lỗi, và trong một vòng lặp như
            thế này lỗi thường gặp là host đọc trước <code>cudaDeviceSynchronize</code> (hoặc trước khi chờ stream của kernel).
            Bốn chính sách của trình khám phá luôn thống nhất về dữ liệu vì cùng lý do: chúng chỉ khác nhau ở các bộ đếm. Quy
            tắc riêng của sách từ mục C.6 vẫn đúng: sắp đặt thứ tự các truy cập đồng thời là việc của bạn.
          </p>
        }
      >
        <p>
          NVIDIA: "Performance hints never impact the correctness of an application. Performance hints only affect performance."
          So when removing a hint changes the output, the hint was hiding a bug, and in a loop like this the usual one is a
          host read before <code>cudaDeviceSynchronize</code> (or a wait on the kernel's stream). The explorer's four policies
          always agree on the data for the same reason: they differ only in the counters. The book's own rule from
          section C.6 still holds: ordering concurrent access is your job.
        </p>
      </Translated>
    ),
    source: 'notes/today/appC-facts.md A13; book §C.6, p. 626 for synchronising concurrent access',
    vi: { source: 'notes/today/appC-facts.md A13; sách §C.6, p. 626 về việc đồng bộ hóa truy cập đồng thời' },
  },
  'appC/C.8/q-signature': {
    kind: 'choice',
    id: 'appC/C.8/q-signature',
    prompt: (
      <Translated
        vi={
          <>
            Vòng lặp decode của bạn prefetch các trang KV managed bằng <code>cudaMemPrefetchAsync</code>. Nó phải build từ một
            mã nguồn cho CUDA 13 và cho ROCm 10.2, nơi một header ánh xạ tên bằng các dòng như{' '}
            <code>#define cudaMemPrefetchAsync hipMemPrefetchAsync</code>. Bạn viết lệnh prefetch thế nào?
          </>
        }
      >
        <>
          Your decode loop prefetches its managed KV pages with <code>cudaMemPrefetchAsync</code>. It must build from one
          source for CUDA 13 and for ROCm 10.2, where a header maps names with lines like{' '}
          <code>#define cudaMemPrefetchAsync hipMemPrefetchAsync</code>. How do you write the prefetch?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Để việc đổi tên lo: viết lời gọi CUDA 13 một lần, và header sẽ biến nó thành lời gọi HIP.">
            Leave it to the rename: write the CUDA 13 call once, and the header turns it into the HIP call.
          </Translated>
        ),
        why: (
          <Translated vi="Tên thì ánh xạ được; chữ ký thì không. Với header mặc định của ROCm 10.2, hipMemPrefetchAsync nhận một device kiểu int, không phải một location và flags.">
            {"The names map; the signatures do not. With ROCm 10.2's default headers hipMemPrefetchAsync takes an int device, not a location and flags."}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Viết dạng CUDA 12 với một device kiểu int, thứ mà cả CUDA 13 và ROCm 10.2 đều chấp nhận.">
            Write the CUDA 12 form with an int device, which both CUDA 13 and ROCm 10.2 accept.
          </Translated>
        ),
        why: (
          <Translated vi="CUDA 13.0 đã bỏ các dạng int: dưới tên gốc, nó nhận một cudaMemLocation và flags. Bản build CUDA thất bại.">
            CUDA 13.0 removed the int forms: under the plain name it takes a cudaMemLocation and flags. The CUDA build fails.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Rẽ nhánh bằng #if: lời gọi CUDA 13 với một cudaMemLocation trên CUDA; trên HIP dùng hipMemPrefetchAsync với device kiểu int, hoặc hipMemPrefetchAsync_v2.">
            {'Branch with #if: the CUDA 13 call with a cudaMemLocation on CUDA; on HIP the int-device hipMemPrefetchAsync, or hipMemPrefetchAsync_v2.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Bỏ lệnh prefetch khỏi bản build HIP; HIP không có lời gọi prefetch, chỉ có lời khuyên.">
            Leave the prefetch out of the HIP build; HIP has no prefetch call, only advice.
          </Translated>
        ),
        why: (
          <Translated vi="HIP có lời gọi đó. Nó nhận các đối số của CUDA 12 dưới tên gốc.">
            HIP has the call. It takes the CUDA 12 arguments under the plain name.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            CUDA 13.0 đổi <code>cudaMemPrefetchAsync</code> và <code>cudaMemAdvise</code> sang nhận một{' '}
            <code>cudaMemLocation</code>, và cho prefetch thêm đối số <code>flags</code>; các dạng <code>int</code> đã không còn.
            HIP giữ hình dạng CUDA 12 dưới các tên gốc, và các dạng location của nó là các biến thể <code>_v2</code>. HIPIFY
            đánh dấu các dạng CUDA 13 là không được hỗ trợ. Vì vậy hãy rẽ nhánh bằng <code>#if</code> theo nền tảng, hoặc ánh
            xạ sang tên <code>_v2</code>. Bỏ lệnh prefetch khỏi một bản build vẫn đúng, vì NVIDIA nói lời gọi này "chỉ nhằm cải
            thiện hiệu năng"; điều sai trong lựa chọn đó là khẳng định HIP không có lời gọi nào. Trích từ các header và tài liệu
            tham chiếu; không có gì ở đây đã được biên dịch.
          </p>
        }
      >
        <p>
          CUDA 13.0 changed <code>cudaMemPrefetchAsync</code> and <code>cudaMemAdvise</code> to take a <code>cudaMemLocation</code>,
          and gave prefetch a <code>flags</code> argument; the <code>int</code> forms are gone. HIP kept the CUDA 12 shape under the
          plain names, and its location forms are the <code>_v2</code> variants. HIPIFY marks the CUDA 13 forms unsupported. So
          branch with <code>#if</code> on the platform, or map to the <code>_v2</code> name. Leaving the prefetch out of one
          build would still be correct, since NVIDIA says the call "only serves to improve performance"; what is wrong in that
          option is the claim that HIP has none. Quoted from the headers and references; nothing here was compiled.
        </p>
      </Translated>
    ),
    source: 'notes/today/appC-facts.md A14, B6',
    vi: { source: 'notes/today/appC-facts.md A14, B6' },
  },
};

export function SC8() {
  return (
    <>
      <Translated
        vi={
          <p>
            Từ Pascal trở đi, GPU có thể nhận một lỗi trang (page fault): một truy cập vào vùng nhớ mà nó không có ánh xạ hợp
            lệ sẽ dừng lại, hệ thống sửa ánh xạ, và truy cập chạy lại. Sách gọi đây là tính năng làm cho Unified Memory liền
            mạch. Tác dụng đầu tiên của nó là gỡ bỏ quy tắc tệ nhất của chế độ hạn chế. Không còn gì phải đẩy sang GPU trước
            một lần khởi chạy nữa, vì bất cứ thứ gì kernel cần đều có thể được nạp khi nó được chạm tới.
          </p>
        }
      >
        <p>
          Starting with Pascal, a GPU can take a page fault: an access to memory it has no valid mapping for stops, the system
          fixes the mapping, and the access runs again. The book calls this the feature that makes Unified Memory seamless. Its
          first effect is to remove the limited mode's worst rule. Nothing has to be flushed to the GPU before a launch any
          more, because anything the kernel needs can be fetched when it is touched.
        </p>
      </Translated>
      <Source section="C.8" pages="628" />
      <Translated
        vi={
          <p>
            Tính nhất quán hoạt động bằng vô hiệu hóa. Khi CPU hoặc GPU sửa một trang managed, nó có thể vô hiệu hóa bản sao của
            bên kia, bằng các cơ chế ánh xạ và bảo vệ trang. Lúc khởi chạy, runtime không còn cập nhật mọi bản sao trên GPU nữa.
            Nếu kernel chạm tới một trang có bản sao GPU đã bị CPU vô hiệu hóa, GPU gặp lỗi trang, trang được cập nhật, và việc
            thực thi tiếp tục.
          </p>
        }
      >
        <p>
          Coherence works by invalidation. When the CPU or the GPU modifies a managed page, it may invalidate the other side's
          copy, using page mapping and protection mechanisms. At a launch the runtime no longer
          brings every GPU copy up to date. If the kernel touches a page whose GPU copy the CPU invalidated, the GPU faults, the
          page is brought up to date, and execution resumes.
        </p>
      </Translated>
      <Source section="C.8" pages="628" />
      <Translated
        vi={
          <p>
            Một trang hoàn toàn không có trong bộ nhớ GPU được xử lý theo cùng cách: nó gây lỗi trang và di chuyển. Hoặc, nếu
            nó chỉ được chạm tới thỉnh thoảng, nó có thể được ánh xạ vào không gian địa chỉ của GPU và đọc qua kênh kết nối, vì
            "ánh xạ khi truy cập đôi khi có thể nhanh hơn di chuyển". Điều đó tổng quát hóa truy cập zero-copy của{' '}
            <a href={href('appC', 'C.2')}>C.2</a> từ một loại bộ nhớ host được ghim đặc biệt sang toàn bộ bộ nhớ host và bộ nhớ
            của các GPU khác, trên toàn hệ thống. Nó cũng hoạt động theo chiều ngược lại: một hàm CPU dereference một trang giờ
            đang nằm trong bộ nhớ GPU vẫn được phục vụ, có thể chậm hơn, điều cho phép bạn gọi các thư viện CPU chưa ai chuyển
            mã. Và vì hai bên dùng cùng giá trị con trỏ, GPU có thể đi theo một cấu trúc liên kết do CPU dựng, và các ứng dụng
            giữ hàng trăm gigabyte "trong bộ nhớ", như CAD, có thể đưa chúng vào tầm với của GPU.
          </p>
        }
      >
        <p>
          A page that is not in GPU memory at all is handled the same way: it faults and migrates. Or, if it will be touched only
          occasionally, it can be mapped into the GPU's address space and read over the interconnect, since "mapping on access can
          sometimes be faster than migration". That generalises the zero-copy access of <a href={href('appC', 'C.2')}>C.2</a>{' '}
          from a special kind of pinned host memory to all of host memory and other GPUs' memory, system-wide. It works the
          other way too: a CPU function that dereferences a page now living in GPU memory is still served, perhaps more slowly,
          which lets you call CPU libraries nobody has ported. And because both sides use the same pointer values, a GPU can
          follow a linked structure the CPU built, and applications that keep hundreds of gigabytes "in core", such as CAD, can
          put it in reach of a GPU.
        </p>
      </Translated>
      <Source section="C.8" pages="628–629" />

      <h2>
        <Translated vi="Những gì NVIDIA ghi nhận ngày nay">What NVIDIA documents today</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hướng dẫn của NVIDIA giờ nêu rõ các quy tắc mà sách để ngầm. Bộ nhớ managed thường được đặt trong bộ nhớ của bộ xử
            lý chạm tới nó đầu tiên, di chuyển khi bộ xử lý khác dùng nó, và được chuyển theo đơn vị trang (nhất quán bằng phần
            mềm) hoặc dòng cache (nhất quán bằng phần cứng, như trên Grace Hopper). Cấp phát vượt dung lượng được phép: bạn có
            thể cấp phát nhiều bộ nhớ managed hơn dung lượng GPU có. Và một lỗi trang phần mềm là thứ bạn sẽ thấy trong hoạt
            hình bên dưới: bên sở hữu mất ánh xạ, bên yêu cầu có được ánh xạ, và trang di chuyển, "một thao tác tốn kém, và
            lượng công việc tỷ lệ với kích thước trang".
          </p>
        }
      >
        <p>
          NVIDIA's guide now states the rules the book leaves implicit. Managed memory is usually placed in the memory of the
          processor that first touches it, migrated when another processor uses it, and moved at the granularity of pages
          (software coherence) or cache lines (hardware coherence, as on Grace Hopper). Oversubscription is allowed: you may
          allocate more managed memory than the GPU has. And a software page fault is what you will see in the animation below:
          the owner loses its mapping, the requester gains one, and the page moves, "an expensive operation, and the amount of
          work is proportional to the page size".
        </p>
      </Translated>
      <NoteGroup>
        <Today checked="2026-11-20 (notes/today/appC-facts.md, A11, A12)" /* i18n-ignore: date and citation only */>
          <Translated
            vi={
              <p>
                Kích thước trang, theo NVIDIA nêu hiện nay và "có thể thay đổi": 4 KiB trên CPU x86_64, 4 đến 64 KiB trên CPU
                Arm, và GPU "ưu tiên trang vật lý 2MiB hoặc lớn hơn". "TLB miss trên GPU nói chung tốn kém hơn đáng kể."
              </p>
            }
          >
            <p>
              Page sizes, as NVIDIA states them now and "subject to change": 4 KiB on x86_64 CPUs, 4 to 64 KiB on Arm CPUs, and
              GPUs "prefer 2MiB physical pages or larger". "TLB misses are generally significantly more expensive on the GPU."
            </p>
          </Translated>
          <Translated
            vi={
              <p>
                Phạm vi: cơ chế lỗi trang kiểu Pascal áp dụng cho bộ nhớ được cấp phát là managed. Bộ nhớ <code>malloc</code>{' '}
                hay <code>mmap</code> thông thường chỉ GPU truy cập được khi có HMM (Linux 6.1.24 trở lên, compute capability
                7.5 trở lên, driver 535 trở lên với open kernel modules) hoặc ATS (Grace Hopper, Grace Blackwell). Bộ nhớ{' '}
                <code>cudaMalloc</code> vẫn nằm ngoài tầm với của CPU ở mọi nơi.
              </p>
            }
          >
            <p>
              Scope: Pascal-style faulting covers memory allocated as managed. Plain <code>malloc</code> or <code>mmap</code> memory
              is GPU-accessible only with HMM (Linux 6.1.24 or later, compute capability 7.5 or later, driver 535 or later with
              open kernel modules) or ATS (Grace Hopper, Grace Blackwell). <code>cudaMalloc</code> memory stays out of the CPU's
              reach everywhere.
            </p>
          </Translated>
        </Today>
        <Simplification>
          <Translated
            vi={
              <p>
                Hoạt hình theo dõi một trang dưới cơ chế lỗi trang rồi di chuyển với nhất quán bằng phần mềm. Không đo thời
                gian, một kích thước trang, và không có hệ thống nhất quán bằng phần cứng.
              </p>
            }
          >
            <p>
              The animation follows one page under fault and migrate with software coherence. No timing, one page size, and
              no hardware-coherent system.
            </p>
          </Translated>
        </Simplification>
      </NoteGroup>
      <PageFaultScene />

      <h2>
        <Translated vi="Di chuyển hay ánh xạ: ai quyết định?">Migrate or map: who decides?</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách nói một trang "có thể được ánh xạ" thay vì di chuyển, và không nói ai chọn. NVIDIA thì có: mặc định driver
            Unified Memory quyết định, theo heuristic, và "mặc định cudaMallocManaged di chuyển theo cơ chế lỗi trang rồi di
            chuyển". Bạn có thể điều khiển nó. <code>cudaMemAdvise</code> với <code>SetPreferredLocation</code> trên host cộng{' '}
            <code>SetAccessedBy</code> cho GPU ghim một vùng vào bộ nhớ CPU và ánh xạ nó cho GPU. <code>SetReadMostly</code>{' '}
            cho cả hai bên giữ một bản sao, "đánh đổi băng thông ghi lấy băng thông đọc". <code>cudaMemPrefetchAsync</code>{' '}
            chuyển một vùng trước khi có ai gặp lỗi trang trên nó. Trên các hệ thống nhất quán bằng phần cứng, một cơ chế thứ
            tư, bộ đếm truy cập, chuyển trang về phía bộ xử lý dùng chúng nhiều nhất.
          </p>
        }
      >
        <p>
          The book says a page "may be mapped" instead of migrated, and does not say who chooses. NVIDIA does: by default the
          Unified Memory driver decides, by heuristics, and "per default cudaMallocManaged migrates based on a fault-and-migrate
          mechanism". You can steer it. <code>cudaMemAdvise</code> with <code>SetPreferredLocation</code> on the host plus{' '}
          <code>SetAccessedBy</code> for the GPU pins a region to CPU memory and maps it for the GPU. <code>SetReadMostly</code>{' '}
          lets both sides hold a copy, "trading off read bandwidth for write bandwidth". <code>cudaMemPrefetchAsync</code> moves
          a range before anyone faults on it. On hardware-coherent systems a fourth mechanism, access counters, moves pages
          toward the processor that uses them most.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/appC-facts.md, A13, A14)" /* i18n-ignore: date and citation only */>
        <Translated
          vi={
            <p>
              Các gợi ý ra đời cùng CUDA 8 và Pascal. Không gợi ý nào cần cho tính đúng đắn: prefetch "không bắt buộc về chức
              năng và chỉ nhằm cải thiện hiệu năng".
            </p>
          }
        >
          <p>
            The hints arrived with CUDA 8 and Pascal. None is needed for correctness: prefetch "is not required for functionality
            and only serves to improve performance".
          </p>
        </Translated>
      </Today>
      <Translated vi={<p>Trước khi khám phá, hãy chốt con số mà hai thời kỳ của sách bất đồng nhiều nhất.</p>}>
        <p>
          Before you explore, settle the number the book's two eras disagree on most.
        </p>
      </Translated>
      <Ask q={questionsSC8['appC/C.8/q-pages-moved']} label={trText('Work it out', 'Hãy tính')} />
      <Translated
        vi={
          <p>
            Trình khám phá chạy một kịch bản gồm các truy cập của CPU và kernel trên mười sáu trang managed dưới bốn chính sách
            cạnh nhau, và giữ một hóa đơn cộng dồn theo mức giá bạn đặt.
          </p>
        }
      >
        <p>
          The explorer runs one script of CPU and kernel accesses on sixteen managed pages under four policies side by side,
          and keeps a running bill at prices you set.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Tính theo trang, không theo byte. Bốn chính sách cố định đại diện cho các heuristic không được ghi nhận của driver,
              và mỗi chính sách áp dụng cho cả mười sáu trang cùng lúc. Chi phí là con số của bạn: sách không đưa ra con số nào.
              Không có prefetch hay chồng lấp. Các hệ thống nhất quán bằng phần cứng, với đơn vị dòng cache và bộ đếm truy cập,
              không được mô hình hóa.
            </p>
          }
        >
          <p>
            Pages, not bytes. Four fixed policies stand in for the driver's undocumented heuristics, and each applies to all
            sixteen pages at once. Costs are your numbers: the book gives none. No prefetching or overlap. Hardware-coherent
            systems, with cache-line granularity and access counters, are not modelled.
          </p>
        </Translated>
      </Simplification>
      <Coherence />
      <Translated
        vi={
          <p>
            Cần tìm gì. Nạp "kernel chạm tới 1 trang trong 16": chuyển hàng loạt di chuyển cả mười sáu trang cho một trang được
            chạm tới. Nạp kịch bản ping-pong: lỗi trang rồi di chuyển chuyển trang bộ đếm ở mỗi lượt, và ánh xạ, thứ không di
            chuyển gì, thắng ở mọi mức giá mà một truy cập từ xa rẻ hơn hai lần di chuyển. Rồi đến vòng lặp decode,{' '}
            {DECODE_TOKENS} token của nó: ánh xạ trả giá cho mọi trang trọng số ở mọi token, còn read-mostly, thứ sao chép
            trọng số một lần, thắng ở mức giá mặc định, trong khi các lần vô hiệu hóa của nó rơi vào trang duy nhất được ghi.
            Chính sách tốt nhất là một tính chất của mẫu truy cập, theo từng vùng, đó là lý do các gợi ý nhận một dải địa chỉ.
          </p>
        }
      >
        <p>
          What to look for. Load "kernel touches 1 page of 16": bulk transfer moves all sixteen for one touched page. Load the
          ping-pong preset: fault and migrate moves the counter page on every turn, and mapping, which moves nothing, wins at any
          prices where a remote access costs less than two migrations. Then the decode loop, {DECODE_TOKENS} tokens of it:
          mapping pays for every weight page on every token, and read-mostly, which copies the weights once, wins at the default
          prices, while its invalidations land on the one page that is written. The best policy is a property of the access
          pattern, region by region, which is why the hints take an address range.
        </p>
      </Translated>
      <Ask q={questionsSC8['appC/C.8/q-pingpong']} />
      <Ask q={questionsSC8['appC/C.8/q-hints']} />

      <h2>
        <Translated vi="Cái bẫy chữ ký hàm">The signature trap</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Nếu bạn dùng các gợi ý này trong một tệp <code>.cu</code> cũng phải build cho HIP, hãy để ý phiên bản. CUDA 13.0 đổi{' '}
            <code>cudaMemPrefetchAsync</code> và <code>cudaMemAdvise</code> sang nhận một <code>cudaMemLocation</code> thay cho
            số hiệu device, và cho prefetch thêm đối số <code>flags</code>. HIP không làm theo dưới cùng các tên đó.
          </p>
        }
      >
        <p>
          If you use these hints in a <code>.cu</code> file that must also build for HIP, mind the version. CUDA 13.0 changed{' '}
          <code>cudaMemPrefetchAsync</code> and <code>cudaMemAdvise</code> to take a <code>cudaMemLocation</code> instead of a
          device number, and gave prefetch a <code>flags</code> argument. HIP did not follow under the same names.
        </p>
      </Translated>
      <table className="data wide" style={{ maxWidth: '58rem' }} aria-label={trText('Prefetch and advise signatures', 'Chữ ký của prefetch và advise')}>
        <thead>
          <tr>
            <th></th>
            <th>CUDA 13.5{/* i18n-ignore: version name */}</th>
            <th>CUDA 12.9{/* i18n-ignore: version name */}</th>
            <th>
              <Translated vi="HIP, header mặc định của ROCm 10.2">HIP, ROCm 10.2 default headers</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>prefetch{/* i18n-ignore: API operation name */}</td>
            <td><code style={{ whiteSpace: 'normal' }}>cudaMemPrefetchAsync(const void *devPtr, size_t count, struct cudaMemLocation location, unsigned int flags, cudaStream_t stream = 0)</code></td>
            <td>
              <Translated
                vi={
                  <>
                    <code style={{ whiteSpace: 'normal' }}>cudaMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream = 0)</code>; dạng location trước đây là <code>cudaMemPrefetchAsync_v2</code>
                  </>
                }
              >
                <>
                  <code style={{ whiteSpace: 'normal' }}>cudaMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream = 0)</code>; the location form was <code>cudaMemPrefetchAsync_v2</code>
                </>
              </Translated>
            </td>
            <td>
              <Translated
                vi={
                  <>
                    <code style={{ whiteSpace: 'normal' }}>hipMemPrefetchAsync(const void* dev_ptr, size_t count, int device, hipStream_t stream)</code>; dạng location là <code>hipMemPrefetchAsync_v2</code>
                  </>
                }
              >
                <>
                  <code style={{ whiteSpace: 'normal' }}>hipMemPrefetchAsync(const void* dev_ptr, size_t count, int device, hipStream_t stream)</code>; the location form is <code>hipMemPrefetchAsync_v2</code>
                </>
              </Translated>
            </td>
          </tr>
          <tr>
            <td>advise{/* i18n-ignore: API operation name */}</td>
            <td><code style={{ whiteSpace: 'normal' }}>cudaMemAdvise(const void *devPtr, size_t count, enum cudaMemoryAdvise advice, struct cudaMemLocation location)</code></td>
            <td>
              <Translated
                vi={
                  <>
                    <code style={{ whiteSpace: 'normal' }}>cudaMemAdvise(..., int device)</code>, cộng thêm <code>cudaMemAdvise_v2</code>
                  </>
                }
              >
                <>
                  <code style={{ whiteSpace: 'normal' }}>cudaMemAdvise(..., int device)</code>, plus <code>cudaMemAdvise_v2</code>
                </>
              </Translated>
            </td>
            <td>
              <Translated
                vi={
                  <>
                    <code style={{ whiteSpace: 'normal' }}>hipMemAdvise(const void* dev_ptr, size_t count, hipMemoryAdvise advice, int device)</code> theo mặc định (<code>HIP_FORCE_API_VERSION</code> dưới 800; mặc định là 717), cộng thêm <code>hipMemAdvise_v2</code>
                  </>
                }
              >
                <>
                  <code style={{ whiteSpace: 'normal' }}>hipMemAdvise(const void* dev_ptr, size_t count, hipMemoryAdvise advice, int device)</code> by default (<code>HIP_FORCE_API_VERSION</code> below 800; the default is 717), plus <code>hipMemAdvise_v2</code>
                </>
              </Translated>
            </td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Đổi tên hoạt động với <code>cudaMallocManaged</code> và <code>cudaMemLocation</code>, những thứ HIPIFY ánh xạ một-một,
            nhưng một <code>#define</code> đơn thuần cho prefetch sẽ biên dịch một lời gọi CUDA 13 với sai tập overload. Hãy rẽ
            nhánh bằng <code>#if</code> theo nền tảng, hoặc ánh xạ sang các tên <code>_v2</code>. Đây là các chữ ký được trích
            dẫn, unverified: không có gì ở đây đã được biên dịch.
          </p>
        }
      >
        <p>
          Renaming works for <code>cudaMallocManaged</code> and <code>cudaMemLocation</code>, which HIPIFY maps one for one, but
          a plain <code>#define</code> of the prefetch compiles a CUDA 13 call against the wrong overload set. Branch with{' '}
          <code>#if</code> on the platform, or map to the <code>_v2</code> names. These are quoted signatures, unverified: nothing here was compiled.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/appC-facts.md, A14, B6)" /* i18n-ignore: date and citation only */>
        <Translated
          vi={
            <p>
              Bảng của HIPIFY đánh dấu các dạng CUDA 13 của cả hai lời gọi là không được hỗ trợ. Bản build HIP của llama.cpp gọi
              dạng <code>int</code> của <code>cudaMemAdvise</code> cho gợi ý coarse-grain của nó.
            </p>
          }
        >
          <p>
            HIPIFY's table marks the CUDA 13 forms of both calls unsupported. llama.cpp's HIP build calls the <code>int</code>{' '}
            form of <code>cudaMemAdvise</code> for its coarse-grain hint.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsSC8['appC/C.8/q-signature']} />

      <h2>
        <Translated vi="Khi mô hình không vừa">When the model does not fit</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Cấp phát vượt dung lượng nghe như lời giải cho một mô hình lớn hơn GPU của bạn: cấp phát nó dạng managed và để các
            trang đến rồi đi. Nhưng decode đọc mọi trọng số một lần mỗi token, theo cùng một thứ tự, và trang nào bị đẩy ra khi
            bộ nhớ đầy sẽ quyết định tất cả.
          </p>
        }
      >
        <p>
          Oversubscription sounds like the answer to a model larger than your GPU: allocate it managed and let pages come and
          go. Decoding, though, reads every weight once per token, in the same order, and which page leaves when memory is full
          decides everything.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Loại bỏ theo ít dùng gần đây nhất là thứ thay thế của chúng tôi. Chính sách loại bỏ của driver không được ghi nhận
              trong các nguồn của chúng tôi, nên vách đứng bên dưới là một tính chất của mô hình cần kiểm chứng trên máy thật,
              không phải một khẳng định về bất kỳ driver nào.
            </p>
          }
        >
          <p>
            Least-recently-used eviction is our stand-in. The driver's eviction policy is not documented in our sources, so the
            cliff below is a property of the model to test on a real machine, not a statement about any driver.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Trình khám phá chạy vòng lặp đó trên một bộ nhớ device 100 trang, với hai cách đặt chỗ: mặc định, lỗi trang rồi di
            chuyển với loại bỏ theo ít dùng gần đây nhất, và một cách được khuyên dùng giữ một phần đầu thường trú và ánh xạ
            phần còn lại. Trước khi kéo thanh trượt: điều gì xảy ra ngay sau 100%?
          </p>
        }
      >
        <p>
          The explorer runs that loop on a device memory of 100 pages, with two placements: the default, fault and migrate with
          least-recently-used eviction, and an advised one that keeps a prefix resident and maps the rest. Before you drag the
          slider: what happens just past 100%?
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Bộ nhớ device chỉ chứa trọng số: không có activation, không có KV cache. Một kích thước trang, không prefetch,
              không chồng lấp truyền dữ liệu với tính toán, và đơn vị đếm là trang, không phải thời gian.
            </p>
          }
        >
          <p>
            Device memory holds weights only: no activations, no KV cache. One page size, no prefetch, no overlap of transfers
            with compute, and the count is pages, not time.
          </p>
        </Translated>
      </Simplification>
      <Ask q={questionsSC8['appC/C.8/q-cliff']} />
      <Overflow />
      <Translated
        vi={
          <p>
            Đó là câu trả lời của mô hình cho câu hỏi vì sao một mô hình hơi quá lớn so với VRAM không nhất thiết chỉ chạy chậm
            hơn một chút: với lỗi trang rồi di chuyển và một quy tắc loại bỏ như thế này, mỗi token có thể phải trả cho toàn bộ
            mô hình ở tốc độ của kênh kết nối. <code>GGML_CUDA_ENABLE_UNIFIED_MEMORY=1</code> của llama.cpp tồn tại để swap sang
            RAM hệ thống "thay vì sập" trên Linux, và đây là sự đánh đổi nó đưa ra: chạy được, với cái giá của đường truyền.
            Trên Windows, driver của NVIDIA có một "Sysmem Fallback Policy" cho phép các vùng cấp phát thông thường tràn sang
            bộ nhớ hệ thống ("shared memory" theo cách gọi của NVIDIA, không phải loại trên chip), "dù ở tốc độ thấp hơn";
            NVIDIA không nêu thứ tương đương nào trên Linux, nên đừng trông chờ. Cách đặt chỗ là thứ quyết định chi phí: giữ
            thường trú phần vừa và ánh xạ phần còn lại, khi đó chỉ phần dư đi qua đường truyền.
          </p>
        }
      >
        <p>
          That is the model's answer to why a model a little too big for VRAM need not run only a little slower: with fault
          and migrate and an eviction rule like this one, every token can pay for the whole model at the speed of the
          interconnect. llama.cpp's <code>GGML_CUDA_ENABLE_UNIFIED_MEMORY=1</code> exists to swap to system RAM "instead of
          crashing" on Linux, and this is the trade it offers: running, at the price of the link. On Windows, NVIDIA's driver
          has a "Sysmem Fallback Policy" that lets ordinary allocations spill into system memory (NVIDIA's "shared memory",
          not the on-chip kind), "albeit at lower speeds";
          NVIDIA states no Linux equivalent, so do not count on one. Placement is what decides the cost: keep what fits
          resident and map the rest, and only the excess crosses the link.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/appC-facts.md, A18, C5, D3, D4, D5)" /* i18n-ignore: date and citation only */>
        <Translated
          vi={
            <p>
              Trên Mac, Metal không ghi nhận cơ chế cấp phát vượt dung lượng nào. <code>recommendedMaxWorkingSetSize</code> là
              lời khuyên, và Apple không nói điều gì xảy ra khi vượt quá. Residency set (macOS 15) làm các vùng cấp phát thường
              trú một cách tường minh. llama.cpp giữ trọng số được ánh xạ bộ nhớ của nó ở trạng thái wired bằng residency set và
              báo <code>recommendedMaxWorkingSetSize</code> làm ngân sách bộ nhớ; MLX cung cấp <code>set_wired_limit</code>.
            </p>
          }
        >
          <p>
            On a Mac, Metal documents no oversubscription mechanism. <code>recommendedMaxWorkingSetSize</code> is advice, and
            Apple does not say what happens above it. Residency sets (macOS 15) make allocations resident explicitly. llama.cpp
            keeps its memory-mapped weights wired with residency sets and reports <code>recommendedMaxWorkingSetSize</code> as its
            memory budget; MLX exposes <code>set_wired_limit</code>.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Trên HIP và Metal">On HIP and Metal</Translated>
      </h2>
      <Translated
        vi={
          <p>
            AMD có cùng các thời kỳ đó, được chọn theo từng GPU và từng tiến trình thay vì theo năm. Với <code>HSA_XNACK=1</code>{' '}
            và một driver có HMM, truy cập của kernel vào một trang managed không thường trú sẽ gây lỗi trang và di chuyển, và
            khi bộ nhớ device đầy, một trang được gửi trả về bộ nhớ host, nên cấp phát vượt dung lượng hoạt động. Một số GPU AMD
            hoàn toàn không nhận lỗi trang được; ở đó, nếu bạn muốn truy cập cục bộ, mọi trang phải được di chuyển trước khi
            kernel được dispatch, tức là chế độ hạn chế của sách trên phần cứng AMD ngày nay. Trên MI300A, nơi CPU và GPU dùng
            chung một HBM, không có gì di chuyển, và XNACK vẫn cần: lỗi trang ở đó thiết lập ánh xạ, không phải di chuyển.
          </p>
        }
      >
        <p>
          AMD has the same eras, chosen per GPU and per process rather than by year. With <code>HSA_XNACK=1</code> and an HMM
          driver, a kernel's access to a non-resident managed page faults and migrates, and when device memory is full a page is
          sent back to host memory, so oversubscription works. Some AMD GPUs cannot take page faults at all; there, if you want
          local accesses, every page has to be migrated before the kernel is dispatched, which is the book's limited mode on AMD
          hardware today. On the MI300A, where CPU and GPU share one HBM, nothing migrates, and XNACK is still needed: faults
          there establish mappings, not moves.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Trên Metal, lỗi trang duy nhất Apple ghi nhận là một lỗi: <code>MTLCommandBufferErrorPageFault</code>, "một lỗi
            trang GPU không thể phục vụ", do không khớp thuộc tính đọc-ghi hoặc truy cập ngoài biên. Việc GPU của Apple có phục
            vụ lỗi trang nào hay không, dưới dạng phân trang theo nhu cầu, không được ghi nhận, nên khóa học này không dạy câu
            trả lời nào.
          </p>
        }
      >
        <p>
          On Metal, the only page fault Apple documents is an error: <code>MTLCommandBufferErrorPageFault</code>, "an
          unserviceable GPU page fault", caused by a read-write attribute mismatch or an out-of-bounds access. Whether Apple GPUs
          service any faults, as demand paging, is not documented, so this course teaches neither answer.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ nói về NVIDIA. AMD lấy từ trang unified memory và bài viết về MI300A của họ (appC-facts B5, B8); Apple lấy
              từ các header của macOS 27.0 SDK (C5).
            </p>
          }
        >
          <p>
            The book covers NVIDIA only. AMD from its unified memory page and MI300A post (appC-facts B5, B8); Apple from the
            macOS 27.0 SDK headers (C5).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Tiếp theo, <a href={href('appC', 'C.9')}>C.9</a> trao cho bạn chính không gian địa chỉ để sắp xếp. Để đo chi phí di
            chuyển trên máy của bạn, lab C.1 trên <a href={href('appC', 'labs')}>trang lab</a> cấp dữ liệu cho một kernel bằng
            sao chép, bộ nhớ ánh xạ và bộ nhớ managed.
          </p>
        }
      >
        <p>
          Next, <a href={href('appC', 'C.9')}>C.9</a> gives you the address space itself to arrange. To measure what migration
          costs on your machine, lab C.1 on the <a href={href('appC', 'labs')}>labs page</a> feeds one kernel with copies, mapped
          memory and managed memory.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Quay lại <a href={href('appC', 'start')}>bản đồ</a>: bước của mục này là bước được đánh dấu C.7–C.8. Bước sau C.9,
            vượt ra ngoài sách, là nơi lỗi trang chạm tới bộ nhớ <code>malloc</code> thông thường.
          </p>
        }
      >
        <p>
          Back to <a href={href('appC', 'start')}>the map</a>: this section's step is the one marked C.7–C.8. The step after
          C.9, beyond the book, is where page faults reach plain <code>malloc</code> memory.
        </p>
      </Translated>
    </>
  );
}
