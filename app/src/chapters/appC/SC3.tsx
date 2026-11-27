import { useState, type MouseEvent } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Bench, Beyond, NoteGroup, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { benchRows, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { ALLOCS, HOP_LABEL, NODE_VALUES, SETTINGS, addresses, chase, hex, type Alloc, type HopVerdict, type Setting } from '../../sim/ptrchase';
import { REGISTERED, SPACE_LABEL, TOP, hex4, inferKind, ranges, type Mode, type Range } from '../../sim/uvacopy';
import { LAB } from './SC2';

// Lesson copy of labs/appC/snippets/c3_chase.cu; SC3.snippets.test.ts fails if it drifts.
const CHASE_CU = `// A linked list that the host built. The kernel can follow it only if every
// node is in memory a kernel may dereference; under UVA (CUDA 4) that means
// cudaHostAlloc or device memory, never plain malloc memory.
struct Node {
    float value;
    Node* next;
};

// One thread walks the list and sums it: pure pointer chasing, no parallelism.
__global__ void sumList(const Node* head, float* sum) {
    float s = 0.0f;
    for (const Node* p = head; p != nullptr; p = p->next) s += p->value;
    *sum = s;
}

void launchSumList(const Node* head, float* sum) {
    sumList<<<1, 1>>>(head, sum);
}`;
const CHASE_TIERS: Record<string, Tier> = { s: 'reg', p: 'reg' };

// ---------------------------------------------------------------- explorer: which way does the copy go?
function rangeFill(r: Range): { fill: string; stroke: string; dash?: string } {
  if (r.registered) return { fill: 'var(--page)', stroke: 'var(--ink)', dash: '4 3' };
  if (r.space === 'host') return { fill: 'var(--bench)', stroke: 'var(--ink-3)' };
  return { fill: 'var(--global-wash)', stroke: 'var(--global)' };
}

function CopyDirection() {
  const [mode, setMode] = useState<Mode>('uva');
  const [reg, setReg] = useState(false);
  const [src, setSrc] = useState(0x3000);
  const [dst, setDst] = useState(0x9000);
  const [placing, setPlacing] = useState<'src' | 'dst'>('src');
  const registered = mode === 'uva' && reg;
  const rs = ranges(mode, registered);
  const r = inferKind(dst, src, mode, registered);
  const W = 640;
  const pad = 14;
  const lanes = mode === 'uva' ? [null] : (['host', 'gpu0', 'gpu1'] as const);
  const laneH = 30;
  const top = 26;
  const bottom = top + lanes.length * (laneH + 8) - 8;
  const H = bottom + 36;
  const X = (a: number) => pad + (a / TOP) * (W - 2 * pad);
  const place = (ev: MouseEvent<SVGSVGElement>) => {
    const box = ev.currentTarget.getBoundingClientRect();
    const x = ((ev.clientX - box.left) / box.width) * W;
    const a = Math.max(0, Math.min(TOP - 0x100, Math.round((((x - pad) / (W - 2 * pad)) * TOP) / 0x100) * 0x100));
    if (placing === 'src') setSrc(a);
    else setDst(a);
  };
  const laneY = (i: number) => top + i * (laneH + 8);
  const marker = (a: number, label: string) => (
    <g key={label}>
      <line x1={X(a)} x2={X(a)} y1={label === 'src' ? top - 6 : top} y2={label === 'src' ? bottom : bottom + 6} stroke="var(--ink)" strokeWidth="2" />
      <text x={Math.min(W - 50, Math.max(50, X(a)))} y={label === 'src' ? top - 10 : bottom + 19} fontSize="13" textAnchor="middle" fontWeight="650">
        {label} {hex4(a)}
      </text>
    </g>
  );
  return (
    <Bench
      title={tr('Which way does the copy go?', 'Lần sao chép đi theo chiều nào?')}
      task={tr(
        'Place src and dst, then switch between one address space and separate ones.',
        'Đặt src và dst, rồi chuyển qua lại giữa một không gian địa chỉ chung và các không gian riêng.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label={tr('Address spaces', 'Không gian địa chỉ')}>
          <button aria-pressed={mode === 'uva'} data-mode="uva" onClick={() => setMode('uva')}>
            <Translated vi="một không gian địa chỉ chung (CUDA 4 trở về sau)">one shared address space (CUDA 4 and later)</Translated>
          </button>
          <button aria-pressed={mode === 'separate'} data-mode="separate" onClick={() => setMode('separate')}>
            <Translated vi="các không gian riêng (trước đó)">separate spaces (before)</Translated>
          </button>
        </span>
        <label className="check ui" style={{ opacity: mode === 'uva' ? 1 : 0.5 }}>
          <input type="checkbox" checked={reg} disabled={mode !== 'uva'} onChange={(e) => setReg(e.target.checked)} />
          <Translated vi="thêm một buffer được đăng ký bằng cudaHostRegister">add a buffer registered with cudaHostRegister</Translated>
        </label>
      </div>
      <div className="row ui" style={{ marginBottom: '0.4rem' }}>
        <span className="muted">
          <Translated vi="Nhấp vào bản đồ để đặt">Clicking the map places</Translated>
        </span>
        <span className="seg" role="group" aria-label={tr('Marker to place', 'Mốc cần đặt')}>
          <button aria-pressed={placing === 'src'} onClick={() => setPlacing('src')}>src{/* i18n-ignore: code identifier */}</button>
          <button aria-pressed={placing === 'dst'} onClick={() => setPlacing('dst')}>dst{/* i18n-ignore: code identifier */}</button>
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Address map', 'Bản đồ địa chỉ')} style={{ width: '100%', maxWidth: W, display: 'block', cursor: 'crosshair' }} onClick={place}>
        {lanes.map((lane, i) => (
          <g key={lane ?? 'one'}>
            <rect x={pad} y={laneY(i)} width={W - 2 * pad} height={laneH} fill="none" stroke="var(--rule)" />
            {rs
              .filter((x) => (lane === null ? true : x.space === lane))
              .map((x) => {
                const st = rangeFill(x);
                return (
                  <g key={x.label}>
                    <rect x={X(x.lo)} y={laneY(i) + (x.registered ? 4 : 0)} width={X(x.hi) - X(x.lo)} height={laneH - (x.registered ? 8 : 0)} fill={st.fill} stroke={st.stroke} strokeDasharray={st.dash} />
                    <text x={(X(x.lo) + X(x.hi)) / 2} y={laneY(i) + laneH / 2 + 5} fontSize={x.registered ? 11 : 13} textAnchor="middle">
                      {x.registered ? (x.registered === 'host-side' ? tr('registered', 'đã đăng ký') : tr('its device address', 'địa chỉ device của nó')) : lane ? tr(SPACE_LABEL[x.space] + "'s space", 'không gian của ' + { host: 'bộ nhớ host', gpu0: 'GPU 0', gpu1: 'GPU 1' }[x.space]) : tr(SPACE_LABEL[x.space], { host: 'bộ nhớ host', gpu0: 'GPU 0', gpu1: 'GPU 1' }[x.space])}
                    </text>
                  </g>
                );
              })}
          </g>
        ))}
        {marker(src, 'src')}
        {marker(dst, 'dst')}
        <text x={pad} y={H - 3} fontSize="11">0x0000</text>
        <text x={W - pad} y={H - 3} fontSize="11" textAnchor="end">0xffff{/* i18n-ignore: hex address */}</text>
      </svg>
      <div className="row" style={{ marginTop: '0.5rem' }}>
        <label className="field ui" style={{ flex: '1 1 14rem' }}>
          src = {hex4(src)}
          <input type="range" min={0} max={TOP - 0x100} step={0x100} value={src} aria-label={tr('src address', 'địa chỉ src')} onChange={(e) => setSrc(+e.target.value)} />
        </label>
        <label className="field ui" style={{ flex: '1 1 14rem' }}>
          dst = {hex4(dst)}
          <input type="range" min={0} max={TOP - 0x100} step={0x100} value={dst} aria-label={tr('dst address', 'địa chỉ dst')} onChange={(e) => setDst(+e.target.value)} />
        </label>
      </div>
      <div className={'narrate' + (r.kind === 'ambiguous' || r.kind === 'unmapped' ? ' bad' : '')} data-kind={r.kind} style={{ marginTop: '0.7rem' }}>
        <p style={{ margin: 0 }}>
          <code>
            cudaMemcpy(dst, src, n, cudaMemcpyDefault)
          </code>{' '}
          →{' '}
          <b>
            {r.kind.startsWith('cudaMemcpy') ? <code>{r.kind}</code> : r.kind === 'ambiguous' ? tr('cannot tell', 'không xác định được') : r.kind === 'unmapped' ? tr('no allocation', 'không có vùng cấp phát') : tr('one buffer, two addresses', 'một buffer, hai địa chỉ')}
          </b>
        </p>
        <p style={{ margin: '0.3rem 0 0' }} data-detail>
          {r.detail}
        </p>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- stepper: follow the list
const HOP_TONE: Record<HopVerdict, string> = { local: 'chip ok', remote: 'chip', fault: 'chip', invalid: 'chip fail' }; // i18n-ignore: CSS class names
const neutralChip = { border: '1px solid var(--rule)', background: 'var(--page)', color: 'var(--ink)' }; // i18n-ignore: CSS values

function FollowList() {
  const [setting, setSetting] = useState<Setting>('uva');
  const [nodes, setNodes] = useState<Alloc[]>(['cudaMalloc', 'cudaHostAlloc', 'malloc', 'cudaMalloc', 'cudaHostAlloc']);
  const [step, setStep] = useState(0);
  const hops = chase(nodes, setting);
  const addr = addresses(nodes);
  const shown = hops.slice(0, step);
  const cur = shown[shown.length - 1];
  const done = step >= hops.length;
  const failAt = hops.findIndex((h) => h.verdict === 'invalid');
  const label = tr(SETTINGS.find((s) => s.id === setting)!.label, { uva: 'CUDA 4, định địa chỉ hợp nhất', pascal: 'Pascal trở về sau, chỉ bộ nhớ managed (Linux, không HMM)', hmm: 'HMM hoặc ATS', 'um-limited': 'Hạn chế (Windows, WSL)' }[setting]);
  const reset = () => setStep(0);
  return (
    <Bench
      title={tr('Follow the list', 'Đi theo danh sách')}
      task={tr(
        'Choose where each node lives and the system, then step the kernel through the list.',
        'Chọn nơi mỗi nút nằm và loại hệ thống, rồi cho kernel đi qua danh sách từng bước.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Hệ thống">System</Translated>
          <select
            style={{ maxWidth: '100%' }}
            value={setting}
            aria-label={tr('system', 'hệ thống')}
            onChange={(e) => {
              setSetting(e.target.value as Setting);
              reset();
            }}
          >
            {SETTINGS.map((s) => (
              <option key={s.id} value={s.id}>
                {tr(s.label, { uva: 'CUDA 4, định địa chỉ hợp nhất', pascal: 'Pascal trở về sau, chỉ bộ nhớ managed (Linux, không HMM)', hmm: 'HMM hoặc ATS', 'um-limited': 'Hạn chế (Windows, WSL)' }[s.id])}
              </option>
            ))}
          </select>
        </label>
        <button className="primary" onClick={() => setStep((x) => Math.min(hops.length, x + 1))} disabled={done}>
          <Translated vi="Bước">Step</Translated>
        </button>
        <button onClick={() => setStep(failAt >= 0 ? failAt + 1 : hops.length)} disabled={done}>
          <Translated vi="Chạy đến khi lỗi">Run to failure</Translated>
        </button>
        <button className="quiet" onClick={reset} disabled={step === 0}>
          <Translated vi="Đặt lại">Reset</Translated>
        </button>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'stretch', marginBottom: '0.8rem' }}>
        {nodes.map((a, i) => {
          const h = shown[i];
          return (
            <div
              key={i}
              className="panel"
              data-node={i}
              data-verdict={h?.verdict ?? ''}
              style={{
                flex: '1 1 9rem',
                minWidth: 0,
                borderColor: h ? (h.verdict === 'invalid' ? 'var(--fail)' : h.verdict === 'local' ? 'var(--ok)' : 'var(--ink-3)') : undefined,
                background: h?.verdict === 'invalid' ? 'var(--fail-wash)' : undefined,
              }}
            >
              <p className="ui" style={{ margin: 0, fontSize: '0.78rem' }}>
                <Translated
                  vi={
                    <>
                      <b>nút {i + 1}</b> · giá trị {NODE_VALUES[i]}
                      {i < nodes.length - 1 ? ' · next →' : ' · next = null'}
                    </>
                  }
                >
                  <>
                    <b>node {i + 1}</b> · value {NODE_VALUES[i]}
                    {i < nodes.length - 1 ? ' · next →' : ' · next = null'}
                  </>
                </Translated>
              </p>
              <select
                value={a}
                aria-label={tr(`node ${i + 1} allocator`, `bộ cấp phát của nút ${i + 1}`)}
                style={{ width: '100%', margin: '0.3rem 0', fontSize: '0.8rem' }}
                onChange={(e) => {
                  const n = [...nodes];
                  n[i] = e.target.value as Alloc;
                  setNodes(n);
                  reset();
                }}
              >
                {ALLOCS.map((x) => (
                  <option key={x} value={x}>
                    {x}
                  </option>
                ))}
              </select>
              <p className="mono" style={{ margin: 0, fontSize: '0.72rem', overflowWrap: 'anywhere' }}>
                {hex(addr[i])}
              </p>
            </div>
          );
        })}
      </div>
      <div className={'narrate' + (cur?.verdict === 'invalid' ? ' bad' : '')} data-narrate>
        {!cur ? (
          <p style={{ margin: 0 }}>
            <Translated
              vi={
                <>
                  {label}. Kernel đã được khởi chạy với địa chỉ của nút 1 làm đối số. Trước khi bấm bước, hãy nói lần đi
                  này sẽ dừng ở nút nào, nếu nó dừng.
                </>
              }
            >
              <>
                {label}. The kernel has been launched with the address of node 1 as its argument. Before you step, say at
                which node the walk will stop, if it stops.
              </>
            </Translated>
          </p>
        ) : (
          <>
            <p style={{ margin: 0 }}>
              <Translated
                vi={
                  <>
                    Bước nhảy {cur.index + 1}: kernel dereference <code>{hex(cur.address)}</code>, địa chỉ nó lấy từ{' '}
                    {cur.via === 'argument' ? 'đối số của nó' : `trường next của nút ${cur.index}`}. Địa chỉ đó nằm trong{' '}
                    {cur.range}.{' '}
                  </>
                }
              >
                <>
                  Hop {cur.index + 1}: the kernel dereferences <code>{hex(cur.address)}</code>, which it got from{' '}
                  {cur.via === 'argument' ? 'its argument' : `node ${cur.index}'s next field`}. That address lies in{' '}
                  {cur.range}.{' '}
                </>
              </Translated>
              <span className={HOP_TONE[cur.verdict]} style={cur.verdict === 'remote' || cur.verdict === 'fault' ? neutralChip : undefined}>
                {tr(HOP_LABEL[cur.verdict], { local: 'đọc cục bộ', remote: 'đọc qua kênh kết nối', fault: 'lỗi trang và di chuyển', invalid: 'dereference không hợp lệ' }[cur.verdict])}
              </span>
            </p>
            <p style={{ margin: '0.3rem 0 0' }}>
              {cur.rule.cost}{' '}
              {cur.verdict === 'invalid' ? (
                <b>
                  <Translated
                    vi={
                      <>
                        Lần đi dừng ở đây. Kernel đã đọc đúng {cur.index} giá trị; tổng đến lúc này là {cur.sum}.
                      </>
                    }
                  >
                    <>
                      The walk stops here. The kernel read {cur.index} value{cur.index === 1 ? '' : 's'} correctly; the sum so far is {cur.sum}.
                    </>
                  </Translated>
                </b>
              ) : done ? (
                <b>
                  <Translated vi={<>Kernel đã đến cuối danh sách: tổng là {cur.sum}.</>}>
                    <>The kernel reached the end of the list: the sum is {cur.sum}.</>
                  </Translated>
                </b>
              ) : (
                <Translated vi={<>Tổng đến lúc này: {cur.sum}.</>}>
                  <>Sum so far: {cur.sum}.</>
                </Translated>
              )}
            </p>
          </>
        )}
      </div>
      {shown.length ? (
        <table className="threads" style={{ marginTop: '0.7rem' }} data-hops>
          <thead>
            <tr>
              <th>
                <Translated vi="bước nhảy">hop</Translated>
              </th>
              <th>
                <Translated vi="địa chỉ">address</Translated>
              </th>
              <th>
                <Translated vi="vùng">range</Translated>
              </th>
              <th>
                <Translated vi="kết quả">result</Translated>
              </th>
              <th>
                <Translated vi="tổng">sum</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            {shown.map((h) => (
              <tr key={h.index} className={h === cur ? 'cur' : undefined} data-hop={h.index} data-verdict={h.verdict}>
                <td>{h.index + 1}</td>
                <td>{hex(h.address)}</td>
                <td>{h.range}</td>
                <td>{tr(HOP_LABEL[h.verdict], { local: 'đọc cục bộ', remote: 'đọc qua kênh kết nối', fault: 'lỗi trang và di chuyển', invalid: 'dereference không hợp lệ' }[h.verdict])}</td>
                <td>{h.sum}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- your Mac's addresses (lab C.1)
const truthy = (v: unknown) => v === 1 || v === true || v === '1' || v === 'true' || v === 'yes';

export function addressRow(reports: Reports): { shared: boolean; nocopy: boolean; when: string } | null {
  const run = reports.metal;
  const row = benchRows(run, LAB).find((r) => r.name === 'addresses');
  if (!row || row.shared_same === undefined || row.nocopy_same === undefined) return null;
  return { shared: truthy(row.shared_same), nocopy: truthy(row.nocopy_same), when: run.generated.slice(0, 10) };
}

function YourAddresses() {
  const { reports, loading } = useReports();
  if (loading) return null;
  const a = addressRow(reports);
  if (!a)
    return (
      <p data-addresses="none">
        <Translated
          vi={
            <>
              <b>Máy Mac của bạn:</b> chưa có quan sát nào. Hãy chạy lab C.1 với <code>--bench</code> trên backend Metal (
              <code>python3 labs/pmpp.py run appC/lab1 --bench</code>) rồi tải lại trang; hàng "addresses" của nó so sánh
              hai giá trị giúp bạn.
            </>
          }
        >
          <>
            <b>Your Mac:</b> no observation yet. Run lab C.1 with <code>--bench</code> on the Metal backend (
            <code>python3 labs/pmpp.py run appC/lab1 --bench</code>) and reload; its "addresses" row compares the two values
            for you.
          </>
        </Translated>
      </p>
    );
  const same = (b: boolean) => (b ? tr('the same value', 'cùng một giá trị') : tr('different values', 'hai giá trị khác nhau'));
  return (
    <p data-addresses="some" data-shared-same={a.shared ? '1' : '0'} data-nocopy-same={a.nocopy ? '1' : '0'}>
      <Translated
        vi={
          <>
            <b>Quan sát trên máy của bạn, không có trong tài liệu</b> (lab C.1, {a.when}): với một buffer shared,{' '}
            <code>contents</code> và <code>gpuAddress</code> là {same(a.shared)}; với một buffer tạo bằng{' '}
            <code>newBufferWithBytesNoCopy</code>, chúng là {same(a.nocopy)}.
            {!a.shared && !a.nocopy ? (
              <>
                {' '}
                Vậy trên máy của bạn, các byte được chia sẻ nhưng CPU và GPU gọi chúng bằng hai địa chỉ khác nhau: đó là
                bức tranh của mục C.2, hai giá trị con trỏ cho một mẩu dữ liệu, không phải bức tranh của mục này.
              </>
            ) : null}
          </>
        }
      >
        <>
          <b>Observed on your machine, not documented</b> (lab C.1, {a.when}): for a shared buffer, <code>contents</code> and{' '}
          <code>gpuAddress</code> were {same(a.shared)}; for a buffer made with <code>newBufferWithBytesNoCopy</code>, they
          were {same(a.nocopy)}.
          {!a.shared && !a.nocopy ? (
            <>
              {' '}
              On your machine, then, the bytes are shared but the CPU and the GPU name them with different addresses: the
              picture of section C.2, two pointer values for one piece of data, not the picture of this section.
            </>
          ) : null}
        </>
      </Translated>
    </p>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSC3: Record<string, Question> = {
  'appC/C.3/q-direction': {
    kind: 'choice',
    id: 'appC/C.3/q-direction',
    prompt: (
      <Translated
        vi={
          <>
            Với UVA: <code>cudaMemcpy(dst, src, n, cudaMemcpyDefault)</code>, trong đó <code>dst</code> đến từ{' '}
            <code>cudaMalloc</code> trên GPU 1 và <code>src</code> từ <code>cudaMallocHost</code>. Runtime làm gì?
          </>
        }
      >
        <>
          Under UVA: <code>cudaMemcpy(dst, src, n, cudaMemcpyDefault)</code>, where <code>dst</code> came from{' '}
          <code>cudaMalloc</code> on GPU 1 and <code>src</code> from <code>cudaMallocHost</code>. What does the runtime do?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Từ host sang device, vào GPU 1, suy ra từ hai giá trị con trỏ.">
            Host to device, into GPU 1, read off the two pointer values.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nó thất bại với lỗi chiều không hợp lệ, vì cudaMemcpyDefault không chỉ ra loại sao chép nào.">
            It fails with an invalid-direction error, because cudaMemcpyDefault names no kind of copy.
          </Translated>
        ),
        why: (
          <Translated vi="cudaMemcpyDefault chính là loại: nó yêu cầu runtime suy ra chiều sao chép, điều mà định địa chỉ hợp nhất cho phép.">
            cudaMemcpyDefault is the kind: it asks the runtime to infer the direction, which unified addressing allows.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một lần sao chép từ host sang device vào device hiện hành, bất kể dst đến từ GPU nào.">
            A host-to-device copy into the current device, whatever GPU dst came from.
          </Translated>
        ),
        why: (
          <Translated vi="Giá trị đích nói là GPU 1; device hiện hành không ảnh hưởng đến nơi các byte đi tới.">
            The destination value says GPU 1; the current device does not matter for where the bytes go.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một lần sao chép từ device sang host, vì runtime đọc đối số đầu tiên, dst, làm nguồn.">
            A device-to-host copy, because the runtime reads the first argument, dst, as the source.
          </Translated>
        ),
        why: (
          <Translated vi="Thứ tự là đích trước, như memcpy, và chiều sao chép được suy ra từ các giá trị: src là bộ nhớ host, dst là bộ nhớ GPU.">
            The order is destination first, as in memcpy, and the direction is read off the values: src is host memory, dst is GPU memory.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Với định địa chỉ hợp nhất, mọi vùng cấp phát nằm trong một không gian địa chỉ ảo, mỗi bộ xử lý trong vùng riêng
            của nó, nên giá trị của một con trỏ cho biết nó nằm ở đâu. <code>cudaMemcpyDefault</code> (giá trị 4) "yêu cầu
            unified virtual addressing".
          </p>
        }
      >
        <p>
          Under unified addressing every allocation lies in one virtual address space, each processor in its own range, so
          a pointer's value says where it lives. <code>cudaMemcpyDefault</code> (value 4) "requires unified virtual
          addressing".
        </p>
      </Translated>
    ),
    source: 'book p. 625; notes/today/ch02-facts.md A4; appC-facts A5',
    vi: { source: 'sách p. 625; notes/today/ch02-facts.md A4; appC-facts A5' },
  },
  'appC/C.3/q-address-not-access': {
    kind: 'choice',
    id: 'appC/C.3/q-address-not-access',
    prompt: (
      <Translated vi={<>Một đồng nghiệp nói: "Với UVA mọi con trỏ đều dùng được ở mọi nơi." Đâu là phản ví dụ trên CUDA ngày nay?</>}>
        <>A teammate says: "With UVA every pointer works everywhere." Which is a counterexample on CUDA today?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mã host dereference một con trỏ cudaMalloc.">Host code dereferencing a cudaMalloc pointer.</Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Một kernel đọc bộ nhớ cudaMallocHost qua chính con trỏ của host.">
            {"A kernel reading cudaMallocHost memory through the host's own pointer."}
          </Translated>
        ),
        why: (
          <Translated vi="Được phép: với định địa chỉ hợp nhất, con trỏ đó dùng trực tiếp được trong kernel, qua kênh kết nối.">
            Allowed: under unified addressing that pointer works in kernels directly, over the interconnect.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một kernel đọc bộ nhớ managed mà host đã ghi.">A kernel reading managed memory that the host wrote.</Translated>
        ),
        why: (
          <Translated vi="Được phép: bộ nhớ managed sinh ra để làm việc đó (mục C.6).">
            Allowed: that is what managed memory is for (section C.6).
          </Translated>
        ),
      },
      {
        label: <Translated vi="cudaMemcpyDefault giữa hai GPU.">cudaMemcpyDefault between two GPUs.</Translated>,
        why: (
          <Translated vi="Được phép: chiều sao chép được suy ra từ hai giá trị.">
            Allowed: the direction is inferred from the two values.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một không gian địa chỉ không có nghĩa là một quyền truy cập. Mã host vẫn không được dereference bộ nhớ{' '}
            <code>cudaMalloc</code>, và NVIDIA nói vậy cho mọi loại hệ thống ngày nay, kể cả hệ thống có tính nhất quán
            phần cứng.
          </p>
        }
      >
        <p>
          One address space is not one access right. Host code still may not dereference <code>cudaMalloc</code> memory,
          and NVIDIA says so for every kind of system today, hardware coherence included.
        </p>
      </Translated>
    ),
    source: 'book p. 625; appC-facts A3, A12',
    vi: { source: 'sách p. 625; appC-facts A3, A12' },
  },
  'appC/C.3/q-chase': {
    kind: 'numeric',
    id: 'appC/C.3/q-chase',
    prompt: (
      <Translated
        vi={
          <>
            CUDA 4. Host dựng một danh sách bốn nút: nút 1 và 2 trong bộ nhớ <code>cudaHostAlloc</code>, nút 3 trong bộ nhớ{' '}
            <code>malloc</code>, nút 4 trong bộ nhớ <code>cudaHostAlloc</code>, rồi truyền nút đầu cho một kernel. Kernel
            đọc đúng bao nhiêu giá trị nút trước lần dereference không hợp lệ đầu tiên?
          </>
        }
      >
        <>
          CUDA 4. The host builds a four-node list: nodes 1 and 2 in <code>cudaHostAlloc</code> memory, node 3 in{' '}
          <code>malloc</code> memory, node 4 in <code>cudaHostAlloc</code> memory, and passes the head to a kernel. How
          many node values does the kernel read correctly before its first invalid dereference?
        </>
      </Translated>
    ),
    answer: 2,
    unit: 'values',
    explain: (
      <Translated
        vi={
          <p>
            Nút 1 và 2 được đọc qua kênh kết nối. Đọc trường <code>next</code> của nút 2 cũng ổn, vì bản thân trường đó
            nằm trong bộ nhớ ánh xạ. Dereference giá trị đó thì không: nó trỏ vào bộ nhớ <code>malloc</code>, thứ mà kernel
            không được chạm tới dưới CUDA 4. Nút 4 không bao giờ được tới.
          </p>
        }
      >
        <p>
          Nodes 1 and 2 are read over the interconnect. Reading node 2's <code>next</code> field is fine too, because the
          field itself lives in mapped memory. Dereferencing that value is not: it points into <code>malloc</code> memory,
          which a kernel may not touch under CUDA 4. Node 4 is never reached.
        </p>
      </Translated>
    ),
    source: 'book p. 625',
    vi: { unit: 'giá trị', source: 'sách p. 625' },
  },
  'appC/C.3/q-hip-host': {
    kind: 'choice',
    id: 'appC/C.3/q-hip-host',
    prompt: (
      <Translated
        vi={
          <>
            Để gỡ lỗi một bộ lấy mẫu, một đồng nghiệp thêm <code>printf("%f\n", d_logits[0]);</code> vào mã host, trong đó{' '}
            <code>d_logits</code> đến từ <code>hipMalloc</code>. Trên máy AMD nó in ra đúng giá trị. Cùng tệp đó cũng được
            build cho CUDA, nơi <code>d_logits</code> đến từ <code>cudaMalloc</code>. Bạn nói gì với họ?
          </>
        }
      >
        <>
          To debug a sampler, a teammate adds <code>printf("%f\n", d_logits[0]);</code> to host code, where{' '}
          <code>d_logits</code> came from <code>hipMalloc</code>. On the AMD machine it prints the right value. The same
          file also builds for CUDA, where <code>d_logits</code> comes from <code>cudaMalloc</code>. What do you tell them?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="AMD ghi nhận việc host đọc bộ nhớ hipMalloc, dưới dạng zero copy; CUDA cấm điều đó với bộ nhớ cudaMalloc. Hãy dùng memcpy trong cả hai bản build.">
            {'AMD documents host reads of hipMalloc memory, as zero copy; CUDA forbids them for cudaMalloc memory. Use a memcpy in both builds.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Ổn trên cả hai: định địa chỉ hợp nhất, bật sẵn trong mọi tiến trình 64-bit, cho phép host đọc bất kỳ con trỏ device nào.">
            {'It is fine on both: unified addressing, on in every 64-bit process, lets the host read any device pointer.'}
          </Translated>
        ),
        why: (
          <Translated vi="Định địa chỉ hợp nhất cho runtime biết một con trỏ nằm ở đâu; nó không cấp quyền truy cập. Mã host CUDA vẫn không được dereference bộ nhớ cudaMalloc.">
            {'Unified addressing tells the runtime where a pointer lives; it grants no access. CUDA host code still may not dereference cudaMalloc memory.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Ổn trên cả hai, chỉ chậm, vì mỗi lần host đọc bộ nhớ device đều đi qua kênh kết nối.">
            {'It is fine on both, just slow, because every host read of device memory crosses the interconnect.'}
          </Translated>
        ),
        why: (
          <Translated vi="Chậm là hành vi được ghi nhận của HIP. Trên CUDA việc đọc hoàn toàn không được phép.">
            {"Slow is HIP's documented behaviour. On CUDA the read is not allowed at all."}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Trên HIP nó chạy được là nhờ may mắn: host đọc bộ nhớ device ở đó cũng là hành vi không xác định, y như trên CUDA.">
            {'It worked by luck on HIP: host reads of device memory are undefined there too, just as they are on CUDA.'}
          </Translated>
        ),
        why: (
          <Translated vi='AMD ghi nhận điều này: "CPU có thể truy cập bộ nhớ được cấp phát qua hipMalloc()". Đây là một trong những chỗ HIP và CUDA thực sự khác nhau.'>
            {'AMD documents it: "CPUs can access memory allocated via hipMalloc()". This is one of the places where HIP and CUDA really differ.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trang unified memory của AMD: "CPU có thể truy cập bộ nhớ được cấp phát qua hipMalloc()". CUDA cấm cùng lần đọc
            đó, và tài liệu của NVIDIA không nói điều gì xảy ra khi mã host phạm quy tắc, nên đừng trông chờ một lỗi rõ
            ràng. Một dòng như thế này qua được mọi bài kiểm thử trên máy AMD và sai ngay khi tệp được build cho CUDA: hãy
            giữ các lần host đọc bộ nhớ device ngoài mã phải build cho cả hai, và ngoài các vòng lặp nóng trên HIP.
          </p>
        }
      >
        <p>
          AMD's unified memory page: "CPUs can access memory allocated via hipMalloc()". CUDA forbids the same read, and
          NVIDIA's documentation does not say what happens when host code breaks the rule, so do not count on a clean
          error. A line like this passes every test on the AMD machine and is wrong the moment the file builds for CUDA:
          keep host reads of device memory out of code that must build for both, and out of hot loops on HIP.
        </p>
      </Translated>
    ),
    source: 'not in the book; appC-facts B3; notes/today/ch02-facts.md A17 and B7',
    vi: { source: 'không có trong sách; appC-facts B3; notes/today/ch02-facts.md A17 và B7' },
  },
};

// ---------------------------------------------------------------- the page
export function SC3() {
  return (
    <>
      <Translated
        vi={
          <p>
            Trước CUDA 4, host và device mỗi bên có không gian địa chỉ ảo riêng, mỗi không gian ánh xạ lên bộ nhớ vật lý
            của riêng nó. Vì vậy một vị trí vật lý có thể có hai địa chỉ ảo khác nhau, mỗi bên một địa chỉ. Đó chính là
            tình huống của mục C.2, nơi con trỏ của host và con trỏ của kernel tới cùng những byte ánh xạ là hai giá trị
            khác nhau.
          </p>
        }
      >
        <p>
          Before CUDA 4, host and device each had their own virtual address space, each mapped onto its own physical
          memory. One physical location could therefore have two different virtual addresses, one on each side. That is
          exactly the situation of section C.2, where the host's pointer and the kernel's pointer to the same mapped
          bytes were different values.
        </p>
      </Translated>
      <Source section="C.3" pages="625" />
      <Translated
        vi={
          <p>
            CUDA 4, năm 2011, giới thiệu <b>không gian địa chỉ ảo hợp nhất (unified virtual address space)</b>: một không
            gian địa chỉ ảo chung cho host và các device, trong đó, theo lời sách, mỗi địa chỉ vật lý chỉ ánh xạ tới một vị
            trí ảo. Giờ runtime có thể biết từ giá trị của một con trỏ rằng nó chỉ tới bộ nhớ host hay bộ nhớ device. Vì
            vậy <code>cudaMemcpy</code> không còn cần được cho biết chiều sao chép: truyền <code>cudaMemcpyDefault</code> và
            nó suy ra chiều từ hai con trỏ.
          </p>
        }
      >
        <p>
          CUDA 4, in 2011, introduced the <b>unified virtual address space</b>: one virtual address space shared by host
          and devices, in which, in the book's words, each physical address maps to only one virtual location. The runtime
          can now tell from a pointer's value whether it refers to host or device memory. So <code>cudaMemcpy</code> no
          longer needs to be told the direction: pass <code>cudaMemcpyDefault</code> and it reads the direction off the two
          pointers.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="C.3" pages="625" />
        <Today
          checked={tr(
            '2026-11-20 against the CUDA 13.5 guide and runtime reference (notes/today/appC-facts.md, A5; ch02-facts A4)',
            '2026-11-20 đối chiếu với hướng dẫn và tài liệu tham chiếu runtime CUDA 13.5 (notes/today/appC-facts.md, A5; ch02-facts A4)',
          )}
        >
          <Translated
            vi={
              <p>
                CUDA 4 ra đời tháng 5 năm 2011. Hướng dẫn ngày nay: "Một không gian địa chỉ ảo duy nhất được dùng cho toàn
                bộ bộ nhớ host và toàn bộ bộ nhớ toàn cục trên mọi GPU", kể cả bộ nhớ từ <code>malloc</code> và{' '}
                <code>mmap</code>, và nó tự động bật trong các tiến trình 64-bit. <code>cudaPointerGetAttributes</code> cho
                bạn biết một con trỏ nằm ở đâu. Các ngoại lệ của "một vị trí": bộ nhớ host được đăng ký và bộ nhớ host
                write-combined có một địa chỉ device riêng.
              </p>
            }
          >
            <p>
              CUDA 4 is from May 2011. Today's guide: "A single virtual address space is used for all host memory and all
              global memory on all GPUs", including memory from <code>malloc</code> and <code>mmap</code>, and it is on
              automatically in 64-bit processes. <code>cudaPointerGetAttributes</code> tells you where a pointer lives. The
              exceptions to "one location": registered and write-combined host memory get a distinct device address.
            </p>
          </Translated>
        </Today>
      </NoteGroup>
      <Translated
        vi={
          <p>
            Hai cái tên, một tính năng: phụ lục C gọi là UVAS, còn chú thích của chương 2 ở trang 35 gọi là UVA. NVIDIA
            dùng cả hai. Sách cũng nói UVAS bắt nguồn từ GMAC, thư viện nghiên cứu của bài báo năm 2010 mà sách trích dẫn;
            đó là lời kể của sách, và các trang của chính NVIDIA không nhắc tới.
          </p>
        }
      >
        <p>
          Two names, one feature: appendix C says UVAS, and chapter 2's footnote on page 35 says UVA. NVIDIA uses both.
          The book also says UVAS came from GMAC, the research library of the 2010 paper it cites; that is the book's
          account, and NVIDIA's own pages do not mention it.
        </p>
      </Translated>
      <Source pages="625" figs={tr('UVA: ch. 2, p. 35, footnote 3', 'UVA: ch. 2, p. 35, chú thích 3')} />

      <h2>
        <Translated vi="Lần sao chép đi theo chiều nào?">Which way does the copy go?</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Đặt một nguồn và một đích trên bản đồ địa chỉ rồi đọc ra <code>cudaMemcpyDefault</code> sẽ làm gì. Sau đó chuyển
            sang các không gian riêng, tình huống trước CUDA 4, và thử lại.
          </p>
        }
      >
        <p>
          Place a source and a destination on the address map and read off what <code>cudaMemcpyDefault</code> would do.
          Then switch to separate spaces, the situation before CUDA 4, and try again.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Địa chỉ gốc và kích thước của các vùng là bịa ra và rất nhỏ. Một host và hai GPU. Mô hình chỉ suy ra chiều sao
              chép; nó không thực hiện hay đo thời gian một lần sao chép.
            </p>
          }
        >
          <p>
            Range bases and sizes are invented and tiny. One host and two GPUs. The model infers the direction only; it does
            not perform or time a copy.
          </p>
        </Translated>
      </Simplification>
      <CopyDirection />
      <Translated
        vi={
          <p>
            Với các không gian riêng, một giá trị chỉ một vị trí trong mọi không gian cùng lúc, nên chiều sao chép phải do
            bạn cung cấp: đó là lý do <code>cudaMemcpy</code> có đối số kind. Dưới một không gian địa chỉ, giá trị là đủ.
            Thêm buffer được đăng ký để thấy ngoại lệ đã được ghi nhận: một buffer ở hai địa chỉ (từ{' '}
            {hex4(REGISTERED.host[0])} và từ {hex4(REGISTERED.device[0])} trên bản đồ), của host và của device.
          </p>
        }
      >
        <p>
          With separate spaces a value names a location in every space at once, so the direction has to come from you:
          that is why <code>cudaMemcpy</code> has a kind argument at all. Under one address space the value is enough.
          Add the registered buffer to see the documented exception: one buffer at two addresses (from{' '}
          {hex4(REGISTERED.host[0])} and from {hex4(REGISTERED.device[0])} on the map), the host's and the device's.
        </p>
      </Translated>
      <Ask q={questionsSC3['appC/C.3/q-direction']} />

      <h2>
        <Translated vi="Một địa chỉ không phải là một quyền truy cập">An address is not an access right</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Điều mà không gian địa chỉ hợp nhất không cho bạn là quyền truy cập. Dưới CUDA 4, mã host vẫn không thể
            dereference một con trỏ từ <code>cudaMalloc</code>, và kernel vẫn không thể dereference bộ nhớ host thông
            thường. Ngoại lệ là bộ nhớ từ <code>cudaHostAlloc</code>: mã host có thể truyền thẳng con trỏ đó cho kernel, và
            mỗi lần truy cập đi qua PCIe. Những con trỏ mà kernel đọc từ bộ nhớ, như các liên kết của một danh sách do host
            dựng, chỉ an toàn nếu chúng cũng trỏ vào bộ nhớ <code>cudaHostAlloc</code>.
          </p>
        }
      >
        <p>
          What the unified address space does not give you is access. Under CUDA 4, host code still cannot dereference a
          pointer from <code>cudaMalloc</code>, and a kernel still cannot dereference ordinary host memory. The exception is
          memory from <code>cudaHostAlloc</code>: host code can pass that pointer straight to a kernel, and each access goes
          over PCIe. Pointers that a kernel reads from memory, such as the links of a list the host built, are safe only if
          they too point into <code>cudaHostAlloc</code> memory.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="C.3" pages="625" />
        <Today
          checked={tr(
            '2026-11-20 against the CUDA 13.5 guide (notes/today/appC-facts.md, A12)',
            '2026-11-20 đối chiếu với hướng dẫn CUDA 13.5 (notes/today/appC-facts.md, A12)',
          )}
        >
          <Translated
            vi={
              <p>
                Vẫn đúng với bộ nhớ <code>cudaMalloc</code> trên mọi loại hệ thống: "Tính nhất quán phần cứng không cho phép
                host truy cập các vùng cấp phát chỉ dành cho GPU, như những vùng tạo bằng cudaMalloc."
              </p>
            }
          >
            <p>
              Still true of <code>cudaMalloc</code> memory on every kind of system: "Hardware coherency does not enable access
              to GPU-only allocations such as those made with cudaMalloc from the host."
            </p>
          </Translated>
        </Today>
      </NoteGroup>
      <Ask q={questionsSC3['appC/C.3/q-address-not-access']} />

      <h2>
        <Translated vi="Đi theo danh sách">Follow the list</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Đây là kernel mà sách nghĩ tới: một luồng (thread) đi dọc một danh sách liên kết do host dựng, vừa đi vừa cộng.
            Mọi bước nhảy sau bước đầu tiên đều dereference một con trỏ mà kernel đọc từ bộ nhớ.
          </p>
        }
      >
        <p>
          Here is the kernel the book has in mind: one thread walking a linked list that the host built, summing as it
          goes. Every hop after the first dereferences a pointer the kernel read from memory.
        </p>
      </Translated>
      <Code
        tabs={[{ label: 'CUDA', code: CHASE_CU, tiers: CHASE_TIERS, status: 'front-end-checked' }]}
        caption={tr(
          'labs/appC/snippets/c3_chase.cu. It has not run on an NVIDIA or AMD GPU.',
          'labs/appC/snippets/c3_chase.cu. Nó chưa chạy trên GPU NVIDIA hay AMD nào.',
        )}
      />
      <Translated
        vi={
          <p>
            Giờ hãy chọn nơi mỗi nút nằm và loại hệ thống, rồi cho kernel đi qua danh sách từng bước. Hãy tìm hai điều: một
            danh sách ổn trong hai bước nhảy và hỏng ở bước thứ ba, và điều cần có để một danh sách có nút{' '}
            <code>malloc</code> chạy được. Mục C.8 trả lời câu hỏi còn bỏ ngỏ của sách ở đây, và bản đồ trên{' '}
            <a href={href('appC', 'start')}>trang mở đầu</a> cho thấy cùng các quy tắc đó.
          </p>
        }
      >
        <p>
          Now choose where each node lives and on which kind of system, and step the kernel through the list. Look for
          two things: a list that is fine for two hops and fails on the third, and what it takes for a list with a{' '}
          <code>malloc</code> node to work at all. Section C.8 settles the book's open question here, and the map on the{' '}
          <a href={href('appC', 'start')}>start page</a> shows the same rules.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một luồng và không đo thời gian. "Dereference không hợp lệ" đại diện cho hành vi không xác định, thứ trên GPU
              có thể làm sập kernel hoặc lặng lẽ trả về dữ liệu rác. Địa chỉ là bịa ra, mỗi bộ cấp phát một vùng.
            </p>
          }
        >
          <p>
            One thread and no timing. "Invalid dereference" stands for undefined behaviour, which on a GPU may crash the
            kernel or quietly return garbage. Addresses are invented, one range per allocator.
          </p>
        </Translated>
      </Simplification>
      <FollowList />
      <Translated
        vi={
          <p>
            Một cấu trúc do host dựng chỉ an toàn cho kernel nếu mọi nút nằm trong bộ nhớ mà kernel được truy cập. Lỗi trang
            của Pascal chỉ giúp được với các nút managed. Một nút trong bộ nhớ <code>malloc</code> thông thường cần HMM hoặc
            ATS, những thứ xuất hiện nhiều năm sau.
          </p>
        }
      >
        <p>
          A structure built by the host is safe for a kernel only if every node lives in memory the kernel may access.
          Pascal's page faults help only for managed nodes. A node in plain <code>malloc</code> memory needs HMM or ATS,
          which arrived years later.
        </p>
      </Translated>
      <Ask q={questionsSC3['appC/C.3/q-chase']} />

      <h2>
        <Translated vi="HIP: đi xa hơn CUDA">HIP: further than CUDA</Translated>
      </h2>
      <Translated
        vi={
          <p>
            HIP có cùng các thành phần: <code>hipMemcpyDefault</code> và <code>hipPointerGetAttributes</code>. Về quyền truy
            cập, nó đi xa hơn CUDA một bước. AMD ghi nhận rằng CPU có thể đọc trực tiếp bộ nhớ <code>hipMalloc</code>, dưới
            dạng "zero copy" qua kênh kết nối: chậm, nhưng được phép, trong khi CUDA cấm. Và một cái bẫy: header HIP chú
            thích <code>hipDeviceAttributeUnifiedAddressing</code> là "Cuda only", nên đừng dùng nó để kiểm tra định địa chỉ
            hợp nhất trên AMD.
          </p>
        }
      >
        <p>
          HIP has the same pieces: <code>hipMemcpyDefault</code> and <code>hipPointerGetAttributes</code>. It goes one step
          further than CUDA on access. AMD documents that the CPU may read <code>hipMalloc</code> memory directly, as "zero
          copy" over the interconnect: slow, but allowed, where CUDA forbids it. And one trap: the HIP header comments{' '}
          <code>hipDeviceAttributeUnifiedAddressing</code> as "Cuda only", so do not use it to test for unified addressing
          on AMD.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Tài liệu HIP của AMD và header ROCm 10.2 (appC-facts B3).</p>}>
          <p>AMD's HIP documentation and the ROCm 10.2 header (appC-facts B3).</p>
        </Translated>
      </Beyond>
      <Ask q={questionsSC3['appC/C.3/q-hip-host']} />

      <h2>
        <Translated vi="Metal: buffer, không phải con trỏ">Metal: buffers, not pointers</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Metal không bao giờ yêu cầu bạn đưa cho kernel một con trỏ CPU thô. Kernel nhận buffer bằng cách gắn (bind)
            chúng, nên câu hỏi địa chỉ của CPU có mang cùng ý nghĩa trên GPU hay không không xuất hiện trong mã thông
            thường. Apple có ghi nhận một địa chỉ ảo GPU 64-bit cho mỗi buffer, <code>gpuAddress</code>, và không nói gì về
            quan hệ của nó với con trỏ <code>contents</code> của buffer.
          </p>
        }
      >
        <p>
          Metal never asks you to hand a kernel a raw CPU pointer. A kernel receives buffers by binding them, so the
          question of whether the CPU's address means the same thing on the GPU does not come up in ordinary code. Apple
          does document a 64-bit GPU virtual address per buffer, <code>gpuAddress</code>, and says nothing about how it
          relates to the buffer's <code>contents</code> pointer.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={<p>Các header Metal của Apple, macOS 27 SDK (appC-facts C3). Lần kiểm tra thực tế thấy quan hệ này không được ghi nhận.</p>}
        >
          <p>Apple's Metal headers, macOS 27 SDK (appC-facts C3). The fact check found the relation undocumented.</p>
        </Translated>
      </Beyond>
      <YourAddresses />
      <Translated
        vi={
          <p>
            Nếu bạn dựng một cấu trúc liên kết cho kernel Metal, hãy đặt nó trong một buffer và liên kết các bản ghi bằng
            độ lệch (offset) tính từ đầu buffer, không bằng con trỏ. Một độ lệch mang cùng ý nghĩa với CPU và GPU, dù hai
            địa chỉ là gì đi nữa.
          </p>
        }
      >
        <p>
          If you build a linked structure for a Metal kernel, put it inside one buffer and link the records by offsets
          from the buffer's start, not by pointers. An offset means the same thing to the CPU and to the GPU whatever the
          two addresses are.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Lời khuyên dùng độ lệch là của chúng tôi.</p>}>
          <p>The advice to use offsets is ours.</p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Điều này xuất hiện ở đâu trong mã của bạn">Where this shows up in your code</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Khi bạn viết một hàm trợ giúp sao chép nhận cả tensor host lẫn tensor device, <code>cudaMemcpyDefault</code> là
            kind đúng: runtime tự suy ra chiều từ các con trỏ, và bạn không thể chọn sai. Với con trỏ managed, NVIDIA khuyên
            ưu tiên nó hơn một kind không khớp với nơi dữ liệu thực sự nằm.
          </p>
        }
      >
        <p>
          When you write a copy helper that accepts either host or device tensors, <code>cudaMemcpyDefault</code> is the
          right kind: the runtime works out the direction from the pointers, and you cannot get it wrong. For managed
          pointers NVIDIA says to prefer it over a kind that does not match where the data actually is.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              NVIDIA: "Ưu tiên cudaMemcpyDefault hơn một gợi ý cudaMemcpyKind không chính xác" (appC-facts A14). Cách đặt
              vấn đề qua hàm trợ giúp sao chép là của chúng tôi.
            </p>
          }
        >
          <p>
            NVIDIA: "Prefer cudaMemcpyDefault over an inaccurate cudaMemcpyKind hint" (appC-facts A14). The copy-helper
            framing is ours.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Quay lại <a href={href('appC', 'start')}>bản đồ</a>: bước này là C.3. So sánh các ô của nó với các ô của C.2: giá
            trị con trỏ đã thay đổi, còn quy tắc truy cập thì hầu như không.
          </p>
        }
      >
        <p>
          Back to <a href={href('appC', 'start')}>the map</a>: this step is C.3. Compare its cells with C.2's: the pointer
          values changed, the access rules hardly did.
        </p>
      </Translated>
    </>
  );
}
