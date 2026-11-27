import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Bench, Beyond, NoteGroup, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  CALLS,
  LLAMA2_7B_BYTES_PER_TOKEN,
  MiB,
  STRATEGIES,
  apply,
  bookExample,
  bookThreeCalls,
  clean,
  grow,
  initial,
  run,
  slots,
  type Dev,
  type Op,
  type Outcome,
  type Strategy,
  type VmmState,
} from '../../sim/vmmgrow';
import { binary } from '../../sim/addrwidth';
import { GrowBufferScene } from './GrowBufferScene';

/** A long identifier that may break anywhere, for the narrow margin notes. */
function Id({ children }: { children: ReactNode }) {
  return <code style={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>{children}</code>;
}

// ---------------------------------------------------------------- quoted signatures
// Lesson copies of labs/appC/snippets/c9_vmm_decls.h and c9_vmm_decls_hip.h, quoted from
// the vendors' references as recorded in notes/today/appC-facts.md (A15, B7). SC9.test.ts
// fails if a copy drifts from its file, and checks that every declaration appears in the
// facts file. Not compiled anywhere.
export const SIGS_CUDA = `// 1. Reserve a range of virtual addresses.
CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                             CUdeviceptr addr, unsigned long long flags);
// 2. Allocate physical memory on a device; you get a handle, not an address.
CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                     const CUmemAllocationProp *prop, unsigned long long flags);
// 3. Put that memory behind part of the range.
CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle handle, unsigned long long flags);
// 4. Let devices read and write it. The book's list leaves this one out.
CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count);
// Teardown, in this order: cuMemUnmap, cuMemRelease, cuMemAddressFree.`;

export const SIGS_HIP = `// The same steps: hipMemAddressReserve, hipMemCreate, hipMemMap, hipMemSetAccess.
hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr,
                                unsigned long long flags);
hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count);
// Our facts file quotes these two declarations; hipMemCreate and hipMemMap are named, not quoted.`;

// ---------------------------------------------------------------- the stepper
const G = 2 * MiB; // one granule; a parameter of the model, not a device value
const N = 8;
const SLOT = 62;
const SIZES = [1, 2, 4, 1.5];

const KIND_CHIP: Record<Outcome['kind'], string> = { ok: 'ok', error: 'fail', crash: 'fail', order: 'fail' };
const KIND_WORD: Record<Outcome['kind'], string> = {
  get ok() {
    return tr('done', 'xong');
  },
  get error() {
    return tr('refused', 'bị từ chối');
  },
  get crash() {
    return tr('crash', 'sập');
  },
  get order() {
    return tr('out of order', 'sai thứ tự');
  },
};

function opText(op: Op): string {
  const g = (b: number) => `${Number((b / G).toFixed(2))}`;
  switch (op.op) {
    case 'reserve':
      return tr(`reserve ${g(op.size)} granules`, `đặt trước ${g(op.size)} granule`);
    case 'create':
      return tr(`create ${g(op.size)} granule${op.size === G ? '' : 's'} on GPU ${op.device}`, `tạo ${g(op.size)} granule trên GPU ${op.device}`);
    case 'map':
      return tr(`map h${op.chunk} at granule ${g(op.offset)}`, `ánh xạ h${op.chunk} tại granule ${g(op.offset)}`);
    case 'setAccess':
      return tr(
        `grant GPU ${op.device} granules ${g(op.offset)} to ${Number(g(op.offset + op.size)) - 1}`,
        `cấp quyền cho GPU ${op.device} các granule ${g(op.offset)} đến ${Number(g(op.offset + op.size)) - 1}`,
      );
    case 'read':
      return tr(
        `kernel on GPU ${op.device} reads granules ${g(op.offset)} to ${Number(g(op.offset + op.size)) - 1}`,
        `kernel trên GPU ${op.device} đọc các granule ${g(op.offset)} đến ${Number(g(op.offset + op.size)) - 1}`,
      );
    case 'unmap':
      return tr(`unmap h${op.chunk}`, `bỏ ánh xạ h${op.chunk}`);
    case 'release':
      return tr(`release h${op.chunk}`, `giải phóng h${op.chunk}`);
    case 'free':
      return tr('free the range', 'giải phóng dải địa chỉ');
  }
}

function Seg<T extends string | number>({ label, value, options, set }: { label: string; value: T; options: { v: T; t: string }[]; set: (v: T) => void }) {
  return (
    <span className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.v)} aria-pressed={value === o.v} onClick={() => set(o.v)}>
          {o.t}
        </button>
      ))}
    </span>
  );
}

function SlotSelect({ label, value, set, from = 0 }: { label: string; value: number; set: (v: number) => void; from?: number }) {
  return (
    <select aria-label={label} value={value} onChange={(e) => set(Number(e.target.value))}>
      {Array.from({ length: N - from }, (_, k) => k + from).map((k) => (
        <option key={k} value={k}>
          {k}
        </option>
      ))}
    </select>
  );
}

function VmmStepper() {
  const [state, setState] = useState<VmmState>(() => initial(G));
  const [log, setLog] = useState<Outcome[]>([]);
  const [createDev, setCreateDev] = useState<Dev>(0);
  const [createSize, setCreateSize] = useState(4);
  const [mapChunk, setMapChunk] = useState(1);
  const [mapAt, setMapAt] = useState(0);
  const [accDev, setAccDev] = useState<Dev>(0);
  const [accFrom, setAccFrom] = useState(0);
  const [accTo, setAccTo] = useState(N - 1);
  const [readDev, setReadDev] = useState<Dev>(0);
  const [readFrom, setReadFrom] = useState(0);
  const [readTo, setReadTo] = useState(3);
  const [tdChunk, setTdChunk] = useState(1);

  const doOp = (op: Op) => {
    const r = apply(state, op);
    setState(r.state);
    setLog([...log, r.outcome]);
    if (op.op === 'create' && r.outcome.kind === 'ok') {
      setMapChunk(state.nextId);
      setTdChunk(state.nextId);
    }
  };
  const preset = (ops: Op[]) => {
    const r = run(G, ops);
    setState(r.state);
    setLog(r.log);
  };
  const last = log[log.length - 1];
  const lastRead = [...log].reverse().find((o) => o.op.op === 'read');
  const sl = slots(state);
  const live = state.chunks.filter((c) => !c.released);
  const W = 30 + N * SLOT + 10;
  const pieceAt = (k: number) => (lastRead && last === lastRead ? lastRead.pieces?.find((p) => k * G >= p.offset && k * G < p.offset + p.size) : undefined);

  return (
    <Bench
      title={tr('Reserve, create, map, grant', 'Đặt trước, tạo, ánh xạ, cấp quyền')}
      task={tr(
        "Run the book's three calls, find why the kernel crashes, fix it, then tear everything down in the documented order.",
        'Chạy ba lời gọi của sách, tìm xem vì sao kernel sập, sửa nó, rồi dỡ bỏ mọi thứ theo thứ tự được ghi nhận.',
      )}
      foot={
        <Translated
          vi={
            <>
              Quy tắc lấy từ hướng dẫn §4.18 và tài liệu tham chiếu Driver API của NVIDIA, và trang HIP của AMD (appC-facts.md,
              A15, B7). Ánh xạ chồng lên các địa chỉ đã được ánh xạ và cấp quyền truy cập trên các địa chỉ chưa ánh xạ bị mô
              hình này từ chối; các nguồn của chúng tôi không nói driver làm gì.
            </>
          }
        >
          <>
            Rules from NVIDIA's guide §4.18 and Driver API reference, and AMD's HIP page (appC-facts.md, A15, B7). Mapping
            over mapped addresses and granting access over unmapped ones are refused by this model; our sources do not say
            what the driver does.
          </>
        </Translated>
      }
    >
      <div className="row" role="group" aria-label={tr('Presets', 'Kịch bản có sẵn')} style={{ marginBottom: '0.9rem' }}>
        <span className="muted">
          <Translated vi="Bắt đầu từ">Start from</Translated>
        </span>
        <button className="quiet" data-preset="three" onClick={() => preset(bookThreeCalls(G))}>
          <Translated vi="Ba lời gọi của sách, rồi một kernel">The book's three calls, then a kernel</Translated>
        </button>
        <button className="quiet" data-preset="book" onClick={() => preset(bookExample(G))}>
          <Translated vi="Ví dụ của sách: một con trỏ trải trên hai GPU">The book's example: one pointer over two GPUs</Translated>
        </button>
        <button
          className="quiet"
          data-preset="empty"
          onClick={() => {
            setState(initial(G));
            setLog([]);
          }}
        >
          <Translated vi="Trống">Nothing</Translated>
        </button>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${W} 148`} role="img" aria-label={tr('The reserved address range, what is mapped behind it, and which GPU may access it', 'Dải địa chỉ đã đặt trước, những gì được ánh xạ phía sau nó, và GPU nào được truy cập nó')} style={{ width: '100%', maxWidth: W, minWidth: 460, display: 'block' }} data-part="range" data-reserved={state.reserved == null ? 0 : state.reserved / G}>
          <text x={30} y={14} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
            {state.reserved == null ? (state.freed ? tr('the range has been freed', 'dải địa chỉ đã được giải phóng') : tr('no addresses reserved yet', 'chưa đặt trước địa chỉ nào')) : tr(`one pointer: base + 0 … base + ${state.reserved / G} granules`, `một con trỏ: base + 0 … base + ${state.reserved / G} granule`)}
          </text>
          {Array.from({ length: N }, (_, k) => {
            const s = sl[k];
            const p = pieceAt(k);
            const reserved = s != null;
            return (
              <g key={k} data-slot={k} data-device={s?.device ?? ''} data-read={p?.status ?? ''}>
                <rect
                  x={30 + k * SLOT}
                  y={22}
                  width={SLOT - 4}
                  height={40}
                  rx="2"
                  fill={!reserved ? 'transparent' : s.device === 0 ? 'var(--global)' : s.device === 1 ? 'var(--global-wash)' : 'var(--cell)'}
                  stroke={p && (p.status === 'unmapped' || p.status === 'no-access') ? 'var(--fail)' : reserved ? 'var(--global)' : 'var(--rule)'}
                  strokeWidth={p ? 2.5 : 1}
                  strokeDasharray={!reserved ? '2 4' : s.device == null ? '4 3' : undefined}
                  opacity={reserved ? 1 : 0.6}
                />
                <text x={30 + k * SLOT + (SLOT - 4) / 2} y={47} fontSize="11" textAnchor="middle" style={{ fill: s?.device === 0 ? 'var(--page)' : 'var(--ink)' }}>
                  {reserved ? (s.chunk != null ? `h${s.chunk}` : tr('empty', 'trống')) : ''}
                </text>
                <text x={30 + k * SLOT + (SLOT - 4) / 2} y={76} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>
                  {k}
                </text>
                {([0, 1] as Dev[]).map((d) => (
                  <text key={d} x={30 + k * SLOT + (SLOT - 4) / 2} y={100 + d * 18} fontSize="12" textAnchor="middle" data-access={`${d}`} data-on={s?.access[d] ? 1 : 0} style={{ fill: s?.access[d] ? 'var(--ok)' : 'var(--ink-3)' }}>
                    {reserved ? (s.access[d] ? '✓' : '·') : ''}
                  </text>
                ))}
                {p ? (
                  <text x={30 + k * SLOT + (SLOT - 4) / 2} y={142} fontSize="10" textAnchor="middle" style={{ fill: p.status === 'unmapped' || p.status === 'no-access' ? 'var(--fail)' : 'var(--ok)' }}>
                    {tr(
                      p.status === 'no-access' ? 'no access' : p.status,
                      { local: 'cục bộ', remote: 'từ xa', unmapped: 'chưa ánh xạ', 'no-access': 'không có quyền' }[p.status],
                    )}
                  </text>
                ) : null}
              </g>
            );
          })}
          <text x={2} y={47} fontSize="10" style={{ fill: 'var(--ink-3)' }}>
            VA
          </text>
          <text x={2} y={104} fontSize="10" style={{ fill: 'var(--ink-3)' }}>
            G0
          </text>
          <text x={2} y={122} fontSize="10" style={{ fill: 'var(--ink-3)' }}>
            G1
          </text>
        </svg>
      </div>
      <Translated
        vi={
          <p className="muted" style={{ margin: '0.3rem 0 0.8rem' }}>
            Hàng trên: các địa chỉ đã đặt trước, mỗi ô một granule, được tô ở nơi có một chunk được ánh xạ (đậm: bộ nhớ của GPU
            0; nhạt: của GPU 1). Hàng G0 và G1: một dấu tích ở nơi GPU đó đã được cấp quyền truy cập.
          </p>
        }
      >
        <p className="muted" style={{ margin: '0.3rem 0 0.8rem' }}>
          Top row: the reserved addresses, one box per granule, filled where a chunk is mapped (solid: GPU 0's memory; pale:
          GPU 1's). Rows G0 and G1: a tick where that GPU has been granted access.
        </p>
      </Translated>

      <p className={'narrate' + (last && last.kind !== 'ok' ? ' bad' : '')} data-out="last" data-kind={last?.kind ?? ''} data-code={last?.code ?? ''} aria-live="polite" style={{ margin: '0 0 0.8rem' }}>
        {last ? (
          <>
            <b>{last.call === 'kernel' ? tr('Kernel', 'Kernel') : <code>{last.call}</code>}:</b> {last.text}
          </>
        ) : (
          tr('Nothing yet. Reserve a range, or start from one of the presets.', 'Chưa có gì. Hãy đặt trước một dải địa chỉ, hoặc bắt đầu từ một kịch bản có sẵn.')
        )}
      </p>
      <div className="row" style={{ marginBottom: '0.8rem' }} data-part="chunks">
        <span className="muted">
          <Translated vi="Các chunk">Chunks</Translated>
        </span>
        {state.chunks.length ? (
          state.chunks.map((c) => {
            const m = state.maps.find((x) => x.chunk === c.id);
            return (
              <span key={c.id} className={'chip ' + (c.released ? '' : 'global')} data-chunk={c.id}>
                <Translated
                  vi={
                    <>
                      h{c.id}: {c.size / G} trên GPU {c.device}
                      {c.released ? ', đã giải phóng' : m ? `, tại ${m.offset / G}` : ', chưa ánh xạ'}
                    </>
                  }
                >
                  <>
                    h{c.id}: {c.size / G} on GPU {c.device}
                    {c.released ? ', released' : m ? `, at ${m.offset / G}` : ', not mapped'}
                  </>
                </Translated>
              </span>
            );
          })
        ) : (
          <span className="muted">
            <Translated vi="không có">none</Translated>
          </span>
        )}
      </div>

      <div className="panel" style={{ display: 'grid', gap: '0.55rem', marginBottom: '1rem' }} data-part="controls">
        <div className="row" style={{ gap: '0.4rem 0.6rem' }}>
          <code style={{ width: '11rem' }}>{CALLS.reserve.cuda}</code>
          <button data-act="reserve" onClick={() => doOp({ op: 'reserve', size: N * G })}>
            <Translated vi={<>Đặt trước {N} granule</>}>
              <>Reserve {N} granules</>
            </Translated>
          </button>
        </div>
        <div className="row" style={{ gap: '0.4rem 0.6rem' }}>
          <code style={{ width: '11rem' }}>{CALLS.create.cuda}</code>
          <Seg label={tr('Create on', 'Tạo trên')} value={createDev} options={[{ v: 0 as Dev, t: tr('GPU 0', 'GPU 0') }, { v: 1 as Dev, t: tr('GPU 1', 'GPU 1') }]} set={setCreateDev} />
          <select aria-label={tr('Chunk size', 'Kích thước chunk')} value={createSize} onChange={(e) => setCreateSize(Number(e.target.value))}>
            {SIZES.map((s) => (
              <option key={s} value={s}>
                <Translated vi={`${s} granule`}>
                  {s} granule{s === 1 ? '' : 's'}
                </Translated>
              </option>
            ))}
          </select>
          <button data-act="create" onClick={() => doOp({ op: 'create', device: createDev, size: createSize * G })}>
            <Translated vi="Tạo">Create</Translated>
          </button>
        </div>
        <div className="row" style={{ gap: '0.4rem 0.6rem' }}>
          <code style={{ width: '11rem' }}>{CALLS.map.cuda}</code>
          <select aria-label={tr('Chunk to map', 'Chunk cần ánh xạ')} value={mapChunk} onChange={(e) => setMapChunk(Number(e.target.value))}>
            {(live.length ? live : [{ id: 1 }]).map((c) => (
              <option key={c.id} value={c.id}>
                h{c.id}
              </option>
            ))}
          </select>
          <span className="muted">
            <Translated vi="tại granule">at granule</Translated>
          </span>
          <SlotSelect label={tr('Map at granule', 'Ánh xạ tại granule')} value={mapAt} set={setMapAt} />
          <button data-act="map" onClick={() => doOp({ op: 'map', chunk: mapChunk, offset: mapAt * G })}>
            <Translated vi="Ánh xạ">Map</Translated>
          </button>
        </div>
        <div className="row" style={{ gap: '0.4rem 0.6rem' }}>
          <code style={{ width: '11rem' }}>{CALLS.setAccess.cuda}</code>
          <Seg label={tr('Grant to', 'Cấp quyền cho')} value={accDev} options={[{ v: 0 as Dev, t: tr('GPU 0', 'GPU 0') }, { v: 1 as Dev, t: tr('GPU 1', 'GPU 1') }]} set={setAccDev} />
          <SlotSelect label={tr('Grant from granule', 'Cấp quyền từ granule')} value={accFrom} set={setAccFrom} />
          <span className="muted">
            <Translated vi="đến">to</Translated>
          </span>
          <SlotSelect label={tr('Grant to granule', 'Cấp quyền đến granule')} value={accTo} set={setAccTo} />
          <button data-act="access" onClick={() => doOp({ op: 'setAccess', device: accDev, offset: accFrom * G, size: Math.max(1, accTo - accFrom + 1) * G })}>
            <Translated vi="Cấp quyền">Grant</Translated>
          </button>
        </div>
        <div className="row" style={{ gap: '0.4rem 0.6rem' }}>
          <code style={{ width: '11rem' }}>kernel</code>
          <Seg label={tr('Kernel on', 'Kernel trên')} value={readDev} options={[{ v: 0 as Dev, t: tr('GPU 0', 'GPU 0') }, { v: 1 as Dev, t: tr('GPU 1', 'GPU 1') }]} set={setReadDev} />
          <span className="muted">
            <Translated vi="đọc">reads</Translated>
          </span>
          <SlotSelect label={tr('Read from granule', 'Đọc từ granule')} value={readFrom} set={setReadFrom} />
          <span className="muted">
            <Translated vi="đến">to</Translated>
          </span>
          <SlotSelect label={tr('Read to granule', 'Đọc đến granule')} value={readTo} set={setReadTo} />
          <button data-act="read" onClick={() => doOp({ op: 'read', device: readDev, offset: readFrom * G, size: Math.max(1, readTo - readFrom + 1) * G })}>
            <Translated vi="Khởi chạy">Launch</Translated>
          </button>
        </div>
        <div className="row" style={{ gap: '0.4rem 0.6rem' }}>
          <span className="muted" style={{ width: '11rem' }}>
            <Translated vi="dỡ bỏ">teardown</Translated>
          </span>
          <select aria-label={tr('Chunk to tear down', 'Chunk cần dỡ bỏ')} value={tdChunk} onChange={(e) => setTdChunk(Number(e.target.value))}>
            {(state.chunks.length ? state.chunks : [{ id: 1 }]).map((c) => (
              <option key={c.id} value={c.id}>
                h{c.id}
              </option>
            ))}
          </select>
          <button data-act="unmap" onClick={() => doOp({ op: 'unmap', chunk: tdChunk })}>
            {CALLS.unmap.cuda}
          </button>
          <button data-act="release" onClick={() => doOp({ op: 'release', chunk: tdChunk })}>
            {CALLS.release.cuda}
          </button>
          <button data-act="free" onClick={() => doOp({ op: 'free' })}>
            {CALLS.free.cuda}
          </button>
        </div>
      </div>
      <div className="panel">
        <p className="muted" style={{ margin: '0 0 0.4rem' }}>
          <Translated vi="Các lời gọi đến lúc này">Calls so far</Translated>
        </p>
        {log.length ? (
          <ol style={{ margin: 0, paddingLeft: '1.4rem', maxHeight: '15rem', overflowY: 'auto' }} data-part="log">
            {log.map((o, k) => (
              <li key={k} data-kind={o.kind}>
                <span className={'chip ' + KIND_CHIP[o.kind]}>{KIND_WORD[o.kind]}</span> {opText(o.op)}
              </li>
            ))}
          </ol>
        ) : (
          <p className="muted" style={{ margin: 0 }}>
            <Translated vi="không có">none</Translated>
          </p>
        )}
        {log.length && clean(state) && state.freed ? (
          <p style={{ margin: '0.7rem 0 0' }} data-out="clean">
            <span className="chip ok">{tr('clean', 'sạch')}</span>{' '}
            <Translated vi="Mọi thứ đã được bỏ ánh xạ, giải phóng bộ nhớ và giải phóng địa chỉ.">
              Everything is unmapped, released and freed.
            </Translated>
          </p>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- growing a KV cache
const MAX_TOKENS = 4096;
const CHUNKS_MIB = [2, 8, 32, 128];
const CW = 720;
const CH = 210;

function GrowKV() {
  const [tokens, setTokens] = useState(1500);
  const [kib, setKib] = useState(LLAMA2_7B_BYTES_PER_TOKEN / 1024);
  const [chunkMiB, setChunkMiB] = useState(2);
  const [strategy, setStrategy] = useState<Strategy>('double');

  const bytesPerToken = Math.max(1, kib) * 1024;
  const chunk = chunkMiB * MiB;
  const reserve = Math.ceil((MAX_TOKENS * bytesPerToken) / chunk) * chunk;
  const results = useMemo(() => {
    const o = { tokens, bytesPerToken, chunk, reserve };
    return Object.fromEntries(STRATEGIES.map((s) => [s.key, grow(s.key, o)])) as Record<Strategy, ReturnType<typeof grow>>;
  }, [tokens, bytesPerToken, chunk, reserve]);
  const r = results[strategy];
  const top = Math.max(...STRATEGIES.map((s) => results[s.key].peak));
  const X = (t: number) => 56 + ((t - 1) / Math.max(1, tokens - 1)) * (CW - 72);
  const Y = (b: number) => CH - 22 - (b / top) * (CH - 66);
  const stride = Math.max(1, Math.floor(r.steps.length / 600));
  const sampled = r.steps.filter((s, k) => k % stride === 0 || s.moved || k === r.steps.length - 1);
  const allocPath = sampled.map((s, k) => `${k ? 'L' : 'M'} ${X(s.token).toFixed(1)} ${Y(s.allocated).toFixed(1)}`).join(' ');
  const livePath = `M ${X(1)} ${Y(r.steps[0].live)} L ${X(tokens)} ${Y(r.finalLive)}`;
  const spikes = r.steps.filter((s) => s.moved);
  const showSpikes = spikes.length <= 200 ? spikes : spikes.filter((_, k) => k % Math.ceil(spikes.length / 200) === 0);

  return (
    <Bench
      title={tr('Growing a KV cache', 'Nới rộng một KV cache')}
      task={tr(
        'Compare the three strategies at 4096 tokens: what is copied, the most memory held at once, and whether the pointer moves.',
        'So sánh ba chiến lược ở 4096 token: lượng được sao chép, lượng bộ nhớ lớn nhất giữ cùng lúc, và con trỏ có bị dời đi không.',
      )}
      foot={
        <Translated
          vi={
            <>
              Mặc định là ví dụ Llama 2 7B của sách ở 16 bit: 2 × 32 lớp × chiều ẩn 4096 × 2 byte = 512 KiB mỗi token (Eq.
              20.7, ch. 20 §20.6, pp. 505–506). Chunk 2 MiB là kích thước mà chú thích trong allocator của PyTorch nêu cho các
              ánh xạ của nó (appC-facts.md, D2). Phần đặt trước phủ {MAX_TOKENS} token.
            </>
          }
        >
          <>
            The default is the book's Llama 2 7B example at 16 bits: 2 × 32 layers × hidden dimension 4096 × 2 bytes = 512 KiB per token (Eq.
            20.7, ch. 20 §20.6, pp. 505–506). The 2 MiB chunk is the size PyTorch's allocator comment gives for its
            mappings (appC-facts.md, D2). The reservation covers {MAX_TOKENS} tokens.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ flex: '1 1 16rem' }}>
          <Translated vi={<>Số token đã sinh: {tokens.toLocaleString('en-US')}</>}>
            <>Tokens generated: {tokens.toLocaleString('en-US')}</>
          </Translated>
          <input type="range" min={1} max={MAX_TOKENS} value={tokens} aria-label={tr('Tokens generated', 'Số token đã sinh')} onChange={(e) => setTokens(Number(e.target.value))} />
        </label>
        <label className="field">
          <Translated vi="KV cache mỗi token (KiB)">KV cache per token (KiB)</Translated>
          <input type="number" min={1} step={1} value={kib} aria-label={tr('KV cache per token in KiB', 'KV cache mỗi token, tính bằng KiB')} onChange={(e) => setKib(Math.max(1, Math.floor(Number(e.target.value) || 1)))} />
        </label>
        <label className="field">
          <Translated vi="Chunk">Chunk</Translated>
          <select value={chunkMiB} aria-label={tr('Chunk size', 'Kích thước chunk')} onChange={(e) => setChunkMiB(Number(e.target.value))}>
            {CHUNKS_MIB.map((c) => (
              <option key={c} value={c}>
                {c} MiB
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <Seg
          label={tr('Strategy', 'Chiến lược')}
          value={strategy}
          options={STRATEGIES.map((s) => ({
            v: s.key,
            t: tr(s.label, { double: 'Cấp phát gấp đôi rồi sao chép', exact: 'Cấp phát lại vừa khít, mỗi token', vmm: 'Đặt trước một lần, ánh xạ từng chunk' }[s.key]),
          }))}
          set={setStrategy}
        />
      </div>

      <div className="panel" style={{ overflowX: 'auto', marginBottom: '1rem' }}>
        <svg viewBox={`0 0 ${CW} ${CH}`} role="img" aria-label={tr('Memory allocated and data held as tokens are generated', 'Bộ nhớ được cấp phát và dữ liệu được giữ khi các token được sinh ra')} style={{ width: '100%', minWidth: 460, display: 'block' }} data-part="chart" data-strategy={strategy}>
          <line x1={56} x2={CW - 16} y1={Y(0)} y2={Y(0)} stroke="var(--rule)" />
          <line x1={56} x2={56} y1={Y(0)} y2={Y(top)} stroke="var(--rule)" />
          <line x1={56} x2={CW - 16} y1={Y(top)} y2={Y(top)} stroke="var(--rule-soft)" />
          <text x={52} y={Y(top) + 4} fontSize="12" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>
            {binary(top).text}
          </text>
          <text x={52} y={Y(0) + 4} fontSize="12" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>
            0
          </text>
          <text x={CW - 16} y={CH - 4} fontSize="12" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>
            {tr(`tokens: 1 to ${tokens.toLocaleString('en-US')}`, `token: 1 đến ${tokens.toLocaleString('en-US')}`)}
          </text>
          {showSpikes.map((s) => (
            <line key={s.token} x1={X(s.token)} x2={X(s.token)} y1={Y(s.allocated)} y2={Y(s.during)} stroke="var(--global)" strokeDasharray="3 2" strokeWidth="1.5" />
          ))}
          <path d={livePath} fill="none" stroke="var(--ink-2)" strokeWidth="1.5" />
          <path d={allocPath} fill="none" stroke="var(--global)" strokeWidth="2.2" />
          <text x={64} y={16} fontSize="12.5" fontWeight="600" style={{ fill: 'var(--global)' }}>
            {strategy === 'vmm' ? tr('mapped', 'đã ánh xạ') : tr('allocated', 'đã cấp phát')}
            {strategy === 'vmm' ? '' : tr('; dashed: old and new together, during a copy', '; nét đứt: cũ và mới cùng lúc, trong một lần sao chép')}
          </text>
          <text x={64} y={33} fontSize="12.5" style={{ fill: 'var(--ink-2)' }}>
            {tr('data held', 'dữ liệu được giữ')}
          </text>
        </svg>
        <p className="ui muted" style={{ margin: '0.5rem 0 0' }} data-out="note">
          {strategy === 'double'
            ? r.peakRatio
              ? tr(
                  `During each copy, old and new together are up to ${Number(r.peakRatio.toPrecision(3))} times the data being copied.`,
                  `Trong mỗi lần sao chép, cũ và mới cộng lại lên tới ${Number(r.peakRatio.toPrecision(3))} lần lượng dữ liệu đang được sao chép.`,
                )
              : tr('No copy yet: the first buffer still has room.', 'Chưa có lần sao chép nào: buffer đầu tiên vẫn còn chỗ.')
            : strategy === 'exact'
              ? tr(
                  'Every token copies everything held so far, so the total copied grows with the square of the length.',
                  'Mỗi token sao chép mọi thứ đang giữ, nên tổng lượng sao chép tăng theo bình phương độ dài.',
                )
              : tr(
                  `Reserved addresses: ${binary(r.reserved ?? 0).text}, with memory behind ${binary(r.finalAllocated).text} of them.`,
                  `Địa chỉ đã đặt trước: ${binary(r.reserved ?? 0).text}, trong đó ${binary(r.finalAllocated).text} có bộ nhớ phía sau.`,
                )}
        </p>
      </div>
      <div style={{ overflowX: 'auto', marginTop: '1rem' }}>
        <table className="data" style={{ margin: 0 }} data-part="compare">
          <thead>
            <tr>
              <th>
                <Translated
                  vi={
                    <>
                      Cả ba, ở {tokens.toLocaleString('en-US')} token (<span data-out="live">{binary(r.finalLive).text}</span> dữ
                      liệu)
                    </>
                  }
                >
                  <>
                    All three, at {tokens.toLocaleString('en-US')} tokens (<span data-out="live">{binary(r.finalLive).text}</span> of data)
                  </>
                </Translated>
              </th>
              <th className="num">
                <Translated vi="Đã sao chép (× dữ liệu)">Copied (× the data)</Translated>
              </th>
              <th className="num">
                <Translated vi="Lớn nhất cùng lúc">Most at once</Translated>
              </th>
              <th className="num">
                <Translated vi="Đang không dùng">Unused now</Translated>
              </th>
              <th className="num">
                <Translated vi="Số lần con trỏ đổi">Pointer changes</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            {STRATEGIES.map((s) => {
              const x = results[s.key];
              return (
                <tr
                  key={s.key}
                  data-strategy={s.key}
                  data-copied={x.copiedTotal}
                  data-peak={x.peak}
                  data-unused={x.unused}
                  data-moves={x.pointerChanges}
                  data-selected={s.key === strategy ? 1 : 0}
                  onClick={() => setStrategy(s.key)}
                  style={{ cursor: 'pointer', background: s.key === strategy ? 'var(--rule-soft)' : undefined }}
                >
                  <td style={{ fontWeight: s.key === strategy ? 600 : 400 }}>
                    {tr(s.label, { double: 'Cấp phát gấp đôi rồi sao chép', exact: 'Cấp phát lại vừa khít, mỗi token', vmm: 'Đặt trước một lần, ánh xạ từng chunk' }[s.key])}
                  </td>
                  <td className="num">
                    {binary(x.copiedTotal).text}
                    {x.copiedTotal ? <span className="muted"> ({Number(x.copyRatio.toPrecision(3))} ×)</span> : null}
                  </td>
                  <td className="num">{binary(x.peak).text}</td>
                  <td className="num">{binary(x.unused).text}</td>
                  <td className="num">{x.pointerChanges.toLocaleString('en-US')}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSC9: Record<string, Question> = {
  'appC/C.9/q-setaccess': {
    kind: 'choice',
    id: 'appC/C.9/q-setaccess',
    prompt: (
      <Translated
        vi={
          <>
            Bạn đã đặt trước một dải địa chỉ, tạo bộ nhớ trên GPU 0 và ánh xạ nó ở offset 0. Kernel đầu tiên trên GPU 0 đọc
            nó thì sập. Còn thiếu gì?
          </>
        }
      >
        <>
          You reserved a range, created memory on GPU 0 and mapped it at offset 0. The first kernel on GPU 0 that reads it
          crashes. What is missing?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="cudaDeviceSynchronize sau lần ánh xạ, để ánh xạ có hiệu lực trước kernel.">
            cudaDeviceSynchronize after the map, so that the mapping is in place before the kernel.
          </Translated>
        ),
        why: (
          <Translated vi="Không có gì ở đây là bất đồng bộ. Ánh xạ đã có hiệu lực; chỉ là device chưa được phép dùng nó.">
            Nothing here is asynchronous. The mapping is in place; the device just may not use it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một cuMemMap thứ hai, lần này từ GPU 0, device sẽ đọc.">
            A second cuMemMap, this time from GPU 0, the device that will read.
          </Translated>
        ),
        why: (
          <Translated vi="Bộ nhớ đã nằm sau các địa chỉ. Ánh xạ lại không thay đổi gì về việc ai được dùng nó.">
            The memory is already behind the addresses. Mapping it again changes nothing about who may use it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="cuMemSetAccess, cấp cho GPU 0 quyền truy cập dải địa chỉ.">cuMemSetAccess, granting GPU 0 access to the range.</Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="cudaDeviceEnablePeerAccess cho GPU 0, để các kernel của nó được nạp từ ánh xạ.">
            cudaDeviceEnablePeerAccess for GPU 0, so that its kernels may load from the mapping.
          </Translated>
        ),
        why: (
          <Translated vi="Cái đó dành cho việc nạp từ các vùng cấp phát thông thường của một GPU khác, còn bộ nhớ ở đây nằm trên chính GPU 0.">
            {"That is for loads into another GPU's ordinary allocations, and the memory here is on GPU 0 itself."}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hướng dẫn của NVIDIA: ánh xạ "không làm cho địa chỉ truy cập được, và sẽ dẫn đến sập chương trình nếu một kernel
            CUDA truy cập vào". Danh sách ba lời gọi của sách bỏ sót lời gọi thứ tư. Địa chỉ, bộ nhớ và quyền truy cập là ba
            thứ tách biệt, mỗi thứ có lời gọi riêng.
          </p>
        }
      >
        <p>
          NVIDIA's guide: mapping "does not make the address accessible, and would result in a program crash if accessed by
          a CUDA kernel". The book's list of three calls leaves the fourth out. Address, memory and access are three
          separate things, each with its own call.
        </p>
      </Translated>
    ),
    source: 'CUDA guide §4.18.3.4, checked 2026-11-20 (notes/today/appC-facts.md, A15); the book, p. 629',
    vi: { source: 'Hướng dẫn CUDA §4.18.3.4, kiểm tra ngày 2026-11-20 (notes/today/appC-facts.md, A15); sách, p. 629' },
  },
  'appC/C.9/q-copied': {
    kind: 'numeric',
    id: 'appC/C.9/q-copied',
    prompt: (
      <Translated
        vi={
          <>
            Một buffer lớn dần mỗi lần một chunk, từ 1 chunk đến 16 chunk. Nó bắt đầu với dung lượng 1 chunk và nhân đôi dung
            lượng mỗi khi đầy, mỗi lần đều sao chép nội dung sang buffer mới. Tổng cộng có bao nhiêu chunk được sao chép?
          </>
        }
      >
        <>
          A buffer grows one chunk at a time from 1 chunk to 16 chunks. It starts with a capacity of 1 chunk and doubles its
          capacity whenever it is full, copying its contents into the new buffer each time. How many chunks are copied in
          all?
        </>
      </Translated>
    ),
    answer: 15,
    unit: 'chunks',
    explain: (
      <Translated
        vi={
          <p>
            Nó dời đi khi chunk thứ 2, 3, 5 và 9 đến, sao chép 1, 2, 4 và 8 chunk: 15, ít hơn kích thước cuối cùng một. Với
            một dải địa chỉ đặt trước và mỗi lần ánh xạ một chunk, đáp án là 0. Để thấy điều đó trong trình khám phá, đặt 2048
            KiB mỗi token, chunk 2 MiB và 16 token: sao chép 30 MiB, tức là 15 chunk.
          </p>
        }
      >
        <p>
          It moves when the 2nd, 3rd, 5th and 9th chunk arrive, copying 1, 2, 4 and 8 chunks: 15, one less than the final
          size. With a reserved range and one chunk mapped at a time, the answer is 0. To see it in the explorer, set 2048
          KiB per token, a 2 MiB chunk and 16 tokens: 30 MiB copied, which is 15 chunks.
        </p>
      </Translated>
    ),
    source: 'Ours: the arithmetic of doubling; src/sim/vmmgrow.ts',
    vi: { unit: 'chunk', source: 'Của chúng tôi: phép tính của việc nhân đôi; src/sim/vmmgrow.ts' },
  },
  'appC/C.9/q-pointer': {
    kind: 'choice',
    id: 'appC/C.9/q-pointer',
    prompt: (
      <Translated
        vi={
          <>
            Một buffer nằm trong một dải địa chỉ đặt trước. Bạn ánh xạ một chunk mới ngay sau chunk cuối. Các kernel đang giữ
            con trỏ gốc của buffer có cần con trỏ mới không?
          </>
        }
      >
        <>
          A buffer lives in a reserved range. You map a new chunk right after the last one. Do kernels that hold the
          buffer's base pointer need a new pointer?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Có: mỗi lần nới rộng ánh xạ bộ nhớ ở địa chỉ mới, nên gốc của buffer bị dời và kernel phải được đưa gốc mới.">
            {"Yes: each growth maps memory at new addresses, so the buffer's base moves and kernels must be given the new one."}
          </Translated>
        ),
        why: (
          <Translated vi="Đó là chiến lược sao chép, nơi dữ liệu chuyển sang một vùng cấp phát mới.">
            That is the copying strategy, where the data moves to a new allocation.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không: địa chỉ gốc giữ nguyên; chỉ là có thêm phần của dải địa chỉ có bộ nhớ phía sau.">
            No: the base address stays; more of the range just has memory behind it.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Có, khi chunk mới nằm trên một GPU khác, có địa chỉ riêng cho bộ nhớ của nó.">
            Yes when the new chunk is on another GPU, whose memory has addresses of its own.
          </Translated>
        ),
        why: (
          <Translated vi="Địa chỉ vẫn nằm trong cùng dải đã đặt trước. Một con trỏ có thể trải qua các chunk trên những device khác nhau, đó chính là ví dụ của sách.">
            {"The address is still inside the same reserved range. One pointer can span chunks on different devices, which is the book's example."}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đặt trước là điều làm cho cách này hoạt động. NVIDIA mô tả cách dùng này là nới rộng một vùng cấp phát giống như
            realloc hay <code>std::vector</code> nới rộng bộ nhớ CPU, mà không dời dữ liệu. Chunk mới vẫn cần lần cấp quyền
            truy cập riêng của nó.
          </p>
        }
      >
        <p>
          Reserving first is what makes this work. NVIDIA describes the use as growing an allocation the way realloc or{' '}
          <code>std::vector</code> grows CPU memory, without moving data. The new chunk does need its own access grant.
        </p>
      </Translated>
    ),
    source: 'NVIDIA, "Introducing Low-Level GPU Virtual Memory Management" and guide §4.18, checked 2026-11-20 (appC-facts.md, A15)',
    vi: {
      source: 'NVIDIA, "Introducing Low-Level GPU Virtual Memory Management" và hướng dẫn §4.18, kiểm tra ngày 2026-11-20 (appC-facts.md, A15)',
    },
  },
  'appC/C.9/q-peer': {
    kind: 'choice',
    id: 'appC/C.9/q-peer',
    prompt: (
      <Translated
        vi={
          <>
            PyTorch chạy với <code>expandable_segments</code> trên một máy hai GPU, và chương trình dựa vào{' '}
            <code>cudaDeviceEnablePeerAccess</code> để nạp trực tiếp giữa các GPU. Mã nguồn của chính PyTorch cảnh báo điều gì?
          </>
        }
      >
        <>
          PyTorch runs with <code>expandable_segments</code> on a two-GPU machine, and the program relies on{' '}
          <code>cudaDeviceEnablePeerAccess</code> for direct loads between the GPUs. What does PyTorch's own source warn?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không có gì thay đổi: một lời gọi cudaDeviceEnablePeerAccess phủ mọi vùng cấp phát trên GPU kia, kể cả bộ nhớ được ánh xạ bằng cuMemMap.">
            {'Nothing changes: one cudaDeviceEnablePeerAccess call covers every allocation on the peer, memory mapped with cuMemMap included.'}
          </Translated>
        ),
        why: (
          <Translated vi="Bộ nhớ được ánh xạ bằng cuMemMap không phải bộ nhớ device thông thường.">
            Memory mapped with cuMemMap is not ordinary device memory.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="expandable_segments tự bật peer access cho mọi GPU mà nó đặt trước bộ nhớ.">
            expandable_segments turns peer access on by itself for every GPU it reserves memory on.
          </Translated>
        ),
        why: (
          <Translated vi="Chú thích nói điều ngược lại: lời gọi peer của runtime không chạm tới bộ nhớ này.">
            The comment says the opposite kind of thing: the runtime peer call does not reach this memory.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="cudaDeviceEnablePeerAccess không chạm tới bộ nhớ được cấp phát bằng cuMemMap; hãy cấp quyền cho GPU kia bằng cuMemSetAccess.">
            {'cudaDeviceEnablePeerAccess does not reach memory allocated with cuMemMap; grant the peer access with cuMemSetAccess.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nạp peer nhanh hơn với bộ nhớ VMM, vì các chunk ánh xạ lớn của nó hợp với đường truyền.">
            Peer loads are faster with VMM memory, because its large mapped chunks suit the link.
          </Translated>
        ),
        why: (
          <Translated vi="Chú thích nói về việc truy cập có hoạt động hay không, không phải tốc độ.">
            The comment is about whether access works at all, not speed.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Chú thích thiết kế của allocator liệt kê điều này như một giới hạn: các API runtime để chia sẻ bộ nhớ như{' '}
            <code>cudaDeviceEnablePeerAccess</code> "không hoạt động với bộ nhớ được cấp phát bằng cuMemMap". Đó là nửa còn
            lại của <a href={href('appC', 'C.5')}>§C.5</a>: với VMM, quyền truy cập được cấp theo từng dải bằng{' '}
            <code>cuMemSetAccess</code>.
          </p>
        }
      >
        <p>
          The allocator's design comment lists it as a limit: runtime APIs for sharing memory such as{' '}
          <code>cudaDeviceEnablePeerAccess</code> "do not work for memory allocated with cuMemMap". It is the other half of{' '}
          <a href={href('appC', 'C.5')}>§C.5</a>: with VMM, access is granted per range with <code>cuMemSetAccess</code>.
        </p>
      </Translated>
    ),
    source: 'PyTorch c10/cuda/CUDACachingAllocator.cpp, checked 2026-11-20 (appC-facts.md, D2, A8)',
    vi: { source: 'PyTorch c10/cuda/CUDACachingAllocator.cpp, kiểm tra ngày 2026-11-20 (appC-facts.md, D2, A8)' },
  },
};

// ---------------------------------------------------------------- the section
export function SC9() {
  return (
    <>
      <Translated
        vi={
          <p>
            Phụ lục kết thúc bằng quyền kiểm soát chính không gian địa chỉ. Theo lời kể của sách, CUDA 11 thêm ba lời gọi cấp
            thấp. <code>cuMemAddressReserve</code> đặt trước một dải địa chỉ ảo. <code>cuMemCreate</code> cấp phát bộ nhớ
            vật lý trên bất kỳ device nào. <code>cuMemMap</code> đặt bộ nhớ đó phía sau bất kỳ vị trí nào trong dải đã đặt
            trước. Cùng nhau, chúng cho phép những cách bố trí mà một <code>cudaMalloc</code> đơn lẻ không làm được: chẳng hạn,
            một khối 3D có các phần nằm trên nhiều device nhưng kernel truy cập được qua một con trỏ. Sách không nêu năm và
            không đưa mã.
          </p>
        }
      >
        <p>
          The appendix ends with control over the address space itself. In the book's account, CUDA 11 added three
          low-level calls. <code>cuMemAddressReserve</code> reserves a range of virtual addresses. <code>cuMemCreate</code>{' '}
          allocates physical memory on any device. <code>cuMemMap</code> puts that memory behind any position in the
          reserved range. Together they allow layouts that a single <code>cudaMalloc</code> cannot give: for example, one 3D
          volume whose parts live on several devices but which a kernel reaches through one pointer. The book gives no year
          and no code.
        </p>
      </Translated>
      <Source section="C.9" pages="629" />
      <Translated
        vi={
          <p>
            Hai chỗ cần sửa. Các lời gọi này xuất hiện trong CUDA 10.2, tháng 11 năm 2019, không phải CUDA 11. Và danh sách
            thiếu một lời gọi: ánh xạ bộ nhớ vào một dải địa chỉ không làm cho nó dùng được. Một kernel chạm tới nó sẽ sập cho
            đến khi <code>cuMemSetAccess</code> cấp quyền truy cập cho device. Việc dỡ bỏ chạy theo thứ tự ngược: bỏ ánh xạ,
            giải phóng bộ nhớ, giải phóng địa chỉ. NVIDIA nói các lời gọi này yêu cầu dùng trực tiếp Driver API, và một device
            cho biết nó có hỗ trợ chúng hay không qua <code>CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED</code>.
          </p>
        }
      >
        <p>
          Two corrections. The calls arrived in CUDA 10.2, in November 2019, not CUDA 11. And the list is one call short:
          mapping memory into a range does not make it usable. A kernel that touches it crashes until{' '}
          <code>cuMemSetAccess</code> grants the device access. Teardown runs in reverse: unmap, release the memory, free the
          addresses. NVIDIA says these calls require the Driver API directly, and a device says whether it supports them
          through <code>CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED</code>.
        </p>
      </Translated>
      <Today
        checked={tr(
          "2026-11-20 against NVIDIA's VMM post, the CUDA 13.5 guide §4.18 and the Driver API reference (appC-facts.md, A15)",
          '2026-11-20 đối chiếu với bài viết về VMM của NVIDIA, hướng dẫn CUDA 13.5 §4.18 và tài liệu tham chiếu Driver API (appC-facts.md, A15)',
        )}
      >
        <Translated vi={<p>CUDA 10.2 (tháng 11 năm 2019), không phải CUDA 11.</p>}>
          <p>CUDA 10.2 (November 2019), not CUDA 11.</p>
        </Translated>
        <Translated vi={<p>"cuMemMap không làm cho địa chỉ truy cập được": cần có <Id>cuMemSetAccess</Id>.</p>}>
          <p>"cuMemMap does not make the address accessible": <Id>cuMemSetAccess</Id> is required.</p>
        </Translated>
      </Today>
      <Code
        tabs={[
          {
            label: 'CUDA Driver API', // i18n-ignore: API name
            code: SIGS_CUDA,
            status: tr('quoted from the reference; not compiled here: unverified', 'trích từ tài liệu tham chiếu; chưa biên dịch ở đây: unverified'),
          },
          {
            label: 'HIP',
            code: SIGS_HIP,
            status: tr('quoted from the ROCm 10.2 header; not compiled here: unverified', 'trích từ header ROCm 10.2; chưa biên dịch ở đây: unverified'),
          },
        ]}
        caption={
          <Translated
            vi={
              <>
                labs/appC/snippets/c9_vmm_decls.h và c9_vmm_decls_hip.h: các khai báo như tài liệu tham chiếu Driver API của
                NVIDIA (v13.5) và header ROCm 10.2 của AMD đưa ra, được ghi trong notes/today/appC-facts.md (A15, B7); các chú
                thích là của chúng tôi. Không có gì trên trang này gọi chúng: kiểm tra front-end CUDA của khóa học không có
                Driver API, và không lab nào dùng các lời gọi này.
              </>
            }
          >
            <>
              labs/appC/snippets/c9_vmm_decls.h and c9_vmm_decls_hip.h: declarations as NVIDIA's Driver API reference
              (v13.5) and AMD's ROCm 10.2 header give them, recorded in notes/today/appC-facts.md (A15, B7); the comments
              are ours. Nothing on this page calls them: the course's CUDA front-end check has no Driver API, and no lab
              uses these calls.
            </>
          </Translated>
        }
      />

      <h2>
        <Translated vi="Đặt trước, tạo, ánh xạ, cấp quyền">Reserve, create, map, grant</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trình chạy từng bước đưa cho bạn một dải {N} granule và bộ nhớ của hai GPU. Granule là đơn vị mà các lời gọi này
            làm việc; ở đây nó là một tham số, không phải giá trị lấy từ device của bạn. Hãy bắt đầu từ ba lời gọi của sách:
            chúng đặt trước, tạo và ánh xạ, rồi một kernel đọc. Tìm xem vì sao nó sập và làm cho nó chạy được. Sau đó nạp ví
            dụ của chính sách, một dải có nửa đầu do GPU 0 cung cấp bộ nhớ và nửa sau do GPU 1, và khởi chạy một kernel trên
            mỗi GPU.
          </p>
        }
      >
        <p>
          The stepper gives you a range of {N} granules and the memory of two GPUs. A granule is the unit these calls work
          in; here it is a parameter, not a value from your device. Start from the book's three calls: they reserve, create
          and map, and then a kernel reads. Find out why it crashes and make it work. Then load the book's own example, a
          range whose first half is backed by GPU 0 and second half by GPU 1, and launch a kernel on each GPU.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Granule là một tham số, không phải giá trị của device. Chỉ thứ tự được ghi nhận mới được áp đặt, không phải điều
              driver làm khi ra ngoài thứ tự đó. Không đo thời gian. Một chunk chỉ được ánh xạ ở một chỗ.
            </p>
          }
        >
          <p>
            The granule is a parameter, not a device value. Only the documented order is enforced, not what a driver does
            out of it. No timing. A chunk is mapped at one place only.
          </p>
        </Translated>
      </Simplification>
      <VmmStepper />
      <Ask q={questionsSC9['appC/C.9/q-setaccess']} />
      <Translated
        vi={
          <p>
            Vậy là ba tài nguyên, mỗi thứ có lời gọi riêng: địa chỉ, bộ nhớ và quyền truy cập. Giữ chúng tách biệt là điều cho
            phép một con trỏ trải trên hai device, và cũng là điều cho phép một buffer lớn dần mà không phải dời đi.
          </p>
        }
      >
        <p>
          Three resources, then, each with its own call: addresses, memory and access. Keeping them apart is what lets one
          pointer span two devices, and also what lets a buffer grow without moving.
        </p>
      </Translated>

      <h2>
        <Translated vi="Một buffer lớn dần mà không phải dời đi">A buffer that grows without moving</Translated>
      </h2>
      <Translated
        vi={
          <p>
            NVIDIA giờ gọi tên cách dùng quan trọng nhất với bạn: các allocator và cache "như quản lý động KV cache trong các
            mô hình ngôn ngữ lớn". Một KV cache có thêm một mục cho mỗi token được sinh ra, và kích thước cuối cùng của nó
            không được biết khi bắt đầu sinh. Với cấp phát thông thường, lớn dần nghĩa là một buffer mới, lớn hơn, và một lần
            sao chép. Với một dải địa chỉ đặt trước, nó nghĩa là ánh xạ thêm một chunk sau chunk cuối.
          </p>
        }
      >
        <p>
          NVIDIA now names the use that matters most to you: allocators and caches "such as dynamically managing key-value
          caches in large language models". A KV cache gains one entry per generated token, and its final size is not known
          when generation starts. With ordinary allocations, growth means a new, larger buffer and a copy. With a reserved
          range, it means mapping one more chunk after the last.
        </p>
      </Translated>
      <NoteGroup>
        <Today
          checked={tr(
            "2026-11-20 against NVIDIA's VMM post and guide §4.18 (appC-facts.md, A15)",
            '2026-11-20 đối chiếu với bài viết về VMM và hướng dẫn §4.18 của NVIDIA (appC-facts.md, A15)',
          )}
        >
          <Translated
            vi={<p>NVIDIA so sánh việc lớn dần này với realloc hay std::vector, mà không dời dữ liệu, và nói NCCL và NVSHMEM dùng VMM.</p>}
          >
            <p>NVIDIA compares the growth to realloc or std::vector, without moving data, and says NCCL and NVSHMEM use VMM.</p>
          </Translated>
        </Today>
        <Simplification>
          <Translated vi={<p>Hoạt hình: một chuỗi 32 token, 4 token mỗi chunk, trên một device, không đo thời gian.</p>}>
            <p>The animation: one sequence of 32 tokens, 4 to a chunk, on one device, with nothing timed.</p>
          </Translated>
        </Simplification>
      </NoteGroup>
      <GrowBufferScene />
      <Translated
        vi={
          <p>
            Trình khám phá chạy cả ba chiến lược trên một kích thước cache thật: ví dụ Llama 2 7B của sách, 512 KiB key và
            value mỗi token ở 16 bit, nên 4096 token chiếm 2 GiB. Hãy chốt một con số trước khi mở nó.
          </p>
        }
      >
        <p>
          The explorer runs all three strategies on a real cache size: the book's Llama 2 7B example, 512 KiB of keys and
          values per token at 16 bits, so 4096 tokens hold 2 GiB. Commit to a number before you open it.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một chuỗi trên một device. Cấp phát và ánh xạ không được đo thời gian, và không có vùng cấp phát nào khác làm
              phân mảnh bộ nhớ. Địa chỉ cho các chiến lược sao chép đến từ một allocator không bao giờ dùng lại lỗ hổng đã giải
              phóng.
            </p>
          }
        >
          <p>
            One sequence on one device. Allocation and mapping are not timed, and there are no other allocations to
            fragment memory. Addresses for the copying strategies come from an allocator that never reuses a freed hole.
          </p>
        </Translated>
      </Simplification>
      <Predict
        id="appC/C.9/p-double-copy"
        prompt={
          <Translated
            vi={
              <>
                Cache lớn dần đến 4096 token, mỗi token 512 KiB, tổng cộng 2 GiB. Nó bắt đầu là một buffer 2 MiB và nhân đôi
                mỗi khi đầy, mỗi lần đều sao chép mọi thứ nó đang giữ. Trong toàn bộ quá trình, nó sao chép bao nhiêu GiB?
              </>
            }
          >
            <>
              The cache grows to 4096 tokens of 512 KiB each, 2 GiB in all. It starts as one 2 MiB buffer and doubles
              whenever it is full, copying everything it holds each time. How many GiB does it copy over the whole run?
            </>
          </Translated>
        }
        unit="GiB"
      >
        {(v) => {
          const d = grow('double', { tokens: 4096, bytesPerToken: LLAMA2_7B_BYTES_PER_TOKEN, chunk: 2 * MiB });
          const gib = d.copiedTotal / 1024 ** 3;
          return (
            <Translated
              vi={
                <p>
                  Mô hình sao chép {d.copiedTotal / MiB} MiB, tức là {Number(gib.toFixed(3))} GiB: 2 + 4 + … + 1024 MiB, thiếu
                  một chunk so với kích thước cuối cùng.{' '}
                  {Math.abs(v - gib) < 0.25
                    ? 'Đúng như bạn đã đoán.'
                    : v > 4
                      ? 'Mỗi lần sao chép chỉ chuyển lượng buffer đang giữ vào lúc đó, và các lượng này giảm một nửa khi lùi về trước, nên cộng lại ít hơn kích thước cuối cùng.'
                      : 'Mỗi lần nhân đôi sao chép bằng tất cả các lần trước cộng lại, cộng thêm một chunk, nên tổng rơi ngay dưới kích thước cuối cùng.'}{' '}
                  Cấp phát vừa khít sẽ sao chép khoảng 4 TiB; đặt trước rồi ánh xạ không sao chép gì.
                </p>
              }
            >
              <p>
                The model copies {d.copiedTotal / MiB} MiB, which is {Number(gib.toFixed(3))} GiB: 2 + 4 + … + 1024 MiB,
                one chunk short of the final size.{' '}
                {Math.abs(v - gib) < 0.25
                  ? 'That is what you had.'
                  : v > 4
                    ? 'Each copy moves only what the buffer held at that moment, and those amounts halve going back in time, so they add up to less than the final size.'
                    : 'Each doubling copies as much as all the earlier copies together, plus one chunk, so the total lands just under the final size.'}{' '}
                Exact fit would copy about 4 TiB; reserve-and-map copies nothing.
              </p>
            </Translated>
          );
        }}
      </Predict>
      <GrowKV />
      <Ask q={questionsSC9['appC/C.9/q-copied']} />
      <Ask q={questionsSC9['appC/C.9/q-pointer']} />

      <h2>
        <Translated vi="Bạn gặp nó ở đâu">Where you meet it</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Caching allocator CUDA của PyTorch có một tùy chọn làm đúng việc này. Với <code>expandable_segments</code> được đặt
            trong <code>PYTORCH_CUDA_ALLOC_CONF</code> (thử nghiệm, mặc định tắt), nó đặt trước không gian địa chỉ cho khoảng
            toàn bộ bộ nhớ GPU và ánh xạ bộ nhớ vật lý vào đó khi chương trình cần, qua <code>cuMemCreate</code>,{' '}
            <code>cuMemAddressReserve</code>, <code>cuMemMap</code> và <code>cuMemSetAccess</code>, với đơn vị 2 MiB theo chú
            thích của chính nó. Cũng chú thích đó nêu cái giá: lời gọi peer-access của runtime không chạm tới bộ nhớ được ánh
            xạ theo cách này.
          </p>
        }
      >
        <p>
          PyTorch's CUDA caching allocator has an option that does this. With <code>expandable_segments</code> set in{' '}
          <code>PYTORCH_CUDA_ALLOC_CONF</code> (experimental, off by default), it reserves address space for about the whole
          GPU's memory and maps physical memory into it as the program needs it, through <code>cuMemCreate</code>,{' '}
          <code>cuMemAddressReserve</code>, <code>cuMemMap</code> and <code>cuMemSetAccess</code>, at a 2 MiB granularity
          by its own comment. The same comment states the price: the runtime's peer-access call does not reach memory mapped
          this way.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>PyTorch: c10/cuda/CUDACachingAllocator.cpp, chú thích thiết kế (appC-facts.md, D2).</p>}>
          <p>PyTorch: c10/cuda/CUDACachingAllocator.cpp, design comment (appC-facts.md, D2).</p>
        </Translated>
        <Translated
          vi={
            <p>
              Backend CUDA của llama.cpp giữ một VMM pool đặt trước 32 GB địa chỉ, mặc định bật cho CUDA và mặc định tắt cho
              HIP (<Id>GGML_HIP_NO_VMM</Id>, D3).
            </p>
          }
        >
          <p>
            llama.cpp's CUDA backend keeps a VMM pool that reserves 32 GB of addresses, on by default for CUDA and off by
            default for HIP (<Id>GGML_HIP_NO_VMM</Id>, D3).
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsSC9['appC/C.9/q-peer']} />
      <Translated
        vi={
          <p>
            Bản thân sách cũng giải quyết cùng vấn đề đó bên trong kernel. Chương về attention của sách mô tả PagedAttention, kỹ
            thuật chia KV cache thành các khối kích thước bằng nhau vì một cache được đặt trước cho đầu vào dài nhất có thể sẽ
            để lại bộ nhớ bị phân mảnh và lãng phí. VMM giữ một dải địa chỉ liền mạch và lấp đầy nó dần; PagedAttention, lấy cảm
            hứng từ cơ chế phân trang của hệ điều hành, làm việc với các khối kích thước cố định được đưa vào bộ nhớ GPU khi phép
            tính cần đến. Xem <a href={href('ch20', '20.7')}>chương 20, §20.7</a>.
          </p>
        }
      >
        <p>
          The book itself takes the same problem on inside the kernel. Its chapter on attention describes PagedAttention,
          which splits the KV cache into equal-size blocks because a cache reserved for the longest possible input leaves
          memory fragmented and wasted. VMM keeps one contiguous range of addresses and fills it in; PagedAttention, inspired
          by an operating system's paging, works in fixed-size blocks that are brought into GPU memory when the
          computation needs them. See{' '}
          <a href={href('ch20', '20.7')}>chapter 20, §20.7</a>.
        </p>
      </Translated>
      <Source section="20.7" pages="509" />

      <h2>
        <Translated vi="Trên GPU AMD và Apple">On AMD and Apple GPUs</Translated>
      </h2>
      <Translated
        vi={
          <p>
            HIP có cùng bốn bước dưới các tên của runtime API, <code>hipMemAddressReserve</code>, <code>hipMemCreate</code>,{' '}
            <code>hipMemMap</code> và <code>hipMemSetAccess</code>, xây trên HSA, trên Linux (AMD liệt kê Windows là đang phát
            triển). Một kích thước không được căn chỉnh khiến <code>hipMemCreate</code> trả về lỗi hết bộ nhớ. Bản build HIP
            của llama.cpp ánh xạ mỗi lời gọi <code>cuMem</code> sang tên <code>hipMem</code> của nó.
          </p>
        }
      >
        <p>
          HIP has the same four steps under runtime-API names, <code>hipMemAddressReserve</code>, <code>hipMemCreate</code>,{' '}
          <code>hipMemMap</code> and <code>hipMemSetAccess</code>, built on HSA, on Linux (AMD lists Windows as under
          development). A size that is not aligned makes <code>hipMemCreate</code> return an out-of-memory error.
          llama.cpp's HIP build maps each <code>cuMem</code> call to its <code>hipMem</code> name.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Kiểm tra ngày 2026-11-20 đối chiếu với trang bộ nhớ ảo HIP của AMD, header ROCm 10.2 và vendors/hip.h của
              llama.cpp (appC-facts.md, B7, D3).
            </p>
          }
        >
          <p>Checked 2026-11-20 against AMD's HIP virtual memory page, the ROCm 10.2 header and llama.cpp's vendors/hip.h (appC-facts.md, B7, D3).</p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Metal 4 có một thứ tương tự gần gũi cho một device, dù không có gì trải trên nhiều device. Một placement sparse
            buffer đóng vai dải địa chỉ đặt trước, một placement heap đóng vai bộ nhớ từ <code>cuMemCreate</code>, và{' '}
            <code>updateBufferMappings</code> trên một <code>MTL4CommandQueue</code> đóng vai <code>cuMemMap</code>. Không có
            bước cấp quyền truy cập. Phép đối chiếu này là của chúng tôi và chúng tôi chưa chạy nó. Một cái bẫy: header ghi hàm
            tạo buffer và phương thức của queue là macOS 26.0, nhưng <code>supportsPlacementSparse</code> là 26.4, nên hãy kiểm
            tra thuộc tính đó lúc chạy trước khi dựa vào bất cứ điều gì ở đây.
          </p>
        }
      >
        <p>
          Metal 4 has a close analogue for one device, though nothing that spans devices. A placement sparse buffer plays
          the reserved range, a placement heap plays the memory from <code>cuMemCreate</code>, and{' '}
          <code>updateBufferMappings</code> on an <code>MTL4CommandQueue</code> plays <code>cuMemMap</code>. There is no
          access step. The mapping is ours and we have not run it. One trap: the header gives the buffer constructor and
          the queue method as macOS 26.0, but <code>supportsPlacementSparse</code> as 26.4, so check that property at run
          time before relying on any of it.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Metal: <Id>newBufferWithLength:options:placementSparsePageSize:</Id>, <Id>MTLHeap</Id> dạng placement,{' '}
              <Id>updateBufferMappings:heap:operations:count:</Id>. Kiểm tra ngày 2026-11-20 đối chiếu với các header của macOS
              27.0 SDK (appC-facts.md, C6).
            </p>
          }
        >
          <p>
            Metal: <Id>newBufferWithLength:options:placementSparsePageSize:</Id>, placement <Id>MTLHeap</Id>,{' '}
            <Id>updateBufferMappings:heap:operations:count:</Id>. Checked 2026-11-20 against the macOS 27.0 SDK headers
            (appC-facts.md, C6).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Quay lại <a href={href('appC', 'start')}>bản đồ</a>: bước này là C.9. Một kernel chỉ dùng được một dải đã ánh xạ
            khi đã được cấp quyền truy cập, và các nguồn của chúng tôi không nói mã host có được dereference nó hay không.
          </p>
        }
      >
        <p>
          Back to <a href={href('appC', 'start')}>the map</a>: this step is C.9. A kernel can use a mapped range only once
          access is granted, and our sources do not say whether host code may dereference one.
        </p>
      </Translated>
    </>
  );
}
