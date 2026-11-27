import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useProgress } from '../../lib/progress';
import { fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BOOK_LIMITS,
  BOOK_SHAPE,
  MAX_REGS_PER_THREAD,
  budget,
  regsForBlocks,
  type LimitKey,
  type Limits,
  type Shape,
} from '../../sim/gemmbudget';

const KB = 1024;
const count = (v: number) => v.toLocaleString('en-US');
const bytes = (v: number) => (v >= KB ? `${count(v)} B (${fmt(v / KB)} KB)` : `${count(v)} B`); // i18n-ignore: numbers and units only

/** A long identifier that may break anywhere, for the narrow margin notes. */
function Id({ children }: { children: ReactNode }) {
  return <code style={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>{children}</code>;
}

// ---------------------------------------------------------------- devices
interface Device {
  key: string;
  label: string;
  kind: 'book' | 'gpu' | 'metal' | 'custom';
  backend?: string;
  limits: Limits;
  note: ReactNode;
}

const BOOK: Device = {
  key: 'book',
  get label() {
    return tr("The book's example SM", 'SM ví dụ của sách');
  },
  kind: 'book',
  limits: BOOK_LIMITS,
  note: (
    <Translated
      vi={
        <>
          Số liệu của sách: 65,536 thanh ghi, tối đa 255 mỗi luồng và 2048 luồng thường trú mỗi SM (§15.7, p. 363), và
          228 KB bộ nhớ chia sẻ mỗi SM trên thiết bị ví dụ của nó (§5.6, p. 116).
        </>
      }
    >
      <>
        The book's figures: 65,536 registers, at most 255 per thread and 2048 resident threads per SM (§15.7, p. 363),
        and 228 KB of shared memory per SM on its example device (§5.6, p. 116).
      </>
    </Translated>
  ),
};

const positive = (v: unknown): number => {
  const n = Number(v);
  return isFinite(n) && n > 0 ? Math.floor(n) : 0;
};

/** The reader's CUDA and HIP devices that report registers and thread slots per SM, and Metal devices. */
function reported(reports: Reports): Device[] {
  return realBackends(reports).flatMap((b): Device[] => {
    const d = reports[b].device;
    const name = String(d.name ?? b);
    const when = reports[b].generated.slice(0, 10);
    if ('maxThreadgroupMemoryLength' in d)
      return [
        {
          key: b,
          get label() {
            return tr(`Yours: ${name} (${b})`, `Của bạn: ${name} (${b})`);
          },
          kind: 'metal',
          backend: b,
          limits: { regsPerSm: 0, threadsPerSm: 0, maxRegsPerThread: 0, sharedPerBlock: positive(d.maxThreadgroupMemoryLength), sharedPerSm: 0 },
          note: (
            <Translated
              vi={
                <>
                  Những gì thiết bị này báo cáo khi các lab ghi nhận nó lần gần nhất ({when}):{' '}
                  <code>maxThreadgroupMemoryLength</code>. Apple không công bố số thanh ghi, tệp thanh ghi hay giới hạn
                  luồng mỗi lõi, nên chỉ bộ nhớ chia sẻ được kiểm tra (ch15-facts C3).
                </>
              }
            >
              <>
                What this device reported when the labs last recorded it ({when}): <code>maxThreadgroupMemoryLength</code>.
                Apple documents no register count, no register file and no per-core thread limit, so only shared memory
                is checked (ch15-facts C3).
              </>
            </Translated>
          ),
        },
      ];
    if (!('regsPerMultiprocessor' in d) || !('maxThreadsPerMultiProcessor' in d)) return [];
    return [
      {
        key: b,
        get label() {
          return tr(`Yours: ${name} (${b})`, `Của bạn: ${name} (${b})`);
        },
        kind: 'gpu',
        backend: b,
        limits: {
          regsPerSm: positive(d.regsPerMultiprocessor),
          threadsPerSm: positive(d.maxThreadsPerMultiProcessor),
          maxRegsPerThread: MAX_REGS_PER_THREAD[b] ?? 0,
          sharedPerBlock: positive(d.sharedMemPerBlock),
          sharedPerSm: positive(d.sharedMemPerMultiprocessor),
        },
        note: (
          <Translated
            vi={
              <>
                Những gì thiết bị này báo cáo khi các lab ghi nhận nó lần gần nhất ({when}). Mức tối đa mỗi luồng là số
                liệu nhà cung cấp công bố cho{' '}
                {b === 'hip' ? 'AMD (256 thanh ghi vector, ch15-facts B3)' : 'NVIDIA (255, ch04-facts A12)'}.
                {b === 'hip'
                  ? ' AMD cấp phát thanh ghi theo wavefront, từng khối hạt (granule), nên tích theo luồng bên dưới chỉ đúng về hình dạng ở đó (ch04-facts B10).'
                  : ''}
              </>
            }
          >
            <>
              What this device reported when the labs last recorded it ({when}). The per-thread maximum is the
              vendor's documented figure for {b === 'hip' ? 'AMD (256 vector registers, ch15-facts B3)' : 'NVIDIA (255, ch04-facts A12)'}.
              {b === 'hip'
                ? ' AMD allocates registers per wavefront in granules, so the per-thread product below is only the right shape there (ch04-facts B10).'
                : ''}
            </>
          </Translated>
        ),
      },
    ];
  });
}

/**
 * What the chapter's labs printed about their own kernels on this backend (labs/ch15/shared/
 * stage_driver.hpp): on CUDA and HIP the row "occupancy" with blocks_per_sm from the occupancy
 * API (-1 when the runtime would not say); on Metal the row "pipeline" with the compiled
 * pipeline's thread cap and static threadgroup memory. Reference solutions only.
 */
function measured(reports: Reports, backend: string | undefined) {
  if (!backend || !reports[backend]) return [];
  return reports[backend].results
    .filter((r) => r.lab.startsWith('ch15/') && r.variant === 'solution')
    .flatMap((r) =>
      r.bench
        // Labs 15.1 to 15.5 write snake_case fields; lab 15.6 spells them as Metal does.
        .filter((row) => (num(row.blocks_per_sm) ?? -1) >= 0 || (num(row.max_threads_per_threadgroup) ?? num(row.maxTotalThreadsPerThreadgroup)) != null)
        .map((row) => ({
          lab: r.lab,
          blocks: num(row.blocks_per_sm),
          threads: num(row.threads_per_block),
          cap: num(row.max_threads_per_threadgroup) ?? num(row.maxTotalThreadsPerThreadgroup),
          staticBytes: num(row.static_threadgroup_bytes) ?? num(row.staticThreadgroupMemoryLength),
        })),
    );
}

// ---------------------------------------------------------------- the calculator
interface Preset {
  key: string;
  label: string;
  s: Partial<Shape>;
}
const PRESETS: Preset[] = [
  {
    key: 'book',
    get label() {
      return tr('The book: 8 × 8 at 255 registers', 'Sách: 8 × 8 với 255 thanh ghi');
    },
    s: { ...BOOK_SHAPE, extra: 175 },
  },
  {
    key: 'tiles',
    get label() {
      return tr('Tile registers only (80)', 'Chỉ thanh ghi của ô (80)');
    },
    s: { ...BOOK_SHAPE, extra: 0 },
  },
  {
    key: 'r128',
    get label() {
      return tr('If it fitted in 128', 'Nếu vừa trong 128');
    },
    s: { ...BOOK_SHAPE, extra: 48 },
  },
  {
    key: 'pipe',
    get label() {
      return tr('Padded, double-buffered (15.6, 15.8)', 'Có đệm, đệm kép (15.6, 15.8)');
    },
    s: { ...BOOK_SHAPE, extra: 175, pad: 1, buffers: 2 },
  },
];
type Field = 'tM' | 'tN' | 'extra' | 'threads' | 'bM' | 'bN' | 'bK' | 'pad';
const FIELDS: { key: Field; label: string; hint?: string }[] = [
  { key: 'tM', label: 'tM' },
  { key: 'tN', label: 'tN' },
  {
    key: 'extra',
    get label() {
      return tr('Extra registers', 'Thanh ghi thêm');
    },
    get hint() {
      return tr('addresses, counters, temporaries', 'địa chỉ, bộ đếm, giá trị tạm');
    },
  },
  {
    key: 'threads',
    get label() {
      return tr('Threads per block', 'Luồng mỗi khối');
    },
  },
  { key: 'bM', label: 'bM' },
  { key: 'bN', label: 'bN' },
  { key: 'bK', label: 'bK' },
  {
    key: 'pad',
    get label() {
      return tr('Padding columns on A_s', 'Số cột đệm trên A_s');
    },
  },
];
const LIMIT_NAME: Record<LimitKey, string> = {
  get registers() {
    return tr('registers', 'thanh ghi');
  },
  get threads() {
    return tr('thread slots', 'khe luồng');
  },
  get shared() {
    return tr('shared memory', 'bộ nhớ chia sẻ');
  },
};
const toInt = (s: string, min = 0) => {
  const n = Math.floor(Number(s));
  return isFinite(n) && n >= min ? n : min;
};
const CUSTOM_FIELDS: { key: keyof Limits; label: string }[] = [
  {
    key: 'regsPerSm',
    get label() {
      return tr('Registers per SM', 'Thanh ghi mỗi SM');
    },
  },
  {
    key: 'threadsPerSm',
    get label() {
      return tr('Thread slots per SM', 'Khe luồng mỗi SM');
    },
  },
  {
    key: 'maxRegsPerThread',
    get label() {
      return tr('Most registers per thread', 'Số thanh ghi tối đa mỗi luồng');
    },
  },
  {
    key: 'sharedPerSm',
    get label() {
      return tr('Shared memory per SM (bytes)', 'Bộ nhớ chia sẻ mỗi SM (byte)');
    },
  },
];

function BudgetBench() {
  const { reports, loading } = useReports();
  const devices = useMemo(() => [BOOK, ...reported(reports)], [reports]);
  const [key, setKey] = useState('book');
  const [custom, setCustom] = useState<Record<keyof Limits, string>>({
    regsPerSm: String(BOOK_LIMITS.regsPerSm),
    threadsPerSm: String(BOOK_LIMITS.threadsPerSm),
    maxRegsPerThread: String(BOOK_LIMITS.maxRegsPerThread),
    sharedPerBlock: '0',
    sharedPerSm: String(BOOK_LIMITS.sharedPerSm),
  });
  const [form, setForm] = useState<Record<Field, string>>(() => {
    const s = PRESETS[0].s as Shape;
    return { tM: String(s.tM), tN: String(s.tN), extra: String(s.extra), threads: String(s.threads), bM: String(s.bM), bN: String(s.bN), bK: String(s.bK), pad: String(s.pad) };
  });
  const [buffers, setBuffers] = useState<1 | 2>(1);

  const dev: Device =
    key === 'custom'
      ? {
          key: 'custom', label: tr('Custom numbers', 'Số tự nhập'), kind: 'custom',
          limits: {
            regsPerSm: toInt(custom.regsPerSm), threadsPerSm: toInt(custom.threadsPerSm), maxRegsPerThread: toInt(custom.maxRegsPerThread),
            sharedPerBlock: 0, sharedPerSm: toInt(custom.sharedPerSm),
          },
          note: (
            <Translated vi={<>Giới hạn bạn đã nhập. Chúng bắt đầu từ SM ví dụ của sách.</>}>
              <>Limits you typed. They start from the book's example SM.</>
            </Translated>
          ),
        }
      : (devices.find((d) => d.key === key) ?? BOOK);
  const shape: Shape = {
    tM: toInt(form.tM, 1), tN: toInt(form.tN, 1), extra: toInt(form.extra), threads: toInt(form.threads, 1),
    bM: toInt(form.bM, 1), bN: toInt(form.bN, 1), bK: toInt(form.bK, 1), pad: toInt(form.pad), buffers,
  };
  const b = budget(shape, dev.limits);
  const metal = dev.kind === 'metal';
  const lim = dev.limits;
  const runs = measured(reports, dev.backend);

  const apply = (p: Preset) => {
    const s = p.s as Shape;
    setForm({ tM: String(s.tM), tN: String(s.tN), extra: String(s.extra), threads: String(s.threads), bM: String(s.bM), bN: String(s.bN), bK: String(s.bK), pad: String(s.pad) });
    setBuffers(s.buffers);
  };
  const active = (p: Preset) => (Object.keys(p.s) as (keyof Shape)[]).every((k) => shape[k] === p.s[k]);
  const field = (label: string, value: string, set: (v: string) => void, hint?: string) => (
    <label className="field" key={label}>
      {label}
      <input type="number" min="0" step="1" value={value} aria-label={label} onChange={(e) => set(e.target.value)} />
      {hint ? <span style={{ fontSize: '0.74rem' }}>{hint}</span> : null}
    </label>
  );

  const names = b.binding.map((k) => LIMIT_NAME[k]);
  const list = names.length > 1 ? names.slice(0, -1).join(', ') + tr(' and ', ' và ') + names[names.length - 1] : (names[0] ?? '');
  let verdict: ReactNode;
  if (metal)
    verdict = (
      <Translated
        vi={
          <p style={{ margin: 0 }}>
            Apple không công bố số thanh ghi, kích thước tệp thanh ghi hay quy tắc bao nhiêu threadgroup dùng chung một
            lõi GPU, nên không thể tính số khối hay occupancy. Chỉ bộ nhớ chia sẻ có thể được kiểm tra, ở bên trái.
          </p>
        }
      >
        <p style={{ margin: 0 }}>
          Apple publishes no register count, no register file size and no rule for how many threadgroups share a GPU
          core, so no block count or occupancy can be computed. Only shared memory can be checked, on the left.
        </p>
      </Translated>
    );
  else if (b.blocks == null)
    verdict = (
      <Translated vi={<p style={{ margin: 0 }}>Nhập số thanh ghi mỗi SM và số khe luồng mỗi SM để có số khối.</p>}>
        <p style={{ margin: 0 }}>Enter registers per SM and thread slots per SM to get a block count.</p>
      </Translated>
    );
  else if (b.blocks === 0)
    verdict = (
      <Translated
        vi={
          <p style={{ margin: 0 }}>
            <span className="chip fail">không vừa</span> Không một khối {count(shape.threads)} luồng nào vừa: nó cần
            nhiều {list} hơn SM có. Trên CUDA, lần khởi chạy thất bại.
          </p>
        }
      >
        <p style={{ margin: 0 }}>
          <span className="chip fail">does not fit</span> Not one block of {count(shape.threads)} threads fits: it needs
          more {list} than the SM has. On CUDA the launch fails.
        </p>
      </Translated>
    );
  else
    verdict = (
      <p style={{ margin: 0 }}>
        <Translated vi={<b>{list.charAt(0).toUpperCase() + list.slice(1)} là giới hạn ràng buộc.</b>}>
          <b>{list.charAt(0).toUpperCase() + list.slice(1)} {b.binding.length > 1 || names[0] === 'thread slots' || names[0] === 'registers' ? 'bind' : 'binds'}.</b>
        </Translated>{' '}
        {b.binding.includes('registers') && b.blocks != null && lim.threadsPerSm > 0 && b.resident! < lim.threadsPerSm ? (
          <Translated
            vi={
              <>
                Để vừa thêm một khối, mỗi luồng sẽ phải xoay xở với{' '}
                <span data-out="next-regs">{count(regsForBlocks(lim.regsPerSm, shape.threads, b.blocks + 1))}</span> thanh ghi.
              </>
            }
          >
            <>
              To fit one more block, a thread would have to make do with{' '}
              <span data-out="next-regs">{count(regsForBlocks(lim.regsPerSm, shape.threads, b.blocks + 1))}</span> registers.
            </>
          </Translated>
        ) : b.resident === lim.threadsPerSm ? (
          <Translated vi={<>Mọi khe luồng đều đã được lấp đầy.</>}>
            <>Every thread slot is filled.</>
          </Translated>
        ) : null}
      </p>
    );

  return (
    <Bench
      title={tr('Register and shared-memory budget', 'Ngân sách thanh ghi và bộ nhớ chia sẻ')}
      task={
        <Translated vi="Bắt đầu từ kernel của sách, rồi tìm xem một luồng cần ít thanh ghi đến mức nào để hai khối vừa.">
          Start from the book's kernel, then find how few registers a thread would need for two blocks to fit.
        </Translated>
      }
      foot={
        <>
          {dev.note}{' '}
          <Translated vi="1 KB là 1024 byte, như trong phép tính của sách.">1 KB is 1024 bytes, as in the book's arithmetic.</Translated>
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-start' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Giới hạn của một SM">Limits of one SM</Translated>
          <select value={key} onChange={(e) => setKey(e.target.value)} aria-label={tr('Limits of one SM', 'Giới hạn của một SM')} style={{ maxWidth: '100%' }}>
            {devices.map((d) => (
              <option key={d.key} value={d.key}>{d.label}</option>
            ))}
            <option value="custom">{tr('Custom numbers', 'Số tự nhập')}</option>
          </select>
        </label>
        {key === 'custom' ? CUSTOM_FIELDS.map((f) => field(f.label, custom[f.key], (v) => setCustom({ ...custom, [f.key]: v }))) : null}
      </div>
      {!loading && devices.length === 1 ? (
        <Translated
          vi={
            <p className="muted" style={{ margin: '0 0 0.8rem' }}>
              Báo cáo của máy này chưa có GPU nào, nên danh sách chỉ có ví dụ của sách và số tự nhập. Hãy chạy một lab
              trên máy có GPU (ví dụ <code>python3 labs/pmpp.py run 15.6 --solution</code>) rồi tải lại để thấy các thiết
              bị của bạn ở đây.
            </p>
          }
        >
          <p className="muted" style={{ margin: '0 0 0.8rem' }}>
            No GPU in this machine's reports yet, so the list offers the book's example and custom numbers. Run a lab
            on a machine with a GPU (for example <code>python3 labs/pmpp.py run 15.6 --solution</code>) and reload to see
            your own devices here.
          </p>
        </Translated>
      ) : null}
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-start' }}>
        {FIELDS.map((f) => field(f.label, form[f.key], (v) => setForm({ ...form, [f.key]: v }), f.hint))}
        <label className="field">
          <Translated vi="Số buffer">Buffers</Translated>
          <select value={buffers} onChange={(e) => setBuffers(Number(e.target.value) as 1 | 2)} aria-label={tr('Buffers', 'Số buffer')}>
            <option value={1}>1</option>
            <option value={2}>{tr('2 (section 15.8)', '2 (mục 15.8)')}</option>
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '1rem' }} role="group" aria-label={tr('Presets', 'Cấu hình mẫu')}>
        <span className="muted">
          <Translated vi="Cấu hình mẫu">Presets</Translated>
        </span>
        {PRESETS.map((p) => (
          <button key={p.key} className="quiet" data-preset={p.key} aria-pressed={active(p)} onClick={() => apply(p)}>
            {p.label}
          </button>
        ))}
      </div>

      <div className="cols plot">
        <div className="panel" style={{ overflowX: 'auto' }}>
          <p style={{ margin: '0 0 0.6rem' }}>
            <span className="chip reg">
              <Translated vi="thanh ghi mỗi luồng">registers per thread</Translated>
            </span>{' '}
            <span data-out="regs" data-regs={b.regs} data-wanted={b.wanted}>
              <Translated
                vi={
                  <>
                    {shape.tM} × {shape.tN} + {shape.tM} + {shape.tN} = {count(b.tileRegs)} cho các ô, + {count(shape.extra)} ={' '}
                    <b>{count(b.wanted)}</b>
                  </>
                }
              >
                {shape.tM} × {shape.tN} + {shape.tM} + {shape.tN} = {count(b.tileRegs)} for the tiles, + {count(shape.extra)} ={' '}
                <b>{count(b.wanted)}</b>
              </Translated>
            </span>
            {b.spills ? (
              <Translated
                vi={
                  <>
                    {' '}
                    <span className="chip fail">vượt giới hạn</span> Một luồng chỉ được có tối đa {count(lim.maxRegsPerThread)};
                    trình biên dịch giữ phần còn lại trong bộ nhớ cục bộ (local memory), vốn là bộ nhớ của device.
                  </>
                }
              >
                <>
                  {' '}
                  <span className="chip fail">over the limit</span> A thread may have at most {count(lim.maxRegsPerThread)}; the
                  compiler keeps the rest in local memory, which is device memory.
                </>
              </Translated>
            ) : null}
          </p>
          {metal ? null : (
            <table className="data" style={{ margin: 0 }}>
              <thead>
                <tr>
                  <th style={{ paddingLeft: '0.5rem' }}>
                    <Translated vi="Giới hạn">Limit</Translated>
                  </th>
                  <th className="num">
                    <Translated vi="Một SM có">One SM has</Translated>
                  </th>
                  <th className="num">
                    <Translated vi="Một khối cần">One block needs</Translated>
                  </th>
                  <th className="num">
                    <Translated vi="Số khối vừa">Blocks that fit</Translated>
                  </th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {(
                  [
                    ['registers', <span className="chip reg"><Translated vi="thanh ghi">registers</Translated></span>, lim.regsPerSm, b.regs * shape.threads],
                    ['threads', tr('thread slots', 'khe luồng'), lim.threadsPerSm, shape.threads],
                    ['shared', <span className="chip shared"><Translated vi="bộ nhớ chia sẻ">shared memory</Translated></span>, lim.sharedPerSm, b.shared],
                  ] as [LimitKey, ReactNode, number, number][]
                ).map(([k, name, cap, need]) => {
                  const fit = b.byLimit[k];
                  const binds = b.binding.includes(k);
                  return (
                    <tr key={k} data-limit={k} data-fit={fit ?? 'none'} style={binds ? { background: 'var(--rule-soft)' } : undefined}>
                      <td style={{ paddingLeft: '0.5rem' }}>{name}</td>
                      <td className="num">{cap > 0 ? (k === 'shared' ? bytes(cap) : count(cap)) : tr('not known', 'không rõ')}</td>
                      <td className="num">{k === 'shared' ? bytes(need) : count(need)}</td>
                      <td className="num">{fit == null ? <span className="muted"><Translated vi="không tính">not counted</Translated></span> : count(fit)}</td>
                      <td style={{ paddingRight: '0.5rem' }}>{binds ? <b><Translated vi="ràng buộc">binds</Translated></b> : null}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
          <p className="muted" style={{ margin: '0.7rem 0 0' }} data-out="shared" data-bytes={b.shared}>
            <Translated
              vi={
                <>
                  Bộ nhớ chia sẻ mỗi khối: A_s {count(shape.bM)} × {count(shape.bK + shape.pad)} float = {bytes(b.sharedA)}, B_s{' '}
                  {count(shape.bK)} × {count(shape.bN)} float = {bytes(b.sharedB)}
                  {buffers === 2 ? ', nhân đôi' : ''}: <b>{bytes(b.shared)}</b>.
                </>
              }
            >
              Shared memory per block: A_s {count(shape.bM)} × {count(shape.bK + shape.pad)} floats = {bytes(b.sharedA)}, B_s{' '}
              {count(shape.bK)} × {count(shape.bN)} floats = {bytes(b.sharedB)}
              {buffers === 2 ? ', twice over' : ''}: <b>{bytes(b.shared)}</b>.
            </Translated>
            {lim.sharedPerBlock > 0 ? (
              <>
                {' '}
                <Translated vi={<>So với giới hạn mỗi khối của thiết bị này là {bytes(lim.sharedPerBlock)}:</>}>
                  Against this device's per-block limit of {bytes(lim.sharedPerBlock)}:
                </Translated>{' '}
                {b.fitsBlock ? (
                  <span className="chip ok" data-out="fits"><Translated vi="vừa">fits</Translated></span>
                ) : (
                  <span className="chip fail" data-out="fits"><Translated vi="không vừa">does not fit</Translated></span>
                )}
              </>
            ) : null}
          </p>
          {!b.covers ? (
            <p style={{ margin: '0.7rem 0 0' }} data-out="covers">
              <span className="chip fail">
                <Translated vi="kiểm tra lại hình dạng">check the shape</Translated>
              </span>{' '}
              <Translated
                vi={
                  <>
                    {count(shape.threads)} luồng × {shape.tM} × {shape.tN} = {count(shape.threads * shape.tM * shape.tN)} đầu
                    ra, nhưng ô của khối có {count(shape.bM * shape.bN)}.
                  </>
                }
              >
                {count(shape.threads)} threads × {shape.tM} × {shape.tN} ={' '}
                {count(shape.threads * shape.tM * shape.tN)} outputs, but the block tile has {count(shape.bM * shape.bN)}.
              </Translated>
            </p>
          ) : null}
        </div>
        <div className="panel" aria-live="polite">
          <p className="muted" style={{ margin: 0 }}>
            <Translated vi="Occupancy (mức chiếm dụng)">Occupancy</Translated>
          </p>
          <p style={{ fontSize: '2rem', fontWeight: 600, margin: '0 0 0.2rem', lineHeight: 1.1 }}>
            <output data-out="occupancy" data-occupancy={b.occupancy == null ? 'none' : fmt(b.occupancy)} data-blocks={b.blocks ?? 'none'} data-binding={b.binding.join('+')}>
              {b.occupancy == null ? tr('not documented', 'không được công bố') : fmt(b.occupancy) + '%'}
            </output>
          </p>
          {b.occupancy != null ? (
            <p className="muted" style={{ margin: '0 0 0.7rem' }}>
              <Translated
                vi={
                  <>
                    {count(b.blocks!)} khối × {count(shape.threads)} luồng = {count(b.resident!)} trên{' '}
                    {count(lim.threadsPerSm)} khe luồng
                  </>
                }
              >
                {count(b.blocks!)} {b.blocks === 1 ? 'block' : 'blocks'} × {count(shape.threads)} threads ={' '}
                {count(b.resident!)} of {count(lim.threadsPerSm)} thread slots
              </Translated>
            </p>
          ) : null}
          {b.threadBound != null ? (
            <p style={{ margin: '0 0 0.7rem' }} data-out="bound" data-bound={b.threadBound}>
              <Translated
                vi={
                  <>
                    Riêng thanh ghi đã chừa chỗ cho {count(lim.regsPerSm)} ÷ {count(b.regs)} = <b>{count(b.threadBound)}</b>{' '}
                    luồng.
                  </>
                }
              >
                Registers alone leave room for {count(lim.regsPerSm)} ÷ {count(b.regs)} = <b>{count(b.threadBound)}</b>{' '}
                threads.
              </Translated>
            </p>
          ) : null}
          <div data-out="verdict">{verdict}</div>
          {dev.kind === 'gpu' || dev.kind === 'metal' ? (
            <div style={{ margin: '0.8rem 0 0' }} data-out="measured" data-runs={runs.length}>
              {runs.length ? (
                runs.map((r, i) =>
                  r.blocks != null ? (
                    <p key={i} style={{ margin: '0 0 0.4rem' }}>
                      <Translated
                        vi={
                          <>
                            {r.lab}: API occupancy báo cáo <b>{r.blocks}</b> khối {r.threads ?? '?'} luồng mỗi SM cho
                            kernel tham chiếu của nó.
                          </>
                        }
                      >
                        {r.lab}: the occupancy API reports <b>{r.blocks}</b> {r.blocks === 1 ? 'block' : 'blocks'} of{' '}
                        {r.threads ?? '?'} threads per SM for its reference kernel.
                      </Translated>
                    </p>
                  ) : (
                    <p key={i} style={{ margin: '0 0 0.4rem' }}>
                      <Translated
                        vi={
                          <>
                            {r.lab}: pipeline đã biên dịch chấp nhận tối đa <b>{r.cap}</b> luồng mỗi threadgroup
                            {r.staticBytes != null ? <> và khai báo {bytes(r.staticBytes)} bộ nhớ threadgroup</> : null}.
                          </>
                        }
                      >
                        {r.lab}: the compiled pipeline accepts at most <b>{r.cap}</b> threads per threadgroup
                        {r.staticBytes != null ? <> and declares {bytes(r.staticBytes)} of threadgroup memory</> : null}.
                      </Translated>
                    </p>
                  ),
                )
              ) : null}
              {runs.length > 0 && dev.kind === 'metal' ? (
                <Translated
                  vi={
                    <p className="muted" style={{ margin: 0 }}>
                      Kernel của các lab khai báo mức trần 256 luồng, nên điều này cho thấy Metal đã không hạ mức đó,
                      chứ không cho biết nó có thể lên cao hơn 256 bao nhiêu.
                    </p>
                  }
                >
                  <p className="muted" style={{ margin: 0 }}>
                    The labs' kernels declare a cap of 256 threads, so this shows that Metal did not lower it, not how
                    far above 256 it would go.
                  </p>
                </Translated>
              ) : null}
              {runs.length > 0 ? null : dev.kind === 'gpu' ? (
                <Translated
                  vi={
                    <p className="muted" style={{ margin: 0 }}>
                      Chưa lab nào của chương 15 báo cáo <code>blocks_per_sm</code> trên thiết bị này. Các lab hỏi API
                      occupancy, vốn biết các khối cấp phát mà ước tính này bỏ qua.
                    </p>
                  }
                >
                  <p className="muted" style={{ margin: 0 }}>
                    No chapter 15 lab has reported <code>blocks_per_sm</code> on this device yet. The labs ask the
                    occupancy API, which knows the allocation chunks this estimate leaves out.
                  </p>
                </Translated>
              ) : (
                <Translated
                  vi={
                    <p className="muted" style={{ margin: 0 }}>
                      Chưa lab nào của chương 15 chạy benchmark trên thiết bị này. Khi đã có, mức trần luồng của pipeline
                      đã biên dịch sẽ hiện ở đây. Metal đặt mức trần đó theo lượng bộ nhớ kernel dùng và có thể hạ nó.
                    </p>
                  }
                >
                  <p className="muted" style={{ margin: 0 }}>
                    No chapter 15 lab has run its benchmark on this device yet. When one has, its compiled pipeline's
                    thread cap appears here. Metal sets that cap from the kernel's memory usage and may lower it.
                  </p>
                </Translated>
              )}
            </div>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

/** The calculator opens once the prediction above it is committed, or on request. */
function GatedBudget() {
  const { progress } = useProgress();
  const [open, setOpen] = useState(false);
  if (open || progress.answers['ch15/15.7/p-blocks'] != null) return <BudgetBench />;
  return (
    <p className="ui" data-out="gate">
      <Translated vi="Máy tính sẽ mở ra với các số của sách khi bạn đã chốt một dự đoán.">
        The calculator opens with the book's numbers once you have committed a prediction.
      </Translated>{' '}
      <button className="quiet" onClick={() => setOpen(true)}>
        <Translated vi="Mở mà không dự đoán">Open it without predicting</Translated>
      </button>
    </p>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS157: Record<string, Question> = {
  'ch15/15.7/q-80': {
    kind: 'numeric',
    id: 'ch15/15.7/q-80',
    prompt: (
      <Translated
        vi={
          <>
            Một luồng của kernel phân ô thanh ghi (mục 15.4) sở hữu một ô đầu ra 8 × 8 và, ở mỗi bước của vòng lặp trong,
            giữ một dải của A và một dải của B. Nó cần ít nhất bao nhiêu thanh ghi chỉ cho các ô này, tính một thanh ghi
            32-bit cho mỗi float?
          </>
        }
      >
        <>
          A thread of the register-tiled kernel (section 15.4) owns an 8 × 8 output tile and, at each step of its inner
          loop, holds one strip of A and one strip of B. What is the fewest registers it needs for these tiles alone,
          counting one 32-bit register per float?
        </>
      </Translated>
    ),
    answer: 80,
    unit: 'registers',
    explain: (
      <Translated
        vi={
          <p>
            Ô đầu ra là 8 × 8 = 64 giá trị, sống từ cặp ô đầu tiên đến lần ghi cuối cùng. Dải A là một cột thuộc 8 hàng của
            A mà luồng phụ trách, 8 giá trị; dải B là một hàng thuộc 8 cột của B, thêm 8 nữa. 64 + 8 + 8 = 80. Đó là mức
            sàn: địa chỉ, bộ đếm vòng lặp và các giá trị tạm của mã đã unroll còn cộng thêm vào.
          </p>
        }
      >
        <p>
          The output tile is 8 × 8 = 64 values, live from the first tile pair to the final store. The A strip is a column of
          the thread's 8 rows of A, 8 values; the B strip is a row of its 8 columns of B, 8 more. 64 + 8 + 8 = 80. It is a
          floor: addresses, loop counters and the temporaries of the unrolled code come on top.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.7, p. 363',
    vi: { unit: 'thanh ghi' },
  },
  'ch15/15.7/q-1536': {
    kind: 'numeric',
    id: 'ch15/15.7/q-1536',
    prompt: (
      <Translated
        vi={
          <>
            Cùng kernel đó, vẫn 255 thanh ghi mỗi luồng và một khối 256 luồng mỗi SM, chạy trên một SM có giới hạn luồng là
            1536 thay vì 2048. Occupancy của nó là bao nhiêu?
          </>
        }
      >
        <>
          The same kernel, still 255 registers per thread and one 256-thread block per SM, runs on an SM whose thread limit
          is 1536 rather than 2048. What is its occupancy?
        </>
      </Translated>
    ),
    answer: 16.7,
    tol: 0.005,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            Thanh ghi vẫn chỉ cho phép một khối, nên 256 luồng thường trú: 256 / 1536 = 16.7%. Kernel không đổi và về
            nguyên tắc tốc độ của nó cũng không đổi; chỉ mẫu số thay đổi. NVIDIA công bố 1536 luồng thường trú mỗi SM cho
            compute capability 8.6, 8.9 và 12.x (RTX 30, 40 và 50), và 1024 cho 7.5 và 10.7, nơi cùng kernel đó đọc ra
            25%. Một con số occupancy ít ý nghĩa nếu không biết nó được tính cho SM nào.
          </p>
        }
      >
        <p>
          Registers still allow only one block, so 256 threads are resident: 256 / 1536 = 16.7%. The kernel did not change
          and neither did its speed in principle; only the denominator did. NVIDIA documents 1536 resident threads per SM
          for compute capabilities 8.6, 8.9 and 12.x (RTX 30, 40 and 50), and 1024 for 7.5 and 10.7, where the same kernel
          reads 25%. An occupancy percentage means little without knowing which SM it was computed for.
        </p>
      </Translated>
    ),
    source: 'our arithmetic; limits from notes/today/ch15-facts.md A3 and ch04-facts.md A12',
    vi: { source: 'phép tính của chúng tôi; giới hạn lấy từ notes/today/ch15-facts.md A3 và ch04-facts.md A12' },
  },
  'ch15/15.7/q-128regs': {
    kind: 'numeric',
    id: 'ch15/15.7/q-128regs',
    prompt: (
      <Translated
        vi={
          <>
            Giả sử trình biên dịch nhét được kernel vào 128 thanh ghi mỗi luồng. Trên SM ví dụ của sách (65,536 thanh ghi,
            2048 khe luồng), bao nhiêu khối 256 luồng có thể thường trú?
          </>
        }
      >
        <>
          Suppose the compiler fitted the kernel in 128 registers per thread. On the book's example SM (65,536 registers,
          2048 thread slots), how many 256-thread blocks could be resident?
        </>
      </Translated>
    ),
    answer: 2,
    unit: 'blocks',
    explain: (
      <Translated
        vi={
          <p>
            Một khối cần 256 × 128 = 32,768 thanh ghi, và 65,536 / 32,768 = 2: 512 luồng, 25%. Đó là phép tính theo mô
            hình của sách. Hai khối có thực sự vừa hay không còn tùy cách phần cứng làm tròn phần cấp phát của mỗi warp,
            điều NVIDIA không công bố, nên với một kernel thật hãy hỏi{' '}
            <code>cudaOccupancyMaxActiveBlocksPerMultiprocessor</code>; các lab in câu trả lời của nó dưới tên{' '}
            <code>blocks_per_sm</code>.
          </p>
        }
      >
        <p>
          One block needs 256 × 128 = 32,768 registers, and 65,536 / 32,768 = 2: 512 threads, 25%. That is the arithmetic
          of the book's model. Whether two blocks really fit depends on how the hardware rounds each warp's allocation,
          which NVIDIA does not publish, so for a real kernel ask{' '}
          <code>cudaOccupancyMaxActiveBlocksPerMultiprocessor</code>; the labs print its answer as{' '}
          <code>blocks_per_sm</code>.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.7, p. 363 (the model); notes/today/ch15-facts.md A3 (the allocation chunks)',
    vi: {
      unit: 'khối',
      source: 'ch. 15 §15.7, p. 363 (mô hình); notes/today/ch15-facts.md A3 (các khối cấp phát)',
    },
  },
  'ch15/15.7/q-two-phases': {
    kind: 'choice',
    id: 'ch15/15.7/q-two-phases',
    prompt: (
      <Translated vi={<>Với một khối mỗi SM, phần cứng không còn làm điều gì cho bạn mà nó đã làm cho kernel của chương 5?</>}>
        <>With one block per SM, what does the hardware no longer do for you that it did for chapter 5's kernel?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Chồng lấp việc nạp của một khối với việc tính toán của một khối khác.">
            {'Overlap one block\'s loading with another block\'s computing.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Hợp nhất các lần nạp của ô đầu vào, việc cần đủ warp đang chạy để gộp địa chỉ của chúng.">
            {'Coalesce the loads of the input tiles, which needs enough warps in flight to merge their addresses.'}
          </Translated>
        ),
        why: (
          <Translated vi="Hợp nhất truy cập liên quan đến cách các địa chỉ của một warp phân bố, không phải số khối thường trú. Các lần nạp ô vẫn được hợp nhất như trước.">
            {'Coalescing is about how one warp\'s addresses fall, not about how many blocks are resident. The tile loads coalesce as well as before.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Giữ ô đầu ra trong thanh ghi: một khối đơn độc phải đẩy các bộ tích lũy ra ngoài để lấy chỗ cho các lần nạp.">
            {'Keep the output tile in registers: a lone block must spill its accumulators to make room for the loads.'}
          </Translated>
        ),
        why: (
          <Translated vi="Ô đầu ra nằm trong thanh ghi chính vì mỗi luồng được cấp nhiều thanh ghi như vậy. Đó là nguyên nhân của occupancy thấp, không phải điều occupancy thấp lấy đi.">
            {'The output tile is in registers precisely because each thread was given so many of them. That is the cause of the low occupancy, not something it takes away.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Che độ trễ của các lần đọc bộ nhớ chia sẻ, việc cần warp của các khối khác để chuyển sang.">
            {'Hide the latency of shared-memory reads, which needs other blocks\' warps to switch to.'}
          </Translated>
        ),
        why: (
          <Translated vi="Phân ô thanh ghi (mục 15.4) đã cắt số lần đọc bộ nhớ chia sẻ đi tám lần, và unroll phơi ra các FMA độc lập để che phần còn lại. Khoảng trống còn lại là toàn bộ pha nạp.">
            {'Register tiling (section 15.4) already cut the shared-memory reads eightfold, and unrolling exposes independent FMAs to cover the rest. The gap that remains is the whole loading phase.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Khi có nhiều khối trên một SM, một khối chờ ô của nó trong khi khối khác tính, và bộ lập lịch xen kẽ các warp của
            chúng. Với một khối, mọi warp của SM đều ở cùng một rào chắn: tất cả cùng nạp, rồi tất cả cùng tính. Phần cứng bộ
            nhớ nhàn rỗi trong lúc tính và các đơn vị số học nhàn rỗi trong lúc nạp. Mục 15.8 sắp xếp lại vòng lặp để một
            khối làm cả hai cùng lúc.
          </p>
        }
      >
        <p>
          With several blocks on an SM, one waits for its tiles while another computes, and the scheduler interleaves
          their warps. With one block, every warp of the SM is at the same barrier: all loading, then all computing. The
          memory hardware idles during the compute and the arithmetic units during the loads. Section 15.8 rearranges the
          loop so that one block does both at once.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.7, p. 364',
  },
  'ch15/15.7/q-launch-bounds': {
    kind: 'choice',
    id: 'ch15/15.7/q-launch-bounds',
    prompt: (
      <Translated
        vi={
          <>
            Vì sao <code>__launch_bounds__(256)</code> trên kernel này quan trọng hơn với bản build HIP của cùng tệp{' '}
            <code>.cu</code>?
          </>
        }
      >
        <>
          Why does <code>__launch_bounds__(256)</code> on this kernel matter more for the HIP build of the same{' '}
          <code>.cu</code> file?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không có nó, HIP biên dịch cho khối lớn nhất được phép và giới hạn số thanh ghi mỗi luồng tương ứng.">
            Without it HIP compiles for the largest block allowed and caps registers per thread to match.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không có nó, HIP từ chối mọi lần khởi chạy có khối không phải 1024 luồng mặc định.">
            Without it HIP rejects any launch whose block is not the default 1024 threads.
          </Translated>
        ),
        why: (
          <Translated vi="Giới hạn này là cận trên cho việc sinh mã và kiểm tra lần khởi chạy. Một khối 256 nằm trong mức mặc định.">
            The bound is an upper limit for code generation and launch validation. A block of 256 is within the default.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="HIP bỏ qua __launch_bounds__, nên trên HIP nó không thay đổi gì trong mã đã biên dịch.">
            HIP ignores __launch_bounds__, so on HIP it changes nothing in the compiled code.
          </Translated>
        ),
        why: (
          <Translated vi="HIP tôn trọng nó; đối số thứ hai của nó thậm chí còn mang nghĩa khác với của CUDA (số wave mỗi đơn vị thực thi, không phải số khối mỗi SM).">
            {'HIP honours it; its second argument even means something different from CUDA\'s (waves per execution unit, not blocks per SM).'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Wavefront của AMD có thể rộng 64, nên một khối 256 luồng gồm 4 wavefront phải được khai báo với trình biên dịch.">
            AMD wavefronts can be 64 wide, so a 256-thread block of 4 wavefronts must be declared to the compiler.
          </Translated>
        ),
        why: (
          <Translated vi="Wavefront của AMD rộng 64 hoặc 32, và mọi bội số của độ rộng đều chạy được mà không cần khai báo. Vấn đề nằm ở ngân sách thanh ghi mà trình biên dịch giả định.">
            AMD wavefronts are 64 or 32 wide, and any multiple of the width works without a declaration. The issue is the register budget the compiler assumes.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            AMD: nếu không có launch bounds, trình biên dịch giả định kích thước khối tối đa của thiết bị, và "giảm
            MAX_THREADS_PER_BLOCK cho phép trình biên dịch dùng nhiều tài nguyên hơn mỗi luồng so với một lần biên dịch
            không bị ràng buộc". Một kernel muốn 80 thanh ghi mỗi luồng trở lên cho các ô của nó chính là trường hợp điều
            này được viết ra. Áp dụng nó cho kernel này là cách đọc của chúng tôi về phát biểu của AMD; điều đó chưa được đo
            ở đây.
          </p>
        }
      >
        <p>
          AMD: if no launch bounds are given, the compiler assumes the device's maximum block size, and "reducing
          MAX_THREADS_PER_BLOCK allows the compiler to use more resources per thread than an unconstrained compilation".
          A kernel that wants 80 or more registers per thread for its tiles is the case this was written for. Applying it
          to this kernel is our reading of AMD's statement; it has not been measured here.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch15-facts.md B3 and ch08-facts.md B4 (not in the book)',
    vi: { source: 'notes/today/ch15-facts.md B3 và ch08-facts.md B4 (không có trong sách)' },
  },
  'ch15/15.7/q-mlx': {
    kind: 'choice',
    id: 'ch15/15.7/q-mlx',
    prompt: (
      <Translated
        vi={
          <>
            GEMM Metal của MLX dùng 64 đến 128 luồng mỗi threadgroup và các ô như 64 × 64 × 16, chọn theo từng thiết bị, so
            với 256 luồng và 128 × 128 × 8 của sách. Bạn có thể kết luận gì về tệp thanh ghi của GPU Apple?
          </>
        }
      >
        <>
          MLX's Metal GEMM uses 64 to 128 threads per threadgroup and tiles such as 64 × 64 × 16, chosen per device,
          against the book's 256 threads and 128 × 128 × 8. What may you conclude about the register file of Apple GPUs?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không gì cả: Apple không công bố số thanh ghi hay quy tắc occupancy, nên hình dạng của MLX là một lựa chọn thiết kế, không phải bằng chứng.">
            {'Nothing: Apple documents no register count or occupancy rule, so MLX\'s shapes are a design, not evidence.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Tệp thanh ghi của nó lớn khoảng một phần tư của một SM NVIDIA, vì threadgroup của MLX lớn bằng một phần tư.">
            {'Its register file is about a quarter the size of an NVIDIA SM\'s, since MLX\'s threadgroups are a quarter as large.'}
          </Translated>
        ),
        why: (
          <Translated vi="Đó là đọc một số liệu phần cứng từ lựa chọn của một thư viện. Hình dạng ô phụ thuộc nhiều thứ: các lệnh ma trận được dùng, bộ nhớ threadgroup, kiểu dữ liệu, tốc độ đo được.">
            {'That reads a hardware figure off a library\'s choice. Tile shapes depend on many things: the matrix instructions used, the threadgroup memory, the data type, measured speed.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="GPU Apple cho phép tối đa 128 luồng mỗi threadgroup, nếu không MLX đã dùng 256 như sách.">
            {'Apple GPUs allow at most 128 threads per threadgroup, or MLX would use the book\'s 256.'}
          </Translated>
        ),
        why: (
          <Translated vi="Metal công bố 1024 là mức tối đa lý thuyết trên Apple4 trở về sau; mức trần thực tế tính theo từng pipeline. MLX đã chọn ít hơn.">
            Metal documents 1024 as the theoretical maximum on Apple4 and later; the real cap is per pipeline. MLX chose fewer.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lựa chọn của MLX là một sự thật về MLX: hình dạng nào các tác giả của nó thấy nhanh, trên những chip nào. Apple
            không công bố số thanh ghi, tệp thanh ghi hay API occupancy cho kernel Metal; Xcode hiển thị occupancy của một
            shader, nhưng không hàm nào trả về nó. Trên Metal, các hình dạng ô được so sánh bằng cách đo chúng.
          </p>
        }
      >
        <p>
          MLX's choice is a fact about MLX: which shape its authors found fast, on which chips. Apple documents no
          register count, register file or occupancy API for Metal kernels; Xcode shows a shader's occupancy, but no
          function returns it. On Metal, tile shapes are compared by measuring them.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch15-facts.md C3 and D3; ch06-facts.md C5 (not in the book)',
    vi: { source: 'notes/today/ch15-facts.md C3 và D3; ch06-facts.md C5 (không có trong sách)' },
  },
};

function PredictReveal({ v }: { v: number }) {
  return (
    <Translated
      vi={
        <p>
          65,536 / 255 = 257, nên thanh ghi của một SM chứa được 257 luồng của kernel này: <b>một khối</b> 256 luồng, dư
          ra phần của một luồng.{' '}
          {v === 1
            ? 'Đó đúng là điều bạn đã dự đoán.'
            : v > 1
              ? 'Khe luồng cho phép 8 khối, và bộ nhớ chia sẻ cho phép nhiều hơn nữa; thanh ghi chỉ cho phép một.'
              : 'Một khối vẫn vừa: 256 × 255 = 65,280 thanh ghi, vừa dưới 65,536.'}{' '}
          Một khối là 256 trên 2048 khe luồng, occupancy 12.5%. Đó cũng là lý do khối có 256 luồng: một khối 512 luồng sẽ
          cần 130,560 thanh ghi và hoàn toàn không thể thường trú.
        </p>
      }
    >
      <p>
        65,536 / 255 = 257, so the registers of one SM hold 257 threads of this kernel: <b>one block</b> of 256, with one
        thread's worth left over.{' '}
        {v === 1
          ? 'That is what you had.'
          : v > 1
            ? 'Thread slots would allow 8 blocks, and shared memory many more; registers allow one.'
            : 'One block does fit: 256 × 255 = 65,280 registers, just under 65,536.'}{' '}
        One block is 256 of 2048 thread slots, 12.5% occupancy. It is also why the block has 256 threads: a block of 512
        would need 130,560 registers and could not be resident at all.
      </p>
    </Translated>
  );
}

// ---------------------------------------------------------------- the section
export function S157() {
  return (
    <>
      <Translated
        vi={
          <p>
            Bốn mục vừa qua đã mua cường độ tính toán bằng bộ lưu trữ trên chip: một ô đầu ra lớn cho mỗi khối giữ trong thanh
            ghi (register), các ô đầu vào đặt tạm trong bộ nhớ chia sẻ (shared memory), các dải chép vào thanh ghi. Cả hai
            nơi lưu trữ đều thuộc về SM, và mọi khối thường trú trên nó đều rút từ đó. Mục này cộng hóa đơn lại, và đi đến
            một kết luận thoạt nhìn có vẻ sai: kernel này được thiết kế để chạy ở occupancy (mức chiếm dụng) thấp.
          </p>
        }
      >
        <p>
          The last four sections bought arithmetic intensity with on-chip storage: a large output tile per block held in
          registers, input tiles staged in shared memory, strips copied into registers. Both stores belong to the SM, and
          every block resident on it draws from them. This section adds up the bill, and arrives at a conclusion that
          looks wrong at first: the kernel is meant to run at low occupancy.
        </p>
      </Translated>
      <Source section="15.7" pages="363" />

      <h2>
        <Translated vi="Bộ nhớ chia sẻ ở đây rẻ">Shared memory is cheap here</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Nhớ lại kết quả của <a href={href('ch15', '15.2')}>mục 15.2</a>. Với mỗi cặp ô đầu vào, một khối thực hiện
            2·m·n·k phép toán và nạp 4·k·(m + n) byte, nên tỉ số là 0.5·m·n/(m + n) và độ sâu k triệt tiêu. Một ô đầu vào
            sâu hơn không mua thêm cường độ, nên bạn giữ nó nông. Với m = n = 128 và k = 8, ô A là 128 × 8 float và ô B là
            8 × 128: mỗi ô 4 KB. Sách nói điều đó không giới hạn occupancy trên một GPU hiện đại. Trên thiết bị ví dụ của
            nó, với 228 KB bộ nhớ chia sẻ mỗi SM, sẽ vừa lượng ô của 28 khối như vậy.
          </p>
        }
      >
        <p>
          Recall the result of <a href={href('ch15', '15.2')}>section 15.2</a>. For each pair of input tiles a block does
          2·m·n·k operations and loads 4·k·(m + n) bytes, so the ratio is 0.5·m·n/(m + n) and the depth k cancels. A
          deeper input tile buys no intensity, so you keep it shallow. With m = n = 128 and k = 8, the A tile is 128 × 8
          floats and the B tile 8 × 128: 4 KB each. The book says that does not limit occupancy on a modern GPU. On its
          example device, with 228 KB of shared memory per SM, 28 blocks' worth of these tiles would fit.
        </p>
      </Translated>
      <Source section="15.7" pages="363" />

      <h2>
        <Translated vi="Thanh ghi thì không">Registers are not</Translated>
      </h2>
      <Translated vi={<p>Giờ hãy đếm xem một luồng phải giữ những gì cùng một lúc. Tính ra trước khi đọc tiếp.</p>}>
        <p>
          Now count what one thread must hold at the same moment. Work it out before reading on.
        </p>
      </Translated>
      <Ask q={questionsS157['ch15/15.7/q-80']} label={tr('Work it out', 'Tính ra')} />
      <Translated
        vi={
          <p>
            Tám mươi là mức sàn. Địa chỉ, bộ đếm vòng lặp và các giá trị tạm của mã đã unroll cũng cần thanh ghi, và việc
            unroll, chính là thứ giữ các ô trong thanh ghi ngay từ đầu (mục 15.3), làm tăng số giá trị sống cùng lúc. Sách
            nói các kernel kiểu này rốt cuộc dùng đến mức tối đa một luồng được có: 255 thanh ghi. Mỗi luồng thường trú trên
            SM lấy chừng ấy thanh ghi từ cùng một tệp thanh ghi.
          </p>
        }
      >
        <p>
          Eighty is a floor. Addresses, loop counters and the temporaries of the unrolled code need registers too, and
          unrolling, which is what keeps the tiles in registers at all (section 15.3), raises the number of values live at
          once. The book says kernels like this end up at the most a thread may have: 255 registers. Every thread resident
          on the SM takes that many from one register file.
        </p>
      </Translated>
      <Source section="15.7" pages="363" />
      <Predict
        id="ch15/15.7/p-blocks"
        prompt={
          <Translated
            vi={
              <>
                SM ví dụ của sách có 65,536 thanh ghi và 2048 khe luồng. Với 255 thanh ghi mỗi luồng, bao nhiêu khối 256
                luồng có thể thường trú trên nó cùng lúc?
              </>
            }
          >
            <>
              The book's example SM has 65,536 registers and 2048 thread slots. At 255 registers per thread, how many
              blocks of 256 threads can be resident on it at once?
            </>
          </Translated>
        }
        unit={tr('blocks', 'khối')}
      >
        {(v) => <PredictReveal v={v} />}
      </Predict>
      <Translated
        vi={
          <p>
            Đó là số liệu của sách. Trên GPU của bạn, câu trả lời dựa vào hai con số khác nhau giữa các dòng NVIDIA, và vào
            một con số NVIDIA không công bố. Máy tính bên dưới làm phép tính cho bất kỳ hình dạng ô và bất kỳ SM nào, bắt
            đầu từ kernel của sách; cấu hình mẫu thứ hai của nó cho thấy ngay cả mức sàn 80 thanh ghi cũng để kernel còn
            cách xa occupancy đầy đủ.
          </p>
        }
      >
        <p>
          Those are the book's figures. On your GPU the answer rests on two numbers that differ between NVIDIA parts, and
          on one that NVIDIA does not publish. The calculator below does the arithmetic for any tile shape and any SM,
          starting from the book's kernel; its second preset shows that even the 80-register floor leaves the kernel
          well short of full occupancy.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch15-facts A3; ch04-facts A12, A14)', '2026-11-20 (ch15-facts mục A3; ch04-facts mục A12, A14)')}>
        <Translated
          vi={
            <p>
              255 mỗi luồng và 64 K mỗi SM đúng trên mọi compute capability NVIDIA liệt kê. 2048 luồng thường trú chỉ trên
              8.0, 9.0, 10.0 và 10.3; 1536 trên RTX 30, 40 và 50; 1024 trên 7.5 và 10.7.
            </p>
          }
        >
          <p>
            255 per thread and 64 K per SM hold on every compute capability NVIDIA lists. 2048 resident threads only on
            8.0, 9.0, 10.0 and 10.3; 1536 on RTX 30, 40 and 50; 1024 on 7.5 and 10.7.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Thanh ghi được cấp phát theo khối có kích thước không được công bố, nên con số vừa chính xác đến từ{' '}
              <Id>cudaOccupancyMaxActiveBlocksPerMultiprocessor</Id> (các lab in nó dưới tên <Id>blocks_per_sm</Id>) hoặc{' '}
              <Id>launch__occupancy_limit_registers</Id> của Nsight Compute.
            </p>
          }
        >
          <p>
            Registers are allocated in chunks of undocumented size, so the exact fit comes from{' '}
            <Id>cudaOccupancyMaxActiveBlocksPerMultiprocessor</Id> (the labs print it as <Id>blocks_per_sm</Id>) or Nsight
            Compute's <Id>launch__occupancy_limit_registers</Id>.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Thiết bị CUDA hoặc HIP của bạn xuất hiện trong danh sách khi một lần chạy lab đã ghi nhận nó, kèm những gì API
            occupancy nói về các kernel thật của lab bên cạnh ước tính. Một thiết bị Metal được hiển thị với những gì Apple
            công bố, tức là chỉ bộ nhớ chia sẻ.
          </p>
        }
      >
        <p>
          Your own CUDA or HIP device appears in its list once a lab run has recorded it, with what the
          occupancy API said about the labs' real kernels beside the estimate. A Metal device is shown with what Apple
          documents, which is shared memory only.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Thanh ghi mỗi luồng nhân số luồng mỗi khối, so với tệp thanh ghi: mô hình của sách. Cấp phát thật tính theo
              warp, theo khối NVIDIA không công bố và theo hạt (granule) AMD định cỡ cho từng target. Số khe khối và bộ nhớ
              chia sẻ mà driver giữ lại không được tính.
            </p>
          }
        >
          <p>
            Registers per thread times threads per block, against the register file: the book's model. Real allocation is
            per warp, in chunks NVIDIA does not publish and granules AMD sizes per target. Block slots and the shared
            memory the driver reserves are not counted.
          </p>
        </Translated>
      </Simplification>
      <GatedBudget />
      <Ask q={questionsS157['ch15/15.7/q-1536']} label={tr('Work it out', 'Tính ra')} />
      <Ask q={questionsS157['ch15/15.7/q-128regs']} label={tr('Work it out', 'Tính ra')} />

      <h2>
        <Translated vi="Occupancy thấp, một cách cố ý">Low occupancy, on purpose</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương 4 dạy bạn muốn có nhiều warp mỗi SM, để một warp chờ bộ nhớ không tốn gì trong khi warp khác chạy. Kernel
            này từ bỏ phần lớn điều đó. Sách đánh giá sự đánh đổi là xứng đáng: các ô thanh ghi lớn là thứ nâng cường độ
            tính toán, và cái lợi đó lớn hơn những gì các warp vắng mặt lẽ ra che được. Hướng dẫn profiling của chính
            NVIDIA ngày nay cũng đứng về phía đó.
          </p>
        }
      >
        <p>
          Chapter 4 taught you to want many warps per SM, so that one waiting on memory costs nothing while another runs.
          This kernel gives most of that up. The book judges the trade worth it: the large register tiles are what raise
          the arithmetic intensity, and that gain outweighs what the missing warps would have hidden. NVIDIA's own
          profiling guidance takes the same side today.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch15-facts A3)', '2026-11-20 (ch15-facts mục A3)')}>
        <Translated
          vi={
            <p>
              Compute Triage Guide: với occupancy thấp và một pipeline bận, "Launching extra warps is counter-productive
              because the busy pipe cannot absorb more work" (khởi chạy thêm warp là phản tác dụng vì pipe đang bận không
              thể nhận thêm việc).
            </p>
          }
        >
          <p>
            The Compute Triage Guide: with low occupancy and a busy pipeline, "Launching extra warps is
            counter-productive because the busy pipe cannot absorb more work."
          </p>
        </Translated>
      </Today>
      <Translated vi={<p>Sách nêu hai cái giá của việc chạy quá ít luồng, và một cách khắc phục cho mỗi cái.</p>}>
        <p>
          The book names two costs of running so few threads, and a remedy for each.
        </p>
      </Translated>
      <ol>
        <Translated
          vi={
            <li>
              <b>Ít lệnh bộ nhớ đang chạy hơn.</b> Ít luồng hơn phát ra ít lệnh nạp và ghi hơn, có thể để băng thông bộ nhớ
              không được dùng hết. Cách khắc phục là để mỗi lệnh chuyển nhiều hơn: nạp và ghi vector 16 byte (chương 6, mục
              6.3). Sách nói những lệnh này đã được thấy "in Section 15.3", trong khi mục đó chỉ gợi ý chúng như bài tập
              (notes, mục 9); lab 15.3 và 15.4 cung cấp mã.
            </li>
          }
        >
          <li>
            <b>Fewer memory instructions in flight.</b> Fewer threads issue fewer loads and stores, which can leave memory
            bandwidth unused. The remedy is to make each instruction move more: vector loads and stores of 16 bytes
            (chapter 6, section 6.3). The book says these were seen "in Section 15.3", which only suggests them as
            exercises (notes, item 9); labs 15.3 and 15.4 supply the code.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Ít warp hơn để che độ trễ.</b> Khi các warp duy nhất trên SM đều chờ, không có gì chạy. Cách khắc phục là
              unroll mạnh tay (chương 6, mục 6.6): nó loại bỏ rẽ nhánh và bày ra các lệnh độc lập mà trình biên dịch có thể
              đặt giữa một lệnh chậm và lần dùng đầu tiên kết quả của nó. Làm tốt thì phần lớn kernel đã biên dịch là một
              chuỗi dài các phép nhân-cộng gộp (FMA) độc lập, và các đơn vị số học hiếm khi phải dừng.
            </li>
          }
        >
          <li>
            <b>Fewer warps to cover latency.</b> When the only warps on the SM all wait, nothing runs. The remedy is
            aggressive unrolling (chapter 6, section 6.6): it removes branches and lays out independent instructions the
            compiler can place between a slow instruction and its first use. Done well, most of the compiled kernel is a
            long run of independent fused multiply-adds, and the arithmetic units rarely stall.
          </li>
        </Translated>
      </ol>
      <Source section="15.7" pages="364" />

      <h2>
        <Translated vi="Điều còn lại: hai pha">What is left: two phases</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một vấn đề vẫn còn sau cả hai cách khắc phục. Thông thường một kernel bị giới hạn bởi tính toán có thể bỏ qua độ
            trễ bộ nhớ, vì trong khi một khối chờ ô của nó, một khối khác trên cùng SM tính toán. Ở đây không có khối nào
            khác. Khối đơn độc luân phiên giữa một pha bị giới hạn bởi bộ nhớ, trong đó các ô được nạp và các đơn vị số học
            nhàn rỗi, và một pha bị giới hạn bởi tính toán, trong đó nó tính và phần cứng bộ nhớ nhàn rỗi. Nếu muốn hai pha
            chồng lấp, chương trình phải tự sắp xếp. <a href={href('ch15', '15.8')}>Mục 15.8</a> làm điều đó, bằng pipeline
            phần mềm.
          </p>
        }
      >
        <p>
          One problem survives both remedies. Normally a compute-bound kernel can ignore memory latency, because while one
          block waits for its tiles, another block on the same SM computes. Here there is no other block. The lone block
          alternates between a memory-bound phase, in which tiles load and the arithmetic units idle, and a compute-bound
          phase, in which it computes and the memory hardware idles. If the two are to overlap, the program has to
          arrange it. <a href={href('ch15', '15.8')}>Section 15.8</a> does, with software pipelining.
        </p>
      </Translated>
      <Source section="15.7" pages="364" />
      <Ask q={questionsS157['ch15/15.7/q-two-phases']} />
      <Translated
        vi={
          <p>
            Hãy nhớ sự đánh đổi này khi bạn đọc mã GEMM trong một thư viện hay một backend của framework. Một mảng lớn các
            bộ tích lũy mỗi luồng và occupancy thấp là thiết kế, không phải lỗi cần sửa, và thư viện template của chính
            NVIDIA nói đúng như vậy.
          </p>
        }
      >
        <p>
          Keep this trade in mind when you read GEMM code in a library or a framework backend. A large array of
          accumulators per thread and a low occupancy are the design, not a defect to fix, and NVIDIA's own template
          library says so in as many words.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (ch15-facts A3, A12)', '2026-11-20 (ch15-facts mục A3, A12)')}>
        <Translated
          vi={
            <p>
              CUTLASS: các bộ tích lũy "typically occupy at least half a thread's total register budget" (thường chiếm ít
              nhất một nửa tổng ngân sách thanh ghi của một luồng), nên occupancy "is relatively low compared to other
              classes of GPU workloads" (tương đối thấp so với các loại tải GPU khác).
            </p>
          }
        >
          <p>
            CUTLASS: the accumulators "typically occupy at least half a thread's total register budget", so occupancy
            "is relatively low compared to other classes of GPU workloads."
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Trên GPU AMD và Apple">On AMD and Apple GPUs</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sự đánh đổi này áp dụng cho AMD, với những con số khác. Một luồng có thể có tới 256 thanh ghi vector, cộng 256
            thanh ghi ma trận trên CDNA. AMD chỉ công bố tệp thanh ghi của một compute unit dưới dạng một khoảng, nên máy
            tính dùng những gì thiết bị của bạn báo cáo, và AMD cấp phát thanh ghi theo wavefront, từng hạt (granule), nên
            tích theo luồng chỉ đúng về hình dạng. Hướng dẫn của chính AMD nói rằng các GEMM xây trên lệnh ma trận của nó
            trên CDNA thường chạy ở occupancy thấp, vì vài warp là đủ để giữ các đơn vị đó và pipeline bộ nhớ bận rộn.
          </p>
        }
      >
        <p>
          The trade carries over to AMD, with different numbers. A thread may have up to 256 vector registers, plus 256
          matrix registers on CDNA. AMD documents the register file of a compute unit only as a range, so the calculator
          uses what your device reports, and it allocates registers per wavefront in granules, so the per-thread product
          is only the right shape. AMD's own guide says that GEMMs built on its matrix instructions on CDNA often run at
          low occupancy, because a few warps are enough to keep those units and the memory pipeline busy.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              HIP và Metal không có trong sách. Đã kiểm tra 2026-11-20: ch15-facts B3, C3; ch04-facts B10, C5; ch06-facts
              C5; ch08-facts B4.
            </p>
          }
        >
          <p>
            HIP and Metal are not in the book. Checked 2026-11-20: ch15-facts B3, C3; ch04-facts B10, C5; ch06-facts C5;
            ch08-facts B4.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Một chi tiết quan trọng với HIP hơn với CUDA. Nếu không được chỉ định khác, trình biên dịch HIP sinh mã cho khối
            lớn nhất mà thiết bị cho phép và phân bổ ngân sách thanh ghi mỗi luồng tương ứng.
          </p>
        }
      >
        <p>
          One detail matters more on HIP than on CUDA. Unless told otherwise, the HIP compiler generates code for the
          largest block the device allows and budgets registers per thread to match.
        </p>
      </Translated>
      <Ask q={questionsS157['ch15/15.7/q-launch-bounds']} />
      <Translated
        vi={
          <p>
            Apple không công bố điều nào trong số này cho Metal: không số thanh ghi, không kích thước tệp thanh ghi, không
            API occupancy. Xcode hiển thị occupancy của một shader, nhưng không hàm nào trả về nó. Hai sự thật đã được công
            bố vẫn liên quan đến một kernel nặng thanh ghi với 256 luồng. Một pipeline có thể nhận mức trần luồng thấp hơn
            mức tối đa của thiết bị ("Metal may lower the thread limit when creating the compute pipeline state"), nên các
            kernel Metal của chương khai báo <code>[[max_total_threads_per_threadgroup(256)]]</code>, đối ứng của{' '}
            <code>__launch_bounds__(256)</code> ở phía CUDA, và các bài test của lab thất bại nếu{' '}
            <code>maxTotalThreadsPerThreadgroup</code> của pipeline trả về dưới 256. Và các ô phải vừa giới hạn bộ nhớ
            threadgroup, đó là điều duy nhất máy tính có thể kiểm tra trên một thiết bị Metal.
          </p>
        }
      >
        <p>
          Apple documents none of this for Metal: no register count, no register file size, no occupancy API. Xcode
          shows a shader's occupancy, but no function returns it. Two documented facts still bear on a register-heavy
          kernel with 256 threads. A pipeline may be given a lower thread cap than the device's maximum ("Metal may lower
          the thread limit when creating the compute pipeline state"), so the chapter's Metal kernels declare{' '}
          <code>[[max_total_threads_per_threadgroup(256)]]</code>, the counterpart of the <code>__launch_bounds__(256)</code>{' '}
          on their CUDA side, and the labs' tests fail if the pipeline's <code>maxTotalThreadsPerThreadgroup</code> comes
          back below 256. And the tiles must fit the threadgroup memory limit, which is the one thing the calculator can
          check on a Metal device.
        </p>
      </Translated>
      <Ask q={questionsS157['ch15/15.7/q-mlx']} />
    </>
  );
}
