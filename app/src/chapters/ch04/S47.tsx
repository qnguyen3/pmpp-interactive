import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, labStatus, num, realBackends, useReports, type BenchRow, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  BOOK, BOOK_BLOCK_SIZES, EX6, EX7, EX7_ASSIGNMENTS, EX8, EX8_KERNELS, MAX_REGS_PER_THREAD, assignment,
  limitsFromReport, notReported, occupancy, pct, regValues, regsAtFullOccupancy, scan, smallestFillingBlock, steps,
  threadValues, type Kernel, type LimitKey, type Limits, type Occupancy, type ScanPoint, type Step,
} from '../../sim/occupancy';

const count = (v: number) => v.toLocaleString('en-US');
const plural = (v: number, one: string, many = one + 's') => `${count(v)} ${v === 1 ? one : many}`;
const toInt = (s: string) => {
  const v = Number(s);
  return Number.isFinite(v) && v > 0 ? Math.floor(v) : 0;
};
const KB = 1024;

// ---------------------------------------------------------------- devices
interface Device {
  key: string;
  label: string;
  limits: Limits;
  /** Where the numbers come from. */
  note: string;
}

const BOOK_DEVICE: Device = {
  key: 'book',
  get label() {
    return tr("The book's example", 'Ví dụ của sách');
  },
  limits: BOOK,
  get note() {
    return tr(
      "The book's example device: 32 blocks, 2048 threads (64 warps) and 65,536 registers per SM, 1024 threads per block (§4.7, pp. 85–86); 228 KB of shared memory per SM (§5.6, p. 116).",
      'Thiết bị ví dụ của sách: 32 khối, 2048 luồng (64 warp) và 65,536 thanh ghi mỗi SM, 1024 luồng mỗi khối (§4.7, pp. 85–86); 228 KB bộ nhớ chia sẻ mỗi SM (§5.6, p. 116).',
    );
  },
};
/** One row of NVIDIA's current table of limits (ch04-facts.md, A12): compute capabilities 8.6 and 8.7. */
export const TABLE_1536: Limits = {
  threadSlots: 1536, blockSlots: 16, blockSlotsDerived: false, regs: 65536, shared: null, width: 32, maxThreadsPerBlock: 1024,
};
const TABLE_DEVICE: Device = {
  key: 't1536',
  get label() {
    return tr("A row of NVIDIA's table: 1536 thread slots, 16 block slots", 'Một hàng trong bảng của NVIDIA: 1536 khe luồng, 16 khe khối');
  },
  limits: TABLE_1536,
  get note() {
    return tr(
      "The row for compute capabilities 8.6 and 8.7 in NVIDIA's table of limits, read on 2026-11-20 (ch04-facts.md, A12): 16 blocks, 48 warps (1536 threads) and 64 K registers per SM, 1024 threads per block. No shared-memory figure is used for it here.",
      'Hàng dành cho compute capability 8.6 và 8.7 trong bảng giới hạn của NVIDIA, đọc ngày 2026-11-20 (ch04-facts.md, A12): 16 khối, 48 warp (1536 luồng) và 64 K thanh ghi mỗi SM, 1024 luồng mỗi khối. Ở đây không dùng con số bộ nhớ chia sẻ nào cho nó.',
    );
  },
};
const EX_DEVICES: Device[] = [
  {
    key: 'ex6',
    get label() { return tr('The device of exercise 6', 'Thiết bị của bài tập 6'); },
    limits: EX6,
    get note() { return tr('Exercise 6 (book p. 91): an SM that holds up to 1536 threads and 4 blocks. The exercise gives no register or shared-memory figure. The warp width of 32 is the book\'s.', 'Bài tập 6 (sách p. 91): một SM chứa tối đa 1536 luồng và 4 khối. Bài tập không cho con số thanh ghi hay bộ nhớ chia sẻ. Độ rộng warp 32 là của sách.'); },
  },
  {
    key: 'ex7',
    get label() { return tr('The device of exercise 7', 'Thiết bị của bài tập 7'); },
    limits: EX7,
    get note() { return tr('Exercise 7 (book p. 92): 64 blocks and 2048 threads per SM. The exercise gives no register or shared-memory figure. The warp width of 32 is the book\'s.', 'Bài tập 7 (sách p. 92): 64 khối và 2048 luồng mỗi SM. Bài tập không cho con số thanh ghi hay bộ nhớ chia sẻ. Độ rộng warp 32 là của sách.'); },
  },
];

/** CUDA and HIP devices in this machine's reports that carry per-SM fields. */
function reportedDevices(reports: Reports): Device[] {
  return realBackends(reports).flatMap((b) => {
    const d = reports[b].device as Record<string, unknown>;
    const limits = limitsFromReport(d, b);
    if (!limits) return [];
    const missing = notReported(d);
    return [
      {
        key: b,
        get label() {
          return tr(`Yours: ${String(d.name ?? b)} (${b})`, `Của bạn: ${String(d.name ?? b)} (${b})`);
        },
        limits,
        get note() {
          return (
            tr(`What this device reported in its device properties, in the run of ${reports[b].generated.slice(0, 10)}.`, `Những gì thiết bị này báo cáo trong thuộc tính thiết bị của nó, ở lần chạy ngày ${reports[b].generated.slice(0, 10)}.`) +
            (b === 'hip' ? tr(' On HIP the block-slot figure is computed from the thread slots, so it is shown and not counted as a limit.', ' Trên HIP con số khe khối được tính từ số khe luồng, nên nó được hiển thị nhưng không tính là một giới hạn.') : '') +
            (missing.length ? tr(` Missing or 0, and so left out: ${missing.join(', ')}.`, ` Thiếu hoặc bằng 0, nên bị bỏ qua: ${missing.join(', ')}.`) : '')
          );
        },
      },
    ];
  });
}

// ---------------------------------------------------------------- the budget table
const LIMIT_NAME: Record<LimitKey, string> = {
  get threads() { return tr('Thread slots', 'Khe luồng'); },
  get blocks() { return tr('Block slots', 'Khe khối'); },
  get regs() { return tr('Registers', 'Thanh ghi'); },
  get shared() { return tr('Shared memory', 'Bộ nhớ chia sẻ'); },
};
const LIMIT_LOWER: Record<LimitKey, string> = {
  get threads() { return tr('thread slots', 'khe luồng'); },
  get blocks() { return tr('block slots', 'khe khối'); },
  get regs() { return tr('registers', 'thanh ghi'); },
  get shared() { return tr('shared memory', 'bộ nhớ chia sẻ'); },
};
const BAR: Record<LimitKey, string> = { threads: 'var(--ink)', blocks: 'var(--ink)', regs: 'var(--reg)', shared: 'var(--shared)' };
const stacked = (bytes: number) => (bytes >= KB && bytes % KB === 0 ? `${count(bytes / KB)} KB` : `${count(bytes)} B`); // i18n-ignore: number and unit

function whyNot(key: LimitKey, why: string | undefined): string {
  if (why === 'derived') return tr('on HIP this figure is computed from the thread slots, so it is not a limit of its own', 'trên HIP con số này được tính từ số khe luồng, nên tự nó không phải một giới hạn');
  if (why === 'unused') return key === 'regs' ? tr('registers per thread is set to 0', 'số thanh ghi mỗi luồng được đặt bằng 0') : tr('this kernel uses none', 'kernel này không dùng');
  return tr('no figure for this device', 'không có con số cho thiết bị này');
}

/** One row per budget: what an SM has, what a block takes, how many blocks that allows, and which one binds. */
function BudgetTable({ o }: { o: Occupancy }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="data" style={{ margin: 0 }}>
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.5rem' }}><Translated vi="Ngân sách">Budget</Translated></th>
            <th className="num"><Translated vi="Một SM có">One SM has</Translated></th>
            <th className="num"><Translated vi="Một khối cần">One block needs</Translated></th>
            <th className="num"><Translated vi="Số khối cho phép">Blocks it allows</Translated></th>
            <th><Translated vi={<>Đang dùng với {count(o.blocks)} khối</>}>In use with {plural(o.blocks, 'block')}</Translated></th>
            <th />
          </tr>
        </thead>
        <tbody>
          {o.rows.map((r) => {
            const binds = o.binding.includes(r.key);
            const used = o.blocks * r.need;
            const share = r.cap ? Math.min(100, (100 * used) / r.cap) : 0;
            const show = r.key === 'shared' ? stacked : count;
            return (
              <tr key={r.key} data-budget={r.key} data-fit={r.fit ?? 'none'} data-binds={binds ? '' : undefined} style={binds ? { background: 'var(--rule-soft)' } : undefined}>
                <td style={{ paddingLeft: '0.5rem' }}>
                  {r.key === 'shared' ? <span className="chip shared">{LIMIT_NAME[r.key]}</span> : r.key === 'regs' ? <span className="chip reg">{LIMIT_NAME[r.key]}</span> : LIMIT_NAME[r.key]}
                </td>
                <td className="num">{r.cap != null ? show(r.cap) : tr('n/a', 'không có')}</td>
                {r.fit == null ? (
                  <td colSpan={3} className="muted">
                    <Translated vi={<>Không tính: {whyNot(r.key, r.why)}.</>}>Not counted: {whyNot(r.key, r.why)}.</Translated>
                  </td>
                ) : (
                  <>
                    <td className="num">{show(r.need)}</td>
                    <td className="num">{count(r.fit)}</td>
                    <td style={{ minWidth: '9rem' }}>
                      <div aria-hidden="true" style={{ height: 8, background: 'var(--rule-soft)', borderRadius: 2, margin: '0.25rem 0' }}>
                        <div style={{ width: share + '%', height: '100%', background: BAR[r.key], borderRadius: 2 }} />
                      </div>
                      <span className="muted">
                        <Translated vi={<>{show(used)} trên {show(r.cap as number)} ({fmt(share)}%)</>}>
                          {show(used)} of {show(r.cap as number)} ({fmt(share)}%)
                        </Translated>
                      </span>
                    </td>
                  </>
                )}
                <td style={{ paddingRight: '0.5rem' }}>{binds ? <b><Translated vi="là giới hạn">binds</Translated></b> : null}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const listOf = (keys: LimitKey[]) => {
  const names = keys.map((k) => LIMIT_LOWER[k]);
  return names.length > 1 ? names.slice(0, -1).join(', ') + tr(' and ', ' và ') + names[names.length - 1] : (names[0] ?? '');
};
const cap1 = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/** One sentence on why occupancy is what it is. */
function verdictOf(o: Occupancy, kernel: Kernel, threadSlots: number | null): ReactNode {
  if (!o.usable)
    return (
      <Translated vi={<>Trước hết, occupancy cần hai con số: số khe luồng của một SM và số luồng trong một khối.</>}>
        <>Occupancy needs two numbers before anything else: the thread slots of one SM and the threads in one block.</>
      </Translated>
    );
  if (o.blocks === 0)
    return (
      <Translated
        vi={
          <>
            <span className="chip fail">không vừa</span> Không một khối nào vừa trên một SM: một khối cần nhiều {listOf(o.binding)} hơn
            SM có. Hướng dẫn của NVIDIA nói một kernel như vậy sẽ khởi chạy thất bại.
          </>
        }
      >
        <>
          <span className="chip fail">does not fit</span> Not even one block fits on an SM: a block needs more {listOf(o.binding)} than
          the SM has. NVIDIA's guide says such a kernel fails to launch.
        </>
      </Translated>
    );
  if (o.idleSlots === 0)
    return (
      <Translated
        vi={
          <>
            <b>Mọi khe luồng đều đã được dùng.</b> {count(o.blocks)} khối, mỗi khối {count(o.warpsPerBlock)} warp.{' '}
            {o.binding.length > 1 ? `${cap1(listOf(o.binding))} cùng cạn một lúc.` : 'Không ngân sách nào khác cạn trước.'}
          </>
        }
      >
        <>
          <b>Every thread slot is taken.</b> {cap1(plural(o.blocks, 'block'))} of {plural(o.warpsPerBlock, 'warp')} each.{' '}
          {o.binding.length > 1 ? `${cap1(listOf(o.binding))} run out together.` : 'No other budget runs out first.'}
        </>
      </Translated>
    );
  if (o.binding.length === 1 && o.binding[0] === 'threads')
    return (
      <Translated
        vi={
          <>
            <b>Khe luồng là giới hạn, nhưng không trọn vẹn.</b> {count(o.blocks)} khối chiếm {count(o.blocks * o.slotsPerBlock)} trên{' '}
            {count(threadSlots ?? 0)} khe, và thêm một khối {count(kernel.threads)} luồng nữa thì không vừa{' '}
            {count(o.idleSlots)} khe còn lại.
          </>
        }
      >
        <>
          <b>Thread slots bind, but not cleanly.</b> {cap1(plural(o.blocks, 'block'))} take {count(o.blocks * o.slotsPerBlock)} of{' '}
          {count(threadSlots ?? 0)} slots, and one more block of {count(kernel.threads)} threads does not fit in the{' '}
          {count(o.idleSlots)} that remain.
        </>
      </Translated>
    );
  return (
    <Translated
      vi={
        <>
          <b>{cap1(listOf(o.binding))} là giới hạn.</b>{' '}
          Một SM chỉ có chỗ cho {count(o.blocks)} khối của kernel này, để trống {count(o.idleSlots)} khe luồng.
        </>
      }
    >
      <>
        <b>
          {cap1(listOf(o.binding))} {o.binding.length === 1 && o.binding[0] === 'shared' ? 'binds' : 'bind'}.
        </b>{' '}
        An SM has room for only {plural(o.blocks, 'block')} of this kernel, which leaves {count(o.idleSlots)} thread slots empty.
      </>
    </Translated>
  );
}

// ---------------------------------------------------------------- the calculator
interface CalcPreset {
  key: string;
  label: string;
  kernel: Kernel;
}
const CALC_PRESETS: CalcPreset[] = [
  ...BOOK_BLOCK_SIZES.map((t) => ({ key: 't' + t, label: String(t), kernel: { threads: t, regs: 0, shared: 0 } })),
  { key: 't48', get label() { return tr('48 (not whole warps)', '48 (không trọn warp)'); }, kernel: { threads: 48, regs: 0, shared: 0 } },
];

function OccupancyCalculator() {
  const { reports, loading } = useReports();
  const mine = useMemo(() => reportedDevices(reports), [reports]);
  const devices = useMemo(() => [BOOK_DEVICE, ...mine, TABLE_DEVICE, ...EX_DEVICES], [mine]);
  const [key, setKey] = useState('book');
  const [custom, setCustom] = useState({
    threadSlots: String(BOOK.threadSlots), blockSlots: String(BOOK.blockSlots), regs: String(BOOK.regs), shared: String(BOOK.shared), width: String(BOOK.width),
  });
  const [work, setWork] = useState({ threads: '256', regs: '0', shared: '0' });

  const customLimits: Limits = {
    threadSlots: toInt(custom.threadSlots) || null,
    blockSlots: toInt(custom.blockSlots) || null,
    blockSlotsDerived: false,
    regs: toInt(custom.regs) || null,
    shared: toInt(custom.shared) || null,
    width: toInt(custom.width) || null,
    maxThreadsPerBlock: null,
  };
  const dev: Device =
    key === 'custom'
      ? { key: 'custom', label: tr('Custom numbers', 'Số tự nhập'), limits: customLimits, note: tr("Limits you typed. They start from the book's example. A field left at 0 is not counted.", 'Các giới hạn bạn tự nhập. Chúng bắt đầu từ ví dụ của sách. Ô nào để 0 thì không được tính.') }
      : (devices.find((d) => d.key === key) ?? BOOK_DEVICE);
  const kernel: Kernel = { threads: toInt(work.threads), regs: toInt(work.regs), shared: toInt(work.shared) };
  const o = occupancy(dev.limits, kernel);
  const active = (p: CalcPreset) => p.kernel.threads === kernel.threads && p.kernel.regs === kernel.regs && p.kernel.shared === kernel.shared;

  const field = (label: string, value: string, set: (v: string) => void, hint?: string) => (
    <label className="field">
      {label}
      <input type="number" min="0" step="1" value={value} aria-label={label} onChange={(e) => set(e.target.value)} />
      {hint ? <span style={{ fontSize: '0.74rem' }}>{hint}</span> : null}
    </label>
  );
  const fillAt = regsAtFullOccupancy(dev.limits);
  const smallest = smallestFillingBlock(dev.limits);

  return (
    <Bench
      title={tr('Occupancy calculator', 'Máy tính occupancy')}
      task={tr('Find the budget that runs out first. Then change one input until a different one does.', 'Tìm ngân sách cạn trước. Rồi đổi một đầu vào cho đến khi một ngân sách khác cạn trước.')}
      foot={
        <Translated vi={<>{dev.note} Thanh ghi được tính bằng số luồng × số thanh ghi mỗi luồng, theo phép tính của sách.</>}>
          <>{dev.note} Registers are counted as threads × registers per thread, the book's arithmetic.</>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-start' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Giới hạn của một SM">Limits of one SM</Translated>
          <select value={key} onChange={(e) => setKey(e.target.value)} aria-label={tr('Limits of one SM', 'Giới hạn của một SM')}>
            {devices.map((d) => (
              <option key={d.key} value={d.key}>
                {d.label}
              </option>
            ))}
            <option value="custom">{tr('Custom numbers', 'Số tự nhập')}</option>
          </select>
        </label>
        {key === 'custom' ? (
          <>
            {field(tr('Thread slots per SM', 'Số khe luồng mỗi SM'), custom.threadSlots, (v) => setCustom({ ...custom, threadSlots: v }))}
            {field(tr('Block slots per SM', 'Số khe khối mỗi SM'), custom.blockSlots, (v) => setCustom({ ...custom, blockSlots: v }))}
            {field(tr('Registers per SM', 'Số thanh ghi mỗi SM'), custom.regs, (v) => setCustom({ ...custom, regs: v }))}
            {field(tr('Shared memory per SM (bytes)', 'Bộ nhớ chia sẻ mỗi SM (byte)'), custom.shared, (v) => setCustom({ ...custom, shared: v }))}
            {field(tr('Threads in a warp', 'Số luồng trong một warp'), custom.width, (v) => setCustom({ ...custom, width: v }))}
          </>
        ) : null}
      </div>
      {!loading && mine.length === 0 ? (
        <p className="muted" data-out="no-device" style={{ margin: '0 0 0.8rem' }}>
          <Translated
            vi={
              <>
                Báo cáo của máy này không có thiết bị CUDA hay HIP nào, nên danh sách không có mục nào của bạn. Hãy chạy <code>./verify.sh</code>{' '}
                trên một máy có bộ công cụ CUDA hoặc ROCm và mở trang này ở đó. Một thiết bị Metal không báo cáo giới hạn nào trong số
                này; lý do nằm ở phía dưới.
              </>
            }
          >
            No CUDA or HIP device in this machine's reports, so the list has no entry of yours. Run <code>./verify.sh</code>{' '}
            on a machine with the CUDA or ROCm toolkit and open this page there. A Metal device reports none of these
            limits; the reason is further down.
          </Translated>
        </p>
      ) : null}
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-start' }}>
        {field(tr('Threads per block', 'Số luồng mỗi khối'), work.threads, (v) => setWork({ ...work, threads: v }))}
        {field(tr('Registers per thread', 'Số thanh ghi mỗi luồng'), work.regs, (v) => setWork({ ...work, regs: v }), tr('0 leaves registers out', '0 nghĩa là bỏ qua thanh ghi'))}
        {field(tr('Shared memory per block (bytes)', 'Bộ nhớ chia sẻ mỗi khối (byte)'), work.shared, (v) => setWork({ ...work, shared: v }), tr('0 leaves it out', '0 nghĩa là bỏ qua'))}
      </div>
      <div className="row" style={{ marginBottom: '1rem' }} role="group" aria-label={tr('Block sizes the book works through', 'Các kích thước khối sách tính qua')}>
        <span className="muted"><Translated vi="Kích thước khối của sách">The book's block sizes</Translated></span>
        {CALC_PRESETS.map((p) => (
          <button key={p.key} className="quiet" data-preset={p.key} aria-pressed={active(p)}
            onClick={() => setWork({ threads: String(p.kernel.threads), regs: String(p.kernel.regs), shared: String(p.kernel.shared) })}>
            {p.label}
          </button>
        ))}
      </div>

      <div className="cols plot">
        <div className="panel">
          <BudgetTable o={o} />
          {o.widthAssumed ? (
            <p className="muted" style={{ margin: '0.6rem 0 0' }}>
              <Translated vi="Thiết bị này không có độ rộng warp, nên một khối được đếm theo từng luồng và không thêm phần đệm.">
                No warp width for this device, so a block is counted thread by thread and no padding is added.
              </Translated>
            </p>
          ) : null}
        </div>
        <div className="panel" aria-live="polite">
          <p className="muted" style={{ margin: 0 }}>
            <Translated vi="Occupancy (theo warp)">Occupancy (warps)</Translated>
          </p>
          <p style={{ fontSize: '2rem', fontWeight: 600, margin: '0 0 0.2rem', lineHeight: 1.1 }}>
            <output data-out="occupancy" data-value={o.usable ? pct(o.byWarps) : 'n/a'} /* i18n-ignore: data attribute */ data-binding={o.binding.join('+')}>
              {o.usable ? pct(o.byWarps) + '%' : tr('n/a', 'không có')}
            </output>
          </p>
          {o.usable ? (
            <p className="muted" style={{ margin: '0 0 0.6rem' }}>
              <Translated
                vi={
                  <>
                    <span data-out="blocks">{count(o.blocks)}</span> khối × {count(o.warpsPerBlock)} warp ={' '}
                    <span data-out="warps">{count(o.residentWarps)}</span> trên {count(o.warpSlots ?? 0)} khe warp.{' '}
                    <span data-out="idle">{count(o.idleSlots)}</span> khe luồng rảnh.
                  </>
                }
              >
                <span data-out="blocks">{count(o.blocks)}</span> {o.blocks === 1 ? 'block' : 'blocks'} × {plural(o.warpsPerBlock, 'warp')} ={' '}
                <span data-out="warps">{count(o.residentWarps)}</span> of {count(o.warpSlots ?? 0)} warp slots.{' '}
                <span data-out="idle">{count(o.idleSlots)}</span> thread slots idle.
              </Translated>
            </p>
          ) : null}
          {o.usable && o.paddedLanes > 0 ? (
            <p data-out="by-threads" data-value={pct(o.byThreads)} style={{ margin: '0 0 0.6rem' }}>
              <Translated
                vi={
                  <>
                    Tính theo luồng thì là <b>{pct(o.byThreads)}%</b>: {count(o.residentThreads)} luồng chiếm {count(o.blocks * o.slotsPerBlock)}{' '}
                    khe, vì warp cuối của mỗi khối mang {o.slotsPerBlock - kernel.threads} lane đệm.
                  </>
                }
              >
                By threads it is <b>{pct(o.byThreads)}%</b>: {count(o.residentThreads)} threads hold {count(o.blocks * o.slotsPerBlock)}{' '}
                slots, because each block's last warp carries {o.slotsPerBlock - kernel.threads} padded lanes.
              </Translated>
            </p>
          ) : null}
          <p style={{ margin: 0 }} data-out="verdict">
            {verdictOf(o, kernel, dev.limits.threadSlots)}
          </p>
          {o.tooLarge ? (
            <p style={{ margin: '0.6rem 0 0' }}>
              <Translated
                vi={
                  <>
                    <span className="chip fail">khởi chạy không hợp lệ</span> Thiết bị này chấp nhận tối đa {count(dev.limits.maxThreadsPerBlock ?? 0)}{' '}
                    luồng trong một khối.
                  </>
                }
              >
                <span className="chip fail">invalid launch</span> This device accepts at most {count(dev.limits.maxThreadsPerBlock ?? 0)}{' '}
                threads in a block.
              </Translated>
            </p>
          ) : null}
          {fillAt != null || smallest != null ? (
            <p className="muted" style={{ margin: '0.6rem 0 0' }}>
              {smallest != null ? (
                <Translated vi={<>Khối nhỏ nhất có thể lấp đầy các khe luồng: <span data-out="smallest">{count(smallest)}</span> luồng. </>}>
                  <>Smallest block that can fill the thread slots: <span data-out="smallest">{count(smallest)}</span> threads. </>
                </Translated>
              ) : null}
              {fillAt != null ? (
                <Translated vi={<>Số thanh ghi tối đa mỗi luồng khi mọi khe đều đầy: <span data-out="fill-regs">{fillAt}</span>.</>}>
                  <>Most registers per thread with every slot filled: <span data-out="fill-regs">{fillAt}</span>.</>
                </Translated>
              ) : null}
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the staircase
const PW = 430;
const PH = 190;
const PL = 40;
const PR = 12;
const PT = 12;
const PB = 28;

function Staircase({ points, x, axis, label }: { points: ScanPoint[]; x: number; axis: 'regs' | 'threads'; label: string }) {
  const n = points.length;
  const w = PW - PL - PR;
  const h = PH - PT - PB;
  const slot = w / n;
  const px = (i: number) => PL + (i + 0.5) * slot;
  const py = (occ: number) => PT + (1 - occ) * h;
  const at = Math.max(0, points.findIndex((p) => p.x === x));
  const cur = points[at];
  const colour = axis === 'regs' ? 'var(--reg)' : 'var(--ink)';
  let path = '';
  points.forEach((p, i) => {
    const x0 = PL + i * slot;
    path += `${i === 0 ? 'M' : 'L'}${x0.toFixed(2)},${py(p.occupancy).toFixed(2)} L${(x0 + slot).toFixed(2)},${py(p.occupancy).toFixed(2)} `;
  });
  const ticks = axis === 'regs' ? [1, 32, 64, 128, 192, 255] : [points[0].x, points[Math.floor((n - 1) / 2)].x, points[n - 1].x];
  return (
    <svg viewBox={`0 0 ${PW} ${PH}`} role="img" aria-label={label} data-plot={axis} data-x={cur?.x} data-occ={cur ? pct(cur.occupancy) : ''} style={{ width: '100%', display: 'block' }}>
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={PL} x2={PW - PR} y1={py(g)} y2={py(g)} stroke="var(--rule-soft)" />
          <text x={PL - 6} y={py(g) + 3.5} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{g * 100}%</text>
        </g>
      ))}
      {axis === 'threads'
        ? points.map((p, i) => (
            <rect key={p.x} x={PL + i * slot + slot * 0.15} y={py(p.occupancy)} width={slot * 0.7} height={Math.max(0, PT + h - py(p.occupancy))}
              fill={i === at ? 'var(--ink)' : 'var(--ink-3)'} opacity={i === at ? 1 : 0.55} />
          ))
        : <path d={path} fill="none" stroke={colour} strokeWidth="1.8" />}
      {ticks.map((t) => {
        const i = points.findIndex((p) => p.x === t);
        return i < 0 ? null : (
          <text key={t} x={px(i)} y={PH - 10} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{t}</text>
        );
      })}
      {cur ? (
        <g>
          <line x1={px(at)} x2={px(at)} y1={PT} y2={PT + h} stroke="var(--ink)" strokeDasharray="3 3" strokeWidth="0.8" />
          <circle cx={px(at)} cy={py(cur.occupancy)} r="4.5" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.8" />
          <text x={Math.min(PW - PR - 4, Math.max(PL + 4, px(at) + (at > n * 0.7 ? -8 : 8)))} y={Math.max(PT + 10, py(cur.occupancy) - 8)}
            fontSize="11" fontWeight="600" textAnchor={at > n * 0.7 ? 'end' : 'start'}
            style={{ paintOrder: 'stroke', stroke: 'var(--page)', strokeWidth: 3.5, strokeLinejoin: 'round' }}>
            {cur.x}: {pct(cur.occupancy)}%
          </text>
        </g>
      ) : null}
    </svg>
  );
}

/** A slider that spans the plot area above it, so that dragging moves the marker under the pointer. */
const SLIDER = { width: `${(100 * (PW - PL - PR)) / PW}%`, marginLeft: `${(100 * PL) / PW}%` } as const;

const stepText = (s: Step) => `${pct(s.from)}% → ${pct(s.to)}%`;

function CliffExplorer() {
  const { reports } = useReports();
  const devices = useMemo(() => [BOOK_DEVICE, ...reportedDevices(reports), TABLE_DEVICE], [reports]);
  const [key, setKey] = useState('book');
  const dev = devices.find((d) => d.key === key) ?? BOOK_DEVICE;
  // Shared memory is left out here so that registers and slots are the only budgets in play.
  const limits: Limits = useMemo(() => ({ ...dev.limits, shared: null }), [dev]);
  const tValues = useMemo(() => threadValues(limits), [limits]);
  const [threadsRaw, setThreads] = useState(512);
  const [regs, setRegs] = useState(31);
  const threads = tValues.includes(threadsRaw) ? threadsRaw : tValues.reduce((a, b) => (Math.abs(b - threadsRaw) < Math.abs(a - threadsRaw) ? b : a), tValues[0]);
  const kernel: Kernel = { threads, regs, shared: 0 };
  const o = occupancy(limits, kernel);
  const byRegs = useMemo(() => scan(limits, { threads, regs: 0, shared: 0 }, 'regs', regValues()), [limits, threads]);
  const byThreads = useMemo(() => scan(limits, { threads: 0, regs, shared: 0 }, 'threads', tValues), [limits, regs, tValues]);
  const regSteps = steps(byRegs);
  const next = regSteps.find((s) => s.at > regs);
  const width = limits.width ?? 1;
  const hasRegs = limits.regs != null;
  const lost = next ? next.from - next.to : 0;

  return (
    <Bench
      title={tr('Find the cliff', 'Tìm vách đá')}
      task={tr('Drag registers per thread up from 31, one at a time. Then change the block size and see where the steps go.', 'Kéo số thanh ghi mỗi luồng tăng dần từ 31, mỗi lần một. Rồi đổi kích thước khối và xem các bậc dời đi đâu.')}
      foot={
        <Translated vi={<>{dev.note} Trình khám phá này bỏ qua bộ nhớ chia sẻ. Thanh ghi được tính bằng số luồng × số thanh ghi mỗi luồng, theo phép tính của sách; con số của một kernel thật đến từ hàm occupancy của nhà cung cấp.</>}>
          <>{dev.note} Shared memory is left out of this explorer. Registers are counted as threads × registers per thread, the book's arithmetic; a real kernel's count comes from the vendor's occupancy function.</>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-start' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Giới hạn của một SM">Limits of one SM</Translated>
          <select value={key} onChange={(e) => setKey(e.target.value)} aria-label={tr('Limits of one SM for the staircase', 'Giới hạn của một SM cho biểu đồ bậc thang')}>
            {devices.map((d) => (
              <option key={d.key} value={d.key}>
                {d.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="cols">
        <div className="panel">
          <p className="ui" style={{ margin: '0 0 0.4rem', fontWeight: 600 }}>
            <Translated vi={<>Occupancy theo số thanh ghi mỗi luồng, ở {count(threads)} luồng mỗi khối</>}>
              Occupancy against registers per thread, at {count(threads)} threads per block
            </Translated>
          </p>
          {hasRegs ? (
            <>
              <Staircase points={byRegs} x={regs} axis="regs" /* i18n-ignore: internal key */ label={tr(`Occupancy against registers per thread at ${threads} threads per block`, `Occupancy theo số thanh ghi mỗi luồng ở ${threads} luồng mỗi khối`)} />
              <label className="field" style={{ display: 'flex', width: '100%', marginTop: '0.3rem' }}>
                <Translated vi={<>Số thanh ghi mỗi luồng = {regs}</>}>Registers per thread = {regs}</Translated>
                <input type="range" min={1} max={MAX_REGS_PER_THREAD} step={1} value={regs} aria-label={tr('Registers per thread', 'Số thanh ghi mỗi luồng')}
                  style={SLIDER} onChange={(e) => setRegs(+e.target.value)} />
              </label>
              <div className="row" style={{ marginTop: '0.5rem' }} role="group" aria-label={tr('Steps along registers per thread', 'Các bậc theo số thanh ghi mỗi luồng')}>
                <span className="muted"><Translated vi="Các bậc tại">Steps at</Translated></span>
                {regSteps.length ? regSteps.map((s) => (
                  <button key={s.at} className="quiet" data-step={s.at} aria-pressed={regs === s.at} onClick={() => setRegs(s.at)} title={stepText(s)}>
                    {s.at}: {stepText(s)}
                  </button>
                )) : <span className="muted"><Translated vi="không có: ở kích thước khối này thanh ghi không bao giờ là giới hạn">none: registers never bind at this block size</Translated></span>}
              </div>
            </>
          ) : (
            <p className="muted" data-out="no-regs" style={{ margin: 0 }}>
              <Translated vi="Thiết bị này không báo cáo số thanh ghi mỗi SM, nên không có bậc thang thanh ghi nào để vẽ cho nó.">
                This device did not report its registers per SM, so there is no register staircase to draw for it.
              </Translated>
            </p>
          )}
        </div>
        <div className="panel">
          <p className="ui" style={{ margin: '0 0 0.4rem', fontWeight: 600 }}>
            <Translated vi={<>Occupancy theo số luồng mỗi khối, ở {regs} thanh ghi mỗi luồng</>}>
              Occupancy against threads per block, at {regs} registers per thread
            </Translated>
          </p>
          <Staircase points={byThreads} x={threads} axis="threads" /* i18n-ignore: internal key */ label={tr(`Occupancy against threads per block at ${regs} registers per thread`, `Occupancy theo số luồng mỗi khối ở ${regs} thanh ghi mỗi luồng`)} />
          <label className="field" style={{ display: 'flex', width: '100%', marginTop: '0.3rem' }}>
            <Translated vi={<>Số luồng mỗi khối = {count(threads)} ({count(threads / width)} warp)</>}>
              Threads per block = {count(threads)} ({plural(threads / width, 'warp')})
            </Translated>
            <input type="range" min={tValues[0]} max={tValues[tValues.length - 1]} step={width} value={threads} aria-label={tr('Threads per block', 'Số luồng mỗi khối')}
              style={SLIDER} onChange={(e) => setThreads(+e.target.value)} />
          </label>
          <Translated
            vi={
              <p className="muted" style={{ margin: '0.5rem 0 0' }}>
                Mỗi số nguyên warp một cột. Biểu đồ này không phải bậc thang: nó trũng xuống ở bất cứ chỗ nào một số nguyên
                khối không thể dùng hết ngân sách đang là giới hạn, như 768 không thể dùng hết 2048 khe luồng.
              </p>
            }
          >
            <p className="muted" style={{ margin: '0.5rem 0 0' }}>
              One bar for each whole number of warps. This one is not a staircase: it dips wherever a whole number of
              blocks cannot use up the budget that binds, as 768 cannot use up 2048 thread slots.
            </p>
          </Translated>
        </div>
      </div>
      <div className="panel" aria-live="polite" style={{ marginTop: '1.2rem' }}>
        <p style={{ margin: '0 0 0.4rem' }}>
          <Translated vi={<>{count(threads)} luồng mỗi khối ở {regs} thanh ghi mỗi luồng:{' '}</>}>
            {count(threads)} threads per block at {regs} registers per thread:{' '}
          </Translated>
          <b>
            <output data-out="cliff-occupancy" data-value={pct(o.byWarps)} data-blocks={o.blocks} data-binding={o.binding.join('+')}>
              <Translated vi={<>{count(o.blocks)} khối, {pct(o.byWarps)}%</>}>
                {plural(o.blocks, 'block')}, {pct(o.byWarps)}%
              </Translated>
            </output>
          </b>
          {o.blocks > 0 ? (
            <Translated vi={<>. {cap1(listOf(o.binding))} là giới hạn.</>}>
              <>. {cap1(listOf(o.binding))} {o.binding.length === 1 && o.binding[0] === 'shared' ? 'binds' : 'bind'}.</>
            </Translated>
          ) : (
            <Translated vi={<>. Không một khối nào vừa.</>}><>. Not even one block fits.</></Translated>
          )}
        </p>
        <p style={{ margin: 0 }} data-out="next-step">
          {!hasRegs ? (
            <Translated vi={<>Thiết bị này không tính thanh ghi.</>}><>Registers are not counted for this device.</></Translated>
          ) : next ? (
            <Translated
              vi={
                <>
                  {next.at - regs === 1 ? <b>Thêm một thanh ghi mỗi luồng</b> : <b>Thêm {next.at - regs} thanh ghi mỗi luồng</b>} là một khối
                  không còn vừa: occupancy đi từ {pct(next.from)}% xuống {pct(next.to)}%, tức là{' '}
                  {Math.abs(lost - 0.5) < 1e-9 ? 'một nửa SM' : Math.abs(lost - 0.25) < 1e-9 ? 'một phần tư SM' : `${pct(lost)}% của SM`}.
                </>
              }
            >
              <>
                {next.at - regs === 1 ? <b>One more register per thread</b> : <b>{next.at - regs} more registers per thread</b>} and a block
                stops fitting: occupancy goes from {pct(next.from)}% to {pct(next.to)}%, which is{' '}
                {Math.abs(lost - 0.5) < 1e-9 ? 'half of the SM' : Math.abs(lost - 0.25) < 1e-9 ? 'a quarter of the SM' : `${pct(lost)}% of the SM`}.
              </>
            </Translated>
          ) : (
            <Translated vi={<>Không còn bậc nào nữa: không phần nào của kernel này vừa ở số thanh ghi này hoặc cao hơn.</>}>
              <>No further step: nothing of this kernel fits at this register count or above.</>
            </Translated>
          )}
        </p>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- exercises 7 and 8
const EX7_CHOICES = (): [string, string][] => [
  ['', tr('your answer', 'câu trả lời của bạn')],
  ['no', tr('not possible', 'không thể')],
  ['25', tr('possible, 25%', 'có thể, 25%')],
  ['50', tr('possible, 50%', 'có thể, 50%')],
  ['75', tr('possible, 75%', 'có thể, 75%')],
  ['100', tr('possible, 100%', 'có thể, 100%')],
];
const EX8_CHOICES = (): [string, string][] => [
  ['', tr('your answer', 'câu trả lời của bạn')],
  ['full', tr('full occupancy', 'occupancy đầy đủ')],
  ['threads', tr('no: thread slots', 'không: khe luồng')],
  ['blocks', tr('no: block slots', 'không: khe khối')],
  ['regs', tr('no: registers', 'không: thanh ghi')],
];
const LETTER = ['a', 'b', 'c', 'd', 'e'];

/** What the model says for each part, in the vocabulary of the answer lists. */
export const EX7_EXPECTED = EX7_ASSIGNMENTS.map((a) => {
  const r = assignment(EX7, a.blocks, a.threads);
  return r.possible ? String(pct(r.occupancy)) : 'no';
});
export const EX8_EXPECTED = EX8_KERNELS.map((k) => {
  const o = occupancy(EX8, k);
  return o.byWarps === 1 ? 'full' : o.binding[0];
});

function Exercises() {
  const [a7, setA7] = useState<string[]>(EX7_ASSIGNMENTS.map(() => ''));
  const [a8, setA8] = useState<string[]>(EX8_KERNELS.map(() => ''));
  const [shown7, setShown7] = useState(false);
  const [shown8, setShown8] = useState(false);
  const mark = (ok: boolean) => <span className={'chip ' + (ok ? 'ok' : 'fail')}>{ok ? tr('right', 'đúng') : tr('not this', 'không phải')}</span>;
  const pick = (value: string, set: (v: string) => void, choices: [string, string][], label: string) => (
    <select value={value} aria-label={label} onChange={(e) => set(e.target.value)}>
      {choices.map(([v, text]) => (
        <option key={v} value={v} disabled={v === ''}>
          {text}
        </option>
      ))}
    </select>
  );
  const right7 = a7.filter((v, i) => v === EX7_EXPECTED[i]).length;
  const right8 = a8.filter((v, i) => v === EX8_EXPECTED[i]).length;

  return (
    <Bench
      title={tr('Exercises 7 and 8', 'Bài tập 7 và 8')}
      task={tr('Answer every part from the limits alone, then check. The bars appear after your answers, not before.', 'Trả lời mọi phần chỉ dựa vào các giới hạn, rồi kiểm tra. Các thanh hiện ra sau khi bạn trả lời, không phải trước.')}
      foot={tr("The devices and kernels are the book's (pp. 91–92). The book prints no solutions: the answers shown are this page's model.", 'Các thiết bị và kernel là của sách (pp. 91–92). Sách không in lời giải: các đáp án hiển thị là của mô hình trang này.')}
    >
      <div className="cols">
        <div className="panel" data-ex="7">
          <Translated
            vi={
              <p style={{ margin: '0 0 0.6rem' }}>
                <b>Bài tập 7.</b> Một thiết bị cho phép {count(EX7.blockSlots as number)} khối và {count(EX7.threadSlots as number)} luồng trên một
                SM. Với mỗi cách gán khối cho một SM: có thể không, và occupancy là bao nhiêu?
              </p>
            }
          >
            <p style={{ margin: '0 0 0.6rem' }}>
              <b>Exercise 7.</b> A device allows {count(EX7.blockSlots as number)} blocks and {count(EX7.threadSlots as number)} threads on an
              SM. For each assignment of blocks to one SM: is it possible, and at what occupancy?
            </p>
          </Translated>
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th><Translated vi="Trên một SM">On one SM</Translated></th>
                <th><Translated vi="Câu trả lời của bạn">Your answer</Translated></th>
                <th>{shown7 ? tr('The model', 'Mô hình') : ''}</th>
              </tr>
            </thead>
            <tbody>
              {EX7_ASSIGNMENTS.map((a, i) => {
                const r = assignment(EX7, a.blocks, a.threads);
                return (
                  <tr key={i} data-part={LETTER[i]}>
                    <td><Translated vi={<>({LETTER[i]}) {a.blocks} khối {a.threads} luồng</>}>({LETTER[i]}) {a.blocks} blocks of {a.threads} threads</Translated></td>
                    <td>{pick(a7[i], (v) => { setA7(a7.map((x, k) => (k === i ? v : x))); setShown7(false); }, EX7_CHOICES(), tr(`Exercise 7 (${LETTER[i]})`, `Bài tập 7 (${LETTER[i]})`))}</td>
                    <td data-model={shown7 ? EX7_EXPECTED[i] : undefined}>
                      {shown7 ? (
                        <Translated
                          vi={
                            <>
                              {mark(a7[i] === EX7_EXPECTED[i])} {a.blocks} trên {EX7.blockSlots} khe khối, {count(a.blocks * a.threads)} trên{' '}
                              {count(EX7.threadSlots as number)} khe luồng: {r.possible ? `có thể, ${pct(r.occupancy)}%` : 'không thể'}
                            </>
                          }
                        >
                          <>
                            {mark(a7[i] === EX7_EXPECTED[i])} {a.blocks} of {EX7.blockSlots} block slots, {count(a.blocks * a.threads)} of{' '}
                            {count(EX7.threadSlots as number)} thread slots: {r.possible ? `possible, ${pct(r.occupancy)}%` : 'not possible'}
                          </>
                        </Translated>
                      ) : null}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="row" style={{ marginTop: '0.7rem' }}>
            <button className="primary" disabled={a7.some((v) => v === '')} onClick={() => setShown7(true)}>
              <Translated vi="Kiểm tra bài tập 7">Check exercise 7</Translated>
            </button>
            {shown7 ? (
              <Translated vi={<span data-out="ex7-score">Đúng {right7} trên {a7.length}.</span>}>
                <span data-out="ex7-score">{right7} of {a7.length} right.</span>
              </Translated>
            ) : (
              <span className="muted"><Translated vi="Hãy trả lời đủ cả năm trước.">Answer all five first.</Translated></span>
            )}
          </div>
          {shown7 ? (
            <Translated
              vi={
                <p className="muted" style={{ margin: '0.6rem 0 0' }}>
                  Cả năm đều nằm trong cả hai giới hạn. Điều khác nhau là SM đầy đến đâu. Ba trong số đó để trống một nửa số khe
                  luồng: với các kích thước khối đó, SM có thể chứa gấp đôi số khối mà cách gán đặt lên nó.
                </p>
              }
            >
              <p className="muted" style={{ margin: '0.6rem 0 0' }}>
                All five fit within both limits. What differs is how full the SM is. Three of them leave half the thread
                slots empty: with those block sizes the SM could hold twice as many blocks as the assignment puts on it.
              </p>
            </Translated>
          ) : null}
        </div>
        <div className="panel" data-ex="8">
          <Translated
            vi={
              <p style={{ margin: '0 0 0.6rem' }}>
                <b>Bài tập 8.</b> Một thiết bị cho phép {count(EX8.threadSlots as number)} luồng, {count(EX8.blockSlots as number)} khối và{' '}
                {count(EX8.regs as number)} thanh ghi trên một SM. Mỗi kernel có đạt được occupancy đầy đủ không? Nếu không, điều gì cản nó?
              </p>
            }
          >
            <p style={{ margin: '0 0 0.6rem' }}>
              <b>Exercise 8.</b> A device allows {count(EX8.threadSlots as number)} threads, {count(EX8.blockSlots as number)} blocks and{' '}
              {count(EX8.regs as number)} registers on an SM. Can each kernel reach full occupancy? If not, what stops it?
            </p>
          </Translated>
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th><Translated vi="Kernel">Kernel</Translated></th>
                <th><Translated vi="Câu trả lời của bạn">Your answer</Translated></th>
              </tr>
            </thead>
            <tbody>
              {EX8_KERNELS.map((k, i) => (
                <tr key={i} data-part={LETTER[i]}>
                  <td>
                    <Translated vi={<>({LETTER[i]}) {k.threads} luồng mỗi khối, {k.regs} thanh ghi mỗi luồng</>}>
                      ({LETTER[i]}) {k.threads} threads per block, {k.regs} registers per thread
                    </Translated>
                  </td>
                  <td data-model={shown8 ? EX8_EXPECTED[i] : undefined}>
                    {pick(a8[i], (v) => { setA8(a8.map((x, j) => (j === i ? v : x))); setShown8(false); }, EX8_CHOICES(), tr(`Exercise 8 (${LETTER[i]})`, `Bài tập 8 (${LETTER[i]})`))}{' '}
                    {shown8 ? mark(a8[i] === EX8_EXPECTED[i]) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="row" style={{ marginTop: '0.7rem' }}>
            <button className="primary" disabled={a8.some((v) => v === '')} onClick={() => setShown8(true)}>
              <Translated vi="Kiểm tra bài tập 8">Check exercise 8</Translated>
            </button>
            {shown8 ? (
              <Translated vi={<span data-out="ex8-score">Đúng {right8} trên {a8.length}.</span>}>
                <span data-out="ex8-score">{right8} of {a8.length} right.</span>
              </Translated>
            ) : (
              <span className="muted"><Translated vi="Hãy trả lời đủ cả ba trước.">Answer all three first.</Translated></span>
            )}
          </div>
        </div>
      </div>
      {shown8
        ? EX8_KERNELS.map((k, i) => {
            const o = occupancy(EX8, k);
            return (
              <div className="panel" key={i} data-ex8-bars={LETTER[i]} style={{ marginTop: '1rem' }}>
                <p style={{ margin: '0 0 0.5rem' }}>
                  <Translated
                    vi={
                      <>
                        <b>8({LETTER[i]})</b> {k.threads} luồng mỗi khối, {k.regs} thanh ghi mỗi luồng:{' '}
                        <b>{count(o.blocks)} khối, {pct(o.byWarps)}%</b>. {verdictOf(o, k, EX8.threadSlots)}
                      </>
                    }
                  >
                    <b>8({LETTER[i]})</b> {k.threads} threads per block, {k.regs} registers per thread:{' '}
                    <b>{plural(o.blocks, 'block')}, {pct(o.byWarps)}%</b>. {verdictOf(o, k, EX8.threadSlots)}
                  </Translated>
                </p>
                <BudgetTable o={o} />
              </div>
            );
          })
        : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- lab 4.4, once it has run
const LAB4 = 'ch04/lab4_blocksize';
const SIZE_KEYS = ['threads_per_block', 'block', 'block_size', 'threadsPerBlock', 'threads_per_threadgroup', 'threadgroup'];

interface SizeTime {
  size: number;
  seconds: number;
}
/** The rows of a lab 4.4 benchmark that give a block size and a time, one per size. */
export function sizeTimes(rows: BenchRow[]): SizeTime[] {
  const out: SizeTime[] = [];
  let name: string | null = null;
  for (const r of rows) {
    const seconds = num(r.seconds);
    const k = SIZE_KEYS.find((key) => num(r[key]) != null);
    if (!k || seconds == null || !(seconds > 0)) continue;
    if (name == null) name = r.name;
    if (r.name !== name) continue; // another series of the same benchmark
    const size = num(r[k]) as number;
    if (!out.some((x) => x.size === size)) out.push({ size, seconds });
  }
  return out.sort((a, b) => a.size - b.size);
}

function BlockSizeReveal({ committed }: { committed: number }) {
  const { reports, loading, reload } = useReports();
  const sets = realBackends(reports).flatMap((b) => {
    const run = reports[b];
    const mine = labStatus(run, LAB4, 'work') === 'pass' ? sizeTimes(benchRows(run, LAB4, 'work')) : [];
    const rows = mine.length >= 2 ? mine : sizeTimes(benchRows(run, LAB4, 'solution'));
    if (rows.length < 2) return [];
    return [{ backend: b, device: String(run.device.name ?? b), date: run.generated.slice(0, 10), own: mine.length >= 2, rows, limits: limitsFromReport(run.device as Record<string, unknown>, b) }];
  });
  if (!sets.length)
    return loading ? (
      <p data-measured="loading"><Translated vi="Đang đọc báo cáo.">Reading reports.</Translated></p>
    ) : (
      <Translated
        vi={
          <p data-measured="none">
            Máy này chưa có số đo nào. Lab 4.4 trên <a href={href('ch04', 'labs')}>trang lab</a> đo thời gian một kernel
            ở nhiều kích thước khối; chạy benchmark của nó bằng <code>python3 labs/pmpp.py run 4.4 --bench</code>, hoặc{' '}
            <code>./verify.sh</code> không có <code>--quick</code>. Dự đoán của bạn đã được lưu và sẽ được đặt bên cạnh
            kết quả.{' '}
            <button className="quiet" onClick={reload}>
              Tải lại kết quả
            </button>
          </p>
        }
      >
        <p data-measured="none">
          No measurement on this machine yet. Lab 4.4 on the <a href={href('ch04', 'labs')}>labs page</a> times one kernel
          at several block sizes; run its benchmark with <code>python3 labs/pmpp.py run 4.4 --bench</code>, or{' '}
          <code>./verify.sh</code> without <code>--quick</code>. Your prediction is stored and will be set beside the
          result.{' '}
          <button className="quiet" onClick={reload}>
            Reload results
          </button>
        </p>
      </Translated>
    );
  return (
    <>
      {sets.map((s) => {
        const fastest = s.rows.reduce((a, b) => (b.seconds < a.seconds ? b : a));
        const slowest = s.rows.reduce((a, b) => (b.seconds > a.seconds ? b : a));
        const ratio = slowest.seconds / fastest.seconds;
        const close = committed > 0 && Math.abs(committed / ratio - 1) <= 0.1;
        return (
          <div key={s.backend} data-measured={fmt(ratio)} data-backend={s.backend}>
            <Translated
              vi={
                <p>
                  Đo trên {s.device} ({s.backend}) với {s.own ? 'kernel của bạn' : 'lời giải tham chiếu'}, ở lần chạy ngày {s.date}:
                  kích thước khối chậm nhất ({count(slowest.size)} luồng) mất thời gian gấp <b>{fmt(ratio)}</b> lần kích thước nhanh nhất (
                  {count(fastest.size)} luồng).{' '}
                  {!(committed > 0)
                    ? 'Dự đoán của bạn không phải một tỉ số dương, nên không có gì để so sánh.'
                    : close
                      ? 'Dự đoán của bạn lệch không quá 10% so với con số đó.'
                      : committed < ratio
                        ? `Con số đó gấp ${fmt(ratio / committed)} lần dự đoán của bạn.`
                        : `Dự đoán của bạn gấp ${fmt(committed / ratio)} lần con số đó.`}
                </p>
              }
            >
              <p>
                Measured on {s.device} ({s.backend}) with {s.own ? 'your kernel' : 'the reference solution'}, in the run of {s.date}:
                the slowest block size ({count(slowest.size)} threads) took <b>{fmt(ratio)}</b> times as long as the fastest (
                {count(fastest.size)} threads).{' '}
                {!(committed > 0)
                  ? 'Your prediction was not a positive ratio, so there is nothing to compare.'
                  : close
                    ? 'Your prediction was within 10% of that.'
                    : committed < ratio
                      ? `That is ${fmt(ratio / committed)} times your prediction.`
                      : `Your prediction is ${fmt(committed / ratio)} times that.`}
              </p>
            </Translated>
            <div style={{ overflowX: 'auto' }}>
              <table className="data" style={{ margin: '0 0 0.8rem' }}>
                <thead>
                  <tr>
                    <th className="num"><Translated vi="Số luồng mỗi khối">Threads per block</Translated></th>
                    <th className="num"><Translated vi="Thời gian (ms)">Time (ms)</Translated></th>
                    <th className="num"><Translated vi="So với nhanh nhất">Against the fastest</Translated></th>
                    <th className="num"><Translated vi="Occupancy chỉ theo khe (mô hình)">Occupancy by slots alone (model)</Translated></th>
                  </tr>
                </thead>
                <tbody>
                  {s.rows.map((r) => {
                    const o = s.limits ? occupancy({ ...s.limits, regs: null, shared: null }, { threads: r.size, regs: 0, shared: 0 }) : null;
                    return (
                      <tr key={r.size} data-size={r.size}>
                        <td className="num">{count(r.size)}</td>
                        <td className="num">{fmt(r.seconds * 1e3)}</td>
                        <td className="num">{fmt(r.seconds / fastest.seconds)}×</td>
                        <td className="num" data-model={o && o.usable ? pct(o.byWarps) : 'none'} /* i18n-ignore: data attribute */>{o && o.usable ? pct(o.byWarps) + '%' : tr('n/a', 'không có')}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="muted">
              {s.limits
                ? tr('The last column is the arithmetic of this section on the limits your device reported, with registers left out: a device report does not carry a kernel\'s register count. Lab 4.4 sets the vendor\'s own occupancy function beside it.', 'Cột cuối là phép tính của mục này trên các giới hạn mà thiết bị của bạn báo cáo, bỏ qua thanh ghi: báo cáo thiết bị không chứa số thanh ghi của kernel. Lab 4.4 đặt hàm occupancy của chính nhà cung cấp bên cạnh nó.')
                : tr('There is no occupancy column for this device: its report has no per-SM limits. On Metal none are documented, so the times are all there is to go on.', 'Thiết bị này không có cột occupancy: báo cáo của nó không có giới hạn mỗi SM. Trên Metal không giới hạn nào được công bố, nên chỉ có thể dựa vào thời gian đo.')}
            </p>
          </div>
        );
      })}
      <Translated
        vi={
          <p className="muted">
            Một số đo từ máy của bạn, không phải giá trị kỳ vọng. Ở chỗ thời gian bám theo occupancy của mô hình, khả năng chịu đựng độ trễ
            là lý do có thể nhất. Ở chỗ không bám theo, có điều gì đó mà mô hình bỏ qua đang tác động, trước hết là băng thông
            bộ nhớ.
          </p>
        }
      >
        <p className="muted">
          A measurement from your machine, not an expected value. Where the times follow the model's occupancy, latency
          tolerance is the likely reason. Where they do not, something the model leaves out is at work, memory bandwidth
          first among them.
        </p>
      </Translated>
    </>
  );
}

// ---------------------------------------------------------------- the occupancy call, from lab 4.4
function OccupancyCall() {
  const code = useLabFile(LAB4 + '/provided/occupancy_api.cu');
  if (!code) return null;
  return (
    <Code
      tabs={[{ label: tr('CUDA and HIP', 'CUDA và HIP'), code, status: tr('front-end-checked; unverified on CUDA and HIP hardware', 'front-end-checked; unverified trên phần cứng CUDA và HIP') }]}
      caption={
        <Translated
          vi={
            <>
              Lời gọi mà benchmark của lab 4.4 thực hiện, nạp từ <code>labs/{LAB4}/provided/occupancy_api.cu</code>. Nó đã
              được clang phân tích cú pháp với các header giả và chưa từng được nvcc hay trình biên dịch HIP biên dịch.
            </>
          }
        >
          <>
            The call lab 4.4's benchmark makes, loaded from <code>labs/{LAB4}/provided/occupancy_api.cu</code>. It has
            been parsed by clang against stub headers and never compiled by nvcc or a HIP compiler.
          </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- questions
const SLOTS_ONLY: Limits = { ...BOOK, regs: null, shared: null };
const O48 = occupancy(SLOTS_ONLY, { threads: 48, regs: 0, shared: 0 });
const CLIFF_A = occupancy(BOOK, { threads: 512, regs: 31, shared: 0 });
const CLIFF_B = occupancy(BOOK, { threads: 512, regs: 33, shared: 0 });
const SMALL = occupancy(SLOTS_ONLY, { threads: 32, regs: 0, shared: 0 });
const EX8C = occupancy(EX8, EX8_KERNELS[2]);

export const questionsS47: Record<string, Question> = {
  'ch04/4.7/q-ex6': {
    kind: 'choice',
    id: 'ch04/4.7/q-ex6',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 6 của sách. Một SM có thể chứa tối đa 1536 luồng và tối đa 4 khối. Kích thước khối nào đặt được nhiều luồng nhất lên
            nó?
          </>
        }
      >
        <>
          Book exercise 6. An SM can hold up to 1536 threads and up to 4 blocks. Which block size puts the most threads on
          it?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="128 luồng mỗi khối">128 threads per block</Translated>,
        why: <Translated vi="Bốn khe khối chứa 4 × 128 = 512 luồng. Khe khối cạn trong khi hai phần ba số khe luồng còn trống.">Four block slots hold 4 × 128 = 512 threads. The block slots run out with two thirds of the thread slots empty.</Translated>,
      },
      {
        label: <Translated vi="256 luồng mỗi khối">256 threads per block</Translated>,
        why: <Translated vi="Bốn khối, 1024 luồng. Lại là khe khối cạn trước.">Four blocks, 1024 threads. Block slots run out first again.</Translated>,
      },
      { label: <Translated vi="512 luồng mỗi khối">512 threads per block</Translated>, correct: true },
      {
        label: <Translated vi="1024 luồng mỗi khối">1024 threads per block</Translated>,
        why: <Translated vi="Một khối vừa; khối thứ hai sẽ cần 2048 khe. 1024 luồng.">One block fits; a second would need 2048 slots. 1024 threads.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đếm từng giới hạn riêng rồi lấy giá trị nhỏ hơn. Ở 512 luồng, khe luồng cho phép 1536 / 512 = 3 khối và
            khe khối cho phép 4, vậy là 3 khối và 1536 luồng: đầy. Hai kích thước nhỏ hơn bị chặn bởi 4 khe
            khối, kích thước lớn nhất bị chặn vì nó không chia hết 1536.
          </p>
        }
      >
        <p>
          Count each limit separately and take the smaller. At 512 threads the thread slots allow 1536 / 512 = 3 blocks and
          the block slots allow 4, so 3 blocks and 1536 threads: full. The two smaller sizes are stopped by the 4 block
          slots, the largest by a size that does not divide 1536.
        </p>
      </Translated>
    ),
    source: 'ch. 4 exercise 6, p. 91; the answer is ours (the book prints none)',
    vi: { source: 'ch. 4, bài tập 6, p. 91; đáp án là của chúng tôi (sách không in đáp án)' },
  },
  'ch04/4.7/q-ex8c': {
    kind: 'numeric',
    id: 'ch04/4.7/q-ex8c',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 8(c) của sách. Một SM cho phép 2048 luồng, 32 khối và 65,536 thanh ghi. Một kernel chạy với 256 luồng mỗi
            khối và dùng 34 thanh ghi mỗi luồng. Occupancy của nó là bao nhiêu phần trăm?
          </>
        }
      >
        <>
          Book exercise 8(c). An SM allows 2048 threads, 32 blocks and 65,536 registers. A kernel runs with 256 threads per
          block and uses 34 registers per thread. What is its occupancy, in percent?
        </>
      </Translated>
    ),
    answer: pct(EX8C.byWarps),
    tol: 0.001,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            Một khối cần 256 × 34 = 8,704 thanh ghi, và 65,536 / 8,704 là 7 và một chút, nên theo thanh ghi thì {EX8C.blocks} khối
            vừa. Khe luồng cho phép 8 và khe khối cho phép 32. Bảy khối là {count(EX8C.residentThreads)} luồng
            trên 2048: {pct(EX8C.byWarps)}%. Thanh ghi là yếu tố giới hạn, và hai thanh ghi vượt quá 32 tốn mất một phần tám
            SM.
          </p>
        }
      >
        <p>
          One block needs 256 × 34 = 8,704 registers, and 65,536 / 8,704 is 7 and a bit, so {EX8C.blocks} blocks fit by
          registers. Thread slots would allow 8 and block slots 32. Seven blocks are {count(EX8C.residentThreads)} threads
          of 2048: {pct(EX8C.byWarps)}%. Registers are the limiting factor, and the two registers above 32 cost an eighth
          of the SM.
        </p>
      </Translated>
    ),
    source: "ch. 4 exercise 8, p. 92, worked with the book's arithmetic of p. 86; the answer is ours",
    vi: { source: 'ch. 4, bài tập 8, p. 92, tính theo phép tính của sách ở p. 86; đáp án là của chúng tôi' },
  },
  'ch04/4.7/q-768': {
    kind: 'numeric',
    id: 'ch04/4.7/q-768',
    prompt: (
      <Translated
        vi={
          <>
            Một thiết bị báo cáo 1536 khe luồng và 16 khe khối mỗi SM. Một kernel được khởi chạy với 768 luồng mỗi khối và
            dùng ít thanh ghi đến mức không đáng kể. Occupancy của nó là bao nhiêu phần trăm?
          </>
        }
      >
        <>
          A device reports 1536 thread slots and 16 block slots per SM. A kernel is launched with 768 threads per block and
          few enough registers that they do not matter. What is its occupancy, in percent?
        </>
      </Translated>
    ),
    answer: pct(occupancy({ ...TABLE_1536, regs: null }, { threads: 768, regs: 0, shared: 0 }).byWarps),
    tol: 0.001,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            1536 / 768 = đúng 2 khối, nằm gọn trong 16 khe khối: mọi khe luồng đều được dùng. Trên ví dụ của sách, cùng
            lần khởi chạy đó cho 75%, vì 768 không chia hết 2048. Một kích thước khối tốt hay xấu là tùy từng thiết bị,
            đó là lý do nên đọc các giới hạn thay vì nhớ chúng.
          </p>
        }
      >
        <p>
          1536 / 768 = 2 blocks exactly, well within 16 block slots: every thread slot is taken. On the book's example the
          same launch gives 75%, because 768 does not divide 2048. A block size is good or bad for a particular device,
          which is the reason to read the limits and not remember them.
        </p>
      </Translated>
    ),
    source: "NVIDIA's table of per-SM limits, read 2026-11-20 (ch04-facts.md, A12); the arithmetic is the book's, p. 86",
    vi: { source: 'Bảng giới hạn mỗi SM của NVIDIA, đọc ngày 2026-11-20 (ch04-facts.md, A12); phép tính là của sách, p. 86' },
  },
  'ch04/4.7/q-twodefs': {
    kind: 'choice',
    id: 'ch04/4.7/q-twodefs',
    prompt: (
      <Translated
        vi={
          <>
            Các khối 48 luồng trên thiết bị ví dụ của sách (32 khe khối, 2048 khe luồng, warp rộng 32). Sách
            định nghĩa occupancy theo warp ở p. 85 và theo luồng trong phần tóm tắt ở p. 90. Mỗi định nghĩa cho ra bao nhiêu?
          </>
        }
      >
        <>
          Blocks of 48 threads on the book's example device (32 block slots, 2048 thread slots, warps of 32). The book
          defines occupancy with warps on p. 85 and with threads in its summary on p. 90. What does each definition give?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="75% theo warp và 75% theo luồng.">75% by warps and 75% by threads.</Translated>,
        why: <Translated vi="Theo luồng thì đúng. Nhưng mỗi khối là hai warp, và 32 khối hai warp chiếm hết 64 khe warp.">By threads, yes. But each block is two warps, and 32 blocks of two warps take all 64 warp slots.</Translated>,
      },
      {
        label: (
          <Translated vi={`${pct(O48.byWarps)}% theo warp và ${pct(O48.byThreads)}% theo luồng.`}>
            {`${pct(O48.byWarps)}% by warps and ${pct(O48.byThreads)}% by threads.`}
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="100% theo cả hai: dù tính cách nào SM cũng đầy.">100% by both: the SM is full either way.</Translated>,
        why: <Translated vi="Mọi khe đều bị chiếm, nhưng 16 lane của mỗi warp thứ hai không chạy luồng nào. Đếm theo luồng sẽ thấy điều đó.">Every slot is held, but 16 lanes of every second warp run no thread. Counting threads shows that.</Translated>,
      },
      {
        label: <Translated vi="50% theo warp và 75% theo luồng.">50% by warps and 75% by threads.</Translated>,
        why: <Translated vi="Một khối 48 luồng là hai warp, không phải một: warp thứ hai được đệm thêm 16 lane không hoạt động (mục 4.4).">A 48-thread block is two warps, not one: the second is padded with 16 inactive lanes (section 4.4).</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Một khối 48 luồng được cắt thành hai warp, warp thứ hai được đệm 16 lane không chạy gì. Cả{' '}
              {O48.blocks} khe khối đều được dùng, tạo ra {O48.residentWarps} warp: mọi khe warp. Vậy theo warp là {pct(O48.byWarps)}
              %. Đếm theo luồng thì là {O48.blocks} × 48 = {count(O48.residentThreads)} trên 2048,{' '}
              {pct(O48.byThreads)}%.
            </p>
            <p>
              Cả hai đều đúng về một điều gì đó. Con số theo warp cho biết bộ lập lịch có bao nhiêu warp để chọn, điều mà
              khả năng chịu đựng độ trễ cần. Con số theo luồng cho biết các khe đó mang bao nhiêu việc.{' '}
              {count(O48.paddedLanes)} lane đệm chiếm khe mà không làm gì. Với kích thước khối là trọn warp, hai
              định nghĩa trùng nhau, thêm một lý do để dùng chúng.
            </p>
          </>
        }
      >
        <>
          <p>
            A block of 48 threads is cut into two warps, the second padded with 16 lanes that run nothing. All{' '}
            {O48.blocks} block slots are used, which makes {O48.residentWarps} warps: every warp slot. So {pct(O48.byWarps)}
            % by warps. Counted in threads it is {O48.blocks} × 48 = {count(O48.residentThreads)} of 2048,{' '}
            {pct(O48.byThreads)}%.
          </p>
          <p>
            Both are right about something. The warp figure says how many warps the scheduler has to choose from, which
            is what latency tolerance needs. The thread figure says how much work those slots carry. The{' '}
            {count(O48.paddedLanes)} padded lanes hold slots and do nothing. With whole-warp block sizes the two
            definitions agree, which is one more reason to use them.
          </p>
        </>
      </Translated>
    ),
    source: "ch. 4 §4.7, p. 85 and §4.9, p. 90; padding from §4.4, p. 75; NVIDIA defines occupancy with warps (ch04-facts.md, A13)",
    vi: { source: 'ch. 4 §4.7, p. 85 và §4.9, p. 90; phần đệm từ §4.4, p. 75; NVIDIA định nghĩa occupancy theo warp (ch04-facts.md, A13)' },
  },
  'ch04/4.7/q-regs64': {
    kind: 'numeric',
    id: 'ch04/4.7/q-regs64',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel cần 64 thanh ghi mỗi luồng. Trên thiết bị ví dụ của sách (65,536 thanh ghi và 2048 khe luồng mỗi
            SM), occupancy cao nhất mà bất kỳ kích thước khối nào có thể cho nó là bao nhiêu phần trăm?
          </>
        }
      >
        <>
          A kernel needs 64 registers per thread. On the book's example device (65,536 registers and 2048 thread slots per
          SM), what is the highest occupancy any block size can give it, in percent?
        </>
      </Translated>
    ),
    answer: 50,
    tol: 0,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            65,536 / 64 = lượng thanh ghi đủ cho 1024 luồng, dù các luồng được gom thành khối thế nào. Đó là một nửa của
            2048 khe luồng. Đổi kích thước khối không giúp được gì; chỉ dùng ít thanh ghi hơn mỗi luồng mới giúp được, và
            cách trình biên dịch làm điều đó (đẩy giá trị ra bộ nhớ) tốn thời gian ở mỗi lần truy cập.
          </p>
        }
      >
        <p>
          65,536 / 64 = 1024 threads' worth of registers, however the threads are grouped into blocks. That is half of the
          2048 thread slots. Changing the block size cannot help; only using fewer registers per thread can, and the
          compiler's way of doing that (spilling values to memory) costs time on every access.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.7, p. 86',
  },
  'ch04/4.7/q-hipslots': {
    kind: 'choice',
    id: 'ch04/4.7/q-hipslots',
    prompt: (
      <Translated
        vi={
          <>
            Một chương trình HIP trên thiết bị AMD đọc <code>maxBlocksPerMultiProcessor</code> và nhận được một số N. N có phải là giới hạn
            về khối cùng loại với con số 32 của sách không?
          </>
        }
      >
        <>
          A HIP program on an AMD device reads <code>maxBlocksPerMultiProcessor</code> and gets some number N. Is N a limit
          on blocks of the same kind as the book's 32?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Có. HIP sao chép trường của CUDA, nên đó là số khe khối của một compute unit.">Yes. HIP mirrors the CUDA field, so it is the number of block slots of a compute unit.</Translated>,
        why: <Translated vi="Cái tên được sao chép. Giá trị thì được tính ra, và từ một thứ khác.">The name is mirrored. The value is computed, and from something else.</Translated>,
      },
      {
        label: (
          <Translated vi="Không. Nó là số khe luồng chia cho độ rộng wavefront: số khe wave, không phải một giới hạn riêng.">
            No. It is thread slots divided by the wavefront width: the number of wave slots, not a separate limit.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Không. Nó đếm số khối cả thiết bị có thể chứa cùng lúc, không phải số khối của một compute unit.">No. It counts the blocks the whole device can hold at once, not the blocks of one compute unit.</Translated>,
        why: <Translated vi="Nó tính cho mỗi compute unit. Điều khác biệt là nó không phải một giới hạn độc lập.">It is per compute unit. What differs is that it is not an independent limit.</Translated>,
      },
      {
        label: (
          <Translated vi="Có, và nó nhỏ hơn con số 32 của NVIDIA, vì wavefront rộng hơn chừa chỗ cho ít khối hơn.">
            {'Yes, and it is smaller than NVIDIA\'s 32, because wider wavefronts leave room for fewer blocks.'}
          </Translated>
        ),
        why: <Translated vi="Không có tài liệu nào nói điều như vậy, và độ rộng thay đổi theo dòng sản phẩm AMD.">Nothing of the kind is documented, and width varies by AMD product family.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trên thiết bị của sách, các khối một warp chạm 32 khe khối trong khi một nửa số khe luồng còn trống. Mô tả compute unit
            của AMD không có giới hạn khe khối, và runtime điền trường này bằng số khe luồng /
            độ rộng. Vậy trường hợp 50% của mục này không có bản tương ứng trên AMD, và một máy tính coi N là giới hạn thứ hai
            sẽ bịa ra một giới hạn. Những gì thực sự là giới hạn trên AMD là khe wave, hai vùng thanh ghi và bộ nhớ chia sẻ.
          </p>
        }
      >
        <p>
          On the book's device, blocks of one warp hit the 32 block slots with half the thread slots empty. AMD's
          description of a compute unit has no block-slot limit, and the runtime fills this field with thread slots /
          width. So the 50% case of this section has no AMD counterpart, and a calculator that treats N as a second limit
          would invent one. What does bind on AMD is wave slots, two register pools and shared memory.
        </p>
      </Translated>
    ),
    source: 'HIP runtime source and docs, read 2026-11-20 (ch04-facts.md, B9, B10); not in the book',
    vi: { source: 'Mã nguồn và tài liệu runtime HIP, đọc ngày 2026-11-20 (ch04-facts.md, B9, B10); không có trong sách' },
  },
};

// ---------------------------------------------------------------- the section
export function S47() {
  return (
    <>
      <Translated
        vi={
          <p>
            <a href={href('ch04', '4.6')}>Mục 4.6</a> kết thúc bằng một hóa đơn. Mỗi warp thường trú trên một SM đều giữ một
            chỗ ở đó, cùng các thanh ghi của nó, chừng nào nó còn ở lại. Mục này làm phép tính sổ sách: với những gì một SM
            có và những gì một khối của kernel bạn yêu cầu, bao nhiêu khối vừa, và khi đó SM đầy đến đâu?
          </p>
        }
      >
        <p>
          <a href={href('ch04', '4.6')}>Section 4.6</a> ended on a bill. Every warp that is resident on an SM holds a
          place there, and its registers, for as long as it stays. This section does the accounting: given what one SM
          has and what one block of your kernel asks for, how many blocks fit, and how full is the SM then?
        </p>
      </Translated>

      <h2><Translated vi="Occupancy">Occupancy</Translated></h2>
      <Translated
        vi={
          <p>
            Thước đo là <b>occupancy (mức chiếm dụng)</b>: số warp được gán cho một SM, chia cho số warp tối đa SM đó có thể
            chứa. Một SM chứa một nửa số warp nó có thể chứa thì ở mức 50%. Theo mục 4.6, điều đó nghĩa là chỉ có một nửa khả năng che
            thời gian chờ bộ nhớ so với mức nó có thể có.
          </p>
        }
      >
        <p>
          The measure is <b>occupancy</b>: the number of warps assigned to an SM, divided by the most warps that SM can
          hold. An SM holding half the warps it could is at 50%. By section 4.6 that means half the cover for waits on
          memory that it could have had.
        </p>
      </Translated>
      <Source section="4.7" pages="85" />
      <Translated
        vi={
          <p>
            Phần tóm tắt của sách nêu cùng tỉ số đó nhưng dùng luồng thay cho warp. Hai cách trùng nhau mỗi khi khối là
            trọn warp, và tách nhau khi không phải vậy. Trang này dùng warp và hiển thị con số kia ở chỗ nó
            khác.
          </p>
        }
      >
        <p>
          The book's summary states the same ratio with threads in place of warps. The two agree whenever blocks are
          whole warps, and part company when they are not. This page uses warps and shows the other figure where it
          differs.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 against the CUDA 13.5 guide (ch04-facts.md, A13)', '2026-11-20, đối chiếu với hướng dẫn CUDA 13.5 (ch04-facts.md, A13)')}>
        <Translated
          vi={
            <p>
              NVIDIA cũng định nghĩa nó theo warp: số warp hoạt động trên số warp hoạt động tối đa mà một SM hỗ trợ. Phiên bản theo
              luồng nằm ở p. 90 của sách.
            </p>
          }
        >
          <p>
            NVIDIA defines it with warps too: active warps over the most active warps an SM supports. The threads
            version is on the book's p. 90.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Những gì được chia sẻ">What is shared out</Translated></h2>
      <Translated vi={<p>Thứ cạn kiệt không phải các streaming processor. Mà là bốn thứ khác, và mỗi thứ là một lượng cố định trên mỗi SM:</p>}>
        <p>The streaming processors are not what runs out. Four other things are, and each is a fixed amount per SM:</p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li><b>Khe khối</b>: bao nhiêu khối có thể thường trú cùng lúc.</li>
            <li><b>Khe luồng</b>: bao nhiêu luồng, cũng tức là bao nhiêu warp.</li>
            <li><b>Thanh ghi (register)</b>: một vùng chung, mọi luồng thường trú đều lấy từ đó.</li>
            <li>
              <b>Bộ nhớ chia sẻ (shared memory)</b>: một vùng chung, mọi khối thường trú đều lấy từ đó.{' '}
              <a href={href('ch05', '5.6')}>Mục 5.6</a> đưa nó vào phép tính này.
            </li>
          </ul>
        }
      >
        <ul>
          <li><b>Block slots</b>: how many blocks can be resident at once.</li>
          <li><b>Thread slots</b>: how many threads, which is the same as how many warps.</li>
          <li><b>Registers</b>: one pool, drawn on by every resident thread.</li>
          <li>
            <b>Shared memory</b>: one pool, drawn on by every resident block.{' '}
            <a href={href('ch05', '5.6')}>Section 5.6</a> adds it to this arithmetic.
          </li>
        </ul>
      </Translated>
      <Translated
        vi={
          <p>
            Không thứ nào bị cắt thành các phần cố định. Một khối lấy những gì nó cần từ mỗi vùng khi được gán và
            trả lại khi xong. Chính điều đó cho phép cùng một SM chứa nhiều khối nhỏ hoặc vài khối lớn. Nếu chia
            phần cố định, một khối nhỏ sẽ lãng phí phần lớn phần của nó, còn một khối lớn hơn một phần thì không chạy được.
          </p>
        }
      >
        <p>
          None of them is cut into fixed shares. A block takes what it needs from each pool when it is assigned and
          gives it back when it finishes. That is what lets the same SM hold many small blocks or a few large ones. With
          fixed shares, a small block would waste most of its share and a block larger than a share could not run at all.
        </p>
      </Translated>
      <Source section="4.7" pages="85" />

      <h2><Translated vi="Khe luồng và khe khối">Thread slots and block slots</Translated></h2>
      <Translated
        vi={
          <p>
            Thiết bị ví dụ của sách cho phép 32 khối và 2048 luồng (64 warp) trên một SM, và 1024 luồng trong một khối.
            Bắt đầu với khối lớn. Hai khối 1024 luồng lấp đầy 2048 khe. Bốn khối 512, tám khối 256,
            mười sáu khối 128 và ba mươi hai khối 64 cũng vậy. Mỗi trường hợp đều là 100%.
          </p>
        }
      >
        <p>
          The book's example device allows 32 blocks and 2048 threads (64 warps) on an SM, and 1024 threads in a block.
          Start with large blocks. Two blocks of 1024 threads fill the 2048 slots. So do four of 512, eight of 256,
          sixteen of 128 and thirty-two of 64. Each of those is 100%.
        </p>
      </Translated>
      <Source section="4.7" pages="85" />
      <Predict
        id="ch04/4.7/p-32"
        prompt={
          <Translated vi={<>Nhỏ thêm một bậc trên cùng thiết bị: khối 32 luồng. Occupancy là bao nhiêu phần trăm?</>}>
            <>
              Go one step smaller on the same device: blocks of 32 threads. What is the occupancy, in percent?
            </>
          </Translated>
        }
        unit="%"
      >
        {(v) => (
          <Translated
            vi={
              <>
                <p>
                  <b>{pct(SMALL.byWarps)}%.</b> Lấp đầy 2048 khe luồng bằng khối 32 luồng sẽ cần 64 khối, mà
                  SM chỉ có 32 khe khối. Vậy là {SMALL.blocks} khối, {count(SMALL.residentThreads)} luồng, một nửa số
                  khe trống.{' '}
                  {v === pct(SMALL.byWarps) ? 'Đúng như bạn đã đoán.' : v === 100 ? 'Riêng khe luồng thì sẽ cho 100%; nhưng giới hạn kia đã chạm trước.' : ''}
                </p>
                <p>
                  Trên thiết bị này, một khối cần ít nhất 2048 / 32 = {smallestFillingBlock(BOOK)} luồng thì mới có thể đạt
                  occupancy đầy đủ.
                </p>
              </>
            }
          >
            <>
              <p>
                <b>{pct(SMALL.byWarps)}%.</b> Filling 2048 thread slots with 32-thread blocks would take 64 blocks, and
                the SM has 32 block slots. So {SMALL.blocks} blocks, {count(SMALL.residentThreads)} threads, half the
                slots empty.{' '}
                {v === pct(SMALL.byWarps) ? 'That is what you had.' : v === 100 ? 'The thread slots alone would give 100%; the other limit got there first.' : ''}
              </p>
              <p>
                On this device a block needs at least 2048 / 32 = {smallestFillingBlock(BOOK)} threads before full
                occupancy is possible at all.
              </p>
            </>
          </Translated>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Đó là cách tương tác thứ nhất giữa các giới hạn: một khối có thể nhỏ đến mức khe khối cạn trước khe
            luồng. Cách thứ hai đơn giản hơn. Một khối 768 luồng không chia hết 2048. Hai khối vừa, khối thứ ba thì
            không, 512 khe bỏ trống, và occupancy là 75% dù không giới hạn nào bị chạm.
          </p>
        }
      >
        <p>
          That is the first way the limits interact: a block can be so small that the block slots run out before the
          thread slots do. The second is plainer. A block of 768 threads does not divide 2048. Two fit, a third does
          not, 512 slots stay empty, and occupancy is 75% with neither limit reached.
        </p>
      </Translated>
      <Source section="4.7" pages="85–86" />
      <Translated
        vi={
          <p>
            Máy tính dưới đây thực hiện phép tính này cho mọi bộ giới hạn. Mỗi hàng là một ngân sách: SM có bao nhiêu, một khối
            lấy bao nhiêu, và riêng ngân sách đó cho phép bao nhiêu khối. SM nhận con số nhỏ nhất trong các con số đó. Các
            nút là bảy kích thước khối sách tính qua, và một kích thước sách không tính.
          </p>
        }
      >
        <p>
          The calculator holds this arithmetic for any limits. Each row is one budget: what an SM has, what one block
          takes, and how many blocks that budget alone would allow. The SM gets the smallest of those counts. The
          buttons are the seven block sizes the book works through, and one it does not.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <>
              <p>
                Thanh ghi được tính theo luồng: số luồng × số thanh ghi mỗi luồng. Đó là mô hình của sách. NVIDIA nói
                thanh ghi được chia cho các warp và không công bố theo lượng nào, nên với một kernel thật,
                hàm occupancy của nhà cung cấp mới là căn cứ.
              </p>
              <p>Lane đệm bị tính một khe luồng và không tính thanh ghi. Bộ nhớ chia sẻ mà driver giữ lại bị bỏ qua.</p>
            </>
          }
        >
          <>
            <p>
              Registers are counted per thread: threads × registers per thread. That is the book's model. NVIDIA says
              registers are shared out among warps and does not publish in what amounts, so for a real kernel the
              vendor's occupancy function is the authority.
            </p>
            <p>Padded lanes are charged a thread slot and no registers. Shared memory the driver reserves is ignored.</p>
          </>
        </Translated>
      </Simplification>
      <OccupancyCalculator />
      <Translated
        vi={
          <p>
            Mọi trường hợp sách tính đều thuộc về thiết bị ví dụ của nó. Đổi các giới hạn thì cùng một kích thước khối
            cho ra đáp án khác. Hãy tự tính tay câu hỏi tiếp theo trước khi chạm vào máy tính.
          </p>
        }
      >
        <p>
          Every one of the book's worked cases belongs to its example device. Change the limits and the same block size
          gives a different answer. Work the next question out by hand before you touch the calculator.
        </p>
      </Translated>
      <Today checked={tr("2026-11-20 against NVIDIA's compute-capability tables (ch04-facts.md, A12, A15)", '2026-11-20, đối chiếu với các bảng compute capability của NVIDIA (ch04-facts.md, A12, A15)')}>
        <Translated
          vi={
            <p>
              Các GPU NVIDIA hiện nay có 16, 24 hoặc 32 khe khối và 1024, 1536 hoặc 2048 khe luồng mỗi SM, tùy theo
              compute capability, và 64 K thanh ghi trên tất cả. Compute capability cao hơn không có nghĩa là mỗi SM có nhiều hơn.
            </p>
          }
        >
          <p>
            Current NVIDIA GPUs have 16, 24 or 32 block slots and 1024, 1536 or 2048 thread slots per SM, depending on
            the compute capability, and 64 K registers on all of them. A higher capability does not mean more per SM.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS47['ch04/4.7/q-768']} />
      <Translated
        vi={
          <p>
            Rồi kiểm tra. Danh sách giới hạn trong máy tính có một bộ thứ hai lấy từ chính bảng của NVIDIA, với 1536 khe
            luồng và 16 khe khối. Chọn nó, bấm 768, rồi 1024, và đặt cả hai cạnh kết quả của ví dụ của sách.
          </p>
        }
      >
        <p>
          Then check it. The list of limits in the calculator has a second set from NVIDIA's own table, with 1536 thread
          slots and 16 block slots. Choose it, press 768, then 1024, and set both beside what the book's example gave.
        </p>
      </Translated>
      <Ask q={questionsS47['ch04/4.7/q-ex6']} />
      <Ask q={questionsS47['ch04/4.7/q-twodefs']} />

      <h2><Translated vi="Thanh ghi">Registers</Translated></h2>
      <Translated
        vi={
          <p>
            Biến tự động trong kernel nằm trong thanh ghi (<a href={href('ch05', '5.2')}>mục 5.2</a>), nên mỗi
            luồng của kernel cần một số thanh ghi nào đó, và con số ấy cố định khi kernel được biên dịch. Thanh ghi
            cũng đến từ một vùng chung trên mỗi SM. Ví dụ của sách có 65,536. Chia cho 2048 khe luồng: một kernel
            cần tối đa {regsAtFullOccupancy(BOOK)} thanh ghi mỗi luồng có thể lấp mọi khe. Một kernel cần 64 thì
            chỉ có tối đa 1024 luồng thường trú, dù kích thước khối là bao nhiêu.
          </p>
        }
      >
        <p>
          Automatic variables in a kernel live in registers (<a href={href('ch05', '5.2')}>section 5.2</a>), so every
          thread of a kernel needs some number of them, and that number is fixed when the kernel is compiled. Registers
          come from a pool per SM as well. The book's example has 65,536. Divide by the 2048 thread slots: a kernel that
          needs up to {regsAtFullOccupancy(BOOK)} registers per thread can fill every slot. A kernel that needs 64 can
          have at most 1024 threads resident, whatever its block size.
        </p>
      </Translated>
      <Source section="4.7" pages="86" />
      <Translated
        vi={
          <p>
            Trình biên dịch có thể giảm con số này bằng cách <b>đẩy ra bộ nhớ (spilling)</b>: giữ một số giá trị trong bộ nhớ và nạp lại khi
            cần. Khi đó nhiều khối hơn vừa. Mỗi lần truy cập một giá trị bị đẩy ra lâu hơn, và sách cảnh báo rằng
            cả lưới có thể chạy chậm hơn.
          </p>
        }
      >
        <p>
          The compiler can bring the count down by <b>spilling</b>: keeping some values in memory and fetching them when
          they are needed. More blocks then fit. Each access to a spilled value takes longer, and the book warns that
          the grid as a whole may come out slower.
        </p>
      </Translated>
      <Ask q={questionsS47['ch04/4.7/q-regs64']} />

      <h2><Translated vi="Vách đá">The cliff</Translated></h2>
      <Translated
        vi={
          <p>
            Occupancy không giảm đều khi kernel lớn dần. Khối hoặc vừa trọn vẹn hoặc không vừa chút nào, nên nó giảm theo bậc, và một
            bậc có thể rất lớn.
          </p>
        }
      >
        <p>
          Occupancy does not fall smoothly as a kernel grows. Blocks fit whole or not at all, so it falls in steps, and a
          step can be large.
        </p>
      </Translated>
      <Source section="4.7" pages="86" />
      <Predict
        id="ch04/4.7/p-cliff"
        prompt={
          <Translated
            vi={
              <>
                Trên thiết bị ví dụ của sách, một kernel chạy với 512 luồng mỗi khối và dùng 31 thanh ghi mỗi luồng. Bốn
                khối vừa: 100%. Bạn khai báo thêm hai biến cục bộ và giờ trình biên dịch cần 33 thanh ghi mỗi luồng.
                Occupancy là bao nhiêu phần trăm?
              </>
            }
          >
            <>
              On the book's example device a kernel runs with 512 threads per block and uses 31 registers per thread. Four
              blocks fit: 100%. You declare two more local variables and the compiler now needs 33 registers per thread.
              What is the occupancy, in percent?
            </>
          </Translated>
        }
        unit="%"
      >
        {(v) => (
          <Translated
            vi={
              <>
                <p>
                  <b>{pct(CLIFF_B.byWarps)}%.</b> Ở 31 thanh ghi, bốn khối cần 2048 × 31 = {count(CLIFF_A.blocks * 512 * 31)}{' '}
                  thanh ghi, dưới 65,536. Ở 33 chúng sẽ cần 2048 × 33 = {count(2048 * 33)}, vượt quá. Khối thứ tư
                  không còn vừa, còn lại {CLIFF_B.blocks} khối ({count(CLIFF_B.residentThreads)} luồng,{' '}
                  {count(CLIFF_B.blocks * 512 * 33)} thanh ghi), và một phần tư SM bỏ trống.{' '}
                  {v === pct(CLIFF_B.byWarps) ? 'Đúng như bạn đã đoán.' : v > 90 ? 'Giảm vài phần trăm sẽ là đáp án nếu mọi thứ giảm đều. Nhưng khối thì không chia nhỏ được.' : ''}
                </p>
                <p>
                  Hai thanh ghi, một phần tư mức song song. Sách gọi một mức tăng nhỏ về tài nguyên dẫn tới
                  sụt giảm lớn như thế này là <b>vách đá hiệu năng (performance cliff)</b>.
                </p>
              </>
            }
          >
            <>
              <p>
                <b>{pct(CLIFF_B.byWarps)}%.</b> At 31 registers the four blocks need 2048 × 31 = {count(CLIFF_A.blocks * 512 * 31)}{' '}
                registers, under 65,536. At 33 they would need 2048 × 33 = {count(2048 * 33)}, which is over. The fourth
                block no longer fits, {CLIFF_B.blocks} blocks stay ({count(CLIFF_B.residentThreads)} threads,{' '}
                {count(CLIFF_B.blocks * 512 * 33)} registers), and a quarter of the SM is empty.{' '}
                {v === pct(CLIFF_B.byWarps) ? 'That is what you had.' : v > 90 ? 'A drop of a few percent would be the smooth answer. Blocks are not divisible.' : ''}
              </p>
              <p>
                Two registers, a quarter of the parallelism. The book calls a small increase in resource use that causes
                a large drop like this a <b>performance cliff</b>.
              </p>
            </>
          </Translated>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Trình khám phá vẽ toàn bộ bậc thang. Bên trái, occupancy khi số thanh ghi mỗi luồng đi từ 1 đến 255 ở
            kích thước khối bạn chọn. Bên phải, occupancy cho mọi kích thước khối ở số thanh ghi bạn chọn. Di chuyển
            thanh trượt nào thì biểu đồ kia được vẽ lại. Hãy tìm ba điều: bậc đầu tiên ở đâu, một bậc duy nhất
            có thể tốn bao nhiêu ở 1024 luồng mỗi khối, và mọi thứ thay đổi thế nào trên các bộ giới hạn khác.
          </p>
        }
      >
        <p>
          The explorer draws the whole staircase. On the left, occupancy as registers per thread go from 1 to 255 at the
          block size you choose. On the right, occupancy for every block size at the register count you choose. Move
          either slider and the other plot is redrawn. Look for three things: where the first step is, how much a single
          step can cost at 1024 threads per block, and what happens to all of it on other limits.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Lại là mô hình của máy tính: thanh ghi tính bằng số luồng × số thanh ghi mỗi luồng, không có độ chi tiết
              cấp phát, và bỏ qua bộ nhớ chia sẻ, nên khe và thanh ghi là những ngân sách duy nhất.
            </p>
          }
        >
          <p>
            The calculator's model again: registers counted as threads × registers per thread, with no allocation
            granularity, and shared memory left out, so slots and registers are the only budgets.
          </p>
        </Translated>
      </Simplification>
      <CliffExplorer />
      <Ask q={questionsS47['ch04/4.7/q-ex8c']} />
      <Translated
        vi={
          <p>
            Khối lớn hơn tạo bậc cao hơn, vì mỗi khối không còn vừa sẽ kéo theo nhiều warp hơn. Và
            vị trí các bậc thuộc về thiết bị: ở 512 luồng mỗi khối, bậc đầu tiên nằm ở 33 thanh ghi trên
            ví dụ của sách và ở 43 trên bộ giới hạn 1536 khe.
            Một kernel được tinh chỉnh để nằm ngay dưới một bậc trên GPU này có thể nằm ngay trên bậc đó ở GPU khác.
          </p>
        }
      >
        <p>
          Larger blocks make taller steps, because each block that stops fitting takes more warps with it. And the
          positions of the steps belong to the device: at 512 threads per block the first one is at 33 registers on the
          book's example and at 43 on the 1536-slot limits.
          A kernel tuned to sit just under a step on one GPU can be just over it on another.
        </p>
      </Translated>

      <h2><Translated vi="Tìm các con số cho một kernel thật">Finding the numbers for a real kernel</Translated></h2>
      <Translated
        vi={
          <p>
            Các ngân sách tương tác với nhau, trình biên dịch quyết định số thanh ghi, và phần cứng chia thanh ghi theo những cách
            mà phép tính này chỉ xấp xỉ. Vì vậy không ai tính tay occupancy của một kernel thật. Sách nêu hai
            công cụ: Occupancy Calculator thuộc profiler Nsight Compute, và các hàm occupancy bạn gọi
            từ mã host, như <code>cudaOccupancyMaxActiveBlocksPerMultiprocessor</code>.
          </p>
        }
      >
        <p>
          The budgets interact, the compiler decides the register count, and the hardware shares registers out in ways
          this arithmetic only approximates. So nobody works out a real kernel's occupancy by hand. The book names two
          tools: the Occupancy Calculator that is part of the Nsight Compute profiler, and occupancy functions you call
          from host code, such as <code>cudaOccupancyMaxActiveBlocksPerMultiprocessor</code>.
        </p>
      </Translated>
      <Source section="4.7" pages="87" />
      <Translated
        vi={
          <p>
            Đây là những thứ bạn sẽ gặp. Không thứ nào được chạy trên máy này: trên CUDA tất cả đều unverified
            ở đây, và tên gọi cùng ý nghĩa lấy từ tài liệu của NVIDIA.
          </p>
        }
      >
        <p>
          These are the ones you will meet. None of them was run on this machine: on CUDA all of this is unverified
          here, and the names and meanings are NVIDIA's documentation.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 against the CUDA 13.5 guide, the nvcc and runtime references and the Nsight Compute 2026.4 docs (ch04-facts.md, A13, A14)',
          '2026-11-20, đối chiếu với hướng dẫn CUDA 13.5, tài liệu tham khảo nvcc và runtime, và tài liệu Nsight Compute 2026.4 (ch04-facts.md, A13, A14)',
        )}
      >
        <Translated vi={<p>Mọi tên trong bảng đều tồn tại với các ý nghĩa này. Occupancy Calculator đã thay thế bảng tính cũ.</p>}>
          <p>All of the names in the table exist with these meanings. The Occupancy Calculator replaced the older spreadsheet.</p>
        </Translated>
      </Today>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-tools>
          <thead>
            <tr>
              <th style={{ width: '38%' }}><Translated vi="Công cụ">What</Translated></th>
              <th><Translated vi="Nó cho bạn biết gì, hoặc làm gì">What it tells you, or does</Translated></th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><code>nvcc --resource-usage</code></td>
              <td><Translated vi="Lúc biên dịch: số thanh ghi và bộ nhớ chia sẻ mỗi kernel cần.">At compile time: the registers and shared memory each kernel needs.</Translated></td>
            </tr>
            <tr>
              <td><Translated vi={<><code>cudaFuncGetAttributes</code>, trường <code>numRegs</code></>}><code>cudaFuncGetAttributes</code>, field <code>numRegs</code></Translated></td>
              <td><Translated vi="Lúc chạy: số thanh ghi mỗi luồng của một kernel dùng.">At run time: the registers each thread of a kernel uses.</Translated></td>
            </tr>
            <tr>
              <td><code>cudaOccupancyMaxActiveBlocksPerMultiprocessor</code></td>
              <td><Translated vi="Với một kernel, kích thước khối và lượng bộ nhớ chia sẻ động: bao nhiêu khối vừa trên một SM. Nó trả về số khối, không phải phần trăm.">For one kernel, block size and amount of dynamic shared memory: how many blocks fit on an SM. It returns a block count, not a percentage.</Translated></td>
            </tr>
            <tr>
              <td><code>cudaOccupancyMaxPotentialBlockSize</code></td>
              <td><Translated vi="Gợi ý một kích thước khối và một lưới nhỏ nhất cho kernel occupancy tốt nhất có thể. Chỉ có cho C++.">Suggests a block size and a smallest grid that give the kernel its best possible occupancy. C++ only.</Translated></td>
            </tr>
            <tr>
              <td><Translated vi={<>Occupancy Calculator trong Nsight Compute (<code>ncu-ui</code>)</>}>Occupancy Calculator in Nsight Compute (<code>ncu-ui</code>)</Translated></td>
              <td><Translated vi="Cùng thông tin đó cho mọi kích thước khối một lúc, dưới dạng bảng và đồ thị.">The same for every block size at once, as tables and a graph.</Translated></td>
            </tr>
            <tr>
              <td><code>__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor, …)</code></td>
              <td><Translated vi="Các cận được viết trên định nghĩa kernel, một trong những cách NVIDIA liệt kê để điều khiển việc dùng thanh ghi. Lần khởi chạy có nhiều luồng mỗi khối hơn cận thứ nhất sẽ thất bại.">Bounds written on the kernel definition, one of the ways NVIDIA lists to steer register use. A launch with more threads per block than the first bound fails.</Translated></td>
            </tr>
            <tr>
              <td><Translated vi={<><code>nvcc --maxrregcount N</code>, hoặc <code>__maxnreg__(N)</code> trên một kernel</>}><code>nvcc --maxrregcount N</code>, or <code>__maxnreg__(N)</code> on one kernel</Translated></td>
              <td><Translated vi="Giới hạn số thanh ghi mỗi luồng. Một kernel cần nhiều hơn có khả năng phải đẩy ra bộ nhớ; NVIDIA nói số khối vừa thêm khi đó vẫn có thể mang lại lợi ích ròng.">Caps registers per thread. A kernel that needs more is likely to spill to memory; NVIDIA says the extra blocks that then fit may still give a net gain.</Translated></td>
            </tr>
          </tbody>
        </table>
      </div>
      <Translated
        vi={
          <p>
            Để đổi đáp án của hàm occupancy thành phần trăm của mục này, làm tròn kích thước khối lên trọn warp
            rồi chia cho số khe luồng: numBlocks × ceil(blockSize / warpSize) × warpSize /
            maxThreadsPerMultiProcessor. Lab 4.4 thực hiện lời gọi đó cho một kernel của bạn và đặt kết quả cạnh mô hình
            của chính bạn:
          </p>
        }
      >
        <p>
          To turn the occupancy function's answer into this section's percentage, round the block size up to whole warps
          and divide by the thread slots: numBlocks × ceil(blockSize / warpSize) × warpSize /
          maxThreadsPerMultiProcessor. Lab 4.4 makes that call for a kernel of yours and sets the result beside your own
          model:
        </p>
      </Translated>
      <OccupancyCall />

      <h2><Translated vi="Trên AMD, qua HIP">On AMD, through HIP</Translated></h2>
      <Translated
        vi={
          <p>
            AMD định nghĩa occupancy theo cùng cách, số warp hoạt động trên số tối đa một compute unit có thể chứa, và HIP có các
            hàm occupancy cùng tên với tiền tố <code>hip</code>. Những gì được chia sẻ thì khác: khe
            wave, một vùng thanh ghi vector, một vùng thanh ghi vô hướng, và bộ nhớ chia sẻ, mà AMD gọi là LDS.
            Thanh ghi được cấp theo wave, theo từng khối kích thước cố định, không theo luồng.
          </p>
        }
      >
        <p>
          AMD defines occupancy the same way, active warps over the most a compute unit can hold, and HIP has the
          occupancy functions under the same names with a <code>hip</code> prefix. What is shared out is different: wave
          slots, a pool of vector registers, a pool of scalar registers, and shared memory, which AMD calls LDS.
          Registers are given out per wave, in fixed-size chunks, not per thread.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ đề cập CUDA. Phần này lấy từ tài liệu HIP của AMD và mã nguồn runtime, đọc ngày 2026-11-20
              (ch04-facts.md, B9, B10). Không gì ở đây được chạy trên phần cứng AMD.
            </p>
          }
        >
          <p>
            The book covers CUDA only. This is from AMD's HIP documentation and the runtime's source, read on 2026-11-20
            (ch04-facts.md, B9, B10). Nothing here was run on AMD hardware.
          </p>
        </Translated>
      </Beyond>
      <Translated vi={<p>Ba hệ quả với mã phải chạy trên cả hai nhà cung cấp:</p>}>
        <p>Three consequences for code that has to run on both vendors:</p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>
              Không có giới hạn khe khối riêng. <code>maxBlocksPerMultiProcessor</code> có trong{' '}
              <code>hipDeviceProp_t</code>, nhưng runtime tính nó bằng số khe luồng chia cho độ rộng wavefront, nên
              nó là số khe wave. Trường hợp khối 32 luồng, 50% ở trên không có bản tương ứng trên AMD.
            </li>
            <li>
              Một khối được làm tròn lên trọn wave trước khi chia bất cứ thứ gì. Trên thiết bị rộng 64, một khối 65 luồng tốn
              128 khe luồng.
            </li>
            <li>
              <code>__launch_bounds__</code> viết giống hệt nhưng đối số thứ hai mang nghĩa khác: số warp tối thiểu
              trên mỗi execution unit (một compute unit có bốn), trong khi của CUDA là số khối tối thiểu
              trên mỗi SM.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            There is no separate block-slot limit. <code>maxBlocksPerMultiProcessor</code> exists in{' '}
            <code>hipDeviceProp_t</code>, but the runtime computes it as thread slots divided by the wavefront width, so
            it is the number of wave slots. The 32-thread, 50% case above has no AMD counterpart.
          </li>
          <li>
            A block is rounded up to whole waves before anything is divided. On a 64-wide device a 65-thread block costs
            128 thread slots.
          </li>
          <li>
            <code>__launch_bounds__</code> is spelt the same and means something else in its second argument: the minimum
            number of warps per execution unit (a compute unit has four), where CUDA's is the minimum number of blocks
            per SM.
          </li>
        </ul>
      </Translated>
      <Ask q={questionsS47['ch04/4.7/q-hipslots']} />

      <h2><Translated vi="Trên Metal không có gì để tính">On Metal there is nothing to compute</Translated></h2>
      <Translated
        vi={
          <p>
            Apple không công bố con số occupancy nào, không giới hạn số luồng thường trú, không khe khối, không vùng thanh ghi, không
            số thanh ghi theo từng kernel và không hàm nào trả về bất kỳ thứ gì trong số đó. Đó là lý do máy tính không có mục
            nào cho thiết bị Metal: không có con số nào để đưa vào, và trang này sẽ không bịa ra chúng.
          </p>
        }
      >
        <p>
          Apple documents no occupancy figure, no limit on resident threads, no block slots, no register pool, no
          per-kernel register count and no function that returns any of them. That is why the calculator has no entry
          for a Metal device: there are no numbers to put in it, and this page will not invent them.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Đã kiểm tra ngày 2026-11-20, đối chiếu với đặc tả Metal Shading Language 4.1, các trang của Apple và header
              của macOS SDK (ch04-facts.md, C5). Không tìm thấy gì, và đó chính là kết quả.
            </p>
          }
        >
          <p>
            Checked 2026-11-20 against the Metal Shading Language specification 4.1, Apple's pages and the macOS SDK
            headers (ch04-facts.md, C5). Nothing was found, which is the finding.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Hành vi được công bố gần nhất là giới hạn luồng của một threadgroup thuộc về kernel đã biên dịch, không
            thuộc về thiết bị. <code>maxTotalThreadsPerThreadgroup</code> là một thuộc tính của pipeline state, và Apple nói nó
            có thể khác nhau giữa các pipeline và phụ thuộc vào việc dùng bộ nhớ của kernel. Điều đó giống với "một kernel cần nhiều hơn
            thì nhận khối nhỏ hơn", nhưng Apple không giải thích nó bằng thanh ghi hay occupancy, và sự giống nhau ấy
            là nhận xét của chúng tôi. Trên Metal, câu hỏi về kích thước khối được giải quyết bằng cách đo.
          </p>
        }
      >
        <p>
          The nearest documented behaviour is that the thread limit of a threadgroup belongs to the compiled kernel, not
          to the device. <code>maxTotalThreadsPerThreadgroup</code> is a property of a pipeline state, and Apple says it
          can differ between pipelines and depends on the kernel's memory use. That resembles "a kernel that needs more
          gets a smaller block", but Apple does not explain it in terms of registers or occupancy, and the resemblance
          is ours. On Metal the block-size question is settled by measuring.
        </p>
      </Translated>

      <h2><Translated vi="Nhiều hơn không phải lúc nào cũng tốt hơn">More is not always better</Translated></h2>
      <Translated
        vi={
          <p>
            Occupancy là phương tiện. Mục 4.6 đã cho thấy nó mua được gì: đủ warp sẵn sàng để che thời gian chờ. Khi kernel đã có
            đủ số đó, thêm warp không thêm được gì ở các đơn vị thực thi, còn thanh ghi hay bộ nhớ chia sẻ chúng chiếm thì phải
            lấy từ đâu đó. Một kernel chờ bộ nhớ ở hầu như mỗi bước muốn SM đầy. Một kernel tính toán
            lâu trên các giá trị trong thanh ghi có thể chạy tốt nhất với ít warp và nhiều thanh ghi cho mỗi warp, đó là cách
            phép nhân ma trận của chương 15 cố ý dừng ở 12.5%.
          </p>
        }
      >
        <p>
          Occupancy is a means. Section 4.6 showed what it buys: enough ready warps to cover the waits. Once a kernel has
          that many, more warps add nothing at the execution units, and the registers or shared memory they take have to
          come from somewhere. A kernel that waits on memory almost every step wants the SM full. A kernel that computes
          for a long time on values in registers may do best with few warps and many registers each, which is how
          chapter 15's matrix multiplication ends up at 12.5% on purpose.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Con số 12.5% là của sách (ch. 15, p. 363). Hướng dẫn của AMD nói thẳng: quá điểm che được độ trễ,
              occupancy cao hơn có thể gây hại, và các kernel bị giới hạn bởi bộ nhớ mới là những kernel được lợi từ nó (ch04-facts.md, B10).
            </p>
          }
        >
          <p>
            The 12.5% is the book's (ch. 15, p. 363). AMD's guide says it outright: past the point of hiding latency,
            more occupancy can hurt, and memory-bound kernels are the ones that gain from it (ch04-facts.md, B10).
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Bài tập của sách">The book's exercises</Translated></h2>
      <Translated
        vi={
          <p>
            Bài tập 6 là một câu hỏi ở trên. Bài tập 7 và 8 ở đây, với một điểm khác so với máy tính: bạn
            trả lời trước.
          </p>
        }
      >
        <p>
          Exercise 6 was a question above. Exercises 7 and 8 are here, with one difference from the calculator: you
          answer first.
        </p>
      </Translated>
      <Source section="4.10" pages="91–92" />
      <Exercises />

      <h2><Translated vi="Đo nó">Measure it</Translated></h2>
      <Translated
        vi={
          <p>
            Tất cả những điều này dự đoán rằng riêng kích thước khối có thể thay đổi tốc độ của một kernel bị giới hạn bởi bộ nhớ, trên một thiết bị mà
            khối nhỏ để trống khe. Lab 4.4 trên <a href={href('ch04', 'labs')}>trang lab</a> kiểm chứng điều đó trên
            GPU của bạn: một kernel cộng các phần tử của bảng được chọn bởi một mảng chỉ số, để gần như mỗi bước đều là một lần nạp từ
            bộ nhớ toàn cục (global memory), đo thời gian ở nhiều kích thước khối. Hãy chốt một con số trước khi chạy.
          </p>
        }
      >
        <p>
          All of this predicts that block size alone can change how fast a memory-bound kernel runs, on a device where
          small blocks leave slots empty. Lab 4.4 on the <a href={href('ch04', 'labs')}>labs page</a> tests that on your
          GPU: one kernel that sums table entries chosen by an index array, so that nearly every step is a load from
          global memory, timed at a range of block sizes. Commit to a number before you run it.
        </p>
      </Translated>
      <Predict
        id="ch04/4.7/p-blocksize"
        prompt={
          <Translated
            vi={
              <>
                Trên thiết bị của bạn, với một kernel chủ yếu chờ bộ nhớ: kích thước khối chậm nhất sẽ mất thời gian gấp bao nhiêu lần
                kích thước nhanh nhất? (1 nghĩa là kích thước khối không tạo khác biệt.)
              </>
            }
          >
            <>
              On your device, for a kernel that mostly waits on memory: how many times as long will the slowest block size
              take as the fastest? (1 means block size makes no difference.)
            </>
          </Translated>
        }
        unit={tr('times', 'lần')}
      >
        {(v) => <BlockSizeReveal committed={v} />}
      </Predict>
      <Translated
        vi={
          <p>
            Một đầu vào của mọi phép tính trên trang này là "giới hạn của một SM". Chúng đến từ đâu, cho từng backend trong
            ba backend, là nội dung của <a href={href('ch04', '4.8')}>mục 4.8</a>.
          </p>
        }
      >
        <p>
          One input to every calculation on this page has been "the limits of one SM". Where they come from, for each of
          the three backends, is <a href={href('ch04', '4.8')}>section 4.8</a>.
        </p>
      </Translated>
    </>
  );
}
