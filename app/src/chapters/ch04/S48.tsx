import { useMemo, useState, type KeyboardEvent, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, labStatus, num, useReports, type BenchRow, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  BOOK_LAUNCH, CUDA_BLOCK_DIMS, CUDA_GRID_DIMS, LAB_PROBLEMS, appleRecipe, checkLaunch, launchLimitsFromReport,
  occupancyLimits, suggest, type LaunchLimits, type RuleStatus,
} from '../../sim/launchconfig';
import { isSynthetic, occupancy, pct } from '../../sim/occupancy';

const count = (v: number) => v.toLocaleString('en-US');
const plural = (v: number, one: string, many = one + 's') => `${count(v)} ${v === 1 ? one : many}`;
const toInt = (s: string) => {
  const v = Number(s);
  return Number.isFinite(v) && v > 0 ? Math.floor(v) : 0;
};
const onKey = (fn: () => void) => (e: KeyboardEvent) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fn();
  }
};
const LAB1 = 'ch04/lab1_launchconfig';

/** The row lab 4.1's benchmark writes with what the reader's own query returned, and whose code produced it. */
function queriedRow(reports: Reports, backend: string): { row: BenchRow; own: boolean } | null {
  const run = reports[backend];
  if (!run) return null;
  const find = (variant: string) => benchRows(run, LAB1, variant).find((r) => r.name === 'queried');
  const mine = labStatus(run, LAB1, 'work') === 'pass' ? find('work') : undefined;
  if (mine) return { row: mine, own: true };
  const ref = find('solution');
  return ref ? { row: ref, own: false } : null;
}

// ---------------------------------------------------------------- "Your device, field by field"
type Family = 'cuda' | 'hip' | 'metal';
interface Where {
  /** The property as the backend spells it. */
  name: string;
  /** Keys of the device report (reports/runs/<backend>.json) that hold it. */
  keys?: string[];
  /** Keys of lab 4.1's "queried" benchmark row that hold it. */
  lab?: string[];
  /** A remark that belongs to this backend. */
  note?: string;
  /** Read from a compiled pipeline, not from the device. */
  pipeline?: boolean;
}
interface FieldRow {
  id: string;
  concept: string;
  use: string;
  cuda: Where | string;
  /** Only where HIP differs from CUDA; otherwise the CUDA entry is used. */
  hip?: Where | string;
  metal: Where | string;
  /** What Metal calls the same thing, where it has it under another name. */
  metalConcept?: string;
  /** The book's figure for its example device, with the page. */
  book?: string;
}

const NONE_METAL = () => tr('No counterpart: Metal documents nothing of the kind.', 'Không có bản tương ứng: Metal không công bố gì thuộc loại này.');
export const FIELDS: FieldRow[] = [
  {
    id: 'name', get concept() { return tr('Name', 'Tên'); }, get use() { return tr('Telling devices apart when there are several.', 'Phân biệt các thiết bị khi có nhiều thiết bị.'); },
    cuda: { name: 'name', keys: ['name'], lab: ['device_name'] }, // i18n-ignore: API field name
    metal: { name: 'MTLDevice.name', keys: ['name'], lab: ['device'] },
  },
  {
    id: 'cc', get concept() { return tr('Architecture number', 'Số hiệu kiến trúc'); }, get use() { return tr('Looking the device up in the vendor\'s tables. It does not rank capacity.', 'Tra thiết bị trong bảng của nhà cung cấp. Nó không xếp hạng năng lực.'); },
    cuda: { name: 'major, minor (compute capability)', keys: ['computeCapability.major', 'computeCapability.minor'], lab: ['major', 'minor'] }, // i18n-ignore: API field names
    hip: { name: 'major, minor', keys: ['computeCapability.major', 'computeCapability.minor'], lab: ['major', 'minor'], get note() { return tr('An AMD architecture number, not a compute capability.', 'Một số hiệu kiến trúc AMD, không phải compute capability.'); } }, // i18n-ignore: API field names
    get metal() { return tr('No counterpart. The device has an architecture name and no number.', 'Không có bản tương ứng. Thiết bị có tên kiến trúc nhưng không có số hiệu.'); },
    book: '9.0 (p. 88)',
  },
  {
    id: 'sms', get concept() { return tr('Units that run blocks', 'Các đơn vị chạy khối'); }, get use() { return tr('With blocks per unit, how many blocks run at once: the waves of section 4.3.', 'Cùng với số khối mỗi đơn vị, cho biết bao nhiêu khối chạy cùng lúc: các đợt của mục 4.3.'); },
    cuda: { name: 'multiProcessorCount', keys: ['multiProcessorCount'], lab: ['multiProcessorCount'], get note() { return tr('SMs.', 'Số SM.'); } },
    hip: { name: 'multiProcessorCount', keys: ['multiProcessorCount'], lab: ['multiProcessorCount'], get note() { return tr('Compute units, or half of them in one driver mode.', 'Số compute unit, hoặc một nửa số đó ở một chế độ driver.'); } },
    get metal() { return tr('No counterpart. No count of cores or compute units is exposed.', 'Không có bản tương ứng. Không có số lõi hay số compute unit nào được cung cấp.'); },
    book: '132 (p. 68)',
  },
  {
    id: 'clock', get concept() { return tr('Clock rate', 'Xung nhịp'); }, get use() { return tr('Nothing in this course. The book offers it, with the SM count, as a rough measure of throughput.', 'Khóa học này không dùng. Sách đưa nó ra, cùng số SM, như một thước đo thô về thông lượng.'); },
    cuda: { name: 'cudaDeviceGetAttribute(…, cudaDevAttrClockRate, …)', get note() { return tr('No longer a member of the struct. The device report does not read it.', 'Không còn là thành viên của struct. Báo cáo thiết bị không đọc nó.'); } },
    hip: { name: 'clockRate', get note() { return tr('Still a member of hipDeviceProp_t. The device report does not read it.', 'Vẫn là thành viên của hipDeviceProp_t. Báo cáo thiết bị không đọc nó.'); } },
    get metal() { return tr('No counterpart.', 'Không có bản tương ứng.'); },
  },
  {
    id: 'width', get concept() { return tr('Width of a warp', 'Độ rộng của một warp'); }, get use() { return tr('Block sizes in whole warps; which threads share a warp (sections 4.4 and 4.5).', 'Kích thước khối theo trọn warp; những luồng nào chung một warp (mục 4.4 và 4.5).'); },
    cuda: { name: 'warpSize', keys: ['warpSize'], lab: ['warpSize'] },
    metal: { name: 'threadExecutionWidth', lab: ['threadExecutionWidth'], pipeline: true, get note() { return tr('A property of a compiled pipeline. The device has none.', 'Một thuộc tính của pipeline đã biên dịch. Thiết bị không có thuộc tính này.'); } },
    get metalConcept() { return tr('Width of a SIMD-group', 'Độ rộng của một SIMD group'); },
    book: '32 (p. 74)',
  },
  {
    id: 'perblock', get concept() { return tr('Threads in a block', 'Số luồng trong một khối'); }, get use() { return tr('Whether a launch is valid at all.', 'Một lần khởi chạy có hợp lệ hay không.'); },
    cuda: { name: 'maxThreadsPerBlock', keys: ['maxThreadsPerBlock'], lab: ['maxThreadsPerBlock'] },
    metal: { name: 'maxTotalThreadsPerThreadgroup', lab: ['maxTotalThreadsPerThreadgroup'], pipeline: true, get note() { return tr('Per pipeline: it can differ from one kernel to the next.', 'Theo từng pipeline: nó có thể khác nhau giữa các kernel.'); } },
    get metalConcept() { return tr('Threads in a threadgroup', 'Số luồng trong một threadgroup'); },
    book: '1024 (p. 85)',
  },
  {
    id: 'dims', get concept() { return tr('Block size along x, y, z', 'Kích thước khối theo x, y, z'); }, get use() { return tr('Bounding the block shapes a program tries.', 'Giới hạn các hình dạng khối mà một chương trình thử.'); },
    cuda: { name: 'maxThreadsDim[0..2]', keys: ['maxThreadsDim.x', 'maxThreadsDim.y', 'maxThreadsDim.z'], lab: ['maxThreadsDim_x', 'maxThreadsDim_y', 'maxThreadsDim_z'] },
    metal: {
      name: 'MTLDevice.maxThreadsPerThreadgroup',
      keys: ['maxThreadsPerThreadgroup.width', 'maxThreadsPerThreadgroup.height', 'maxThreadsPerThreadgroup.depth'],
      lab: ['maxThreadsPerThreadgroup_width', 'maxThreadsPerThreadgroup_height', 'maxThreadsPerThreadgroup_depth'],
      get note() { return tr('For the simplest possible kernel. Not guaranteed for yours.', 'Cho kernel đơn giản nhất có thể. Không được đảm bảo cho kernel của bạn.'); },
    },
    get metalConcept() { return tr('Threadgroup size along x, y, z', 'Kích thước threadgroup theo x, y, z'); },
  },
  {
    id: 'grid', get concept() { return tr('Grid size along x, y, z', 'Kích thước lưới theo x, y, z'); }, get use() { return tr('Whether one launch can cover the data.', 'Một lần khởi chạy có phủ hết dữ liệu được không.'); },
    cuda: { name: 'maxGridSize[0..2]', keys: ['maxGridSize.x', 'maxGridSize.y', 'maxGridSize.z'], lab: ['maxGridSize_x', 'maxGridSize_y', 'maxGridSize_z'] },
    get metal() { return tr('No counterpart among the device properties Apple documents.', 'Không có bản tương ứng trong các thuộc tính thiết bị Apple công bố.'); },
  },
  {
    id: 'threadslots', get concept() { return tr('Thread slots of one SM', 'Số khe luồng của một SM'); }, get use() { return tr('The denominator of occupancy (section 4.7).', 'Mẫu số của occupancy (mục 4.7).'); },
    cuda: { name: 'maxThreadsPerMultiProcessor', keys: ['maxThreadsPerMultiProcessor'], lab: ['maxThreadsPerMultiProcessor'], get note() { return tr('Not in the book\'s list.', 'Không có trong danh sách của sách.'); } },
    hip: { name: 'maxThreadsPerMultiProcessor', keys: ['maxThreadsPerMultiProcessor'], lab: ['maxThreadsPerMultiProcessor'], get note() { return tr('Wave slots times the width. May read 0.', 'Số khe wave nhân với độ rộng. Có thể đọc ra 0.'); } },
    get metal() { return NONE_METAL(); },
    book: '2048 (p. 85)',
  },
  {
    id: 'blockslots', get concept() { return tr('Block slots of one SM', 'Số khe khối của một SM'); }, get use() { return tr('Whether small blocks run out of slots before threads do.', 'Khối nhỏ có cạn khe khối trước khe luồng hay không.'); },
    cuda: { name: 'maxBlocksPerMultiProcessor', keys: ['maxBlocksPerMultiProcessor'], lab: ['maxBlocksPerMultiProcessor'], get note() { return tr('Not in the book\'s list.', 'Không có trong danh sách của sách.'); } },
    hip: { name: 'maxBlocksPerMultiProcessor', keys: ['maxBlocksPerMultiProcessor'], lab: ['maxBlocksPerMultiProcessor'], get note() { return tr('Derived: thread slots divided by the width. Not a limit of its own.', 'Được suy ra: số khe luồng chia cho độ rộng. Tự nó không phải một giới hạn.'); } },
    get metal() { return NONE_METAL(); },
    book: '32 (p. 85)',
  },
  {
    id: 'regssm', get concept() { return tr('Registers of one SM', 'Số thanh ghi của một SM'); }, get use() { return tr('Registers per thread that full occupancy allows: divide by the thread slots.', 'Số thanh ghi mỗi luồng mà occupancy đầy đủ cho phép: chia cho số khe luồng.'); },
    cuda: { name: 'regsPerMultiprocessor', keys: ['regsPerMultiprocessor'], lab: ['regsPerMultiprocessor'], get note() { return tr('Not in the book\'s list; the book describes this figure under regsPerBlock.', 'Không có trong danh sách của sách; sách mô tả con số này dưới tên regsPerBlock.'); } },
    hip: { name: 'regsPerMultiprocessor', keys: ['regsPerMultiprocessor'], lab: ['regsPerMultiprocessor'], get note() { return tr('Vector registers only. The scalar pool is not reported.', 'Chỉ thanh ghi vector. Vùng thanh ghi vô hướng không được báo cáo.'); } },
    get metal() { return NONE_METAL(); },
    book: '65,536 (p. 86)',
  },
  {
    id: 'regsblock', get concept() { return tr('Registers one block may hold', 'Số thanh ghi một khối được giữ'); }, get use() { return tr('An upper bound for one block. Not the pool.', 'Một cận trên cho một khối. Không phải cả vùng chung.'); },
    cuda: { name: 'regsPerBlock', keys: ['regsPerBlock'], lab: ['regsPerBlock'] },
    hip: { name: 'regsPerBlock', keys: ['regsPerBlock'], lab: ['regsPerBlock'], get note() { return tr('Filled with the same value as regsPerMultiprocessor. May read 0.', 'Được điền cùng giá trị với regsPerMultiprocessor. Có thể đọc ra 0.'); } },
    get metal() { return NONE_METAL(); },
  },
  {
    id: 'sharedblock', get concept() { return tr('Shared memory for one block', 'Bộ nhớ chia sẻ cho một khối'); }, get use() { return tr('How large a tile can be (section 5.6).', 'Một ô (tile) có thể lớn đến đâu (mục 5.6).'); },
    cuda: { name: 'sharedMemPerBlock', keys: ['sharedMemPerBlock'], lab: ['sharedMemPerBlock'] },
    metal: { name: 'MTLDevice.maxThreadgroupMemoryLength', keys: ['maxThreadgroupMemoryLength'], lab: ['maxThreadgroupMemoryLength'] },
    get metalConcept() { return tr('Threadgroup memory for one threadgroup', 'Bộ nhớ threadgroup cho một threadgroup'); },
  },
  {
    id: 'sharedsm', get concept() { return tr('Shared memory of one SM', 'Bộ nhớ chia sẻ của một SM'); }, get use() { return tr('The fourth occupancy budget (section 5.6).', 'Ngân sách occupancy thứ tư (mục 5.6).'); },
    cuda: { name: 'sharedMemPerMultiprocessor', keys: ['sharedMemPerMultiprocessor'], get note() { return tr('Not in the book\'s list.', 'Không có trong danh sách của sách.'); } },
    get metal() { return NONE_METAL(); },
    book: '228 KB (ch. 5, p. 116)', // i18n-ignore: figure and citation
  },
];

interface Shown {
  name: string;
  note?: string;
  none?: string;
  /** The value in the device report, or why there is none. */
  value: string | null;
  missing?: string;
  zero?: boolean;
  /** The value from lab 4.1's own query. */
  lab: string | null;
  pipeline?: boolean;
}

/** One row of the table for one backend, with nothing filled in that the reports do not contain. */
export function shownField(f: FieldRow, family: Family, device: Record<string, unknown> | null, lab: BenchRow | null): Shown {
  const w = family === 'metal' ? f.metal : family === 'hip' ? (f.hip ?? f.cuda) : f.cuda;
  if (typeof w === 'string') return { name: '', none: w, value: null, lab: null };
  const read = (src: Record<string, unknown> | null, keys?: string[]) => {
    if (!src || !keys || !keys.every((k) => k in src)) return null;
    return keys.map((k) => String(src[k])).join(f.id === 'cc' ? '.' : ' × ');
  };
  const value = read(device, w.keys);
  const zero = value != null && w.keys != null && f.id !== 'name' && f.id !== 'cc' && w.keys.every((k) => Number(device?.[k]) === 0);
  return {
    name: w.name,
    note: w.note,
    value,
    zero,
    missing:
      value != null
        ? undefined
        : !w.keys
          ? w.pipeline
            ? tr('not a device property', 'không phải thuộc tính của thiết bị')
            : tr('not read by the device report', 'báo cáo thiết bị không đọc')
          : tr('not in this report', 'không có trong báo cáo này'),
    lab: read(lab as Record<string, unknown> | null, w.lab),
    pipeline: w.pipeline,
  };
}

function FieldTable() {
  const { reports, loading } = useReports();
  const backends = useMemo(
    () => Object.keys(reports).filter((b) => !isSynthetic(b, reports[b].device as Record<string, unknown>)).sort(),
    [reports],
  );
  const [key, setKey] = useState<string | null>(null);
  const chosen = key ?? backends[0] ?? 'book';
  const isBook = chosen === 'book' || !reports[chosen];
  const family: Family = isBook ? 'cuda' : chosen === 'metal' || 'maxThreadgroupMemoryLength' in reports[chosen].device ? 'metal' : chosen === 'hip' ? 'hip' : 'cuda';
  const device = isBook ? null : (reports[chosen].device as Record<string, unknown>);
  const q = isBook ? null : queriedRow(reports, chosen);
  const rows = FIELDS.map((f) => ({ f, s: shownField(f, family, device, q?.row ?? null) }));
  const none = rows.filter((r) => r.s.none).length;

  return (
    <Bench
      title={tr('Your device, field by field', 'Thiết bị của bạn, từng trường một')}
      task={tr("Switch between the backends you have and the book's example. Which rows have a value on one and nothing on another?", 'Chuyển qua lại giữa các backend bạn có và ví dụ của sách. Những hàng nào có giá trị ở bên này mà trống ở bên kia?')}
      foot={
        <Translated
          vi={
            <>
              Các giá trị được đọc từ <code>reports/runs/</code> trên máy này, như <code>./verify.sh</code> (hoặc lần{' '}
              <code>labs/pmpp.py run</code> đầu tiên trên một backend) và benchmark của lab 4.1 đã ghi. Không có gì được điền vào khi báo cáo không có. Các trình giả lập (emulator) CPU cũng ghi một
              báo cáo thiết bị; giá trị của nó không mô tả GPU nào nên không được đưa ra ở đây.
            </>
          }
        >
          <>
            Values are read from <code>reports/runs/</code> on this machine, as <code>./verify.sh</code> (or the first{' '}
            <code>labs/pmpp.py run</code> on a backend) and lab 4.1's benchmark wrote them. Nothing is filled in when a report does not have it. The CPU emulators also write a
            device report; its values describe no GPU and it is not offered here.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Device', 'Thiết bị')}>
          {backends.map((b) => (
            <button key={b} aria-pressed={chosen === b} data-device={b} onClick={() => setKey(b)}>
              <Translated vi={<>Của bạn: {String(reports[b].device.name ?? b)} ({b})</>}>
                Yours: {String(reports[b].device.name ?? b)} ({b})
              </Translated>
            </button>
          ))}
          <button aria-pressed={isBook} data-device="book" onClick={() => setKey('book')}>
            <Translated vi="Ví dụ của sách">The book's example</Translated>
          </button>
        </span>
      </div>
      {!loading && backends.length === 0 ? (
        <p className="muted" style={{ margin: '0 0 0.8rem' }}>
          <Translated vi={<>Máy này chưa có báo cáo thiết bị GPU nào. Hãy chạy <code>./verify.sh --quick</code>, rồi tải lại trang này.</>}>
            No GPU device report on this machine yet. Run <code>./verify.sh --quick</code>, then reload this page.
          </Translated>
        </p>
      ) : null}
      <p style={{ margin: '0 0 0.8rem' }} data-out="field-summary" data-none={none}>
        {isBook ? (
          <Translated vi={<>Thiết bị ví dụ của sách, với các con số sách đưa ra cho nó. Ở chỗ sách không đưa ra, ô sẽ ghi rõ.</>}>
            <>The book's example device, with the figures the book gives for it. Where the book gives none, the cell says so.</>
          </Translated>
        ) : family === 'metal' ? (
          <Translated
            vi={
              <>
                Một thiết bị Metal. <b>{none} trên {FIELDS.length}</b> hàng hoàn toàn không có bản tương ứng, và hai hàng nữa chỉ tồn tại trên một
                pipeline đã biên dịch.{' '}
                {q ? <>Các giá trị pipeline lấy từ benchmark của lab 4.1 ({q.own ? 'queryMetal của chính bạn' : 'queryMetal của lời giải tham chiếu; của bạn sẽ hiện khi nó qua các bài kiểm thử'}).</> : <>Các giá trị pipeline sẽ hiện khi benchmark của lab 4.1 đã chạy.</>}
              </>
            }
          >
            <>
              A Metal device. <b>{none} of {FIELDS.length}</b> rows have no counterpart at all, and two more exist only on a compiled
              pipeline.{' '}
              {q ? <>The pipeline values are from lab 4.1's benchmark ({q.own ? 'your own queryMetal' : "the reference solution's queryMetal; your own appears once it passes its tests"}).</> : <>The pipeline values appear once lab 4.1's benchmark has run.</>}
            </>
          </Translated>
        ) : (
          <Translated
            vi={
              <>
                {family === 'hip' ? 'Một thiết bị HIP. Các tên là của CUDA; một số giá trị được tính khác đi, như các ghi chú cho biết.' : 'Một thiết bị CUDA.'}{' '}
                {q ? <>Cột cuối là những gì {q.own ? 'queryDevices của chính bạn' : 'queryDevices của lời giải tham chiếu'} trả về trong benchmark của lab 4.1.</> : <>Một cột cho truy vấn của chính bạn sẽ hiện khi benchmark của lab 4.1 đã chạy.</>}
              </>
            }
          >
            <>
              {family === 'hip' ? 'A HIP device. The names are CUDA\'s; several values are computed differently, as the notes say.' : 'A CUDA device.'}{' '}
              {q ? <>The last column is what {q.own ? 'your own queryDevices' : "the reference solution's queryDevices"} returned in lab 4.1's benchmark.</> : <>A column for your own query appears once lab 4.1's benchmark has run.</>}
            </>
          </Translated>
        )}
      </p>
      <div className="panel" style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }} data-family={family} data-device={chosen}>
          <thead>
            <tr>
              <th style={{ width: '17%' }}><Translated vi="Đại lượng">What</Translated></th>
              <th style={{ width: '27%' }}>
                {isBook ? tr('Field (CUDA)', 'Trường (CUDA)') : family === 'metal' ? tr('Property (Metal)', 'Thuộc tính (Metal)') : tr(`Field (${family === 'hip' ? 'HIP' : 'CUDA'})`, `Trường (${family === 'hip' ? 'HIP' : 'CUDA'})`)}
              </th>
              <th style={{ width: '16%' }}>{isBook ? tr("The book's figure", 'Con số của sách') : tr('Reported', 'Được báo cáo')}</th>
              {q ? <th style={{ width: '14%' }}><Translated vi="Truy vấn của lab 4.1">Lab 4.1's query</Translated></th> : null}
              <th><Translated vi="Chương này dùng nó để làm gì">What this chapter does with it</Translated></th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ f, s }) => (
              <tr key={f.id} data-field={f.id} data-kind={s.none ? 'none' : s.pipeline ? 'pipeline' : 'device'}>
                <td>{family === 'metal' && f.metalConcept ? f.metalConcept : f.concept}</td>
                {s.none ? (
                  <td colSpan={q ? 3 : 2} className="muted" data-value="none">
                    {s.none}
                  </td>
                ) : (
                  <>
                    <td>
                      <code>{s.name}</code>
                      {s.note ? <div className="muted" style={{ fontSize: '0.78rem' }}>{s.note}</div> : null}
                    </td>
                    <td data-value={isBook ? (f.book ?? 'none') : (s.value ?? 'none')}>
                      {isBook ? (
                        f.book ?? <span className="muted"><Translated vi="sách không có con số">no figure in the book</Translated></span>
                      ) : s.value != null ? (
                        <>
                          {s.value}
                          {s.zero ? <span className="muted"><Translated vi=" (không được báo cáo)"> (not reported)</Translated></span> : null}
                        </>
                      ) : (
                        <span className="muted">{s.missing}</span>
                      )}
                    </td>
                    {q ? <td data-lab={s.lab ?? 'none'}>{s.lab ?? <span className="muted">{s.pipeline || f.id === 'clock' || f.id === 'sharedsm' ? tr('not queried', 'không truy vấn') : tr('not in the row', 'không có trong hàng')}</span>}</td> : null}
                  </>
                )}
                <td className="muted">{f.use}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- erratum hunt: the field list on p. 89
// The loop is the book's (p. 88). The lines inside it are ours: one for each field the
// book lists on p. 89, with the book's description as the comment. Not compiled.
export const AS_PRINTED = `int devCount;
cudaGetDeviceCount(&devCount);
cudaDeviceProp devProp;
for(unsigned int i = 0; i < devCount; i++) {
    cudaGetDeviceProperties(&devProp, i);
    int perBlock = devProp.maxThreadsPerBlock;   // most threads in a block
    int sms      = devProp.multiProcessorCount;  // SMs in the device
    int clock    = devProp.clockRate;            // clock frequency
    int blockX   = devProp.maxThreadsDim[0];     // most threads along x
    int gridX    = devProp.maxGridSize[0];       // most blocks along x
    int regs     = devProp.regsPerBlock;         // registers in each SM
    int width    = devProp.warpSize;             // threads in a warp
}`;
type Hunt = 'compile' | 'describe';
export const HUNT_ANSWER: Record<Hunt, number> = { compile: 8, describe: 11 };
const HUNT_PROMPT: Record<Hunt, string> = {
  get compile() { return tr('Which line stops this file compiling against current CUDA headers?', 'Dòng nào khiến tệp này không biên dịch được với các header CUDA hiện tại?'); },
  get describe() { return tr('Which line reads a field that is not what its comment says?', 'Dòng nào đọc một trường không đúng như chú thích của nó nói?'); },
};
/** Why a line is not the answer, by 1-based line number. */
const LINE_FINE: Record<number, Record<Hunt, string>> = {
  1: { get compile() { return tr('An int declaration compiles anywhere.', 'Một khai báo int biên dịch được ở bất cứ đâu.'); }, get describe() { return tr('There is no field on this line.', 'Dòng này không có trường nào.'); } },
  2: { get compile() { return tr('cudaGetDeviceCount is unchanged. What the page leaves out is its return value, an error code.', 'cudaGetDeviceCount không đổi. Điều trang sách bỏ qua là giá trị trả về của nó, một mã lỗi.'); }, get describe() { return tr('There is no field on this line.', 'Dòng này không có trường nào.'); } },
  3: { get compile() { return tr('The struct still has this name.', 'Struct vẫn mang tên này.'); }, get describe() { return tr('There is no field on this line.', 'Dòng này không có trường nào.'); } },
  4: {
    get compile() { return tr('It compiles. It compares an unsigned index with an int count, which some compilers warn about; NVIDIA\'s own version of this loop declares i as int.', 'Nó biên dịch được. Nó so sánh một chỉ số unsigned với một số đếm int, điều một số trình biên dịch cảnh báo; phiên bản vòng lặp này của chính NVIDIA khai báo i là int.'); },
    get describe() { return tr('There is no field on this line.', 'Dòng này không có trường nào.'); },
  },
  5: { get compile() { return tr('The call is unchanged. Like the count, it returns an error code the page does not look at.', 'Lời gọi không đổi. Giống lời gọi đếm, nó trả về một mã lỗi mà trang sách không xem tới.'); }, get describe() { return tr('There is no field on this line.', 'Dòng này không có trường nào.'); } },
  6: { get compile() { return tr('maxThreadsPerBlock is still a member.', 'maxThreadsPerBlock vẫn là một thành viên.'); }, get describe() { return tr('The comment is what the field means.', 'Chú thích đúng với ý nghĩa của trường.'); } },
  7: { get compile() { return tr('multiProcessorCount is still a member.', 'multiProcessorCount vẫn là một thành viên.'); }, get describe() { return tr('The comment is what the field means on NVIDIA. On AMD it counts compute units.', 'Chú thích đúng với ý nghĩa của trường trên NVIDIA. Trên AMD nó đếm số compute unit.'); } },
  8: { compile: '', get describe() { return tr('The comment is right about what the field was. The trouble with this line is of the other kind.', 'Chú thích đúng về việc trường này từng là gì. Vấn đề của dòng này thuộc loại còn lại.'); } },
  9: { get compile() { return tr('maxThreadsDim is still a member.', 'maxThreadsDim vẫn là một thành viên.'); }, get describe() { return tr('The comment is what the field means.', 'Chú thích đúng với ý nghĩa của trường.'); } },
  10: { get compile() { return tr('maxGridSize is still a member.', 'maxGridSize vẫn là một thành viên.'); }, get describe() { return tr('The comment is what the field means.', 'Chú thích đúng với ý nghĩa của trường.'); } },
  11: { get compile() { return tr('regsPerBlock is still a member, so this compiles. Look at what it holds.', 'regsPerBlock vẫn là một thành viên, nên dòng này biên dịch được. Hãy xem nó chứa gì.'); }, describe: '' },
  12: { get compile() { return tr('warpSize is still a member.', 'warpSize vẫn là một thành viên.'); }, get describe() { return tr('The comment is what the field means.', 'Chú thích đúng với ý nghĩa của trường.'); } },
  13: { get compile() { return tr('A closing brace.', 'Một dấu ngoặc nhọn đóng.'); }, get describe() { return tr('There is no field on this line.', 'Dòng này không có trường nào.'); } },
};

function FieldHunt() {
  const lines = AS_PRINTED.split('\n');
  const [mode, setMode] = useState<Hunt>('compile');
  const [picks, setPicks] = useState<Record<Hunt, number | null>>({ compile: null, describe: null });
  const [answers, setAnswers] = useState(false);
  const pick = (n: number) => setPicks({ ...picks, [mode]: n });
  const found = (h: Hunt) => picks[h] === HUNT_ANSWER[h];
  const state = (n: number): 'idle' | 'right' | 'wrong' | 'missed' => {
    for (const h of ['compile', 'describe'] as Hunt[]) {
      if (picks[h] === n && h === mode) return found(h) ? 'right' : 'wrong';
      if (HUNT_ANSWER[h] === n && (found(h) || answers)) return found(h) ? 'right' : 'missed';
    }
    return 'idle';
  };
  const reveal = (h: Hunt): ReactNode =>
    h === 'compile' ? (
      <Translated
        vi={
          <>
            <b>Dòng 8.</b> <code>clockRate</code> bị đánh dấu lỗi thời rồi bị gỡ khỏi <code>cudaDeviceProp</code> trong CUDA 13.0, nên một
            tệp đọc nó không còn biên dịch được. Giá trị này vẫn lấy được, từng thuộc tính một:{' '}
            <code>cudaDeviceGetAttribute(&amp;clock, cudaDevAttrClockRate, i)</code>, tính bằng kilohertz.
          </>
        }
      >
        <>
          <b>Line 8.</b> <code>clockRate</code> was deprecated and then removed from <code>cudaDeviceProp</code> in CUDA 13.0, so a
          file that reads it no longer compiles. The value is still available, one attribute at a time:{' '}
          <code>cudaDeviceGetAttribute(&amp;clock, cudaDevAttrClockRate, i)</code>, in kilohertz.
        </>
      </Translated>
    ) : (
      <Translated
        vi={
          <>
            <b>Dòng 11.</b> <code>regsPerBlock</code> là số thanh ghi tối đa một khối được giữ. Vùng thanh ghi của một SM, thứ mà
            chú thích mô tả và occupancy cần, là <code>regsPerMultiprocessor</code>. Sách nói cái tên này
            gây hiểu nhầm; ngày nay có hai trường và tên gọi là chính xác.
          </>
        }
      >
        <>
          <b>Line 11.</b> <code>regsPerBlock</code> is the most registers one block may hold. The pool of an SM, which is what
          the comment describes and what occupancy needs, is <code>regsPerMultiprocessor</code>. The book says the name is
          misleading; today there are two fields and the name is accurate.
        </>
      </Translated>
    );
  return (
    <Bench
      title={tr('The field list on p. 89', 'Danh sách trường ở p. 89')}
      task={tr('Two lines here are wrong in different ways. Choose a question, then click the line.', 'Hai dòng ở đây sai theo hai kiểu khác nhau. Chọn một câu hỏi, rồi nhấp vào dòng đó.')}
      foot={tr(
        "The loop is as printed on p. 88. The lines inside it are ours, one for each field the book lists on p. 89, with the book's description as the comment. What is said about line 8 comes from NVIDIA's CUDA 13.0 release notes and 13.5 reference (ch04-facts.md, A18). Nothing here has been compiled by nvcc: the course's front-end check cannot settle it, because its stub header declares only the fields the course reads. The corrected loop above is front-end-checked.",
        'Vòng lặp đúng như in ở p. 88. Các dòng bên trong là của chúng tôi, mỗi dòng cho một trường mà sách liệt kê ở p. 89, với mô tả của sách làm chú thích. Những gì nói về dòng 8 lấy từ ghi chú phát hành CUDA 13.0 và tài liệu tham khảo 13.5 của NVIDIA (ch04-facts.md, A18). Không gì ở đây được nvcc biên dịch: phép kiểm tra front-end của khóa học không giải quyết được, vì header giả của nó chỉ khai báo các trường khóa học đọc. Vòng lặp đã sửa ở trên là front-end-checked.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Question', 'Câu hỏi')}>
          {(['compile', 'describe'] as Hunt[]).map((h) => (
            <button key={h} aria-pressed={mode === h} data-hunt={h} onClick={() => setMode(h)}>
              {h === 'compile' ? tr('A. Does not compile', 'A. Không biên dịch được') : tr('B. Description is off', 'B. Mô tả bị sai')}
            </button>
          ))}
        </span>
        <button className="quiet" onClick={() => setAnswers(true)} disabled={answers}>
          <Translated vi="Hiện cả hai đáp án">Show both answers</Translated>
        </button>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.2rem', alignItems: 'flex-start' }}>
        <div className="mono" data-listing="p89"
          style={{ flex: '3 1 34rem', minWidth: 0, background: 'var(--cell)', border: '1px solid var(--rule)', borderRadius: 4, padding: '0.5rem 0', overflowX: 'auto', fontSize: '0.8rem', lineHeight: 1.75, whiteSpace: 'pre' }}>
          {lines.map((text, i) => {
            const n = i + 1;
            const st = state(n);
            return (
              <div key={n} role="button" tabIndex={0} data-line={n} data-state={st} aria-pressed={picks[mode] === n}
                onClick={() => pick(n)} onKeyDown={onKey(() => pick(n))}
                style={{
                  cursor: 'pointer', paddingRight: '0.8rem',
                  background: st === 'right' ? 'var(--ok-wash)' : st === 'wrong' ? 'var(--fail-wash)' : undefined,
                  outline: st === 'missed' ? '2px dashed var(--fail)' : undefined, outlineOffset: -2,
                }}>
                <span className="muted" style={{ display: 'inline-block', width: '2.6rem', paddingRight: '0.9rem', textAlign: 'right', userSelect: 'none' }}>
                  {String(n).padStart(2, '0')}
                </span>
                {text}
              </div>
            );
          })}
        </div>
        <div className="panel" role="status" style={{ flex: '1 1 16rem', minWidth: 0 }} data-out="hunt" data-found={(['compile', 'describe'] as Hunt[]).filter(found).length}>
          <p style={{ margin: '0 0 0.6rem' }}>
            <b>{HUNT_PROMPT[mode]}</b>
          </p>
          {picks[mode] == null && !answers ? (
            <p className="muted" style={{ margin: 0 }}><Translated vi="Nhấp vào một dòng của đoạn mã.">Click a line of the listing.</Translated></p>
          ) : null}
          {(['compile', 'describe'] as Hunt[]).map((h) => {
            const p = picks[h];
            if (found(h) || answers)
              return (
                <p key={h} data-finding={h} style={{ margin: '0 0 0.6rem' }}>
                  <span className={'chip ' + (found(h) ? 'ok' : 'fail')}>{found(h) ? tr('found', 'đã tìm ra') : tr('shown', 'đã hiện')}</span> {reveal(h)}
                </p>
              );
            if (p != null && h === mode)
              return (
                <p key={h} data-miss={h} style={{ margin: '0 0 0.6rem' }}>
                  <span className="chip fail"><Translated vi={<>không phải dòng {p}</>}>not line {p}</Translated></span> {LINE_FINE[p][h]}
                </p>
              );
            return null;
          })}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- "Derive a launch configuration"
interface LimitChoice {
  key: string;
  label: string;
  limits: LaunchLimits;
  note: string;
}
const BOOK_CHOICE: LimitChoice = {
  key: 'book',
  get label() {
    return tr("The book's example", 'Ví dụ của sách');
  },
  limits: BOOK_LAUNCH,
  get note() {
    return tr(
      "The book's example device: warps of 32, 1024 threads per block, 2048 thread slots and 32 block slots per SM (pp. 74, 85), with CUDA's documented block and grid dimensions.",
      'Thiết bị ví dụ của sách: warp rộng 32, 1024 luồng mỗi khối, 2048 khe luồng và 32 khe khối mỗi SM (pp. 74, 85), cùng các chiều khối và lưới mà CUDA công bố.',
    );
  },
};
const TABLE_CHOICE: LimitChoice = {
  key: 't1536',
  get label() {
    return tr("A row of NVIDIA's table: 1536 thread slots, 16 block slots", 'Một hàng trong bảng của NVIDIA: 1536 khe luồng, 16 khe khối');
  },
  limits: { ...BOOK_LAUNCH, threadSlots: 1536, blockSlots: 16 },
  get note() {
    return tr(
      "The row for compute capabilities 8.6 and 8.7 in NVIDIA's table of limits, read on 2026-11-20 (ch04-facts.md, A12).",
      'Hàng dành cho compute capability 8.6 và 8.7 trong bảng giới hạn của NVIDIA, đọc ngày 2026-11-20 (ch04-facts.md, A12).',
    );
  },
};
const WIDE_CHOICE: LimitChoice = {
  key: 'w64',
  get label() {
    return tr("A 64-wide device from lab 4.1's test table", 'Một thiết bị rộng 64 trong bảng kiểm thử của lab 4.1');
  },
  limits: { ...BOOK_LAUNCH, width: 64, maxThreadsPerBlock: 256, maxThreadsDim: [256, 256, 256] },
  get note() {
    return tr(
      'One of the made-up device descriptions lab 4.1 tests your code against: warps of 64, 256 threads per block. It describes no product.',
      'Một trong các mô tả thiết bị giả định mà lab 4.1 dùng để kiểm thử mã của bạn: warp rộng 64, 256 luồng mỗi khối. Nó không mô tả sản phẩm nào.',
    );
  },
};

function reportedLimits(reports: Reports): LimitChoice[] {
  return Object.keys(reports).sort().flatMap((b) => {
    const d = reports[b].device as Record<string, unknown>;
    const limits = launchLimitsFromReport(d, b);
    if (!limits) return [];
    return [{
      key: b,
      get label() {
        return tr(`Yours: ${String(d.name ?? b)} (${b})`, `Của bạn: ${String(d.name ?? b)} (${b})`);
      },
      limits,
      get note() {
        return tr(
          `What this device reported in the run of ${reports[b].generated.slice(0, 10)}. A field that is missing or 0 is treated as not reported and its rule is skipped.`,
          `Những gì thiết bị này báo cáo ở lần chạy ngày ${reports[b].generated.slice(0, 10)}. Trường nào thiếu hoặc bằng 0 được coi là không báo cáo và quy tắc của nó bị bỏ qua.`,
        );
      },
    }];
  });
}

const CHIP: Record<RuleStatus, { cls: string; text: string }> = {
  pass: { cls: 'chip ok', get text() { return tr('holds', 'thỏa'); } }, // i18n-ignore: CSS class
  fail: { cls: 'chip fail', get text() { return tr('broken', 'vi phạm'); } }, // i18n-ignore: CSS class
  skipped: { cls: 'chip', get text() { return tr('skipped', 'bỏ qua'); } },
};

function LaunchExplorer() {
  const { reports } = useReports();
  const mine = useMemo(() => reportedLimits(reports), [reports]);
  const choices = useMemo(() => [BOOK_CHOICE, ...mine, TABLE_CHOICE, WIDE_CHOICE], [mine]);
  const [key, setKey] = useState('book');
  const [custom, setCustom] = useState({ width: '32', perBlock: '1024', threadSlots: '2048', blockSlots: '32' });
  const [rows, setRows] = useState(1);
  const [cols, setCols] = useState(4096);
  const [bx, setBx] = useState(16);
  const [by, setBy] = useState(16);

  const choice: LimitChoice =
    key === 'custom'
      ? {
          key: 'custom',
          label: tr('Custom numbers', 'Số tự nhập'),
          limits: {
            width: toInt(custom.width) || null, maxThreadsPerBlock: toInt(custom.perBlock) || null, maxThreadsDim: CUDA_BLOCK_DIMS,
            maxGridSize: CUDA_GRID_DIMS, threadSlots: toInt(custom.threadSlots) || null, blockSlots: toInt(custom.blockSlots) || null, blockSlotsDerived: false,
          },
          note: tr("Limits you typed, with CUDA's documented block and grid dimensions. A field left at 0 is treated as not reported.", 'Các giới hạn bạn tự nhập, cùng các chiều khối và lưới mà CUDA công bố. Ô nào để 0 được coi là không báo cáo.'),
        }
      : (choices.find((c) => c.key === key) ?? BOOK_CHOICE);
  const limits = choice.limits;
  const c = checkLaunch(limits, rows, cols, { x: bx, y: by });
  const s = useMemo(() => suggest(limits, rows, cols), [limits, rows, cols]);
  const occ = occupancy(occupancyLimits(limits), { threads: c.threadsPerBlock, regs: 0, shared: 0 });
  const maxX = Math.min(1024, limits.maxThreadsDim[0] ?? 1024);
  const maxY = Math.min(1024, limits.maxThreadsDim[1] ?? 1024);
  const [asked, setAsked] = useState(false);

  const numField = (label: string, value: number, set: (v: number) => void, max: number) => (
    <label className="field">
      {label}
      <input type="number" min="1" max={max} step="1" value={value} aria-label={label} onChange={(e) => { set(Math.max(1, Math.min(max, toInt(e.target.value) || 1))); setAsked(false); }} />
    </label>
  );
  const slider = (label: string, value: number, set: (v: number) => void, max: number) => (
    <label className="field" style={{ flex: '1 1 14rem' }}>
      {label} = {value}
      <input type="range" min={1} max={max} step={1} value={Math.min(value, max)} aria-label={label} style={{ width: '100%' }}
        onChange={(e) => { set(+e.target.value); setAsked(false); }} />
    </label>
  );
  const text = (label: string, value: string, set: (v: string) => void) => (
    <label className="field">
      {label}
      <input type="number" min="0" step="1" value={value} aria-label={label} onChange={(e) => set(e.target.value)} />
    </label>
  );

  // The launched threads as a box, the working ones as its filled part: each axis is scaled on its own.
  const BW = 300;
  const BH = 120;
  const fx = cols / (c.grid.x * c.block.x);
  const fy = rows / (c.grid.y * c.block.y);

  return (
    <Bench
      title={tr('Derive a launch configuration', 'Suy ra một cấu hình khởi chạy')}
      task={tr('One thread per output, x over the outputs, y over the batch. Make every rule hold at batch 1, then at batch 64, then on another device.', 'Mỗi đầu ra một luồng, x theo các đầu ra, y theo batch. Làm cho mọi quy tắc đều thỏa ở batch 1, rồi ở batch 64, rồi trên một thiết bị khác.')}
      foot={
        <Translated vi={<>{choice.note} Các quy tắc là những quy tắc mà bài kiểm thử của lab 4.1 in ra theo tên. Chúng là mặc định đầu tiên của khóa học này cho một kernel như thế này, không phải khuyến nghị của nhà cung cấp.</>}>
          <>{choice.note} The rules are the ones lab 4.1's tests print by name. They are this course's first default for a kernel like this one, not a vendor's recommendation.</>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-start' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Giới hạn">Limits</Translated>
          <select value={key} onChange={(e) => { setKey(e.target.value); setAsked(false); }} aria-label={tr('Launch limits', 'Giới hạn khởi chạy')}>
            {choices.map((d) => (
              <option key={d.key} value={d.key}>
                {d.label}
              </option>
            ))}
            <option value="custom">{tr('Custom numbers', 'Số tự nhập')}</option>
          </select>
        </label>
        {key === 'custom' ? (
          <>
            {text('warpSize', custom.width, (v) => setCustom({ ...custom, width: v }))}
            {text('maxThreadsPerBlock', custom.perBlock, (v) => setCustom({ ...custom, perBlock: v }))}
            {text(tr('Thread slots per SM', 'Số khe luồng mỗi SM'), custom.threadSlots, (v) => setCustom({ ...custom, threadSlots: v }))}
            {text(tr('Block slots per SM', 'Số khe khối mỗi SM'), custom.blockSlots, (v) => setCustom({ ...custom, blockSlots: v }))}
          </>
        ) : null}
        {numField(tr('Rows (batch)', 'Số hàng (batch)'), rows, setRows, 65536)}
        {numField(tr('Columns (outputs)', 'Số cột (đầu ra)'), cols, setCols, 1048576)}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }} role="group" aria-label={tr('Problems lab 4.1 tests', 'Các bài toán lab 4.1 kiểm thử')}>
        <span className="muted"><Translated vi="Các kích thước của lab 4.1">Lab 4.1's shapes</Translated></span>
        {LAB_PROBLEMS.map((p) => (
          <button key={p.rows + 'x' + p.cols} className="quiet" data-problem={`${p.rows}x${p.cols}`} aria-pressed={rows === p.rows && cols === p.cols}
            onClick={() => { setRows(p.rows); setCols(p.cols); setAsked(false); }}>
            {count(p.rows)} × {count(p.cols)}
          </button>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '1rem', alignItems: 'flex-end' }}>
        {slider(tr('Block x', 'Khối x'), bx, setBx, maxX)}
        {slider(tr('Block y', 'Khối y'), by, setBy, maxY)}
        {numField(tr('x, exactly', 'x, chính xác'), bx, setBx, maxX)}
        {numField(tr('y, exactly', 'y, chính xác'), by, setBy, maxY)}
        <button className="primary" data-act="suggest" onClick={() => { setAsked(true); if (s) { setBx(s.x); setBy(s.y); } }}>
          <Translated vi="Gợi ý một cấu hình">Suggest one</Translated>
        </button>
      </div>
      {asked ? (
        <p data-out="suggestion" style={{ margin: '-0.3rem 0 1rem' }}>
          {s ? (
            <Translated
              vi={
                <>
                  Đặt thành {s.x} × {s.y}: các hàng gồm trọn warp, cao bằng mức các giới hạn và batch cho phép, và không rộng hơn mức một
                  quy tắc đòi hỏi. Đó là một cấu hình vượt qua, không phải cấu hình duy nhất.
                </>
              }
            >
              <>
                Set to {s.x} × {s.y}: rows of whole warps, as tall as the limits and the batch allow, and no wider than a
                rule demands. It is one configuration that passes, not the only one.
              </>
            </Translated>
          ) : limits.width == null ? (
            <Translated vi={<>Không có gợi ý: các giới hạn này không có độ rộng warp để dựng hàng.</>}>
              <>No suggestion: these limits have no warp width to build rows from.</>
            </Translated>
          ) : (
            <Translated vi={<>Không khối nào trong các giới hạn này thỏa mọi quy tắc cho bài toán này.</>}>
              <>No block within these limits satisfies every rule for this problem.</>
            </Translated>
          )}
        </p>
      ) : null}

      <div className="cols plot">
        <div className="panel">
          <table className="data" style={{ margin: 0 }} data-out="rules" data-ok={c.ok ? 'yes' : 'no'} data-failed={c.failed.join(',')} data-skipped={c.skipped.join(',')}>
            <tbody>
              {c.rules.map((r) => (
                <tr key={r.key} data-rule={r.key} data-status={r.status}>
                  <td style={{ width: '5.2rem' }}><span className={CHIP[r.status].cls}>{CHIP[r.status].text}</span></td>
                  <td style={{ width: '38%' }}>{tr(r.name, `${({ 'per-block': 'không vượt maxThreadsPerBlock', dims: 'không vượt maxThreadsDim', warps: 'số warp nguyên', divides: 'chia hết số khe luồng', 'block-slots': 'khe khối không cạn trước', rows: 'không cao hơn số hàng', grid: 'không vượt maxGridSize' } as Record<string, string>)[r.key] ?? r.name} (${r.name})`)}</td>
                  <td className="muted">{r.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel" aria-live="polite">
          <p style={{ margin: '0 0 0.5rem' }} data-out="launch-verdict">
            <b>
              {c.ok
                ? c.skipped.length
                  ? tr(`No rule broken; ${plural(c.skipped.length, 'rule')} could not be tested.`, `Không quy tắc nào bị vi phạm; ${count(c.skipped.length)} quy tắc không kiểm tra được.`)
                  : tr('Every rule holds.', 'Mọi quy tắc đều thỏa.')
                : tr(`${plural(c.failed.length, 'rule')} broken.`, `${count(c.failed.length)} quy tắc bị vi phạm.`)}
            </b>
          </p>
          <Translated
            vi={
              <p style={{ margin: '0 0 0.5rem' }}>
                Khối {c.block.x} × {c.block.y} = <span data-out="threads-per-block">{count(c.threadsPerBlock)}</span> luồng. Lưới{' '}
                <span data-out="grid">{count(c.grid.x)} × {count(c.grid.y)}</span> = {count(c.blocks)} khối.
              </p>
            }
          >
            <p style={{ margin: '0 0 0.5rem' }}>
              Block {c.block.x} × {c.block.y} = <span data-out="threads-per-block">{count(c.threadsPerBlock)}</span> threads. Grid{' '}
              <span data-out="grid">{count(c.grid.x)} × {count(c.grid.y)}</span> = {plural(c.blocks, 'block')}.
            </p>
          </Translated>
          <svg viewBox={`0 0 ${BW + 2} ${BH + 2}`} role="img" aria-label={tr('Launched threads and the part of them that owns an output', 'Các luồng được khởi chạy và phần trong số đó sở hữu một đầu ra')} style={{ width: '100%', maxWidth: BW, display: 'block', margin: '0 0 0.5rem' }}>
            <rect x={1} y={1} width={BW} height={BH} fill="var(--rule-soft)" stroke="var(--ink-3)" strokeDasharray="3 3" />
            <rect x={1} y={1} width={Math.max(1.5, BW * fx)} height={Math.max(1.5, BH * fy)} fill="var(--ink)" />
          </svg>
          <Translated
            vi={
              <p style={{ margin: '0 0 0.5rem' }}>
                <span data-out="launched">{count(c.launched)}</span> luồng được khởi chạy, <span data-out="working">{count(c.working)}</span> sở hữu một
                đầu ra, <b><span data-out="surplus">{count(c.surplus)}</span> dư thừa</b>
                {c.launched > 0 ? ` (${pct(c.surplus / c.launched)}%)` : ''}. Hộp là mọi luồng được khởi chạy; phần tô đặc là những luồng có
                việc.
              </p>
            }
          >
            <p style={{ margin: '0 0 0.5rem' }}>
              <span data-out="launched">{count(c.launched)}</span> threads launched, <span data-out="working">{count(c.working)}</span> own an
              output, <b><span data-out="surplus">{count(c.surplus)}</span> surplus</b>
              {c.launched > 0 ? ` (${pct(c.surplus / c.launched)}%)` : ''}. The box is every launched thread; the filled part is the ones with
              work.
            </p>
          </Translated>
          <p className="muted" style={{ margin: 0 }} data-out="launch-occupancy" data-value={occ.usable ? pct(occ.byWarps) : 'n/a'} /* i18n-ignore: data attribute */>
            {occ.usable
              ? tr(
                  `Occupancy by slots alone at ${count(c.threadsPerBlock)} threads per block: ${pct(occ.byWarps)}%. Registers can lower it; no device field says how many a kernel uses.`,
                  `Occupancy chỉ theo khe ở ${count(c.threadsPerBlock)} luồng mỗi khối: ${pct(occ.byWarps)}%. Thanh ghi có thể làm nó thấp hơn; không trường thiết bị nào cho biết một kernel dùng bao nhiêu thanh ghi.`,
                )
              : tr('No occupancy figure: these limits have no thread slots.', 'Không có con số occupancy: các giới hạn này không có khe luồng.')}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- code shown
function DeviceLoop() {
  const code = useLabFile('ch04/snippets/device_loop.cu');
  return (
    <Code
      tabs={[{ label: tr('CUDA host', 'Host CUDA'), code: code ?? tr('// loading…', '// đang tải…'), status: tr('front-end-checked; unverified on CUDA hardware', 'front-end-checked; unverified trên phần cứng CUDA') }]}
      caption={
        <Translated
          vi={
            <>
              Vòng lặp ở p. 88 với mã trả về được kiểm tra, chỉ số kiểu <code>int</code> và thêm các trường theo SM. Nạp
              từ <code>labs/ch04/snippets/device_loop.cu</code>.
            </>
          }
        >
          <>
            The loop of p. 88 with return codes checked, an <code>int</code> index and the per-SM fields added. Loaded
            from <code>labs/ch04/snippets/device_loop.cu</code>.
          </>
        </Translated>
      }
    />
  );
}

/** The part of lab 4.1's Metal host file that asks the device and the pipeline. */
export function metalQueryExcerpt(file: string): string {
  const from = file.indexOf('// ---------- part 1: ask');
  const to = file.indexOf('// ---------- part 2');
  return from >= 0 && to > from ? file.slice(from, to).trimEnd() : file;
}

function MetalQuery() {
  const file = useLabFile(LAB1 + '/solution/launch.mm');
  const { reports } = useReports();
  if (!file) return null;
  return (
    <Code
      tabs={[{ label: tr('Metal host', 'Host Metal'), code: metalQueryExcerpt(file), status: solutionTabStatus(reports, LAB1, 'launch.mm') }]}
      caption={
        <Translated
          vi={
            <>
              Từ lời giải tham chiếu của lab 4.1, <code>labs/{LAB1}/solution/launch.mm</code>. Trạng thái cạnh thẻ được
              đọc từ lần chạy lab gần nhất của máy này.
            </>
          }
        >
          <>
            From lab 4.1's reference solution, <code>labs/{LAB1}/solution/launch.mm</code>. The status beside the tab is
            read from this machine's last lab run.
          </>
        </Translated>
      }
    />
  );
}

/** Apple's recipe on the values lab 4.1 read from the reader's own pipeline. Shows nothing it was not given. */
function MetalRecipe() {
  const { reports } = useReports();
  const q = queriedRow(reports, 'metal');
  const w = q ? num(q.row.threadExecutionWidth) : undefined;
  const most = q ? num(q.row.maxTotalThreadsPerThreadgroup) : undefined;
  const shape = w != null && most != null ? appleRecipe(w, most) : null;
  return (
    <p data-out="metal-recipe" data-shape={shape ? `${shape.x}x${shape.y}` : 'none'}>
      {shape && q ? (
        <Translated
          vi={
            <>
              Trên máy này, benchmark của lab 4.1 ({q.own ? 'kernel của bạn' : 'lời giải tham chiếu'}) đọc được độ rộng {w} và
              giới hạn {count(most as number)} từ pipeline của nó, nên công thức cho ra một threadgroup {shape.x} × {shape.y}. Hai
              con số đó thuộc về kernel ấy. Một kernel khác có thể báo cáo con số khác.
            </>
          }
        >
          <>
            On this machine, lab 4.1's benchmark ({q.own ? 'your kernel' : 'the reference solution'}) read a width of {w} and a
            limit of {count(most as number)} from its pipeline, so the recipe gives a threadgroup of {shape.x} × {shape.y}. Those
            two numbers belong to that kernel. Another kernel may report others.
          </>
        </Translated>
      ) : (
        <Translated
          vi={
            <>
              Khi benchmark của lab 4.1 đã chạy trên một thiết bị Metal, hai con số mà pipeline của nó báo cáo, và threadgroup
              mà công thức tạo ra từ chúng, sẽ hiện ở đây.
            </>
          }
        >
          <>
            Once lab 4.1's benchmark has run on a Metal device, the two numbers its pipeline reported, and the threadgroup
            the recipe makes of them, appear here.
          </>
        </Translated>
      )}
    </p>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS48: Record<string, Question> = {
  'ch04/4.8/q-ex9': {
    kind: 'choice',
    id: 'ch04/4.8/q-ex9',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 9 của sách. Một sinh viên nói họ đã nhân hai ma trận 1024 × 1024 với các khối luồng 32 × 32, mỗi phần tử
            đầu ra một luồng, trên một thiết bị cho phép tối đa 512 luồng mỗi khối và tối đa 8 khối mỗi SM. Bạn phản ứng
            thế nào?
          </>
        }
      >
        <>
          Book exercise 9. A student says they multiplied two 1024 × 1024 matrices with 32 × 32 thread blocks, one thread
          per output element, on a device that allows up to 512 threads per block and up to 8 blocks per SM. How do you
          react?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Ổn: 1024 / 32 = 32 khối theo mỗi trục, một lưới 32 × 32 gồm 1024 khối.">Fine: 1024 / 32 = 32 blocks along each axis, a 32 × 32 grid of 1024 blocks.</Translated>,
        why: <Translated vi="Phép tính lưới thì đúng. Vấn đề nằm ở khối: hãy đếm số luồng của nó.">The grid arithmetic is right. The block is the problem: count its threads.</Translated>,
      },
      {
        label: (
          <Translated vi="Nó không thể đã chạy như mô tả: một khối 32 × 32 là 1024 luồng, vượt mức 512 của thiết bị.">
            It cannot have run as described: a 32 × 32 block is 1024 threads, over the device's 512.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Nó chạy, nhưng chậm: chỉ 8 khối vừa trên một SM, nên 1024 khối phải chạy nhiều đợt.">It runs, but slowly: only 8 blocks fit on an SM, so the 1024 blocks take many waves.</Translated>,
        why: <Translated vi="Tám khối mỗi SM không cản trở lưới có kích thước bất kỳ. Kích thước khối mới là thứ thiết bị từ chối.">Eight blocks per SM is no obstacle to a grid of any size. The block size is what the device refuses.</Translated>,
      },
      {
        label: <Translated vi="Nó chạy: runtime tách mỗi khối 1024 luồng thành hai khối 512 cho vừa giới hạn.">It runs: the runtime splits each 1024-thread block into two blocks of 512 to fit the limit.</Translated>,
        why: <Translated vi="Thiết bị không bao giờ tách một khối. Một khối vượt giới hạn khiến lần khởi chạy thất bại.">A device never splits a block. A block that exceeds the limit makes the launch fail.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>maxThreadsPerBlock</code> là giới hạn trên tích các chiều của khối. 32 × 32 = 1024 vượt quá 512,
            nên lần khởi chạy bị từ chối và không luồng nào chạy. Nếu sinh viên thấy có kết quả, thì hoặc hình dạng khối hoặc thiết bị
            không như họ nghĩ, hoặc không ai kiểm tra lỗi của lần khởi chạy và buffer kết quả còn giữ giá trị cũ.
            Truy vấn giới hạn trước, và kiểm tra lần khởi chạy sau đó, lẽ ra đã phát hiện điều này.
          </p>
        }
      >
        <p>
          <code>maxThreadsPerBlock</code> is a limit on the product of the block's dimensions. 32 × 32 = 1024 is over 512,
          so the launch is rejected and no thread runs. If the student saw output, either the block shape or the device
          was not what they think, or nobody checked the launch for an error and the result buffer held old values.
          Querying the limit first, and checking the launch afterwards, would have caught it.
        </p>
      </Translated>
    ),
    source: 'ch. 4 exercise 9, p. 92; the answer is ours (the book prints none)',
    vi: { source: 'ch. 4, bài tập 9, p. 92; đáp án là của chúng tôi (sách không in đáp án)' },
  },
  'ch04/4.8/q-regsfield': {
    kind: 'choice',
    id: 'ch04/4.8/q-regsfield',
    prompt: (
      <Translated
        vi={
          <>
            Bạn muốn biết số thanh ghi mỗi luồng vẫn cho phép occupancy đầy đủ trên một thiết bị CUDA. Bạn chia trường nào
            cho <code>maxThreadsPerMultiProcessor</code>?
          </>
        }
      >
        <>
          You want the number of registers per thread that still allows full occupancy on a CUDA device. Which field do
          you divide by <code>maxThreadsPerMultiProcessor</code>?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi={<><code>regsPerBlock</code>, như danh sách trường của sách nói.</>}>
            <><code>regsPerBlock</code>, as the book's field list says.</>
          </Translated>
        ),
        why: <Translated vi="Đó là số tối đa một khối được giữ. Sách mô tả tổng của SM dưới tên này.">That is the most one block may hold. The book describes the SM total under this name.</Translated>,
      },
      { label: <><code>regsPerMultiprocessor</code></>, correct: true },
      { label: <><code>maxBlocksPerMultiProcessor</code></>, why: <Translated vi="Đó là khe khối. Nó không nói gì về thanh ghi.">That is block slots. It says nothing about registers.</Translated> },
      { label: <><code>sharedMemPerMultiprocessor</code></>, why: <Translated vi="Bộ nhớ chia sẻ là một vùng riêng, tính bằng byte.">Shared memory is a separate pool, counted in bytes.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Vùng chung được chia cho các warp thường trú là của SM, và trường dành cho nó là{' '}
            <code>regsPerMultiprocessor</code>. <code>regsPerBlock</code> là mức trần cho một khối đơn lẻ. Trên mọi compute
            capability mà NVIDIA liệt kê hiện nay, hai trường mang cùng giá trị, 64 K, nên mã đọc nhầm trường vẫn cho
            đáp án đúng lúc này. Hãy đọc trường mang đúng ý nghĩa bạn cần.
          </p>
        }
      >
        <p>
          The pool that is shared out among resident warps is the SM's, and the field for it is{' '}
          <code>regsPerMultiprocessor</code>. <code>regsPerBlock</code> is a cap on a single block. On every compute
          capability NVIDIA lists today the two hold the same value, 64 K, so code that reads the wrong one gives the
          right answer for now. Read the one that means what you want.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.8, p. 89; corrected against the CUDA 13.5 reference, 2026-11-20 (ch04-facts.md, A18)',
    vi: { source: 'ch. 4 §4.8, p. 89; đã sửa theo tài liệu tham khảo CUDA 13.5, 2026-11-20 (ch04-facts.md, A18)' },
  },
  'ch04/4.8/q-metalwidth': {
    kind: 'choice',
    id: 'ch04/4.8/q-metalwidth',
    prompt: (
      <Translated
        vi={
          <>
            <code>python3 labs/pmpp.py doctor</code> in ra một báo cáo thiết bị cho GPU của máy Mac của bạn. Vì sao báo cáo đó không thể
            chứa độ rộng của một SIMD group?
          </>
        }
      >
        <>
          <code>python3 labs/pmpp.py doctor</code> prints a device report for your Mac's GPU. Why can that report not
          contain the width of a SIMD-group?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Apple coi độ rộng SIMD group là thông tin riêng, nên không API Metal công khai nào trả nó về cho chương trình.">Apple treats the SIMD-group width as private, so no public Metal API returns it to a program.</Translated>,
        why: <Translated vi="Có một API trả về nó. Hãy xem bạn hỏi loại đối tượng nào.">An API does return it. Look at what kind of object you ask.</Translated>,
      },
      {
        label: (
          <Translated vi="Metal cung cấp độ rộng trên một pipeline đã biên dịch, còn báo cáo chạy trước khi bất kỳ kernel nào được biên dịch.">
            Metal exposes the width on a compiled pipeline, and the report runs before any kernel is compiled.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Mọi GPU Apple đều có độ rộng SIMD group là 32, nên báo cáo không có gì riêng của thiết bị để thêm vào.">Every Apple GPU has a SIMD-group width of 32, so there is nothing device-specific for a report to add.</Translated>,
        why: (
          <Translated vi="Apple không công bố giá trị nào, và chính ví dụ định kích thước của hãng dùng 16. Cách được công bố là hỏi một pipeline đã biên dịch. Một số dự án giả định 32; đó là giả định của họ.">
            Apple documents no value, and its own sizing example uses 16. The documented way is to ask a compiled pipeline. Some projects assume 32; that is their assumption.
          </Translated>
        ),
      },
      {
        label: <Translated vi="doctor đọc mô tả GPU từ hệ điều hành, vốn không ghi lại độ rộng.">doctor reads the GPU's description from the operating system, which does not record the width.</Translated>,
        why: <Translated vi="Nó tạo thiết bị Metal và đọc các thuộc tính của thiết bị. Độ rộng không nằm trong số đó.">It creates the Metal device and reads its properties. The width is not among them.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trên CUDA, độ rộng là một trường của thiết bị. Trên Metal nó là <code>threadExecutionWidth</code>, một thuộc tính của{' '}
            <code>MTLComputePipelineState</code>, thứ chỉ tồn tại sau khi một kernel đã được biên dịch cho thiết bị, và
            kernel có thể đọc cùng con số đó qua <code>[[threads_per_simdgroup]]</code>. Báo cáo thiết bị được ghi trước khi có bất kỳ
            kernel nào. Lab 4.1 biên dịch một kernel và đọc nó.
          </p>
        }
      >
        <p>
          On CUDA the width is a field of the device. On Metal it is <code>threadExecutionWidth</code>, a property of an{' '}
          <code>MTLComputePipelineState</code>, which exists only after a kernel has been compiled for the device, and the
          kernel can read the same number as <code>[[threads_per_simdgroup]]</code>. A device report is written before any
          kernel exists. Lab 4.1 compiles one and reads it.
        </p>
      </Translated>
    ),
    source: 'Metal headers and Apple documentation, read 2026-11-20 (ch04-facts.md, C1, C2, C5); not in the book',
    vi: { source: 'Header của Metal và tài liệu của Apple, đọc ngày 2026-11-20 (ch04-facts.md, C1, C2, C5); không có trong sách' },
  },
  'ch04/4.8/q-cc': {
    kind: 'choice',
    id: 'ch04/4.8/q-cc',
    prompt: (
      <Translated
        vi={
          <>
            Thiết bị X báo cáo compute capability 12.0 và thiết bị Y báo cáo 9.0. Bạn kết luận được gì về số luồng
            mà một SM của mỗi thiết bị có thể chứa?
          </>
        }
      >
        <>
          Device X reports compute capability 12.0 and device Y reports 9.0. What can you conclude about how many threads
          an SM of each can hold?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="X chứa nhiều hơn: compute capability cao hơn nghĩa là mỗi SM có nhiều tài nguyên hơn.">X holds more: a higher compute capability means more resources per SM.</Translated>,
        why: (
          <Translated vi="Sách nói “nói chung”. Trong các bảng hiện nay, với cặp này thì ngược lại.">
            {'The book says "generally". In today\'s tables it is the other way round for this pair.'}
          </Translated>
        ),
      },
      {
        label: <Translated vi="Chúng chứa như nhau: số khe luồng mỗi SM không đổi kể từ 8.0.">They hold the same: thread slots per SM have not changed since 8.0.</Translated>,
        why: (
          <Translated vi="Chúng khác nhau giữa các hàng trong bảng của NVIDIA: 1024, 1536 hoặc 2048.">
            {'They differ between rows of NVIDIA\'s table: 1024, 1536 or 2048.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không kết luận được gì nếu không truy vấn: con số này xếp thứ tự kiến trúc, không phải năng lực mỗi SM.">
            Nothing, without querying: the number orders architectures, not capacity per SM.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Y chứa nhiều hơn, vì compute capability thấp hơn là dòng dành cho trung tâm dữ liệu.">Y holds more, because lower compute capabilities are data-centre parts.</Translated>,
        why: <Translated vi="Trong các bảng hiện tại Y đúng là chứa nhiều hơn, nhưng không phải vì một lý do mà con số này mã hóa.">Y does hold more in the current tables, but not for a reason the number encodes.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Compute capability cho bạn biết thiết bị thuộc kiến trúc nào, và do đó có những tính năng nào. Sách nói thêm rằng
            mức cao hơn nói chung mang lại nhiều tài nguyên hơn cho mỗi SM, đó là xu hướng lúc sách được viết. Đó không phải
            quy tắc: trong các bảng hiện tại của NVIDIA, các hàng 12.x có ít khe luồng mỗi SM hơn 9.0. Hãy đọc{' '}
            <code>maxThreadsPerMultiProcessor</code> và các trường lân cận từ chính thiết bị bạn đang dùng.
          </p>
        }
      >
        <p>
          Compute capability tells you which architecture a device is, and so which features it has. The book adds that
          higher levels generally bring more resources per SM, which was the trend when it was written. It is not a
          rule: in NVIDIA's current tables the 12.x rows have fewer thread slots per SM than 9.0. Read{' '}
          <code>maxThreadsPerMultiProcessor</code> and its neighbours from the device you are on.
        </p>
      </Translated>
    ),
    source: "ch. 4 §4.8, p. 88; NVIDIA's compute-capability tables, read 2026-11-20 (ch04-facts.md, A12, A15)",
    vi: { source: 'ch. 4 §4.8, p. 88; các bảng compute capability của NVIDIA, đọc ngày 2026-11-20 (ch04-facts.md, A12, A15)' },
  },
  'ch04/4.8/q-zero': {
    kind: 'choice',
    id: 'ch04/4.8/q-zero',
    prompt: (
      <Translated
        vi={
          <>
            Một chương trình HIP in ra <code>maxThreadsPerMultiProcessor = 0</code>. Một hàm định kích thước lần khởi chạy dựa trên
            giới hạn của thiết bị nên xử lý điều đó thế nào?
          </>
        }
      >
        <>
          A HIP program prints <code>maxThreadsPerMultiProcessor = 0</code>. What should a routine that sizes launches from
          the device's limits do with that?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Coi thiết bị là không chạy được kernel và từ chối khởi chạy.">Treat the device as unable to run kernels and refuse to launch.</Translated>,
        why: <Translated vi="Thiết bị vẫn chạy kernel. Chỉ một trường trong mô tả của nó bị trống.">The device runs kernels. One field of its description is empty.</Translated>,
      },
      {
        label: (
          <Translated vi="Dùng như bình thường: runtime đã báo cáo nó, nên đó là giới hạn thật của thiết bị.">
            {'Use it as usual: the runtime reported it, so it is the device\'s real limit.'}
          </Translated>
        ),
        why: <Translated vi="Chia cho không, hay cho một giới hạn bằng không, không cho ra kích thước khối nào dùng được.">Dividing by zero, or by a limit of zero, gives no block size anyone wants.</Translated>,
      },
      {
        label: (
          <Translated vi="Coi trường đó là không được báo cáo, và định kích thước lần khởi chạy dựa trên các trường có sẵn.">
            Treat the field as not reported, and size the launch from the fields that are there.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Thay bằng 2048, con số của thiết bị ví dụ của sách, làm mặc định an toàn.">
            {'Substitute 2048, the book\'s example device, as a safe default.'}
          </Translated>
        ),
        why: (
          <Translated vi="Đó là con số của một thiết bị. Một giới hạn bịa ra còn tệ hơn không có.">
            {'That is one device\'s number. A made-up limit is worse than none.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trang tham khảo của AMD nói HIP trả về 0 cho trường này, cho <code>regsPerBlock</code> và cho kích thước bộ nhớ đệm L2.
            Mã nguồn runtime ở phiên bản chúng tôi đọc thì gán cả ba. Bạn nhận được giá trị nào tùy vào bản cài đặt, nên mã
            phải chấp nhận cả hai. Số không ở đây nghĩa là "không có thông tin", và cách xử lý trung thực là bỏ qua các quy tắc cần
            trường đó, không phải bịa ra một con số. Bài kiểm thử của lab 4.1 có cả những thiết bị được mô tả theo cách này.
          </p>
        }
      >
        <p>
          AMD's reference page says HIP returns 0 for this field, for <code>regsPerBlock</code> and for the L2 cache size.
          The runtime source at the release we read assigns all three. Which you get depends on the installation, so code
          has to accept both. Zero here means "no information", and the honest response is to skip the rules that need
          the field, not to invent a number. Lab 4.1's tests include devices described this way.
        </p>
      </Translated>
    ),
    source: 'HIP reference and runtime source, read 2026-11-20 (ch04-facts.md, B9); not in the book',
    vi: { source: 'Tài liệu tham khảo HIP và mã nguồn runtime, đọc ngày 2026-11-20 (ch04-facts.md, B9); không có trong sách' },
  },
};

// ---------------------------------------------------------------- the section
export function S48() {
  return (
    <>
      <Translated
        vi={
          <p>
            Mọi giới hạn mà chương này đã dùng đều thuộc về một thiết bị: số luồng mỗi khối, độ rộng của một warp, số khe và
            số thanh ghi của một SM. Các con số của sách là của thiết bị ví dụ của nó. Của bạn thì khác, và một chương trình bạn phát hành
            sẽ gặp những thiết bị mà cả bạn lẫn chúng tôi đều chưa thấy. Vì vậy chương trình phải hỏi.
          </p>
        }
      >
        <p>
          Every limit this chapter has used belongs to a device: threads per block, the width of a warp, the slots and
          registers of an SM. The book's numbers are its example device's. Yours are different, and a program you ship
          will meet devices neither of us has seen. So a program asks.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Đó là toàn bộ động cơ sách đưa ra, và thế là đủ: cùng một đoạn mã chạy trên nhiều hệ thống, và nó phải
            tìm hiểu mỗi hệ thống có gì để tận dụng tốt một hệ thống mạnh mà vẫn chạy được trên một hệ thống khiêm tốn.
          </p>
        }
      >
        <p>
          That is the whole motivation the book gives, and it is enough: the same code runs on many systems, and it has
          to find out what each offers in order to use a capable one well and still work on a modest one.
        </p>
      </Translated>
      <Source section="4.8" pages="87" />

      <h2><Translated vi="Compute capability">Compute capability</Translated></h2>
      <Translated
        vi={
          <p>
            NVIDIA tóm tắt những gì một SM có trong một số phiên bản, gọi là <b>compute capability</b> của thiết bị. Hai mốc
            tham chiếu của sách là 8.0 cho A100 và 9.0 cho H100, và sách nói mức cao hơn nói chung mang lại nhiều
            tài nguyên hơn cho mỗi SM.
          </p>
        }
      >
        <p>
          NVIDIA summarises what an SM offers in a version number, the device's <b>compute capability</b>. The book's
          two reference points are 8.0 for the A100 and 9.0 for the H100, and it says higher levels generally bring more
          resources per SM.
        </p>
      </Translated>
      <Source section="4.8" pages="88" />
      <Translated
        vi={
          <p>
            Hãy dùng con số này để tra một thiết bị, hoặc để kiểm tra một tính năng xuất hiện cùng một kiến trúc. Đừng dùng nó
            để xếp hạng thiết bị theo năng lực.
          </p>
        }
      >
        <p>
          Use the number to look a device up, or to test for a feature that arrived with an architecture. Do not use it
          to rank devices by capacity.
        </p>
      </Translated>
      <Today checked={tr("2026-11-20 against NVIDIA's compute-capability tables (ch04-facts.md, A12, A15)", '2026-11-20, đối chiếu với các bảng compute capability của NVIDIA (ch04-facts.md, A12, A15)')}>
        <Translated
          vi={
            <p>
              Cả hai giá trị đều được xác nhận, và có các giá trị mới hơn đến 12.x. Con số này xếp thứ tự kiến trúc, không phải năng lực:
              các hàng 12.x trong bảng của NVIDIA có ít khe luồng mỗi SM hơn 9.0.
            </p>
          }
        >
          <p>
            Both values are confirmed, and newer ones exist up to 12.x. The number orders architectures, not capacity:
            the 12.x rows of NVIDIA's table have fewer thread slots per SM than 9.0.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS48['ch04/4.8/q-cc']} />

      <h2><Translated vi="Đếm các thiết bị, rồi hỏi từng thiết bị">Count the devices, then ask each one</Translated></h2>
      <Translated
        vi={
          <p>
            Hai lời gọi làm việc này. <code>cudaGetDeviceCount</code> ghi số thiết bị CUDA vào một{' '}
            <code>int</code>. Các thiết bị được đánh số từ 0, và <code>cudaGetDeviceProperties</code> điền một struct{' '}
            <code>cudaDeviceProp</code> cho thiết bị bạn chỉ định. Mã host lặp qua chúng và quyết định thiết bị nào
            đủ tốt.
          </p>
        }
      >
        <p>
          Two calls do the work. <code>cudaGetDeviceCount</code> writes the number of CUDA devices into an{' '}
          <code>int</code>. Devices are numbered from 0, and <code>cudaGetDeviceProperties</code> fills a{' '}
          <code>cudaDeviceProp</code> struct for the one you name. The host code loops over them and decides which are
          good enough.
        </p>
      </Translated>
      <Source section="4.8" pages="88" />
      <DeviceLoop />
      <Translated vi={<p>Đoạn này khác trang sách ở ba điểm, không điểm nào là ý của sách:</p>}>
        <p>This differs from the page in three ways, none of them the book's idea:</p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>
              Cả hai lời gọi đều trả về mã lỗi, và trang sách bỏ qua cả hai. Hướng dẫn của NVIDIA nói phải kiểm tra từng cái. Ở đây, một
              số đếm không đọc được cho ra không thiết bị nào, và một thiết bị không đọc được thì bị bỏ qua.
            </li>
            <li>
              Vòng lặp in trong sách khai báo chỉ số là <code>unsigned int</code> và so sánh nó với số đếm kiểu <code>int</code>.{' '}
              Điều đó biên dịch được, kèm cảnh báo từ một số trình biên dịch. Phiên bản vòng lặp của chính NVIDIA dùng{' '}
              <code>int</code>, và bản này cũng vậy.
            </li>
            <li>
              Nó in thêm bốn trường theo SM mà danh sách của sách không có, vì mục 4.7 cần chúng.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            Both calls return an error code, and the page ignores both. NVIDIA's guide says to check every one. Here a
            count that cannot be read gives zero devices and a device that cannot be read is skipped.
          </li>
          <li>
            The printed loop declares its index <code>unsigned int</code> and compares it with the <code>int</code>{' '}
            count. That compiles, with a warning from some compilers. NVIDIA's own version of the loop uses{' '}
            <code>int</code>, and so does this one.
          </li>
          <li>
            It prints four per-SM fields that the book's list does not have, because section 4.7 needs them.
          </li>
        </ul>
      </Translated>
      <Today checked={tr('2026-11-20 against the CUDA 13.5 runtime reference and guide (ch04-facts.md, A16, A17)', '2026-11-20, đối chiếu với tài liệu tham khảo runtime và hướng dẫn CUDA 13.5 (ch04-facts.md, A16, A17)')}>
        <Translated
          vi={
            <>
              <p>
                Sách cho rằng một PC thông thường có từ hai thiết bị CUDA trở lên vì có GPU tích hợp. Chúng tôi không tìm thấy nguồn nào của
                NVIDIA cho điều đó, và số đếm chỉ tính thiết bị NVIDIA: unverified, nên không nhắc lại ở đây.
              </p>
              <p>Dù vậy vẫn cứ lặp. Có những máy có nhiều GPU.</p>
            </>
          }
        >
          <>
            <p>
              The book expects two or more CUDA devices in a typical PC because of integrated GPUs. We found no NVIDIA
              source for that, and the count covers NVIDIA devices only: unverified, so it is not repeated here.
            </p>
            <p>Loop anyway. Machines with several GPUs exist.</p>
          </>
        </Translated>
      </Today>

      <h2><Translated vi="Các trường, và mỗi trường quyết định điều gì">The fields, and what each decides</Translated></h2>
      <Translated vi={<p>Sách chọn ra bảy trường. Mỗi trường trả lời một câu hỏi bạn đã phải đặt ra trong chương này:</p>}>
        <p>The book picks out seven fields. Each answers a question you have already had to ask in this chapter:</p>
      </Translated>
      <Source section="4.8" pages="89" />
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-book-fields>
          <thead>
            <tr>
              <th style={{ width: '30%' }}><Translated vi="Trường">Field</Translated></th>
              <th><Translated vi="Nó quyết định điều gì">What it decides</Translated></th>
            </tr>
          </thead>
          <tbody>
            <tr><td><code>maxThreadsPerBlock</code></td><td><Translated vi="Một kích thước khối có hợp lệ hay không. Một số thiết bị cho phép 1024, số khác có thể cho phép ít hơn.">Whether a block size is legal at all. Some devices allow 1024, others may allow fewer.</Translated></td></tr>
            <tr><td><code>maxThreadsDim[0..2]</code></td><td><Translated vi="Khối lớn nhất theo x, y và z: cận cho bất kỳ chương trình nào thử nhiều hình dạng khối.">The largest block along x, y and z: the bounds for any program that tries several block shapes.</Translated></td></tr>
            <tr><td><code>maxGridSize[0..2]</code></td><td><Translated vi="Lưới lớn nhất theo mỗi trục: một lần khởi chạy có phủ hết dữ liệu được không hay phải chia nhỏ công việc.">The largest grid along each axis: whether one launch can cover the data or the work has to be split.</Translated></td></tr>
            <tr><td><code>multiProcessorCount</code></td><td><Translated vi="Số SM. Cùng với số khối mỗi SM, nó cho biết số khối chạy cùng lúc.">The number of SMs. With blocks per SM it gives the blocks that run at once.</Translated></td></tr>
            <tr><td><code>warpSize</code></td><td><Translated vi="Độ rộng của một warp, để định kích thước khối theo trọn warp.">The width of a warp, for sizing blocks in whole warps.</Translated></td></tr>
            <tr><td><code>regsPerBlock</code></td><td><Translated vi="Theo sách: số thanh ghi của mỗi SM, để đánh giá thanh ghi có giới hạn occupancy hay không.">The book: the registers of each SM, for judging whether registers will limit occupancy.</Translated></td></tr>
            <tr><td><code>clockRate</code></td><td><Translated vi="Theo sách: tần số xung nhịp, cùng với số SM gợi ý mức tối đa phần cứng có thể đạt.">The book: the clock frequency, which with the SM count hints at the most the hardware can do.</Translated></td></tr>
          </tbody>
        </table>
      </div>
      <Translated
        vi={
          <p>
            Hai mục trong danh sách đó không đứng vững trước bộ công cụ hiện tại. Trước khi đọc đó là mục nào, hãy tự tìm ra. Đoạn
            mã là những gì bạn sẽ gõ theo pp. 88 và 89.
          </p>
        }
      >
        <p>
          Two entries of that list do not survive contact with a current toolkit. Before reading which, find them. The
          listing is what you would type from pp. 88 and 89.
        </p>
      </Translated>
      <FieldHunt />
      <Translated
        vi={
          <p>
            Vậy: đọc xung nhịp, nếu bạn thực sự cần, bằng <code>cudaDeviceGetAttribute</code>. Lấy số thanh ghi mỗi SM
            từ <code>regsPerMultiprocessor</code>. Và thêm các trường sách không liệt kê nhưng occupancy cần:{' '}
            <code>maxThreadsPerMultiProcessor</code>, <code>maxBlocksPerMultiProcessor</code>,{' '}
            <code>regsPerMultiprocessor</code> và <code>sharedMemPerMultiprocessor</code>.
          </p>
        }
      >
        <p>
          So: read the clock rate, if you want it at all, with <code>cudaDeviceGetAttribute</code>. Take registers per SM
          from <code>regsPerMultiprocessor</code>. And add the fields the book does not list but occupancy needs:{' '}
          <code>maxThreadsPerMultiProcessor</code>, <code>maxBlocksPerMultiProcessor</code>,{' '}
          <code>regsPerMultiprocessor</code> and <code>sharedMemPerMultiprocessor</code>.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 against the CUDA 13.5 reference and the CUDA 13.0 release notes (ch04-facts.md, A18)', '2026-11-20, đối chiếu với tài liệu tham khảo CUDA 13.5 và ghi chú phát hành CUDA 13.0 (ch04-facts.md, A18)')}>
        <Translated
          vi={
            <>
              <p>
                <code>clockRate</code> đã bị gỡ khỏi struct trong CUDA 13.0. <code>regsPerBlock</code> được ghi là
                số thanh ghi khả dụng cho mỗi khối; <code>regsPerMultiprocessor</code> là vùng thanh ghi của SM.
              </p>
              <p>
                Sách nói con số theo khối thấp hơn tổng của SM ở một số compute capability. Trong các bảng hiện nay, cả hai
                đều là 64 K ở mọi nơi.
              </p>
            </>
          }
        >
          <>
            <p>
              <code>clockRate</code> was removed from the struct in CUDA 13.0. <code>regsPerBlock</code> is documented as
              the registers available per block; <code>regsPerMultiprocessor</code> is the SM's pool.
            </p>
            <p>
              The book says the block figure is below the SM total on some compute capabilities. In today's tables both
              are 64 K everywhere.
            </p>
          </>
        </Translated>
      </Today>
      <Ask q={questionsS48['ch04/4.8/q-regsfield']} />
      <Translated
        vi={
          <p>
            Sách nêu thêm một lời gọi, không phải một trường: chương trình dùng thread block cluster có thể hỏi{' '}
            <code>cudaOccupancyMaxPotentialClusterSize</code> về cluster lớn nhất một kernel có thể có. Cluster đã được
            giới thiệu ở <a href={href('ch04', '4.2')}>mục 4.2</a> và quay lại ở chương 5.
          </p>
        }
      >
        <p>
          The book names one more call, not a field: programs that use thread block clusters can ask{' '}
          <code>cudaOccupancyMaxPotentialClusterSize</code> for the largest cluster a kernel can have. Clusters were
          introduced in <a href={href('ch04', '4.2')}>section 4.2</a> and come back in chapter 5.
        </p>
      </Translated>
      <Source section="4.8" pages="89" />

      <h2><Translated vi="Các thiết bị của bạn nói gì">What your own devices say</Translated></h2>
      <Translated
        vi={
          <p>
            <code>python3 labs/pmpp.py doctor</code> hỏi mọi backend trên máy này và in ra các câu trả lời, còn{' '}
            <code>./verify.sh</code> lưu chúng ở nơi trang này đọc được.
            Bảng đặt chúng cạnh công dụng của từng trường. Hãy xem nó một lần cho mỗi backend bạn có, và một lần cho ví dụ
            của sách, rồi để ý những hàng nào bị trống.
          </p>
        }
      >
        <p>
          <code>python3 labs/pmpp.py doctor</code> asks every backend on this machine and prints the answers, and{' '}
          <code>./verify.sh</code> saves them where this page can read them.
          The table sets them beside what each is for. Look at it once per backend you have, and once for the book's
          example, and note which rows go empty.
        </p>
      </Translated>
      <FieldTable />

      <h2><Translated vi="Cùng truy vấn đó qua HIP">The same query through HIP</Translated></h2>
      <Translated
        vi={
          <p>
            HIP giữ nguyên hình dạng của tất cả những điều này: <code>hipGetDeviceCount</code>, <code>hipGetDeviceProperties</code>, một{' '}
            <code>hipDeviceProp_t</code> với cùng tên thành viên. Tệp nguồn <code>.cu</code> duy nhất của khóa học đến được
            chúng qua một header đổi tên. Dù vậy vẫn có ba khác biệt chạm tới mã của bạn.
          </p>
        }
      >
        <p>
          HIP keeps the shape of all of this: <code>hipGetDeviceCount</code>, <code>hipGetDeviceProperties</code>, a{' '}
          <code>hipDeviceProp_t</code> with the same member names. The course's single <code>.cu</code> source reaches
          them through a header of renames. Three differences reach your code all the same.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ đề cập CUDA. Phần này lấy từ tài liệu tham khảo HIP của AMD và mã nguồn runtime ở một phiên bản, đọc ngày
              2026-11-20 (ch04-facts.md, B9). Không phần nào được chạy trên phần cứng AMD.
            </p>
          }
        >
          <p>
            The book covers CUDA only. This is from AMD's HIP reference and the runtime source at one release, read on
            2026-11-20 (ch04-facts.md, B9). None of it was run on AMD hardware.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <ul>
            <li>
              <code>clockRate</code> vẫn là thành viên của struct HIP. Một tệp nguồn dùng chung đọc{' '}
              <code>prop.clockRate</code> sẽ build được cho HIP và thất bại với CUDA.
            </li>
            <li>
              Một số trường được tính ra, không phải tra cứu. <code>maxBlocksPerMultiProcessor</code> là số khe luồng chia
              cho độ rộng. <code>regsPerBlock</code> và <code>regsPerMultiprocessor</code> được điền cùng một
              giá trị, và nó chỉ đếm thanh ghi vector. <code>multiProcessorCount</code> đếm số compute unit.
            </li>
            <li>
              Trang tham khảo của AMD nói ba thuộc tính, trong đó có <code>maxThreadsPerMultiProcessor</code> và{' '}
              <code>regsPerBlock</code>, luôn trả về 0. Mã nguồn chúng tôi đọc thì có gán chúng. Mã phải chạy được
              với cả hai trường hợp.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            <code>clockRate</code> is still a member of the HIP struct. A shared source file that reads{' '}
            <code>prop.clockRate</code> builds for HIP and fails for CUDA.
          </li>
          <li>
            Several fields are computed, not looked up. <code>maxBlocksPerMultiProcessor</code> is thread slots divided
            by the width. <code>regsPerBlock</code> and <code>regsPerMultiprocessor</code> are filled with the same
            value, and it counts vector registers only. <code>multiProcessorCount</code> counts compute units.
          </li>
          <li>
            AMD's reference page says three properties, <code>maxThreadsPerMultiProcessor</code> and{' '}
            <code>regsPerBlock</code> among them, always return 0. The source we read assigns them. Code has to work
            with either.
          </li>
        </ul>
      </Translated>
      <Ask q={questionsS48['ch04/4.8/q-zero']} />

      <h2><Translated vi="Metal: thiết bị biết ít hơn, pipeline biết nhiều hơn">Metal: the device knows less, the pipeline knows more</Translated></h2>
      <Translated
        vi={
          <p>
            Metal cũng có vòng lặp này. <code>MTLCreateSystemDefaultDevice()</code> trả về thiết bị mà hệ thống sẽ chọn,
            còn <code>MTLCopyAllDevices()</code> trả về tất cả. Sau đó một thiết bị cho bạn biết tên của nó, threadgroup lớn nhất
            nó hỗ trợ cho kernel đơn giản nhất có thể, một kernel được khai báo bao nhiêu bộ nhớ threadgroup,
            và bạn có thể dùng bao nhiêu bộ nhớ. Sự giống nhau với <code>cudaDeviceProp</code> dừng ở đó. Không có
            số lõi, không có độ rộng SIMD, không có xung nhịp, không có số thanh ghi và không có giới hạn số luồng thường trú.
          </p>
        }
      >
        <p>
          Metal has the loop too. <code>MTLCreateSystemDefaultDevice()</code> returns the device the system would pick,
          and <code>MTLCopyAllDevices()</code> returns all of them. A device then tells you its name, the largest
          threadgroup it supports for the simplest possible kernel, how much threadgroup memory a kernel may declare,
          and how much memory you can use. That is where the resemblance to <code>cudaDeviceProp</code> ends. There is
          no count of cores, no SIMD width, no clock rate, no register count and no limit on resident threads.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ đề cập CUDA. Lấy từ tài liệu của Apple và các header của macOS SDK, đọc ngày 2026-11-20
              (ch04-facts.md, C1, C2, C5, C6). Header đã được tìm kiếm cho từng thuộc tính còn thiếu.
            </p>
          }
        >
          <p>
            The book covers CUDA only. From Apple's documentation and the macOS SDK headers, read on 2026-11-20
            (ch04-facts.md, C1, C2, C5, C6). The header was searched for each of the missing properties.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Hai con số mà một lần khởi chạy cần nhất lại nằm trên một đối tượng khác. Biên dịch một kernel và bạn có một pipeline state,
            và pipeline có <code>threadExecutionWidth</code>, độ rộng của một SIMD group, cùng{' '}
            <code>maxTotalThreadsPerThreadgroup</code>, giới hạn thật của một threadgroup cho kernel đó. Apple nói con số
            thứ hai có thể khác nhau giữa các pipeline. Vì vậy trên Metal bạn không thể định kích thước lần khởi chạy chỉ từ thiết bị.
            Bạn hỏi chính kernel bạn sắp chạy.
          </p>
        }
      >
        <p>
          The two numbers a launch needs most are on a different object. Compile a kernel and you get a pipeline state,
          and the pipeline has <code>threadExecutionWidth</code>, the width of a SIMD-group, and{' '}
          <code>maxTotalThreadsPerThreadgroup</code>, the real limit on a threadgroup for that kernel. Apple says the
          second can differ from one pipeline to another. So on Metal you cannot size a launch from the device alone.
          You ask the kernel you are about to run.
        </p>
      </Translated>
      <MetalQuery />
      <Translated
        vi={
          <p>
            Apple công bố một công thức biến hai con số đó thành một threadgroup: cho nó rộng bằng một SIMD group,{' '}
            w = <code>threadExecutionWidth</code>, và cao nhất mà pipeline cho phép, h ={' '}
            <code>maxTotalThreadsPerThreadgroup</code> / w.
          </p>
        }
      >
        <p>
          Apple publishes a recipe for turning the two numbers into a threadgroup: make it one SIMD-group wide,{' '}
          w = <code>threadExecutionWidth</code>, and as tall as the pipeline allows, h ={' '}
          <code>maxTotalThreadsPerThreadgroup</code> / w.
        </p>
      </Translated>
      <MetalRecipe />
      <Translated
        vi={
          <p>
            Có một công dụng của vòng lặp trong sách không áp dụng được. Trên các máy Mac thời Intel, chương trình có thể phân biệt GPU
            tích hợp với GPU rời bằng các thuộc tính thiết bị như <code>lowPower</code>. Các thuộc tính đó bị đánh dấu lỗi thời
            trong SDK hiện tại với thông báo "Not applicable on Apple Silicon".
          </p>
        }
      >
        <p>
          One thing the book's loop was for does not carry over. On Intel-era Macs a program could tell an integrated
          GPU from a discrete one with device properties such as <code>lowPower</code>. Those properties are deprecated
          in the current SDK with the message "Not applicable on Apple Silicon".
        </p>
      </Translated>
      <Ask q={questionsS48['ch04/4.8/q-metalwidth']} />

      <h2><Translated vi="Từ giới hạn đến một lần khởi chạy">From limits to a launch</Translated></h2>
      <Translated
        vi={
          <p>
            Truy vấn chỉ đáng công khi câu trả lời thay đổi cách bạn khởi chạy. Lấy kernel mà khóa học này cứ quay
            lại: một lớp tuyến tính, mỗi đầu ra một luồng, x theo các đầu ra và y theo batch. Lab 3.3
            khởi chạy nó với khối 16 × 16, vì các hình của sách làm vậy. Đó có phải lựa chọn tốt hay không tùy vào
            thiết bị và batch, và đây là các quy tắc chương này cho bạn để đánh giá:
          </p>
        }
      >
        <p>
          Querying is only worth the trouble if the answers change what you launch. Take the kernel this course keeps
          coming back to: a linear layer, one thread per output, x across the outputs and y across the batch. Lab 3.3
          launched it with 16 × 16 blocks, because the book's figures do. Whether that is a good choice depends on the
          device and on the batch, and these are the rules this chapter gives you for judging it:
        </p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>Khối hợp lệ: nằm trong <code>maxThreadsPerBlock</code> và <code>maxThreadsDim</code> (mục này).</li>
            <li>Nó gồm trọn warp, nên không lane nào là phần đệm (mục 4.4).</li>
            <li>
              Nó có thể lấp đầy khe luồng của một SM: kích thước của nó chia hết số khe đó, và đủ lớn để khe khối không
              cạn trước (mục 4.7).
            </li>
            <li>Nó không cao hơn batch, nên không hàng luồng nào được khởi chạy chỉ để không làm gì.</li>
            <li>Lưới phủ bài toán nằm trong <code>maxGridSize</code>.</li>
          </ul>
        }
      >
        <ul>
          <li>The block is legal: within <code>maxThreadsPerBlock</code> and <code>maxThreadsDim</code> (this section).</li>
          <li>It is whole warps, so no lane is padding (section 4.4).</li>
          <li>
            It can fill the thread slots of an SM: its size divides them, and it is large enough that the block slots do
            not run out first (section 4.7).
          </li>
          <li>It is no taller than the batch, so no row of threads is launched to do nothing.</li>
          <li>The grid that covers the problem is within <code>maxGridSize</code>.</li>
        </ul>
      </Translated>
      <Translated
        vi={
          <p>
            Trình khám phá bắt đầu từ thói quen: batch 1, 4096 đầu ra, 16 × 16. Hãy đọc xem nó vi phạm quy tắc nào và bao nhiêu
            luồng được khởi chạy không có việc gì làm. Rồi tự sửa nó trước khi bấm nút.
          </p>
        }
      >
        <p>
          The explorer starts at the habit: batch 1, 4096 outputs, 16 × 16. Read which rule it breaks and how many of
          the launched threads have nothing to do. Then repair it by hand before pressing the button.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <>
              <p>
                Các quy tắc này nhắm tới việc lấp đầy khe luồng, phù hợp với kernel chờ bộ nhớ. Một kernel làm việc trên
                thanh ghi trong thời gian dài có thể muốn điều ngược lại (mục 4.6).
              </p>
              <p>Thanh ghi và bộ nhớ chia sẻ bị bỏ qua: không trường thiết bị nào cho biết số thanh ghi của một kernel.</p>
              <p>Dành cho CUDA và HIP. Trên Metal, giới hạn đến từ pipeline, và công thức của Apple là điểm xuất phát.</p>
            </>
          }
        >
          <>
            <p>
              These rules aim at full thread slots, which suits a kernel that waits on memory. A kernel that works on
              registers for long stretches may want the opposite (section 4.6).
            </p>
            <p>Registers and shared memory are left out: no device field gives a kernel's register count.</p>
            <p>For CUDA and HIP. On Metal the limits come from the pipeline, and Apple's recipe is the starting point.</p>
          </>
        </Translated>
      </Simplification>
      <LaunchExplorer />
      <Translated
        vi={
          <p>
            Ở batch 1, một khối 16 × 16 đặt mười lăm trong mười sáu hàng luồng nằm ngoài điểm cuối của dữ liệu. Chúng
            bắt đầu, trượt phép kiểm tra biên rồi dừng, và vì một khối được cấp tài nguyên như một đơn vị, các khe của chúng vẫn bị
            chiếm cho đến khi hàng duy nhất có việc của khối xong. Một khối 64 × 1 trên các giới hạn
            của sách khởi chạy đúng những luồng có việc. Và đáp án thay đổi theo thiết bị: ở nơi 16 khe khối
            chia nhau 1536 khe luồng, 64 luồng mỗi khối là quá ít và khối hẹp nhất vượt qua là rộng 96.
          </p>
        }
      >
        <p>
          At batch 1 a 16 × 16 block puts fifteen rows of threads out of every sixteen past the end of the data. They
          start, fail the bounds check and stop, and because a block is given its resources as a unit, their slots stay
          taken until the block's one working row has finished. A 64 × 1 block on the book's
          limits launches exactly the threads that have work. And the answer moves with the device: where 16 block slots
          share 1536 thread slots, 64 threads per block is too few and the narrowest block that passes is 96 wide.
        </p>
      </Translated>
      <Ask q={questionsS48['ch04/4.8/q-ex9']} />
      <Translated
        vi={
          <p>
            Lab 4.1 trên <a href={href('ch04', 'labs')}>trang lab</a> là mục này dưới dạng mã: viết truy vấn, viết một
            hàm chọn cấu hình thỏa các quy tắc này cho bất kỳ mô tả thiết bị nào được đưa vào, và đo thời gian nó
            so với 16 × 16 trên GPU của bạn.
          </p>
        }
      >
        <p>
          Lab 4.1 on the <a href={href('ch04', 'labs')}>labs page</a> is this section as code: write the query, write a
          function that picks a configuration passing these rules for any device description it is handed, and time it
          against 16 × 16 on your GPU.
        </p>
      </Translated>

      <h2><Translated vi="Nơi bạn sẽ gặp điều này trong các backend thật">Where you will meet this in real backends</Translated></h2>
      <Translated
        vi={
          <p>
            Có lẽ bạn đã đọc struct này mà không biết.{' '}
            <code>torch.cuda.get_device_properties(0)</code> trả về một binding trực tiếp của <code>cudaDeviceProp</code>, với
            các thuộc tính như <code>multi_processor_count</code>, <code>max_threads_per_multi_processor</code>,{' '}
            <code>warp_size</code> và <code>regs_per_multiprocessor</code>. <code>clock_rate</code> của nó được đọc theo cách
            mới, qua <code>cudaDeviceGetAttribute</code>.
          </p>
        }
      >
        <p>
          You have probably already read this struct without knowing it.{' '}
          <code>torch.cuda.get_device_properties(0)</code> returns a direct binding of <code>cudaDeviceProp</code>, with
          attributes such as <code>multi_processor_count</code>, <code>max_threads_per_multi_processor</code>,{' '}
          <code>warp_size</code> and <code>regs_per_multiprocessor</code>. Its <code>clock_rate</code> is read the new
          way, through <code>cudaDeviceGetAttribute</code>.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Đọc từ nhánh mặc định của từng dự án ngày 2026-11-20 (ch04-facts.md, D1, D2, D4). Tệp có thể bị dời đi; hãy tìm theo
              tên định danh.
            </p>
          }
        >
          <p>
            Read from each project's default branch on 2026-11-20 (ch04-facts.md, D1, D2, D4). Files move; search for
            the identifier.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            llama.cpp chạy vòng lặp của sách một lần lúc khởi động và giữ ba con số cho mỗi thiết bị: <code>nsm</code> từ{' '}
            <code>multiProcessorCount</code>, <code>smpb</code> từ <code>sharedMemPerBlock</code> và{' '}
            <code>warp_size</code>, thứ sau đó nó dùng trên host để chọn giữa các kernel. Backend Metal của nó làm đúng điều
            phần trên mô tả: sau khi biên dịch mỗi kernel, nó ghi log{' '}
            <code>maxTotalThreadsPerThreadgroup</code> và <code>threadExecutionWidth</code> của pipeline dưới tên <code>th_max</code> và{' '}
            <code>th_width</code>, và giới hạn kích thước threadgroup bằng con số thứ nhất.
          </p>
        }
      >
        <p>
          llama.cpp runs the book's loop once at start-up and keeps three numbers per device: <code>nsm</code> from{' '}
          <code>multiProcessorCount</code>, <code>smpb</code> from <code>sharedMemPerBlock</code> and{' '}
          <code>warp_size</code>, which it later uses on the host to choose between kernels. Its Metal backend does what
          the section above describes: after compiling each kernel it logs the pipeline's{' '}
          <code>maxTotalThreadsPerThreadgroup</code> and <code>threadExecutionWidth</code> as <code>th_max</code> and{' '}
          <code>th_width</code>, and caps threadgroup sizes by the first.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Vậy là khép lại vòng của chương. Một SM giữ nhiều warp hơn số nó chạy để che thời gian chờ; nó giữ được bao nhiêu
            là chuyện của khe và thanh ghi; và đó là những con số bạn đọc từ thiết bị, hoặc trên Metal là từ
            pipeline, trước khi chọn cách khởi chạy. Các <a href={href('ch04', 'labs')}>lab</a> đưa cả ba điều này vào
            kiểm chứng trên phần cứng của chính bạn.
          </p>
        }
      >
        <p>
          That closes the chapter's loop. An SM holds more warps than it runs so that waits are hidden; how many it can
          hold is a matter of slots and registers; and those are numbers you read from the device, or on Metal from the
          pipeline, before you choose a launch. The <a href={href('ch04', 'labs')}>labs</a> put each of the three to the
          test on your own hardware.
        </p>
      </Translated>
    </>
  );
}
