import { useMemo, useState, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useReports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { CPU_WIDTHS, GPU_WIDTHS, GiB, binary, coverage, isEmulator, memoryFigures, smallestWidth } from '../../sim/addrwidth';
import { ROWS, checklist, granularityText, paradigm, paradigmFromRun, type Attrs } from '../../sim/umparadigm';

/** A long identifier that may break anywhere, for the narrow margin notes. */
function Id({ children }: { children: ReactNode }) {
  return <code style={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>{children}</code>;
}

// ---------------------------------------------------------------- does it all fit
const GIB_STEPS = Array.from({ length: 20 }, (_, k) => 2 ** k); // 1 GiB .. 512 TiB
const SW = 640;

function DoesItFit() {
  const { reports } = useReports();
  const yours = useMemo(() => memoryFigures(reports).filter((f) => f.field === 'totalGlobalMem'), [reports]);
  const [gpuBits, setGpuBits] = useState(49);
  const [cpuBits, setCpuBits] = useState(48);
  const [gib, setGib] = useState(64);

  const bytes = gib * GiB;
  const c = coverage(gpuBits, cpuBits, bytes);
  const need = smallestWidth(cpuBits, bytes);
  const scale = Math.max(c.capacity, c.needed);
  const px = (b: number) => (b / scale) * (SW - 20);
  const cpuW = px(c.cpu);
  const gpuW = Math.max(6, px(c.gpu));
  const capW = px(c.capacity);
  const step = GIB_STEPS.reduce((best, v, k) => (Math.abs(Math.log2(v) - Math.log2(gib)) < Math.abs(Math.log2(GIB_STEPS[best]) - Math.log2(gib)) ? k : best), 0);

  return (
    <Bench
      title={tr('Does it all fit?', 'Có vừa hết không?')}
      task={tr(
        "Find the smallest GPU width that holds a 48-bit CPU space and your GPU's memory, then try Linux's default 47 bits.",
        'Tìm độ rộng GPU nhỏ nhất chứa được một không gian CPU 48-bit và bộ nhớ GPU của bạn, rồi thử mức mặc định 47 bit của Linux.',
      )}
      foot={tr(
        'Widths: the book (48-bit CPUs, 49-bit Pascal, pp. 626–628) and the Linux 5-level paging documentation (47 and 56 bits, appC-facts.md A7).',
        'Các độ rộng: sách (CPU 48-bit, Pascal 49-bit, pp. 626–628) và tài liệu phân trang 5 cấp của Linux (47 và 56 bit, appC-facts.md A7).',
      )}
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-start' }}>
        <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
          <Translated vi="Độ rộng địa chỉ ảo của GPU">GPU virtual address width</Translated>
          <span className="seg" role="group" aria-label={tr('GPU virtual address width', 'Độ rộng địa chỉ ảo của GPU')}>
            {GPU_WIDTHS.map((b) => (
              <button key={b} aria-pressed={gpuBits === b} onClick={() => setGpuBits(b)}>
                {b}
              </button>
            ))}
          </span>
        </div>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Không gian người dùng của CPU">CPU user space</Translated>
          <select style={{ maxWidth: '100%' }} value={cpuBits} aria-label={tr('CPU user space', 'Không gian người dùng của CPU')} onChange={(e) => setCpuBits(Number(e.target.value))}>
            {CPU_WIDTHS.map((w) => (
              <option key={w.bits} value={w.bits}>
                {w.label}
              </option>
            ))}
          </select>
        </label>
        <label className="field" style={{ minWidth: '14rem' }}>
          <Translated vi={<>Bộ nhớ GPU cần đặt: {binary(bytes).text}</>}>GPU memory to place: {binary(bytes).text}</Translated>
          <input
            type="range"
            min={0}
            max={GIB_STEPS.length - 1}
            value={step}
            aria-label={tr('GPU memory to place', 'Bộ nhớ GPU cần đặt')}
            onChange={(e) => setGib(GIB_STEPS[Number(e.target.value)])}
          />
        </label>
      </div>
      {yours.length ? (
        <div className="row" role="group" aria-label={tr('Your devices', 'Thiết bị của bạn')} style={{ marginBottom: '0.8rem' }}>
          <span className="muted">
            <Translated vi="Của bạn">Yours</Translated>
          </span>
          {yours.map((f) => (
            <button key={f.backend} className="quiet" data-yours={f.backend} onClick={() => setGib(f.bytes / GiB)}>
              {f.name}: {binary(f.bytes).text}
            </button>
          ))}
        </div>
      ) : (
        <Translated
          vi={
            <p className="muted" style={{ margin: '0 0 0.8rem' }}>
              Chưa có device CUDA hay HIP nào được ghi nhận trên máy này, nên không có giá trị đặt sẵn cho bộ nhớ GPU của bạn.
              Trên Apple silicon, GPU không có bộ nhớ riêng để đặt: nó dùng bộ nhớ của hệ thống.
            </p>
          }
        >
          <p className="muted" style={{ margin: '0 0 0.8rem' }}>
            No CUDA or HIP device is recorded on this machine, so there is no preset for your GPU's memory. On Apple silicon
            the GPU has no memory of its own to place: it uses the system's.
          </p>
        </Translated>
      )}

      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${SW} 118`} role="img" aria-label={tr(
            "The GPU's virtual address space with the CPU range and the GPU range packed into it",
            'Không gian địa chỉ ảo của GPU, với dải CPU và dải GPU được xếp vào trong đó',
          )} style={{ width: '100%', minWidth: 480, display: 'block' }} data-part="strip" data-fits={c.fits ? 1 : 0}>
          <text x={10} y={14} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
            <Translated vi={<>Địa chỉ ảo GPU {gpuBits}-bit: {binary(c.capacity).text}</>}>
              {gpuBits}-bit GPU virtual addresses: {binary(c.capacity).text}
            </Translated>
          </text>
          <rect x={10} y={22} width={capW} height={44} fill="var(--cell)" stroke="var(--ink-2)" strokeWidth="1.5" data-part="capacity" />
          <rect x={10} y={30} width={Math.min(cpuW, capW)} height={28} fill="var(--rule-soft)" stroke="var(--ink-3)" />
          {cpuW > capW ? <rect x={10 + capW} y={30} width={cpuW - capW} height={28} fill="var(--fail-wash)" stroke="var(--fail)" /> : null}
          <rect x={10 + cpuW} y={30} width={gpuW} height={28} fill={c.gpuOutside > 0 ? 'var(--fail)' : 'var(--global)'} data-part="gpu" />
          <text x={14} y={48} fontSize="11">
            <Translated vi={<>Không gian người dùng CPU, {binary(c.cpu).text}</>}>CPU user space, {binary(c.cpu).text}</Translated>
          </text>
          <text x={Math.min(SW - 10, 10 + cpuW + gpuW + 4)} y={82} fontSize="11" textAnchor={10 + cpuW + gpuW > SW - 160 ? 'end' : 'start'} style={{ fill: c.gpuOutside > 0 ? 'var(--fail)' : 'var(--global)' }}>
            <Translated
              vi={
                <>
                  Bộ nhớ GPU, {binary(c.gpu).text}
                  {c.gpu / scale < 0.01 ? ' (vẽ rộng hơn tỉ lệ thật)' : ''}
                </>
              }
            >
              GPU memory, {binary(c.gpu).text}
              {c.gpu / scale < 0.01 ? ' (drawn wider than to scale)' : ''}
            </Translated>
          </text>
          <text x={10} y={108} fontSize="12" fontWeight="600" data-part="verdict" style={{ fill: c.fits ? 'var(--ok)' : 'var(--fail)' }}>
            {c.fits
              ? tr(`Fits, with ${binary(c.spare).text} to spare.`, `Vừa, còn dư ${binary(c.spare).text}.`)
              : c.cpuOutside > 0
                ? tr(
                    `Does not fit: the CPU range alone is ${Math.round(c.cpu / c.capacity).toLocaleString('en-US')} times the GPU's address space.`,
                    `Không vừa: riêng dải CPU đã gấp ${Math.round(c.cpu / c.capacity).toLocaleString('en-US')} lần không gian địa chỉ của GPU.`,
                  )
                : tr(
                    `Does not fit: ${binary(c.over).text} of GPU memory falls past the end.`,
                    `Không vừa: ${binary(c.over).text} bộ nhớ GPU rơi ra ngoài phần cuối.`,
                  )}
          </text>
        </svg>
      </div>
      <p style={{ margin: '0.6rem 0 0' }} data-out="smallest" data-bits={need}>
        <Translated
          vi={
            <>
              Độ rộng nhỏ nhất chứa được dải CPU này cộng {binary(bytes).text}: <b>{need} bit</b>.
            </>
          }
        >
          Smallest width that holds this CPU range plus {binary(bytes).text}: <b>{need} bits</b>.
        </Translated>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- which paradigm
const ATTR_NAMES: { key: keyof Attrs; cuda: string }[] = [
  { key: 'concurrentManagedAccess', cuda: 'cudaDevAttrConcurrentManagedAccess' },
  { key: 'pageableMemoryAccess', cuda: 'cudaDevAttrPageableMemoryAccess' },
  { key: 'usesHostPageTables', cuda: 'cudaDevAttrPageableMemoryAccessUsesHostPageTables' },
];
const PRESET: Record<string, Attrs> = {
  limited: { concurrentManagedAccess: false, pageableMemoryAccess: false, usesHostPageTables: false },
  managed: { concurrentManagedAccess: true, pageableMemoryAccess: false, usesHostPageTables: false },
  hmm: { concurrentManagedAccess: true, pageableMemoryAccess: true, usesHostPageTables: false },
  ats: { concurrentManagedAccess: true, pageableMemoryAccess: true, usesHostPageTables: true },
};

function WhichParadigm() {
  const { reports, loading } = useReports();
  const [a, setA] = useState<Attrs>(PRESET.managed);
  const [note, setNote] = useState<ReactNode>(null);
  const p = paradigm(a);
  const checks = checklist(a);

  const yours = () => {
    const found = Object.keys(reports)
      .filter((b) => !isEmulator(b) && b !== 'metal')
      .map((b) => ({ b, attrs: paradigmFromRun(reports[b]), name: String(reports[b].device.name ?? b), when: reports[b].generated }))
      .filter((x) => x.attrs);
    if (found.length) {
      const f = found[0];
      setA(f.attrs as Attrs);
      setNote(
        <Translated
          vi={
            <>
              Của bạn: {f.name} ({f.b}), theo báo cáo của lab C.1 ngày {f.when.slice(0, 10)}.
              {found.length > 1 ? ` Các backend khác có báo cáo: ${found.slice(1).map((x) => x.b).join(', ')}.` : ''}
            </>
          }
        >
          <>
            Yours: {f.name} ({f.b}), as lab C.1 reported it on {f.when.slice(0, 10)}.
            {found.length > 1 ? ` Other backends with a report: ${found.slice(1).map((x) => x.b).join(', ')}.` : ''}
          </>
        </Translated>,
      );
      return;
    }
    const metal = Object.keys(reports).includes('metal');
    setNote(
      <Translated
        vi={
          <>
            Chưa có kết quả lab C.1 nào từ máy CUDA hay HIP. Hãy chạy{' '}
            <code>python3 labs/pmpp.py run appC/lab1 --bench</code> trên một máy như vậy, rồi quay lại.
            {metal
              ? ' Device Metal của bạn không có các thuộc tính đó: Metal không có managed memory trên Apple silicon, và bộ nhớ CPU chỉ tới được GPU khi được bọc trong một buffer.'
              : ''}
          </>
        }
      >
        <>
          No lab C.1 result from a CUDA or HIP machine yet. Run{' '}
          <code>python3 labs/pmpp.py run appC/lab1 --bench</code> on one, then come back.
          {metal
            ? ' Your Metal device has no such attributes: Metal has no managed memory on Apple silicon, and CPU memory reaches the GPU only when it is wrapped in a buffer.'
            : ''}
        </>
      </Translated>,
    );
  };

  return (
    <Bench
      title={tr('Which paradigm is this machine?', 'Máy này thuộc mô hình (paradigm) nào?')}
      task={tr(
        'Set the switches for Linux without HMM, for WSL and for Grace Hopper, and compare what works.',
        'Đặt các công tắc cho Linux không có HMM, cho WSL và cho Grace Hopper, rồi so sánh những gì chạy được.',
      )}
      foot={tr(
        "The table and the attribute names are NVIDIA's (CUDA guide 2.6.2.1); what works on each row comes from guide sections 2.6.2, 4.1.2 and 4.1.3 (appC-facts.md, A10 to A14). HIP has attributes of the same names and meanings (B5).",
        'Bảng và tên các thuộc tính là của NVIDIA (hướng dẫn CUDA 2.6.2.1); những gì chạy được ở mỗi hàng lấy từ các mục 2.6.2, 4.1.2 và 4.1.3 của hướng dẫn (appC-facts.md, A10 đến A14). HIP có các thuộc tính cùng tên và cùng ý nghĩa (B5).',
      )}
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-start' }}>
        {ATTR_NAMES.map((n) => (
          <label key={n.key} className="check" style={{ alignItems: 'flex-start' }}>
            <input
              type="checkbox"
              data-attr={n.key}
              checked={a[n.key]}
              onChange={(e) => {
                setA({ ...a, [n.key]: e.target.checked });
                setNote(null);
              }}
            />
            <span>
              <Id>{n.cuda}</Id> = {a[n.key] ? 1 : 0}
            </span>
          </label>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <button className="primary" onClick={yours} disabled={loading}>
          <Translated vi="Của bạn">Yours</Translated>
        </button>
        {note ? (
          <span data-out="yours" style={{ flex: '1 1 20rem' }}>
            {note}
          </span>
        ) : null}
      </div>

      <div className="cols plot">
        <div className="panel" style={{ overflowX: 'auto' }}>
          <p className="muted" style={{ margin: '0 0 0.2rem' }}>
            <Translated vi="Mô hình">Paradigm</Translated>
          </p>
          <p style={{ fontSize: '1.25rem', fontWeight: 600, margin: '0 0 0.6rem', lineHeight: 1.25 }}>
            <output data-out="paradigm" data-key={p.row.key} data-listed={p.listed ? 1 : 0}>
              {p.row.name}
            </output>
          </p>
          {!p.listed ? (
            <p style={{ margin: '0 0 0.6rem' }}>
              <span className="chip fail">
                <Translated vi="không có trong bảng">not in the table</Translated>
              </span>{' '}
              {p.note}
            </p>
          ) : null}
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th style={{ paddingLeft: '0.4rem' }}>
                  <Translated vi="Mô hình, và nơi NVIDIA nói bạn gặp nó">Paradigm, and where NVIDIA says you find it</Translated>
                </th>
                <th>
                  <Translated vi="Thuộc tính">Attributes</Translated>
                </th>
              </tr>
            </thead>
            <tbody>
              {ROWS.map((r) => (
                <tr
                  key={r.key}
                  data-row={r.key}
                  data-on={r.key === p.row.key ? 1 : 0}
                  onClick={() => {
                    setA(PRESET[r.key]);
                    setNote(null);
                  }}
                  style={{ cursor: 'pointer', background: r.key === p.row.key ? 'var(--rule-soft)' : undefined }}
                >
                  <td style={{ paddingLeft: '0.4rem' }}>
                    <span style={{ fontWeight: r.key === p.row.key ? 650 : 500 }}>{r.name}</span>
                    <br />
                    <span className="muted">{r.where}</span>
                  </td>
                  <td style={{ minWidth: '9rem' }}>{r.condition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <p className="muted" style={{ margin: '0 0 0.4rem' }}>
            <Translated vi="Những gì chạy được ở đây">What works here</Translated>
          </p>
          <ul style={{ listStyle: 'none', padding: 0, margin: 0 }} data-part="checks">
            {checks.map((k) => (
              <li key={k.key} data-check={k.key} data-answer={k.answer} style={{ marginBottom: '0.55rem' }}>
                <span className={'chip ' + (k.answer === 'yes' ? 'ok' : k.answer === 'no' ? 'fail' : '')}>{tr(k.answer, ({ yes: 'có', no: 'không', 'not stated': 'không được nêu' } as Record<string, string>)[k.answer] ?? k.answer)}</span> {k.what}
                <br />
                <span className="muted">{k.why}</span>
              </li>
            ))}
            <li data-check="granularity">
              <span className="chip">
                <Translated vi="đơn vị">unit</Translated>
              </span>{' '}
              <Translated vi={<>Dữ liệu được quản lý di chuyển hoặc giữ nhất quán theo đơn vị: {granularityText(a)}.</>}>
                Managed data moves or stays coherent in: {granularityText(a)}.
              </Translated>
            </li>
          </ul>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSC7: Record<string, Question> = {
  'appC/C.7/q-49': {
    kind: 'numeric',
    id: 'appC/C.7/q-49',
    prompt: (
      <Translated
        vi={
          <>
            Độ rộng địa chỉ nhỏ nhất, tính bằng bit, chứa được hai dải riêng biệt mỗi dải 2<sup>48</sup> byte là bao nhiêu?
          </>
        }
      >
        <>
          What is the smallest address width, in bits, that holds two separate ranges of 2<sup>48</sup> bytes each?
        </>
      </Translated>
    ),
    answer: 49,
    unit: 'bits',
    explain: (
      <Translated
        vi={
          <p>
            2 × 2<sup>48</sup> = 2<sup>49</sup>. Một dải là không gian địa chỉ CPU 48-bit; dải kia có chỗ cho bộ nhớ của mọi
            GPU mà vẫn còn dư nhiều. Đó là lập luận của sách cho 49 bit của Pascal.
          </p>
        }
      >
        <p>
          2 × 2<sup>48</sup> = 2<sup>49</sup>. One range is a 48-bit CPU address space; the other has room for every GPU's
          memory with plenty to spare. That is the book's reasoning for Pascal's 49 bits.
        </p>
      </Translated>
    ),
    source: 'App. C §C.7, pp. 627–628',
    vi: { unit: 'bit', source: 'Phụ lục C §C.7, pp. 627–628' },
  },
  'appC/C.7/q-paradigm': {
    kind: 'choice',
    id: 'appC/C.7/q-paradigm',
    prompt: (
      <Translated
        vi={
          <>
            Một device báo cáo <code>cudaDevAttrConcurrentManagedAccess</code> = 1 và{' '}
            <code>cudaDevAttrPageableMemoryAccess</code> = 0. Điều nào đúng?
          </>
        }
      >
        <>
          A device reports <code>cudaDevAttrConcurrentManagedAccess</code> = 1 and{' '}
          <code>cudaDevAttrPageableMemoryAccess</code> = 0. Which holds?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mọi thứ đều chạy, kể cả con trỏ malloc và mmap: truy cập managed đồng thời nghĩa là mô hình cao nhất.">
            Everything works, malloc and mmap pointers included: concurrent managed access means the top paradigm.
          </Translated>
        ),
        why: (
          <Translated vi="Hỗ trợ đầy đủ bao quát các phép cấp phát managed. Bộ nhớ từ malloc cần PageableMemoryAccess = 1.">
            Full support covers managed allocations. Memory from malloc needs PageableMemoryAccess = 1.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Đó là mô hình giới hạn: managed memory di chuyển hàng loạt ở mỗi lần khởi chạy.">
            It is the limited paradigm: managed memory moves in bulk at each launch.
          </Translated>
        ),
        why: <Translated vi="Đó là ConcurrentManagedAccess = 0.">That is ConcurrentManagedAccess = 0.</Translated>,
      },
      {
        label: (
          <Translated vi="Unified Memory đầy đủ chỉ cho các phép cấp phát managed: CPU truy cập đồng thời và di chuyển khi có lỗi trang, nhưng không dùng được con trỏ malloc trong kernel.">
            Full Unified Memory for managed allocations only: concurrent CPU access and migration on fault, but no malloc
            pointers in kernels.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Managed memory không được hỗ trợ, vì PageableMemoryAccess = 0 loại trừ nó.">
            Managed memory is not supported, because PageableMemoryAccess = 0 rules it out.
          </Translated>
        ),
        why: (
          <Translated vi="Cả hai thuộc tính này đều mô tả managed memory được hỗ trợ như thế nào; không thuộc tính nào nói nó vắng mặt.">
            Both of these attributes describe how managed memory is supported; neither says it is missing.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Bảng của NVIDIA xếp tổ hợp này vào hàng thứ hai. Cơ chế lỗi trang kiểu Pascal bao quát bộ nhớ được cấp phát dạng
            managed; bộ nhớ do hệ thống cấp phát cần truy cập bộ nhớ phân trang (pageable), tức là HMM hoặc ATS.
          </p>
        }
      >
        <p>
          NVIDIA's table puts this combination in its second row. Pascal-class page faulting covers memory allocated as
          managed; system-allocated memory needs pageable memory access, which means HMM or ATS.
        </p>
      </Translated>
    ),
    source: 'CUDA guide 2.6.2.1 and 4.1.2, checked 2026-11-20 (notes/today/appC-facts.md, A12, A14)',
    vi: { source: 'Hướng dẫn CUDA 2.6.2.1 và 4.1.2, kiểm tra ngày 2026-11-20 (notes/today/appC-facts.md, A12, A14)' },
  },
  'appC/C.7/q-pascal-malloc': {
    kind: 'choice',
    id: 'appC/C.7/q-pascal-malloc',
    prompt: (
      <Translated
        vi={
          <>
            Một GPU Pascal trong một máy Linux không có HMM. Một kernel đi theo một danh sách liên kết mà host dựng bằng{' '}
            <code>new</code>. Điều gì xảy ra?
          </>
        }
      >
        <>
          A Pascal GPU in a Linux machine without HMM. A kernel follows a linked list that the host built with{' '}
          <code>new</code>. What happens?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Nó chạy: GPU gây lỗi trang ở mỗi nút, và driver di chuyển trang của nút đó sang bộ nhớ GPU.">
            It works: the GPU faults on each node, and the driver migrates that node's page to GPU memory.
          </Translated>
        ),
        why: (
          <Translated vi="Cơ chế lỗi trang trên máy này chỉ bao quát các phép cấp phát managed. Bộ nhớ từ new không phải managed.">
            Page faulting on this machine covers managed allocations only. Memory from new is not managed.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó chạy, chậm hơn, vì mỗi nút được đọc tại chỗ qua đường kết nối.">
            It works, more slowly, because each node is read in place over the interconnect.
          </Translated>
        ),
        why: (
          <Translated vi="Đó sẽ là bộ nhớ được ánh xạ (zero-copy), thứ mà new cũng không cho bạn.">
            That would be mapped (zero-copy) memory, which new does not give you either.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó thất bại: cơ chế lỗi trang của Pascal bao quát managed memory; new cần HMM hoặc ATS.">
            It fails: Pascal's faulting covers managed memory; new needs HMM or ATS.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nó chạy khi toàn bộ danh sách còn vừa bộ nhớ GPU, và thất bại khi vượt quá.">
            It works while the whole list fits in GPU memory, and fails once it outgrows it.
          </Translated>
        ),
        why: (
          <Translated vi="Kích thước không phải vấn đề. GPU hoàn toàn không có quyền truy cập vùng nhớ đó.">
            Size is not the issue. The GPU has no access to that memory at all.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Sách nói từ Pascal trở đi, GPU có thể duyệt một cấu trúc liên kết do host dựng, và bỏ ngỏ việc cái gì đã cấp phát
            nó. Hướng dẫn của NVIDIA trả lời: "the GPU cannot access system-allocated memory" (GPU không truy cập được bộ nhớ
            do hệ thống cấp phát) trên một máy như vậy. Hãy dựng danh sách trong managed memory, hoặc chạy ở nơi{' '}
            <code>PageableMemoryAccess</code> bằng 1.
          </p>
        }
      >
        <p>
          The book says from Pascal on a GPU can traverse a host-built linked structure, and leaves open what allocated it.
          NVIDIA's guide answers: "the GPU cannot access system-allocated memory" on such a machine. Build the list in
          managed memory, or run where <code>PageableMemoryAccess</code> is 1.
        </p>
      </Translated>
    ),
    source: 'App. C §C.8, pp. 628–629, against the CUDA guide 4.1.2 (notes/today/appC-facts.md, A12)',
    vi: { source: 'Phụ lục C §C.8, pp. 628–629, đối chiếu với hướng dẫn CUDA 4.1.2 (notes/today/appC-facts.md, A12)' },
  },
  'appC/C.7/q-wsl': {
    kind: 'choice',
    id: 'appC/C.7/q-wsl',
    prompt: (
      <Translated vi={<>Cùng chương trình đó chạy dưới WSL trên cùng GPU. Hành vi managed memory nào được áp dụng?</>}>
        <>The same program runs under WSL on the same GPU. Which managed-memory behaviour applies?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Giống như Linux gốc trên GPU đó, vì mô hình là thuộc tính của GPU.">
            The same as native Linux on that GPU, since the paradigm is a property of the GPU.
          </Translated>
        ),
        why: (
          <Translated vi="Mô hình phụ thuộc vào nền tảng cũng như GPU. NVIDIA liệt kê Windows và WSL trong số các hệ thống giới hạn.">
            The paradigm depends on the platform as well as the GPU. NVIDIA lists Windows and WSL among the limited systems.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="HMM, hoặc ít nhất là Unified Memory đầy đủ: WSL chạy một nhân Linux thật, nên quy tắc của Linux được áp dụng.">
            HMM, or at least full Unified Memory: WSL runs a real Linux kernel, so the Linux rules apply.
          </Translated>
        ),
        why: (
          <Translated vi="NVIDIA liệt kê WSL cùng với Windows, trong mô hình giới hạn.">
            NVIDIA lists WSL with Windows, in the limited paradigm.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Mô hình giới hạn: di chuyển hàng loạt lúc khởi chạy, CPU không truy cập được khi GPU đang chạy, không dùng vượt quá bộ nhớ.">
            The limited one: bulk moves at launch, no CPU access while the GPU runs, no oversubscription.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Hoàn toàn không có managed memory.">No managed memory at all.</Translated>,
        why: (
          <Translated vi="Managed memory chạy được ở đó, với các quy tắc giới hạn.">
            Managed memory works there, with the limited rules.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>ConcurrentManagedAccess</code> bằng 0 trên Windows và WSL. Đây là các quy tắc mà sách mô tả cho Kepler và
            Maxwell, ngày nay vẫn có hiệu lực ở bất cứ đâu thuộc tính đó bằng 0.
          </p>
        }
      >
        <p>
          <code>ConcurrentManagedAccess</code> is 0 on Windows and WSL. These are the rules the book describes for Kepler
          and Maxwell, still in force today wherever that attribute is 0.
        </p>
      </Translated>
    ),
    source: 'CUDA guide 2.6.2.3 and 4.1.3, checked 2026-11-20 (notes/today/appC-facts.md, A10, A14)',
    vi: { source: 'Hướng dẫn CUDA 2.6.2.3 và 4.1.3, kiểm tra ngày 2026-11-20 (notes/today/appC-facts.md, A10, A14)' },
  },
};

// ---------------------------------------------------------------- the section
export function SC7() {
  return (
    <>
      <Translated
        vi={
          <p>
            Theo lời kể của sách, lần mở rộng tiếp theo đến cùng kiến trúc Pascal năm 2016. Cơ chế dịch địa chỉ của nó tăng
            lên địa chỉ ảo 49-bit: đủ cho không gian địa chỉ 48-bit của một CPU và, bên cạnh đó, bộ nhớ riêng của GPU. Khi đó
            CPU và GPU có thể, theo lời sách, "truly share the pointer values" (thực sự dùng chung giá trị con trỏ), và GPU có
            thể đi qua các cấu trúc dữ liệu dựa trên con trỏ nằm trong bộ nhớ host.
          </p>
        }
      >
        <p>
          By the book's account, the next widening came with the Pascal architecture in 2016. Its address translation grew
          to 49-bit virtual addresses: enough for the 48-bit address space of a CPU and, beside it, the GPU's own memory.
          CPUs and GPUs could then, in the book's words, "truly share the pointer values", and a GPU could walk
          pointer-based data structures that live in host memory.
        </p>
      </Translated>
      <Source section="C.7" pages="627–628" />
      <Translated
        vi={
          <p>
            Chính lời kể của NVIDIA đồng ý về độ rộng và năm, và gắn nó với CUDA 8: theo lời họ, 49 bit là "sufficient to
            enable GPUs to access the entire system memory plus the memory of all GPUs in the system" (đủ để GPU truy cập toàn
            bộ bộ nhớ hệ thống cộng với bộ nhớ của mọi GPU trong hệ thống).
          </p>
        }
      >
        <p>
          NVIDIA's own account agrees on the width and the year, and ties it to CUDA 8: 49 bits are, in its words,
          "sufficient to enable GPUs to access the entire system memory plus the memory of all GPUs in the system".
        </p>
      </Translated>
      <Today
        checked={tr(
          "2026-11-20 against NVIDIA's Pascal Unified Memory post and the toolkit archive (appC-facts.md, A11)",
          'ngày 2026-11-20, đối chiếu với bài viết về Unified Memory trên Pascal của NVIDIA và kho lưu trữ toolkit (appC-facts.md, A11)',
        )}
      >
        <Translated
          vi={
            <p>
              CUDA 8: tháng 9 năm 2016 (kho lưu trữ toolkit). Mốc 2016 của sách cho Pascal đứng vững (P100 overview, © 2016).
            </p>
          }
        >
          <p>
            CUDA 8: September 2016 (toolkit archive). The book's 2016 for Pascal stands (P100 overview, © 2016).
          </p>
        </Translated>
      </Today>
      <Ask q={questionsSC7['appC/C.7/q-49']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Trình khám phá xếp nối tiếp dải người dùng của một CPU và một lượng bộ nhớ GPU vào không gian địa chỉ ảo của GPU.
            Hãy thử 48 bit với một CPU 48-bit, rồi 49. Sau đó chọn không gian người dùng mặc định của Linux: trên một máy có
            năm cấp bảng trang, Linux vẫn giữ chương trình dưới mốc 47 bit trừ khi nó yêu cầu nhiều hơn, và 47 cộng một lượng
            bộ nhớ GPU vừa trong 48. Chúng tôi không kiểm chứng được CUDA có xử lý một con trỏ host trên 47 bit hay không.
          </p>
        }
      >
        <p>
          The explorer packs a CPU's user range and some GPU memory end to end into a GPU's virtual address space. Try 48
          bits against a 48-bit CPU, then 49. Then pick Linux's default user space: on a machine with five levels of page
          tables, Linux still keeps a program below the 47-bit line unless it asks for more, and 47 plus some GPU memory
          fits in 48. Whether CUDA handles a host pointer above 47 bits we could not verify.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Các dải được xếp nối tiếp nhau. Bố cục thật dành riêng một số vùng, và các nguồn của chúng tôi không ghi lại
              chúng.
            </p>
          }
        >
          <p>The ranges are packed end to end. Real layouts reserve regions, and our sources do not document them.</p>
        </Translated>
      </Simplification>
      <DoesItFit />

      <h2>
        <Translated vi="Gọi tên được một địa chỉ không có nghĩa là được phép dùng nó">
          Naming an address is not permission to use it
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            "Dùng chung giá trị con trỏ" dễ khiến người ta hiểu theo một cách mà sách chưa bao giờ khẳng định: rằng kernel có
            thể đi theo bất kỳ con trỏ host nào. Sách nói một GPU Pascal có thể duyệt một cấu trúc liên kết do host dựng, và
            bỏ ngỏ việc cái gì đã cấp phát nó. Tài liệu của NVIDIA giải quyết điều này. Cơ chế lỗi trang của Pascal bao quát
            bộ nhớ được cấp phát dạng managed. Kernel chỉ với tới bộ nhớ <code>malloc</code>, <code>new</code> và{' '}
            <code>mmap</code> thông thường trên các hệ thống có HMM, một cơ chế phần mềm trong Linux, hoặc ATS, một cơ chế
            phần cứng, trên Grace Hopper và Grace Blackwell. Và theo chiều ngược lại, bộ nhớ từ <code>cudaMalloc</code> vẫn
            nằm ngoài tầm với của mã host ngay cả khi có tính nhất quán bằng phần cứng.
          </p>
        }
      >
        <p>
          "Share pointer values" invites a reading the book never commits to: that a kernel can follow any host pointer.
          The book says a Pascal GPU can traverse a host-built linked structure, and leaves open what allocated it. NVIDIA's
          documentation settles it. Pascal's page faulting covers memory allocated as managed. A kernel reaches plain{' '}
          <code>malloc</code>, <code>new</code> and <code>mmap</code> memory only on systems with HMM, which is a software
          mechanism in Linux, or ATS, which is hardware, on Grace Hopper and Grace Blackwell. And in the other direction,
          memory from <code>cudaMalloc</code> stays out of host code's reach even with hardware coherence.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 against the CUDA 13.5 guide 2.6.2 and 4.1 (appC-facts.md, A12)',
          'ngày 2026-11-20, đối chiếu với hướng dẫn CUDA 13.5 mục 2.6.2 và 4.1 (appC-facts.md, A12)',
        )}
      >
        <Translated
          vi={
            <p>
              HMM cần Linux 6.1.24+, 6.2.11+ hoặc 6.3+, compute capability 7.5+, và driver 535+ với các mô-đun nhân mã nguồn
              mở.
            </p>
          }
        >
          <p>HMM needs Linux 6.1.24+, 6.2.11+ or 6.3+, compute capability 7.5+, and driver 535+ with the open kernel modules.</p>
        </Translated>
        <Translated
          vi={
            <p>
              Kiểm tra: <Id>nvidia-smi -q | grep Addressing</Id> in ra "Addressing Mode : HMM".
            </p>
          }
        >
          <p>
            Check: <Id>nvidia-smi -q | grep Addressing</Id> prints "Addressing Mode : HMM".
          </p>
        </Translated>
      </Today>
      <Ask q={questionsSC7['appC/C.7/q-pascal-malloc']} />

      <h2>
        <Translated vi="Bạn có loại máy nào?">Which machine do you have?</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn không cần lịch sử phần cứng để biết quy tắc nào áp dụng ở nơi mã của bạn chạy. Ba thuộc tính device trả lời
            điều đó: <code>cudaDevAttrConcurrentManagedAccess</code>, <code>cudaDevAttrPageableMemoryAccess</code> và{' '}
            <code>cudaDevAttrPageableMemoryAccessUsesHostPageTables</code>. Hướng dẫn của NVIDIA chia máy thành bốn mô hình
            theo giá trị của chúng. Mô hình đầu tiên là bức tranh Kepler và Maxwell của sách, ngày nay vẫn còn trên Windows,
            WSL và một số hệ thống Tegra. Ba mô hình còn lại khác nhau ở chỗ GPU dùng được bộ nhớ nào và tính nhất quán được
            giữ thế nào.
          </p>
        }
      >
        <p>
          You do not need the hardware history to tell which rules apply where your code runs. Three device attributes
          answer it: <code>cudaDevAttrConcurrentManagedAccess</code>, <code>cudaDevAttrPageableMemoryAccess</code> and{' '}
          <code>cudaDevAttrPageableMemoryAccessUsesHostPageTables</code>. NVIDIA's guide sorts machines into four
          paradigms by their values. The first is the book's Kepler and Maxwell picture, which survives today on Windows,
          WSL and some Tegra systems. The other three differ in which memory the GPU can use and in how coherence is kept.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 against the CUDA 13.5 guide 2.6.2.1 (appC-facts.md, A14)',
          'ngày 2026-11-20, đối chiếu với hướng dẫn CUDA 13.5 mục 2.6.2.1 (appC-facts.md, A14)',
        )}
      >
        <Translated
          vi={
            <p>
              Truy vấn từng thuộc tính bằng <Id>cudaDeviceGetAttribute</Id>. HIP có đúng ba thuộc tính đó dưới tên{' '}
              <Id>hipDeviceAttribute…</Id> (B5).
            </p>
          }
        >
          <p>Query each with <Id>cudaDeviceGetAttribute</Id>. HIP has the same three under <Id>hipDeviceAttribute…</Id> names (B5).</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Đặt các công tắc và đọc những gì chạy được. Nút "Của bạn" nạp những gì <a href={href('appC', 'labs')}>lab C.1</a>{' '}
            đã báo cáo trên một máy CUDA hoặc HIP, nếu bạn đã chạy nó ở đó.
          </p>
        }
      >
        <p>
          Set the switches and read what works. The "Yours" button loads what <a href={href('appC', 'labs')}>lab C.1</a>{' '}
          reported on a CUDA or HIP machine, if you have run it there.
        </p>
      </Translated>
      <WhichParadigm />
      <Ask q={questionsSC7['appC/C.7/q-paradigm']} />
      <Ask q={questionsSC7['appC/C.7/q-wsl']} />
      <Translated
        vi={
          <p>
            Vậy cùng một mã nguồn CUDA có thể hành xử theo ba cách trên cùng một GPU: dưới WSL với các quy tắc giới hạn, trên
            Linux không có HMM với chỉ managed memory, và trên một Linux mới hơn có HMM dùng được mọi con trỏ. Một chương
            trình dựa vào nhiều hơn các quy tắc giới hạn nên kiểm tra các thuộc tính này lúc khởi động thay vì giả định.
          </p>
        }
      >
        <p>
          So one CUDA source can behave three ways on the same GPU: under WSL with the limited rules, on Linux without HMM
          with managed memory only, and on a newer Linux with HMM using any pointer. A program that relies on more than the
          limited rules should check these attributes at start-up instead of assuming.
        </p>
      </Translated>

      <h2>
        <Translated vi="Nơi bạn gặp nó">Where you meet it</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trên các hệ thống HMM và ATS, bộ nhớ gắn với tệp từ <code>mmap</code> cũng được tính là unified memory. Khi đó một
            kernel có thể đọc tại chỗ trọng số của một tệp mô hình được ánh xạ bộ nhớ, không cần bước nạp nào ở giữa. Đó là
            điều backend Metal của llama.cpp làm trên Mac: nó bọc tệp trọng số đã mmap trong các buffer Metal thay vì sao chép.
            Trên một máy PCIe không có HMM, kernel CUDA không thể đi theo một con trỏ vào tệp được ánh xạ, vì bộ nhớ do hệ
            thống cấp phát không truy cập được từ GPU ở đó.
          </p>
        }
      >
        <p>
          On HMM and ATS systems, file-backed memory from <code>mmap</code> counts as unified memory too. A kernel could
          then read the weights of a memory-mapped model file in place, with no load step in between. That is what
          llama.cpp's Metal backend does on a Mac: it wraps the mmapped weight file in Metal buffers instead of copying it.
          On a PCIe machine without HMM, a CUDA kernel cannot follow a pointer into the mapped file, because
          system-allocated memory is not GPU-accessible there.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Hướng dẫn của NVIDIA: với truy cập bộ nhớ phân trang, "This includes file-backed memory allocations created with
              mmap" (điều này bao gồm cả các phép cấp phát gắn với tệp tạo bằng mmap) (A12). Bài viết về HMM của NVIDIA chạy
              lại Hình C.1 của sách với <Id>malloc</Id> thông thường (A9).
            </p>
          }
        >
          <p>
            NVIDIA's guide: with pageable memory access, "This includes file-backed memory allocations created with mmap"
            (A12). NVIDIA's HMM post reruns the book's Fig. C.1 with plain <Id>malloc</Id> (A9).
          </p>
        </Translated>
        <Translated vi={<p>llama.cpp bọc dải được ánh xạ bằng <Id>newBufferWithBytesNoCopy</Id> (D4).</p>}>
          <p>llama.cpp wraps the mapped range with <Id>newBufferWithBytesNoCopy</Id> (D4).</p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Trên GPU AMD và Apple">On AMD and Apple GPUs</Translated>
      </h2>
      <Translated
        vi={
          <p>
            GPU của AMD đi theo cùng ý tưởng dưới những công tắc khác. Kernel chỉ dùng được bộ nhớ từ <code>new</code> hoặc{' '}
            <code>malloc</code> trên CDNA2 trở lên, với <code>HSA_XNACK=1</code> được đặt và một driver nhân có hỗ trợ HMM, và
            lần truy cập GPU đầu tiên vào một trang như vậy là một lỗi trang có thể phục hồi. Ba thuộc tính tồn tại trong HIP
            với ý nghĩa như của CUDA. Trên Metal không có các thuộc tính như vậy: kernel chỉ dùng tại chỗ được bộ nhớ CPU đã
            cấp phát khi nó được bọc trong một buffer bằng <code>newBufferWithBytesNoCopy</code>, vốn cần bộ nhớ căn theo
            trang (xem <a href={href('ch02', '2.4')}>chương 2, §2.4</a>).
          </p>
        }
      >
        <p>
          AMD's GPUs follow the same idea under other switches. A kernel can use memory from <code>new</code> or{' '}
          <code>malloc</code> only on CDNA2 or later, with <code>HSA_XNACK=1</code> set and a kernel driver with HMM support,
          and the first GPU access to such a page is a recoverable page fault. The three attributes exist in HIP with
          CUDA's meanings. On Metal there are no such attributes: a kernel uses memory the CPU allocated in place only once
          it is wrapped in a buffer with <code>newBufferWithBytesNoCopy</code>, which needs page-aligned memory (see{' '}
          <a href={href('ch02', '2.4')}>chapter 2, §2.4</a>).
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Kiểm tra ngày 2026-11-20, đối chiếu với trang unified memory của HIP từ AMD và header ROCm 10.2 (appC-facts.md,
              B5), cùng header MTLDevice của Apple (ch02-facts.md, C2).
            </p>
          }
        >
          <p>Checked 2026-11-20 against AMD's HIP unified memory page and the ROCm 10.2 header (appC-facts.md, B5), and Apple's MTLDevice header (ch02-facts.md, C2).</p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Quay lại <a href={href('appC', 'start')}>bản đồ</a>: C.7 và C.8 chung bước được đánh dấu C.7–C.8, bước thay đổi
            hàng managed memory. Bộ nhớ <code>malloc</code> thông thường phải chờ tới bước nằm ngoài sách, HMM và ATS.
          </p>
        }
      >
        <p>
          Back to <a href={href('appC', 'start')}>the map</a>: C.7 and C.8 share the step marked C.7–C.8, which changes the
          managed-memory row. Plain <code>malloc</code> memory has to wait for the step beyond the book, HMM and ATS.
        </p>
      </Translated>
    </>
  );
}
