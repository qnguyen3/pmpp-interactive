import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { fmt, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr, useLocale } from '../../lib/i18n';
import {
  BOOK_DEVICE,
  BOOK_DEVICE_TODAY,
  TERMS,
  deviceFromReport,
  isGpuBackend,
  levels,
  noReport,
  residentThreads,
  threadsPerCore,
  totalCores,
  type Device,
  type Level,
  type LevelId,
  type Vocabulary,
} from '../../sim/blocksched';

const VOCABS: Vocabulary[] = ['cuda', 'hip', 'metal'];
const PROVENANCE: Record<Level['provenance'], () => string> = {
  book: () => tr('the book states it', 'sách nêu ra'),
  reported: () => tr('your device reported it', 'thiết bị của bạn báo cáo'),
  'vendor-doc': () => tr("AMD's documentation, CDNA family", 'tài liệu của AMD, dòng CDNA'),
  none: () => tr('nobody says', 'không ai nói'),
};
const where = (w: string) => (w.startsWith('4.') ? href('ch04', w) : href('ch05', 'start'));

/** The count cell of a level row, in words. */
function countText(l: Level): string {
  if (l.count != null) return l.id === 'device' ? '1' : `${fmt(l.count, 6)} ${l.per}`;
  if (l.id === 'onchip' || l.id === 'global') return l.per;
  return l.name == null ? tr('not documented', 'không có tài liệu') : tr('not reported', 'không được báo cáo');
}

// Fig. 4.1 redrawn. The picture is a schematic: it always draws two clusters of
// four units with sixteen cores each, and the labels say how many there really are.
function Picture({ ls, sel, onPick, known }: { ls: Level[]; sel: LevelId; onPick: (id: LevelId) => void; known: boolean }) {
  const L = Object.fromEntries(ls.map((l) => [l.id, l])) as Record<LevelId, Level>;
  const stroke = (id: LevelId) => ({
    stroke: sel === id ? 'var(--ink)' : 'var(--ink-3)',
    strokeWidth: sel === id ? 2.4 : 1,
    strokeDasharray: L[id].name == null ? '4 3' : undefined,
  });
  const pick = (id: LevelId) => ({ onClick: () => onPick(id), style: { cursor: 'pointer' } });
  const units = [0, 1, 2, 3];
  const ghost = known ? 1 : 0.45;
  return (
    <svg viewBox="0 0 560 318" role="img" aria-label={tr("Schematic of the device, redrawn from the book's Fig. 4.1", 'Sơ đồ thiết bị, vẽ lại từ Fig. 4.1 của sách')} style={{ width: '100%', maxWidth: 560, display: 'block' }}>
      <g {...pick('device')}>
        <rect x="4" y="20" width="552" height="292" rx="5" fill="var(--page)" {...stroke('device')} />
        <text x="12" y="14" fontSize="12" fontWeight="600">{L.device.name}</text>
      </g>
      {[0, 1].map((c) => {
        const cx = 16 + c * 268;
        return (
          <g key={c} opacity={ghost}>
            <g {...pick('cluster')}>
              <rect x={cx} y="42" width="260" height="150" rx="4" fill="none" {...stroke('cluster')} strokeDasharray={L.cluster.name == null ? '4 3' : '6 3'} />
              <text x={cx + 8} y="56" fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{L.cluster.name ?? ''}</text>
            </g>
            {units.map((u) => {
              const ux = cx + 10 + u * 61;
              return (
                <g key={u}>
                  <rect x={ux} y="64" width="56" height="120" rx="3" fill="var(--cell)" {...stroke('unit')} {...pick('unit')} />
                  <rect x={ux + 5} y="69" width="46" height="11" rx="1.5" fill="var(--rule)" stroke="var(--ink-3)" strokeWidth="0.6" {...pick('unit')} />
                  <text x={ux + 28} y="77.5" fontSize="7.5" textAnchor="middle" style={{ fill: 'var(--ink-2)', pointerEvents: 'none' }}><Translated vi="điều khiển">control</Translated></text>
                  {[0, 1].map((g) => (
                    <g key={g} {...pick('group')}>
                      <rect x={ux + 4 + g * 25} y="84" width="23" height="48" rx="2" fill="none" {...stroke('group')} />
                      {Array.from({ length: 8 }, (_, i) => (
                        <rect key={i} x={ux + 6.5 + g * 25 + (i % 2) * 9.5} y={86.5 + Math.floor(i / 2) * 11} width="8" height="9.5" rx="1"
                          fill={sel === 'core' ? 'var(--ink-2)' : 'var(--bench)'} stroke={sel === 'core' ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={sel === 'core' ? 1.2 : 0.7}
                          onClick={(e) => { e.stopPropagation(); onPick('core'); }} style={{ cursor: 'pointer' }} />
                      ))}
                    </g>
                  ))}
                  <g {...pick('onchip')}>
                    <rect x={ux + 5} y="137" width="46" height="42" rx="2" fill="var(--page)" {...stroke('onchip')} />
                    <rect x={ux + 8} y="141" width="40" height="15" rx="1.5" fill="var(--reg-wash)" stroke="var(--reg)" strokeWidth="0.8" />
                    <rect x={ux + 8} y="160" width="40" height="15" rx="1.5" fill="var(--shared-wash)" stroke="var(--shared)" strokeWidth="0.8" />
                  </g>
                </g>
              );
            })}
          </g>
        );
      })}
      {!known ? (
        <g>
          <rect x="150" y="100" width="260" height="34" rx="3" fill="var(--page)" stroke="var(--ink-3)" />
          <text x="280" y="121" fontSize="12" textAnchor="middle"><Translated vi="không có gì bên trong thiết bị được ghi trong tài liệu">nothing inside the device is documented</Translated></text>
        </g>
      ) : null}
      {/* memory controllers and global memory */}
      <line x1="30" x2="530" y1="204" y2="204" stroke="var(--ink-3)" />
      {[0, 1, 2, 3, 4, 5].map((k) => (
        <g key={k}>
          <line x1={63 + k * 87} x2={63 + k * 87} y1="204" y2="214" stroke="var(--ink-3)" />
          <rect x={30 + k * 87} y="214" width="66" height="20" rx="2" fill="var(--cell)" stroke="var(--ink-3)" />
          <text x={63 + k * 87} y="227.5" fontSize="8" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}><Translated vi="bộ điều khiển">controller</Translated></text>
          <line x1={63 + k * 87} x2={63 + k * 87} y1="234" y2="254" stroke="var(--global)" strokeWidth="1.4" />
        </g>
      ))}
      <g {...pick('global')}>
        <rect x="16" y="254" width="528" height="48" rx="3" fill="var(--global-wash)" stroke="var(--global)" strokeWidth={sel === 'global' ? 2.4 : 1.2} />
        <text x="280" y="283" fontSize="12.5" fontWeight="600" textAnchor="middle" style={{ fill: 'var(--global)' }}>{L.global.name}</text>
      </g>
    </svg>
  );
}

function LevelExplorer() {
  const { reports } = useReports();
  const [vocab, setVocab] = useState<Vocabulary>('cuda');
  const [pickDevice, setPickDevice] = useState(0);
  const [sel, setSel] = useState<LevelId>('unit');

  const devices: Device[] = useMemo(() => {
    const mine = Object.keys(reports).filter(isGpuBackend)
      .map((b) => deviceFromReport(b, reports[b].device))
      .filter((d) => d.vocabulary === vocab);
    if (vocab === 'cuda') return [BOOK_DEVICE, ...mine];
    return mine.length ? mine : [noReport(vocab)];
  }, [reports, vocab]);
  const d = devices[Math.min(pickDevice, devices.length - 1)];
  const locale = useLocale();
  const ls = useMemo(() => levels(d), [d, locale]);
  const cur = ls.find((l) => l.id === sel)!;
  const hardware = ls.filter((l) => ['cluster', 'unit', 'group', 'core'].includes(l.id));
  const askable = hardware.filter((l) => l.ask != null).length;
  const known = hardware.some((l) => l.name != null);
  const t = TERMS[vocab];
  const optionLabel = (x: Device) =>
    x.source === 'book' ? tr("the book's example", 'ví dụ của sách') : x.source === 'report' ? tr(`your device: ${x.name}`, `thiết bị của bạn: ${x.name}`) : tr(`no ${t.api} device report on this machine`, `không có báo cáo thiết bị ${t.api} trên máy này`);

  return (
    <Bench
      title={tr('The device, level by level', 'Thiết bị, từng tầng một')}
      task={tr('Switch the backend and the device. Which of these sizes can a program ask for?', 'Đổi backend và thiết bị. Một chương trình có thể hỏi những kích thước nào trong số này?')}
      foot={
        <Translated
          vi={
            <>
              Hình là Fig. 4.1 của sách dưới dạng sơ đồ với số ô cố định, tô xám ở chỗ nhà cung cấp không ghi tài liệu
              về tầng đó; bảng cho biết thực tế có bao nhiêu. Trong hình của sách,{' '}
              <span className="chip reg">thanh ghi</span> và <span className="chip shared">bộ nhớ chia sẻ</span> nằm trên
              chip còn <span className="chip global">bộ nhớ toàn cục</span> nằm ngoài chip.
            </>
          }
        >
          <>
            The picture is the book's Fig. 4.1 as a schematic with a fixed number of boxes, greyed where a vendor documents
            no such level; the table says how many there are. In the book's picture,{' '}
            <span className="chip reg">registers</span> and <span className="chip shared">shared memory</span> are on the
            chip and <span className="chip global">global memory</span> is off it.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span className="seg" role="group" aria-label={tr('Backend', 'Backend')}>
          {VOCABS.map((v) => (
            <button key={v} aria-pressed={v === vocab} onClick={() => { setVocab(v); setPickDevice(0); }}>
              {TERMS[v].api}
            </button>
          ))}
        </span>
        <label className="field">
          <Translated vi="Thiết bị">Device</Translated>
          <select value={Math.min(pickDevice, devices.length - 1)} onChange={(e) => setPickDevice(+e.target.value)} aria-label={tr('Device', 'Thiết bị')}>
            {devices.map((x, i) => <option key={i} value={i}>{optionLabel(x)}</option>)}
          </select>
        </label>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
        <div>
          <Picture ls={ls} sel={sel} onPick={setSel} known={known} />
        </div>
        <div>
          <table className="data levels" style={{ margin: 0, background: 'var(--page)', width: '100%' }} aria-label={tr('Levels of the device', 'Các tầng của thiết bị')}>
            <thead>
              <tr>
                <th style={{ paddingLeft: '0.5rem', width: '36%' }}><Translated vi="Tầng">Level</Translated></th>
                <th style={{ width: '27%' }}><Translated vi="Bao nhiêu">How many</Translated></th>
                <th><Translated vi="Chương trình hỏi bằng">A program asks with</Translated></th>
              </tr>
            </thead>
            <tbody>
              {ls.map((l) => (
                <tr key={l.id} data-level={l.id} data-count={l.count ?? ''} style={l.id === sel ? { background: 'var(--rule-soft)' } : undefined}>
                  <td style={{ paddingLeft: '0.5rem' }}>
                    <button className="quiet" aria-pressed={l.id === sel} onClick={() => setSel(l.id)} style={{ whiteSpace: 'normal', textAlign: 'left' }}>
                      {l.name ?? tr(`no counterpart of the ${l.bookName} documented`, `không có tài liệu về tầng tương ứng với ${l.bookName}`)}
                    </button>
                  </td>
                  <td>
                    {countText(l)}
                    {l.count != null ? <div className="muted says">{PROVENANCE[l.provenance]()}</div> : null}
                  </td>
                  <td>{l.ask ? <code style={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>{l.ask}</code> : <span className="muted"><Translated vi="không có truy vấn">no query</Translated></span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="summary" role="status" style={{ margin: '0.8rem 0 0' }}>
            <Translated vi={<>Trong bốn tầng phần cứng bên dưới thiết bị, một chương trình {t.api} có thể hỏi kích thước của <b>{askable}</b>.</>}>
              Of the four hardware levels under the device, a {t.api} program can ask for the size of <b>{askable}</b>.
            </Translated>
            {d.source === 'report' && d.threadsPerUnit != null ? (
              <Translated vi={<> Thiết bị của bạn còn báo cáo {fmt(d.threadsPerUnit, 6)} luồng đang chạy dở trên mỗi {t.unit}{d.blocksPerUnit != null ? <> và {fmt(d.blocksPerUnit, 6)} khối</> : null}.</>}>
                <> Your device also reports {fmt(d.threadsPerUnit, 6)} threads in progress per {t.unit}{d.blocksPerUnit != null ? <> and {fmt(d.blocksPerUnit, 6)} blocks</> : null}.</>
              </Translated>
            ) : null}
          </p>
        </div>
      </div>
      <div className="panel" style={{ marginTop: '1rem' }} data-selected={cur.id}>
        <p style={{ margin: '0 0 0.4rem' }}>
          <b>{cur.name ?? tr(`No counterpart of the ${cur.bookName} documented`, `Không có tài liệu về tầng tương ứng với ${cur.bookName}`)}</b>
          {cur.holds.length ? tr(': what the rest of the chapter places here', ': những gì phần còn lại của chương đặt ở đây') : null}
        </p>
        {cur.holds.length ? (
          <ul style={{ margin: '0 0 0.5rem', paddingLeft: '1.2em' }}>
            {cur.holds.map((h) => (
              <li key={h.what}>
                {h.what} (<a href={where(h.where)}>{h.where.startsWith('4.') ? tr('section ', 'phần ') + h.where : tr(h.where, ({ 'chapter 5': 'chương 5' } as Record<string, string>)[h.where] ?? h.where)}</a>)
              </li>
            ))}
          </ul>
        ) : null}
        <p className="muted" style={{ margin: 0 }}>{cur.note}</p>
      </div>
    </Bench>
  );
}

export const questionsS41: Record<string, Question> = {
  'ch04/4.1/q-cores': {
    kind: 'choice',
    id: 'ch04/4.1/q-cores',
    prompt: (
      <Translated
        vi={
          <>
            Một đồng nghiệp đọc ví dụ của sách và nói: "Thiết bị có 16,896 lõi, nên tại mỗi thời điểm có tối đa 16,896
            luồng của kernel của tôi đang chạy dở." Cần sửa lại điều gì?
          </>
        }
      >
        <>
          A colleague reads the book's example and says: "The device has 16,896 cores, so at most 16,896 threads of my
          kernel are in progress at any moment." What is the correction?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Quá thấp: số lõi giới hạn những gì thực thi trong một chu kỳ, còn mỗi SM có thể giữ tới 2048 luồng đang chạy dở.">
            {'Too low: cores bound what executes in one cycle, and each SM can keep up to 2048 threads in progress.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không cần sửa: mỗi lõi chạy một luồng, phần còn lại của lưới chờ trong danh sách của runtime.">
            {'No correction: one thread runs per core, and the rest of the grid waits on the runtime\'s list.'}
          </Translated>
        ),
        why: (
          <Translated vi="Đó là cách bạn định kích thước một thread pool trên CPU. Trong ví dụ, một SM giữ số luồng gấp mười sáu lần số lõi của nó.">
            {'That is how you would size a CPU thread pool. An SM holds sixteen times more threads than it has cores in the example.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Quá cao: một SM chạy từng luồng một, nên tối đa chỉ có 132 luồng đang chạy dở.">
            {'Too high: an SM runs one thread at a time, so at most 132 threads are in progress.'}
          </Translated>
        ),
        why: (
          <Translated vi="Một SM không phải là một lõi. Trong ví dụ, nó có 128 lõi.">
            {'An SM is not a core. It has 128 of them in the example.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Quá cao: một khối chứa tối đa 1024 luồng, và thiết bị chạy từng khối một.">
            {'Too high: a block holds at most 1024 threads, and the device runs one block at a time.'}
          </Translated>
        ),
        why: (
          <Translated vi="Giới hạn đó là cho từng khối. Nhiều khối đang chạy dở cùng lúc.">
            {'That limit is per block. Many blocks are in progress at once.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hai con số riêng biệt. Những gì có thể thực thi trong cùng một chu kỳ xung nhịp bị giới hạn bởi số lõi: 132 ×
            128 = 16,896. Những gì đang chạy dở bị giới hạn bởi lượng mà các SM giữ thường trú: 132 × 2048 = 270,336 luồng
            trong ví dụ. Khoảng cách đó là có chủ ý (phần 4.6), và các lõi cũng không phải là những bộ xử lý độc lập: chúng
            dùng chung logic điều khiển, nên chạy theo nhóm cùng thực thi một lệnh (phần 4.4).
          </p>
        }
      >
        <p>
          Two separate numbers. What can execute in the same clock cycle is bounded by the cores: 132 × 128 = 16,896. What
          is in progress is bounded by what the SMs keep resident: 132 × 2048 = 270,336 threads in the example. The gap is
          deliberate (section 4.6), and the cores are not independent processors either: they share control logic, so
          they run in groups that execute one instruction together (section 4.4).
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.1, p. 68 and §4.6, p. 85 (figures are the book\'s example device)',
    vi: { source: 'ch. 4 §4.1, p. 68 và §4.6, p. 85 (các con số là của thiết bị ví dụ của sách)' },
  },
  'ch04/4.1/q-mpcount': {
    kind: 'choice',
    id: 'ch04/4.1/q-mpcount',
    prompt: (
      <Translated
        vi={
          <>
            Cùng một chương trình in ra <code>multiProcessorCount</code> trên một thiết bị NVIDIA qua CUDA và trên một
            thiết bị AMD qua HIP. Hai con số đó đếm cái gì?
          </>
        }
      >
        <>
          The same program prints <code>multiProcessorCount</code> on an NVIDIA device through CUDA and on an AMD device
          through HIP. What do the two numbers count?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Cả hai đều đếm lõi: cùng một tên trường, nên số lớn hơn là thiết bị lớn hơn.">
            {'Cores on both: one field name, so the larger number is the larger device.'}
          </Translated>
        ),
        why: <Translated vi="Không cái nào là số lõi.">{'Neither is a core count.'}</Translated>,
      },
      {
        label: (
          <Translated vi="Bên này là SM; bên kia là compute unit, hoặc một nửa số đó. Hai con số không so sánh được với nhau.">
            {'SMs on one; compute units, or half of them, on the other. The two are not comparable.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Cả hai đều là SM: HIP quy đổi phần cứng AMD sang đơn vị của NVIDIA cho trường này.">
            {'SMs on both: HIP translates the AMD hardware into NVIDIA\'s units for this field.'}
          </Translated>
        ),
        why: (
          <Translated vi="HIP giữ tên trường. Nó điền vào đó đơn vị riêng của AMD.">
            {'HIP keeps the field name. It fills it with AMD\'s own unit.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Số GPU trong máy, thứ mà cả hai runtime báo cáo theo cùng một cách.">
            {'The number of GPUs in the machine, which both runtimes report the same way.'}
          </Translated>
        ),
        why: (
          <Translated vi="Đó là thứ mà cudaGetDeviceCount và hipGetDeviceCount trả về.">
            {'That is what cudaGetDeviceCount and hipGetDeviceCount return.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Cái tên được giữ lại; đơn vị thì không. Trên CUDA, nó là số SM. Header của HIP nói nó bằng số compute unit,
            hoặc một nửa số đó khi thiết bị chạy ở chế độ workgroup-processor, nơi một bộ xử lý chứa hai CU. Một SM và một
            CU có cấu tạo bên trong khác nhau, nên tỉ lệ giữa hai con số không nói gì về tốc độ tương đối. Hãy dùng con số
            này cho đúng việc nó làm tốt trên cả hai nhà cung cấp: định kích thước một lần khởi chạy theo thiết bị bạn
            đang dùng.
          </p>
        }
      >
        <p>
          The name carries over; the unit does not. On CUDA it is the number of SMs. HIP's header says it equals the
          number of compute units, or half that number when the device runs in workgroup-processor mode, where one
          processor holds two CUs. An SM and a CU are built differently inside, so a ratio of the two counts says nothing
          about relative speed. Use the number for what it is good for on either vendor: sizing a launch against the
          device you are on.
        </p>
      </Translated>
    ),
    source: 'book p. 89 for CUDA; notes/today/ch04-facts.md B1 and ch05-facts.md B2 for HIP',
    vi: { source: 'sách p. 89 cho CUDA; notes/today/ch04-facts.md B1 và ch05-facts.md B2 cho HIP' },
  },
  'ch04/4.1/q-global': {
    kind: 'choice',
    id: 'ch04/4.1/q-global',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel nhận đối số <code>float* M</code>. Trong hình của thiết bị, dữ liệu phía sau <code>M</code> nằm ở
            đâu, và mỗi lần truy cập nó phải đi qua cái gì?
          </>
        }
      >
        <>
          A kernel takes an argument <code>float* M</code>. In the picture of the device, where does the data behind{' '}
          <code>M</code> live, and what does every access to it cross?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Trong bộ nhớ trên chip của SM mà luồng đang chạy; không đi qua gì cả.">
            {'In the on-chip memory of the SM the thread runs on; nothing is crossed.'}
          </Translated>
        ),
        why: (
          <Translated vi="Bộ nhớ trên chip chứa thanh ghi và bộ nhớ chia sẻ. Một đối số con trỏ không trỏ tới đó.">
            {'On-chip memory holds registers and shared memory. A pointer argument does not point there.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Trong bộ nhớ host; mỗi lần truy cập đều đi qua bus giữa CPU và GPU.">
            {'In host memory; every access crosses the bus between CPU and GPU.'}
          </Translated>
        ),
        why: (
          <Translated vi="Host đã chép nó sang device một lần, trước khi khởi chạy. Kernel không bao giờ chạm vào bộ nhớ host.">
            {'The host copied it to the device once, before the launch. The kernel never touches host memory.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Trong bộ nhớ toàn cục, ngoài chip; mỗi lần truy cập đều đi ra qua các bộ điều khiển bộ nhớ.">
            {'In global memory, off the chip; every access goes out through the memory controllers.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Trong thanh ghi của các luồng đọc nó, được nạp sẵn khi kernel bắt đầu.">
            {'In the registers of the threads that read it, filled when the kernel starts.'}
          </Translated>
        ),
        why: (
          <Translated vi="Một giá trị vào thanh ghi sau khi đã được nạp về. Bản thân mảng nằm ở nơi khác.">
            {'A value lands in a register after it has been fetched. The array itself is elsewhere.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Bộ nhớ toàn cục là thanh rộng ở đáy hình: một kho chung cho cả thiết bị, được truy cập qua các bộ điều khiển
            bộ nhớ. So với phép tính số học trên một thanh ghi, một lần truy cập nó mất nhiều thời gian, và đó chính là thao
            tác mà phần 4.6 xoay quanh: một SM nên làm gì với một luồng đang chờ một lần truy cập như vậy?
          </p>
        }
      >
        <p>
          Global memory is the wide bar at the bottom of the picture: one pool for the whole device, reached through the
          memory controllers. Compared with arithmetic on a register, an access to it takes a long time, and it is the
          operation section 4.6 is built around: what should an SM do with a thread that is waiting for one?
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.1, p. 68 and §4.6, p. 83',
    vi: { source: 'ch. 4 §4.1, p. 68 và §4.6, p. 83' },
  },
};

export function S41() {
  const q = questionsS41;
  const book = BOOK_DEVICE;
  return (
    <>
      <Translated
        vi={
          <p>
            Nhìn từ một kernel, GPU là một mảng các <b>streaming multiprocessor</b>, gọi tắt là SM. Mỗi SM là một tập các
            lõi (core) đơn giản dùng chung một khối logic điều khiển và một kho bộ nhớ trên chip. Sách gọi các lõi này là{' '}
            <b>streaming processor</b>. Bạn cũng sẽ gặp chúng dưới tên "CUDA core"; sách tránh cái tên đó để không ai nhầm
            chúng với lõi CPU, và phần 4.4 cho thấy chúng khác nhau đến mức nào.
          </p>
        }
      >
        <p>
          Seen from a kernel, a GPU is an array of <b>streaming multiprocessors</b>, SMs for short. Each SM is a set of
          simple cores that share one piece of control logic and one pool of memory on the chip. The book calls the cores{' '}
          <b>streaming processors</b>. You will also meet them as "CUDA cores"; the book avoids that name so nobody takes
          them for CPU cores, and section 4.4 shows how different they are.
        </p>
      </Translated>
      <Source section="4.1" pages="68" figs={tr('Fig. 4.1, redrawn in the explorer below', 'Fig. 4.1, được vẽ lại trong trình khám phá bên dưới')} />
      <Translated
        vi={
          <p>
            Bên ngoài các SM là <b>bộ nhớ toàn cục (global memory)</b> của thiết bị: hàng gigabyte DRAM mà mọi luồng của
            mọi lưới đều truy cập được, qua nhiều bộ điều khiển bộ nhớ cùng nhau tạo nên băng thông của nó. GPU đời cũ dùng
            bộ nhớ GDDR; từ thế hệ Pascal của NVIDIA, một GPU có thể dùng một trong các biến thể HBM, đóng gói cùng chip
            GPU. Từ đây trở đi sách gọi tất cả là DRAM, và khóa học này cũng vậy. Dùng nó cho tốt là nội dung chương 6.
          </p>
        }
      >
        <p>
          Outside the SMs sits the device's <b>global memory</b>: gigabytes of DRAM that every thread of every grid can
          reach, through several memory controllers that together give it its bandwidth. Older GPUs used GDDR memory;
          since NVIDIA's Pascal generation a GPU may use one of the HBM variants, packaged with the GPU chip. The book
          calls all of it DRAM from here on, and so does this course. Using it well is chapter 6.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch04-facts.md, A4)', '2026-11-20 (notes/today/ch04-facts.md, mục A4)')}>
        <Translated vi={<p>Trên thiết bị ví dụ của sách, H100, HBM3 chỉ có ở biến thể SXM5; biến thể PCIe dùng HBM2e.</p>}>
          <p>On the book's example device, the H100, HBM3 is on the SXM5 variant only; the PCIe variant has HBM2e.</p>
        </Translated>
      </Today>

      <h2><Translated vi="Ví dụ của sách">The book's example</Translated></h2>
      <Translated
        vi={
          <p>
            Sách gắn số liệu vào hình bằng một thiết bị: {book.units} SM, mỗi SM {book.coresPerUnit} streaming processor,
            tổng cộng {fmt(totalCores(book)!, 6)}. Từ thế hệ Hopper, các SM còn được gom thành{' '}
            <b>GPU processing cluster</b> (GPC), thiết bị này có {book.clusters} cụm. Hãy coi mỗi con số này là thuộc tính
            của riêng thiết bị đó. Thiết bị của bạn có những con số khác, và trình khám phá bên dưới cho thấy nó sẽ cho bạn
            biết những con số nào.
          </p>
        }
      >
        <p>
          The book puts numbers on the picture with one device: {book.units} SMs of {book.coresPerUnit} streaming
          processors each, {fmt(totalCores(book)!, 6)} in all. Since the Hopper generation the SMs are also grouped into{' '}
          <b>GPU processing clusters</b> (GPCs), {book.clusters} on this device. Treat every one of these as a property of
          that one device. Yours has other numbers, and the explorer below shows which of them it will tell you.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch04-facts.md, A1, A2)', '2026-11-20 (notes/today/ch04-facts.md, mục A1, A2)')}>
        <Translated
          vi={
            <p>
              Con số 132 là của H100 SXM5; biến thể PCIe có {BOOK_DEVICE_TODAY.unitsOtherVariant}. Một tầng mà sách bỏ
              qua: TPC gồm {BOOK_DEVICE_TODAY.unitsPerTpc} SM; {BOOK_DEVICE_TODAY.tpcs} TPC tạo nên con số 132. Cách chúng
              chia ra 8 GPC không được công bố (unverified).
            </p>
          }
        >
          <p>
            The 132 is the H100 SXM5; the PCIe variant has {BOOK_DEVICE_TODAY.unitsOtherVariant}. A level the book leaves
            out, the TPC of {BOOK_DEVICE_TODAY.unitsPerTpc} SMs: {BOOK_DEVICE_TODAY.tpcs} TPCs make the 132. Their split
            over 8 GPCs is not published (unverified).
          </p>
        </Translated>
      </Today>
      <Predict
        id="ch04/4.1/p-resident"
        prompt={
          <Translated
            vi={
              <>
                Mỗi SM trong ví dụ của sách có {book.coresPerUnit} lõi. Bạn đoán một SM có thể có bao nhiêu luồng đang chạy
                dở cùng lúc?
              </>
            }
          >
            <>
              Each SM of the book's example has {book.coresPerUnit} cores. How many threads do you expect one SM can have
              in progress at once?
            </>
          </Translated>
        }
        unit={tr('threads', 'luồng')}
      >
        {(v) => (
          <>
            <p>
              <Translated vi={<>Ví dụ của sách: <b>{fmt(book.threadsPerUnit!, 6)}</b> luồng mỗi SM, {threadsPerCore(book)} luồng cho mỗi lõi (p. 85).{' '}</>}>
                The book's example: <b>{fmt(book.threadsPerUnit!, 6)}</b> threads per SM, {threadsPerCore(book)} for every
                core (p. 85).{' '}
              </Translated>
              {v === book.coresPerUnit
                ? tr('One thread per core is the natural guess, and it is how you would size a thread pool on a CPU.', 'Mỗi lõi một luồng là phỏng đoán tự nhiên, và đó là cách bạn định kích thước một thread pool trên CPU.')
                : v < book.threadsPerUnit!
                  ? tr('You expected fewer.', 'Bạn đã đoán ít hơn.')
                  : v === book.threadsPerUnit
                    ? tr('Exactly.', 'Chính xác.')
                    : tr('You expected even more.', 'Bạn đã đoán còn nhiều hơn.')}
            </p>
            <Translated
              vi={
                <p>
                  Một SM không thể thực thi {fmt(book.threadsPerUnit!, 6)} luồng trong cùng một chu kỳ: nó chỉ có{' '}
                  {book.coresPerUnit} lõi. Nó giữ tất cả chúng ở trạng thái đang chạy dở và mỗi lúc thực thi một phần, để
                  một luồng đang chờ bộ nhớ không làm kẹt gì cả. Trên toàn thiết bị, đó là {fmt(residentThreads(book)!, 6)}{' '}
                  luồng đang chạy dở trên {fmt(totalCores(book)!, 6)} lõi. <a href={href('ch04', '4.6')}>Phần 4.6</a> giải
                  thích vì sao nó được thiết kế như vậy.
                </p>
              }
            >
              <p>
                An SM cannot execute {fmt(book.threadsPerUnit!, 6)} threads in the same cycle: it has {book.coresPerUnit}{' '}
                cores. It keeps them all in progress and executes some of them at a time, so that a thread waiting for
                memory holds nothing up. Across the device that is {fmt(residentThreads(book)!, 6)} threads in progress on{' '}
                {fmt(totalCores(book)!, 6)} cores. <a href={href('ch04', '4.6')}>Section 4.6</a> explains why it is built
                this way.
              </p>
            </Translated>
          </>
        )}
      </Predict>

      <h2><Translated vi="Một thiết bị sẽ cho bạn biết gì">What a device will tell you</Translated></h2>
      <Translated
        vi={
          <p>
            Một kernel muốn chạy tốt trên nhiều thiết bị phải tự định kích thước theo những gì thiết bị báo cáo. Vì vậy,
            điều quan trọng là những kích thước nào trong hình này được báo cáo. Hãy chọn một backend và một thiết bị, rồi
            chọn một tầng để xem phần còn lại của chương đặt gì ở đó.
          </p>
        }
      >
        <p>
          A kernel that is to run well on more than one device has to size itself from what the device reports. So it
          matters which sizes of this picture are reported at all. Pick a backend and a device, then select a level to see
          what the rest of the chapter places there.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Đây là một bảng tên và số lượng, không phải mô hình phần cứng. Nó chỉ biết những gì sách nêu về ví dụ của
              sách, những gì báo cáo thiết bị của bạn chứa, và một mô tả từ tài liệu của AMD.
            </p>
          }
        >
          <p>
            This is a table of names and counts, not a model of hardware. It knows only what the book states about its
            example, what your device's report contains, and one description from AMD's documentation.
          </p>
        </Translated>
      </Simplification>
      <LevelExplorer />

      <h2><Translated vi="Ba hệ quả">Three consequences</Translated></h2>
      <Translated vi={<p>Phần còn lại của chương mở ra ba điều mà hình này đã ngầm chứa.</p>}>
        <p>The rest of the chapter unpacks three things this picture already implies.</p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>
              <b>Một khối đi trọn vẹn tới một SM.</b> Bộ nhớ trên chip và logic điều khiển thuộc về một SM, nên các luồng
              cần chia sẻ bộ nhớ hay chờ nhau phải nằm trên cùng một SM (
              <a href={href('ch04', '4.2')}>4.2</a>, <a href={href('ch04', '4.3')}>4.3</a>).
            </li>
            <li>
              <b>Các lõi dùng chung khối điều khiển, nên luồng chạy theo nhóm.</b> Một lệnh được nạp về điều khiển nhiều
              lõi cùng lúc. Các nhóm đó là warp (<a href={href('ch04', '4.4')}>4.4</a>), và một nhóm mà các luồng muốn
              những lệnh khác nhau sẽ phải trả giá (<a href={href('ch04', '4.5')}>4.5</a>).
            </li>
            <li>
              <b>Một SM giữ nhiều luồng hơn số lõi của nó.</b> Đó là điều cho phép nó tiếp tục tính toán trong khi các luồng
              chờ bộ nhớ toàn cục (<a href={href('ch04', '4.6')}>4.6</a>), và số luồng nó giữ được là một ngân sách mà
              kernel của bạn tiêu (<a href={href('ch04', '4.7')}>4.7</a>).
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            <b>A block goes whole to one SM.</b> The on-chip memory and the control logic belong to an SM, so threads
            that are to share memory or wait for each other have to be on the same one (
            <a href={href('ch04', '4.2')}>4.2</a>, <a href={href('ch04', '4.3')}>4.3</a>).
          </li>
          <li>
            <b>Cores share control, so threads run in groups.</b> One fetched instruction drives many cores at once. The
            groups are warps (<a href={href('ch04', '4.4')}>4.4</a>), and a group whose threads want different
            instructions pays for it (<a href={href('ch04', '4.5')}>4.5</a>).
          </li>
          <li>
            <b>An SM holds more threads than it has cores.</b> That is what lets it keep computing while threads wait on
            global memory (<a href={href('ch04', '4.6')}>4.6</a>), and how many it can hold is a budget your kernel
            spends (<a href={href('ch04', '4.7')}>4.7</a>).
          </li>
        </ul>
      </Translated>
      <Ask q={q['ch04/4.1/q-cores']} />
      <Ask q={q['ch04/4.1/q-global']} />

      <h2><Translated vi="Trên các backend khác">On other backends</Translated></h2>
      <Translated
        vi={
          <p>
            AMD mô tả trong tài liệu một hình có cùng dạng với những cái tên khác: shader engine chứa shader array, shader
            array chứa <b>compute unit</b> (CU), đơn vị mà một khối chạy trên đó. Với dòng CDNA, AMD mô tả một CU gồm bốn bộ
            xử lý SIMD, mỗi bộ 16 lane. Trong HIP, trường <code>multiProcessorCount</code> tồn tại với cùng cách viết và
            đếm số CU, hoặc một nửa số đó khi thiết bị chạy ở chế độ workgroup-processor.
          </p>
        }
      >
        <p>
          AMD documents a picture of the same shape with other names: shader engines contain shader arrays, which contain{' '}
          <b>compute units</b> (CUs), the units a block runs on. For its CDNA family AMD describes a CU as four SIMD
          processors of 16 lanes each. Under HIP the field <code>multiProcessorCount</code> exists with the same spelling
          and counts CUs, or half of them when the device runs in workgroup-processor mode.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ nói về phần cứng NVIDIA. Thuật ngữ của AMD và Apple lấy từ tài liệu của họ
              (notes/today/ch04-facts.md, B1, C1). Việc đặt SM cạnh CU, và processing block cạnh SIMD, là phép so sánh của
              chúng tôi, và nó chỉ gần đúng: một SIMD 16 lane cần bốn chu kỳ cho một lệnh của một wavefront 64 luồng.
            </p>
          }
        >
          <p>
            The book covers NVIDIA hardware only. AMD's and Apple's terms are from their documentation
            (notes/today/ch04-facts.md, B1, C1). Setting SM beside CU, and processing block beside SIMD, is our
            comparison, and it is loose: a 16-lane SIMD takes four cycles for one instruction of a 64-thread wavefront.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Metal không mô tả đơn vị nào như vậy. Một <code>MTLDevice</code> có tên, các giới hạn bộ nhớ và kích thước
            threadgroup tối đa, nhưng không có số lõi hay số compute unit. Trên Metal, bạn định kích thước một lần khởi chạy
            từ các thuộc tính của kernel đã biên dịch, đó là nội dung <a href={href('ch04', '4.8')}>phần 4.8</a>.
          </p>
        }
      >
        <p>
          Metal documents no such unit. An <code>MTLDevice</code> has a name, memory limits and a maximum threadgroup
          size, and no count of cores or compute units. On Metal you size a launch from properties of the compiled
          kernel, which is <a href={href('ch04', '4.8')}>section 4.8</a>.
        </p>
      </Translated>
      <Ask q={q['ch04/4.1/q-mpcount']} />

      <h2><Translated vi="Từ dấu nhắc Python">From a Python prompt</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn không cần C++ để xem phía thiết bị của mình trong chuyện này. Trên một bản PyTorch dựng cho CUDA,{' '}
            <code>torch.cuda.get_device_properties(0).multi_processor_count</code> in ra số SM, và cùng đối tượng đó có{' '}
            <code>max_threads_per_multi_processor</code> và <code>warp_size</code>. Đó là một binding trực tiếp của cấu trúc
            mà phần 4.8 truy vấn từ C++.
          </p>
        }
      >
        <p>
          You do not need C++ to see your own device's side of this. On a CUDA build of PyTorch,{' '}
          <code>torch.cuda.get_device_properties(0).multi_processor_count</code> prints the SM count, and the same object
          has <code>max_threads_per_multi_processor</code> and <code>warp_size</code>. It is a direct binding of the
          structure section 4.8 queries from C++.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Của chúng tôi, đọc trong mã nguồn PyTorch ngày 2026-11-20 (notes/today/ch04-facts.md, D4). Khóa học này chưa
              chạy nó trên máy CUDA: unverified.
            </p>
          }
        >
          <p>
            Ours, read in PyTorch's source on 2026-11-20 (notes/today/ch04-facts.md, D4). Not run on a CUDA machine by
            this course: unverified.
          </p>
        </Translated>
      </Beyond>
    </>
  );
}
